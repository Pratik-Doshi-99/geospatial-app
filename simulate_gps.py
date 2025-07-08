"""
Simulate GPS traces of users around buildings and entrances
"""

import os
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import uuid
import warnings
warnings.filterwarnings('ignore')

class GPSTraceSimulator:
    def __init__(self):
        # GPS parameters
        self.sampling_rate = 1.0  # seconds between pings
        self.gps_noise_std = 3.0  # meters of GPS noise
        self.walking_speed = 1.4  # m/s (average walking speed)
        
        # User behavior parameters
        self.entrance_attraction_radius = 20  # meters
        self.building_exit_probability = 0.7  # probability of exiting after entering
        self.random_walk_probability = 0.3   # probability of random walk vs direct path
        
        # Area to user count mapping
        self.area_to_users = [
            (0, 100, (2, 5)),         # Small: 2-5 users
            (100, 500, (5, 15)),      # Medium: 5-15 users
            (500, 1500, (10, 30)),    # Large: 10-30 users
            (1500, 5000, (20, 50)),   # Very large: 20-50 users
            (5000, float('inf'), (30, 80))  # Huge: 30-80 users
        ]
    
    def get_user_count(self, area_m2: float) -> int:
        """Determine number of users based on building area"""
        for min_area, max_area, (min_users, max_users) in self.area_to_users:
            if min_area <= area_m2 < max_area:
                return np.random.randint(min_users, max_users + 1)
        return 10  # Default
    
    def add_gps_noise(self, point: Point) -> Point:
        """Add realistic GPS noise to a point"""
        # Convert noise from meters to degrees (rough approximation)
        noise_degrees = self.gps_noise_std / 111000
        
        noise_x = np.random.normal(0, noise_degrees)
        noise_y = np.random.normal(0, noise_degrees)
        
        return Point(point.x + noise_x, point.y + noise_y)
    
    def generate_path_to_entrance(self, start: Point, entrance: Point, 
                                building_polygon: Polygon) -> List[Point]:
        """Generate a path from start point to entrance"""
        path_points = []
        
        # Decide path type
        if np.random.random() < self.random_walk_probability:
            # Random walk with bias toward entrance
            current = start
            steps = int(start.distance(entrance) * 111000 / self.walking_speed / self.sampling_rate)
            
            for i in range(steps):
                # Direction to entrance with random component
                dx = entrance.x - current.x
                dy = entrance.y - current.y
                
                # Normalize and add randomness
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    dx /= dist
                    dy /= dist
                
                # Add random walk component
                dx += np.random.normal(0, 0.3)
                dy += np.random.normal(0, 0.3)
                
                # Step size in degrees
                step_size = self.walking_speed * self.sampling_rate / 111000
                
                new_point = Point(
                    current.x + dx * step_size,
                    current.y + dy * step_size
                )
                
                path_points.append(self.add_gps_noise(new_point))
                current = new_point
        else:
            # Direct path
            line = LineString([start, entrance])
            total_distance = line.length
            steps = int(total_distance * 111000 / self.walking_speed / self.sampling_rate)
            
            for i in range(steps + 1):
                fraction = i / max(steps, 1)
                point = line.interpolate(fraction, normalized=True)
                path_points.append(self.add_gps_noise(Point(point.x, point.y)))
        
        return path_points
    
    def generate_indoor_trace(self, entrance: Point, building_polygon: Polygon, 
                            duration_seconds: int) -> List[Point]:
        """Generate GPS trace for user inside building"""
        trace_points = []
        
        # GPS is less reliable indoors - increase noise
        indoor_noise_multiplier = 2.0
        
        # Generate random walk inside building
        current = entrance
        steps = int(duration_seconds / self.sampling_rate)
        
        for _ in range(steps):
            # Random walk step
            angle = np.random.uniform(0, 2 * np.pi)
            step_size = self.walking_speed * self.sampling_rate / 111000
            
            new_x = current.x + np.cos(angle) * step_size
            new_y = current.y + np.sin(angle) * step_size
            new_point = Point(new_x, new_y)
            
            # Keep point inside building (with some tolerance for GPS error)
            if building_polygon.buffer(0.0001).contains(new_point):
                current = new_point
            
            # Add extra noise for indoor GPS
            noise_x = np.random.normal(0, self.gps_noise_std * indoor_noise_multiplier / 111000)
            noise_y = np.random.normal(0, self.gps_noise_std * indoor_noise_multiplier / 111000)
            
            noisy_point = Point(current.x + noise_x, current.y + noise_y)
            trace_points.append(noisy_point)
        
        return trace_points
    
    def generate_user_trace(self, user_id: str, building_id: str, 
                          building_polygon: Polygon, entrances: List[Dict]) -> Dict:
        """Generate complete GPS trace for one user"""
        # Select random entrance
        entrance_data = np.random.choice(entrances)
        entrance_point = entrance_data['geometry']
        
        # Generate starting point near building
        start_distance = np.random.uniform(50, 200) / 111000  # 50-200m away
        start_angle = np.random.uniform(0, 2 * np.pi)
        start_point = Point(
            entrance_point.x + np.cos(start_angle) * start_distance,
            entrance_point.y + np.sin(start_angle) * start_distance
        )
        
        # Generate approach path
        approach_path = self.generate_path_to_entrance(
            start_point, entrance_point, building_polygon
        )
        
        # Simulate time inside building
        indoor_duration = np.random.randint(60, 600)  # 1-10 minutes
        indoor_trace = self.generate_indoor_trace(
            entrance_point, building_polygon, indoor_duration
        )
        
        # Optionally generate exit path
        exit_trace = []
        if np.random.random() < self.building_exit_probability:
            # Choose exit entrance (might be different)
            exit_entrance = np.random.choice(entrances)['geometry']
            
            # Generate path away from building
            exit_distance = np.random.uniform(50, 150) / 111000
            exit_angle = np.random.uniform(0, 2 * np.pi)
            exit_point = Point(
                exit_entrance.x + np.cos(exit_angle) * exit_distance,
                exit_entrance.y + np.sin(exit_angle) * exit_distance
            )
            
            exit_trace = self.generate_path_to_entrance(
                exit_entrance, exit_point, building_polygon
            )
        
        # Combine all trace segments
        full_trace = approach_path + indoor_trace + exit_trace
        
        # Add timestamps
        start_time = datetime.now() - timedelta(hours=np.random.randint(0, 24))
        trace_data = []
        
        for i, point in enumerate(full_trace):
            timestamp = start_time + timedelta(seconds=i * self.sampling_rate)
            
            trace_data.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [point.x, point.y]
                },
                'properties': {
                    'user_id': user_id,
                    'building_id': building_id,
                    'timestamp': timestamp.isoformat(),
                    'ping_index': i,
                    'entrance_id': entrance_data['entrance_id'],
                    'is_indoor': i >= len(approach_path) and i < len(approach_path) + len(indoor_trace)
                }
            })
        
        return {
            'type': 'FeatureCollection',
            'features': trace_data,
            'properties': {
                'user_id': user_id,
                'building_id': building_id,
                'total_pings': len(trace_data),
                'duration_seconds': len(trace_data) * self.sampling_rate,
                'entrance_used': entrance_data['entrance_id']
            }
        }

def main():
    """Generate GPS traces for all buildings"""
    # Create output directory
    os.makedirs('data/gps_traces', exist_ok=True)
    
    # Initialize simulator
    simulator = GPSTraceSimulator()
    
    # Load entrance data
    entrance_files = [f for f in os.listdir('data/entrances') 
                     if f.endswith('_entrances.geojson')]
    
    print(f"Found {len(entrance_files)} buildings with entrances")
    
    # Statistics
    total_users = 0
    total_pings = 0
    
    # Process each building
    for entrance_file in entrance_files:
        building_id = entrance_file.replace('_entrances.geojson', '')
        
        print(f"\nProcessing {building_id}")
        
        try:
            # Load entrances
            entrance_gdf = gpd.read_file(f'data/entrances/{entrance_file}')
            
            # Load building polygon
            building_gdf = gpd.read_file(f'data/shapefiles/{building_id}.shp')
            
            if len(entrance_gdf) == 0 or len(building_gdf) == 0:
                print(f"  Skipping - no data found")
                continue
            
            # Get building area
            total_area = sum(building_gdf['area_m2'])
            
            # Determine number of users
            n_users = simulator.get_user_count(total_area)
            print(f"  Building area: {total_area:.0f}m²")
            print(f"  Generating traces for {n_users} users")
            
            # Generate traces for each user
            building_traces = []
            
            for i in range(n_users):
                user_id = str(uuid.uuid4())[:8]
                
                # Select a random building polygon if multiple
                building_idx = np.random.randint(0, len(building_gdf))
                building_polygon = building_gdf.iloc[building_idx]['geometry']
                
                # Get entrances for this building
                building_entrances = entrance_gdf[
                    entrance_gdf['building_id'].str.contains(f'b{building_idx}')
                ].to_dict('records')
                
                if not building_entrances:
                    building_entrances = entrance_gdf.to_dict('records')
                
                # Generate trace
                trace = simulator.generate_user_trace(
                    user_id, building_id, building_polygon, building_entrances
                )
                
                building_traces.append(trace)
                total_pings += len(trace['features'])
            
            # Save traces
            output_path = f'data/gps_traces/{building_id}_traces.json'
            with open(output_path, 'w') as f:
                json.dump({
                    'building_id': building_id,
                    'n_users': n_users,
                    'traces': building_traces
                }, f, indent=2)
            
            print(f"  ✓ Generated {n_users} user traces")
            total_users += n_users
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"GPS trace generation complete!")
    print(f"Total users simulated: {total_users}")
    print(f"Total GPS pings: {total_pings}")
    print(f"Average pings per user: {total_pings/max(total_users, 1):.0f}")

# Test cases
if __name__ == "__main__":
    # Test 1: User count determination
    print("Test 1: Area to user count mapping")
    simulator = GPSTraceSimulator()
    test_areas = [50, 300, 1000, 3000, 8000]
    for area in test_areas:
        counts = [simulator.get_user_count(area) for _ in range(5)]
        print(f"  {area}m² -> {min(counts)}-{max(counts)} users")
    
    # Test 2: GPS noise simulation
    print("\nTest 2: GPS noise simulation")
    test_point = Point(-122.4194, 37.7749)
    noisy_points = [simulator.add_gps_noise(test_point) for _ in range(10)]
    distances = [test_point.distance(p) * 111000 for p in noisy_points]
    print(f"  Average noise: {np.mean(distances):.1f}m")
    print(f"  Std deviation: {np.std(distances):.1f}m")
    
    # Test 3: Generate single user trace
    print("\nTest 3: Generate test user trace")
    test_building = Polygon([
        (-122.4190, 37.7750),
        (-122.4185, 37.7750),
        (-122.4185, 37.7755),
        (-122.4190, 37.7755)
    ])
    
    test_entrances = [{
        'geometry': Point(-122.4187, 37.7750),
        'entrance_id': 0,
        'type': 'main'
    }]
    
    test_trace = simulator.generate_user_trace(
        'test_user', 'test_building', test_building, test_entrances
    )
    
    print(f"  Generated trace with {len(test_trace['features'])} GPS pings")
    print(f"  Duration: {test_trace['properties']['duration_seconds']:.0f} seconds")
    
    # Run main pipeline
    print("\n" + "="*50)
    print("Running main pipeline...")
    print("="*50)
    main()