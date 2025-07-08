"""
Generate random entrances for buildings based on their polygon shapes
"""

import os
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class EntranceGenerator:
    def __init__(self):
        # Mapping of building area (m²) to number of entrances
        self.area_to_entrances = [
            (0, 100, 1),        # Small buildings: 1 entrance
            (100, 500, 2),      # Medium buildings: 2 entrances
            (500, 1500, 3),     # Large buildings: 3 entrances
            (1500, 5000, 4),    # Very large buildings: 4 entrances
            (5000, float('inf'), 5)  # Huge buildings: 5 entrances
        ]
        
        # Entrance placement rules
        self.min_entrance_spacing = 10  # meters
        self.entrance_offset = 0.5      # meters from building edge
    
    def get_entrance_count(self, area_m2: float) -> int:
        """Determine number of entrances based on building area"""
        for min_area, max_area, num_entrances in self.area_to_entrances:
            if min_area <= area_m2 < max_area:
                return num_entrances
        return 2  # Default
    
    def sample_points_on_boundary(self, polygon, n_points: int) -> List[Point]:
        """Sample evenly spaced points along polygon boundary"""
        boundary = polygon.exterior
        total_length = boundary.length
        
        # Calculate spacing between points
        if n_points == 1:
            # Single point at midpoint
            return [boundary.interpolate(0.5, normalized=True)]
        
        # Multiple points with even spacing
        points = []
        segment_length = total_length / n_points
        
        # Add some randomness to avoid perfect symmetry
        for i in range(n_points):
            # Base position with some random offset
            base_distance = (i + 0.5) * segment_length
            random_offset = np.random.uniform(-segment_length * 0.3, segment_length * 0.3)
            distance = (base_distance + random_offset) % total_length
            
            point = boundary.interpolate(distance)
            points.append(point)
        
        return points
    
    def ensure_minimum_spacing(self, points: List[Point], min_distance: float) -> List[Point]:
        """Ensure minimum spacing between entrance points"""
        if len(points) <= 1:
            return points
        
        # Iteratively remove points that are too close
        spaced_points = [points[0]]
        
        for point in points[1:]:
            too_close = False
            for existing_point in spaced_points:
                if point.distance(existing_point) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                spaced_points.append(point)
        
        return spaced_points
    
    def add_entrance_metadata(self, entrance_point: Point, building_polygon, 
                            entrance_id: int) -> Dict:
        """Add metadata to entrance point"""
        # Find the closest edge segment
        boundary = building_polygon.exterior
        coords = list(boundary.coords)
        
        min_dist = float('inf')
        closest_segment = None
        segment_angle = 0
        
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            dist = entrance_point.distance(segment)
            
            if dist < min_dist:
                min_dist = dist
                closest_segment = segment
                
                # Calculate angle of the segment
                dx = coords[i + 1][0] - coords[i][0]
                dy = coords[i + 1][1] - coords[i][1]
                segment_angle = np.arctan2(dy, dx)
        
        # Calculate entrance orientation (perpendicular to building edge)
        entrance_angle = segment_angle + np.pi / 2
        
        # Determine entrance type based on position
        entrance_types = ['main', 'secondary', 'service', 'emergency']
        entrance_type = entrance_types[min(entrance_id, len(entrance_types) - 1)]
        
        return {
            'entrance_id': entrance_id,
            'type': entrance_type,
            'orientation': np.degrees(entrance_angle),
            'edge_distance': min_dist,
            'geometry': entrance_point
        }
    
    def generate_entrances_for_building(self, building_polygon, 
                                      building_id: str, area_m2: float) -> List[Dict]:
        """Generate entrance points for a single building"""
        # Determine number of entrances
        n_entrances = self.get_entrance_count(area_m2)
        
        # Sample points on boundary
        entrance_points = self.sample_points_on_boundary(building_polygon, n_entrances)
        
        # Ensure minimum spacing
        min_spacing_degrees = self.min_entrance_spacing / 111000  # Rough conversion
        entrance_points = self.ensure_minimum_spacing(entrance_points, min_spacing_degrees)
        
        # Create entrance features
        entrances = []
        for i, point in enumerate(entrance_points):
            entrance_data = self.add_entrance_metadata(point, building_polygon, i)
            entrance_data['building_id'] = building_id
            entrance_data['building_area_m2'] = area_m2
            entrances.append(entrance_data)
        
        return entrances

def main():
    """Generate entrances for all buildings"""
    # Create output directory
    os.makedirs('data/entrances', exist_ok=True)
    
    # Initialize generator
    generator = EntranceGenerator()
    
    # Process statistics
    total_entrances = 0
    building_stats = []
    
    # Get all shapefiles
    shapefile_dir = 'data/shapefiles'
    shapefiles = [f for f in os.listdir(shapefile_dir) if f.endswith('.shp')]
    
    print(f"Found {len(shapefiles)} building shapefiles")
    
    # Process each building file
    for shapefile in shapefiles:
        building_id = shapefile.replace('.shp', '')
        shapefile_path = os.path.join(shapefile_dir, shapefile)
        
        print(f"\nProcessing {building_id}")
        
        try:
            # Load building polygons
            gdf = gpd.read_file(shapefile_path)
            
            if len(gdf) == 0:
                print(f"  No buildings in shapefile")
                continue
            
            # Generate entrances for each building in the file
            all_entrances = []
            
            for idx, row in gdf.iterrows():
                building_polygon = row['geometry']
                area_m2 = row.get('area_m2', building_polygon.area * 111000 * 111000)
                
                # Generate entrances
                entrances = generator.generate_entrances_for_building(
                    building_polygon, 
                    f"{building_id}_b{idx}", 
                    area_m2
                )
                
                all_entrances.extend(entrances)
                
                print(f"  Building {idx}: {area_m2:.0f}m² -> {len(entrances)} entrances")
            
            # Create GeoDataFrame for entrances
            if all_entrances:
                entrance_gdf = gpd.GeoDataFrame(all_entrances, crs='EPSG:4326')
                
                # Save as shapefile
                entrance_shapefile = f'data/entrances/{building_id}_entrances.shp'
                entrance_gdf.to_file(entrance_shapefile)
                
                # Save as GeoJSON
                entrance_geojson = f'data/entrances/{building_id}_entrances.geojson'
                entrance_gdf.to_file(entrance_geojson, driver='GeoJSON')
                
                print(f"  ✓ Generated {len(all_entrances)} total entrances")
                total_entrances += len(all_entrances)
                
                # Collect statistics
                building_stats.append({
                    'building_id': building_id,
                    'n_buildings': len(gdf),
                    'n_entrances': len(all_entrances),
                    'avg_entrances_per_building': len(all_entrances) / len(gdf)
                })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Save statistics
    stats_path = 'data/entrances/entrance_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'total_entrances': total_entrances,
            'buildings_processed': len(building_stats),
            'building_stats': building_stats
        }, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Entrance generation complete!")
    print(f"Total entrances generated: {total_entrances}")
    print(f"Buildings processed: {len(building_stats)}")

# Test cases
if __name__ == "__main__":
    # Test 1: Entrance count mapping
    print("Test 1: Area to entrance count mapping")
    generator = EntranceGenerator()
    test_areas = [50, 200, 800, 2000, 10000]
    for area in test_areas:
        count = generator.get_entrance_count(area)
        print(f"  {area}m² -> {count} entrances")
    
    # Test 2: Point sampling on boundary
    print("\nTest 2: Boundary point sampling")
    from shapely.geometry import Polygon
    
    # Create test polygon (square)
    test_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    points = generator.sample_points_on_boundary(test_polygon, 4)
    print(f"  Generated {len(points)} points on square boundary")
    for i, point in enumerate(points):
        print(f"    Point {i}: ({point.x:.2f}, {point.y:.2f})")
    
    # Test 3: Generate entrances for test building
    print("\nTest 3: Generate entrances for test building")
    test_building = Polygon([
        (-122.4190, 37.7750),
        (-122.4185, 37.7750),
        (-122.4185, 37.7755),
        (-122.4190, 37.7755)
    ])
    test_area = 1000  # m²
    
    entrances = generator.generate_entrances_for_building(
        test_building, "test_building", test_area
    )
    
    print(f"  Generated {len(entrances)} entrances:")
    for ent in entrances:
        print(f"    - {ent['type']} entrance at "
              f"({ent['geometry'].x:.6f}, {ent['geometry'].y:.6f}), "
              f"orientation: {ent['orientation']:.1f}°")
    
    # Run main pipeline
    print("\n" + "="*50)
    print("Running main pipeline...")
    print("="*50)
    main()