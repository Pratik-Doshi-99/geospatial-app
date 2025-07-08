"""
Generate raster files for buildings in SF Bay Area localities
"""

import os
import numpy as np
import requests
from PIL import Image
import math
import json
from typing import List, Tuple, Dict
import time

# SF Bay Area bounds
SF_BAY_BOUNDS = {
    'north': 37.9298,
    'south': 37.3594,
    'east': -121.6317,
    'west': -122.5194
}

# OSM tile server
TILE_SERVER = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ZOOM_LEVEL = 18  # Good for building-level detail

def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates"""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def tile_to_lat_lon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates back to lat/lon (NW corner)"""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon

def generate_locality_bounds(n_localities: int = 100) -> List[Dict[str, float]]:
    """Generate random locality bounding boxes within SF Bay Area"""
    localities = []
    
    # Define some key areas in SF Bay with higher density
    key_areas = [
        {'lat': 37.7749, 'lon': -122.4194, 'name': 'SF_Downtown'},
        {'lat': 37.8044, 'lon': -122.2712, 'name': 'Oakland'},
        {'lat': 37.3382, 'lon': -121.8863, 'name': 'San_Jose'},
        {'lat': 37.4419, 'lon': -122.1430, 'name': 'Palo_Alto'},
        {'lat': 37.7858, 'lon': -122.4064, 'name': 'SOMA'},
        {'lat': 37.8199, 'lon': -122.4783, 'name': 'Golden_Gate'},
        {'lat': 37.5485, 'lon': -122.3186, 'name': 'San_Mateo'},
        {'lat': 37.6879, 'lon': -122.4702, 'name': 'Daly_City'}
    ]
    
    # Generate localities around key areas
    for i in range(n_localities):
        # Pick a random key area
        area = key_areas[i % len(key_areas)]
        
        # Add some random offset (about 0.01 degrees ~ 1km)
        lat_offset = np.random.uniform(-0.01, 0.01)
        lon_offset = np.random.uniform(-0.01, 0.01)
        
        center_lat = area['lat'] + lat_offset
        center_lon = area['lon'] + lon_offset
        
        # Create a small bounding box (about 200-500m)
        box_size = np.random.uniform(0.002, 0.005)
        
        locality = {
            'id': f"building_{i:03d}",
            'north': center_lat + box_size/2,
            'south': center_lat - box_size/2,
            'east': center_lon + box_size/2,
            'west': center_lon - box_size/2,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'area_name': area['name']
        }
        localities.append(locality)
    
    return localities

def download_tile(x: int, y: int, zoom: int, max_retries: int = 3) -> Image.Image:
    """Download a single tile with retry logic"""
    url = TILE_SERVER.format(z=zoom, x=x, y=y)
    headers = {'User-Agent': 'GeospatialMLPipeline/1.0'}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return Image.open(requests.get(url, headers=headers, stream=True).raw)
            elif response.status_code == 429:  # Rate limited
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            print(f"Error downloading tile {x},{y}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    # Return blank tile if download fails
    return Image.new('RGB', (256, 256), color=(200, 200, 200))

def create_raster_from_bounds(bounds: Dict[str, float], output_path: str) -> Dict:
    """Create a raster image from bounding box coordinates"""
    # Get tile coordinates for bounds
    x_min, y_max = lat_lon_to_tile(bounds['north'], bounds['west'], ZOOM_LEVEL)
    x_max, y_min = lat_lon_to_tile(bounds['south'], bounds['east'], ZOOM_LEVEL)
    
    # Ensure we have at least one tile
    x_max = max(x_max, x_min)
    y_max = max(y_max, y_min)
    
    # Download and stitch tiles
    width = (x_max - x_min + 1) * 256
    height = (y_max - y_min + 1) * 256
    
    raster = Image.new('RGB', (width, height))
    
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile = download_tile(x, y, ZOOM_LEVEL)
            raster.paste(tile, ((x - x_min) * 256, (y - y_min) * 256))
            time.sleep(0.1)  # Rate limiting
    
    # Save raster
    raster.save(output_path)
    
    # Calculate geospatial metadata
    nw_lat, nw_lon = tile_to_lat_lon(x_min, y_min, ZOOM_LEVEL)
    se_lat, se_lon = tile_to_lat_lon(x_max + 1, y_max + 1, ZOOM_LEVEL)
    
    metadata = {
        'bounds': {
            'north': nw_lat,
            'south': se_lat,
            'east': se_lon,
            'west': nw_lon
        },
        'tile_bounds': {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max
        },
        'zoom': ZOOM_LEVEL,
        'width': width,
        'height': height,
        'pixel_per_degree_lat': height / (nw_lat - se_lat),
        'pixel_per_degree_lon': width / (se_lon - nw_lon)
    }
    
    return metadata

def main():
    """Generate raster files for SF Bay Area localities"""
    # Create output directories
    os.makedirs('data/rasters', exist_ok=True)
    os.makedirs('data/metadata', exist_ok=True)
    
    # Generate locality bounds
    print("Generating locality bounds...")
    localities = generate_locality_bounds(100)
    
    # Save locality information
    with open('data/metadata/localities.json', 'w') as f:
        json.dump(localities, f, indent=2)
    
    print(f"Generated {len(localities)} localities")
    
    # Generate raster for each locality
    all_metadata = {}
    
    for i, locality in enumerate(localities):
        print(f"\nProcessing {locality['id']} ({i+1}/{len(localities)})")
        print(f"  Area: {locality['area_name']}")
        print(f"  Center: ({locality['center_lat']:.4f}, {locality['center_lon']:.4f})")
        
        output_path = f"data/rasters/{locality['id']}.png"
        
        try:
            metadata = create_raster_from_bounds(locality, output_path)
            metadata['locality'] = locality
            all_metadata[locality['id']] = metadata
            print(f"  ✓ Saved raster: {output_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Save all metadata
    with open('data/metadata/raster_metadata.json', 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nCompleted! Generated {len(all_metadata)} raster files")

# Test cases
if __name__ == "__main__":
    # Test 1: Coordinate conversion
    print("Test 1: Coordinate conversion")
    lat, lon = 37.7749, -122.4194  # SF Downtown
    x, y = lat_lon_to_tile(lat, lon, ZOOM_LEVEL)
    print(f"  ({lat}, {lon}) -> Tile ({x}, {y})")
    lat2, lon2 = tile_to_lat_lon(x, y, ZOOM_LEVEL)
    print(f"  Tile ({x}, {y}) -> ({lat2:.4f}, {lon2:.4f})")
    
    # Test 2: Generate small set of localities
    print("\nTest 2: Generate test localities")
    test_localities = generate_locality_bounds(5)
    for loc in test_localities:
        print(f"  {loc['id']}: {loc['area_name']} - "
              f"({loc['center_lat']:.4f}, {loc['center_lon']:.4f})")
    
    # Test 3: Download single test raster
    print("\nTest 3: Generate single test raster")
    test_bounds = {
        'north': 37.7760,
        'south': 37.7740,
        'east': -122.4180,
        'west': -122.4200
    }
    os.makedirs('data/test', exist_ok=True)
    test_metadata = create_raster_from_bounds(test_bounds, 'data/test/test_raster.png')
    print(f"  Generated test raster with size: {test_metadata['width']}x{test_metadata['height']}")
    
    # Run main pipeline
    print("\n" + "="*50)
    print("Running main pipeline...")
    print("="*50)
    main()