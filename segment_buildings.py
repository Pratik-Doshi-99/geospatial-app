"""
Run SAM segmentation on raster files and save as shapefiles
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import cv2
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Note: In production, you would use the actual SAM model
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# For this demo, we'll simulate SAM output

class BuildingSegmenter:
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # In production:
        # self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
        # self.sam.to(self.device)
        # self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    
    def simulate_sam_segmentation(self, image: np.ndarray) -> List[Dict]:
        """
        Simulate SAM segmentation output
        In production, this would use the actual SAM model
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection and morphological operations
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours (simulate building detection)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Generate masks for each contour
        masks = []
        h, w = image.shape[:2]
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 500:  # Filter small areas
                continue
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Simulate confidence score based on area and shape
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            confidence = min(0.5 + circularity * 0.3 + min(area / 50000, 0.2), 0.95)
            
            # Approximate polygon to reduce points
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            masks.append({
                'segmentation': mask > 0,
                'bbox': cv2.boundingRect(contour),
                'area': area,
                'predicted_iou': confidence,
                'stability_score': confidence * 0.9,
                'contour': approx
            })
        
        return masks
    
    def masks_to_polygons(self, masks: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """Convert segmentation masks to polygons with confidence scores"""
        polygons = []
        
        for mask_data in masks:
            if mask_data['predicted_iou'] < self.confidence_threshold:
                continue
            
            # Get contour points
            contour = mask_data['contour']
            if len(contour) < 3:
                continue
            
            # Convert to polygon coordinates
            points = [(float(p[0][0]), float(p[0][1])) for p in contour]
            
            try:
                polygon = Polygon(points)
                if polygon.is_valid and polygon.area > 0:
                    polygons.append({
                        'geometry': polygon,
                        'confidence': mask_data['predicted_iou'],
                        'area_pixels': mask_data['area'],
                        'bbox': mask_data['bbox']
                    })
            except:
                continue
        
        return polygons
    
    def pixel_to_geo_coords(self, pixel_coords: List[Tuple[float, float]], 
                           metadata: Dict) -> List[Tuple[float, float]]:
        """Convert pixel coordinates to geographic coordinates"""
        geo_coords = []
        
        bounds = metadata['bounds']
        width = metadata['width']
        height = metadata['height']
        
        for px, py in pixel_coords:
            # Normalize pixel coordinates
            norm_x = px / width
            norm_y = py / height
            
            # Convert to lat/lon
            lon = bounds['west'] + norm_x * (bounds['east'] - bounds['west'])
            lat = bounds['north'] - norm_y * (bounds['north'] - bounds['south'])
            
            geo_coords.append((lon, lat))  # GeoJSON uses (lon, lat) order
        
        return geo_coords
    
    def process_raster(self, raster_path: str, metadata: Dict) -> gpd.GeoDataFrame:
        """Process a single raster file and return building polygons"""
        # Load image
        image = np.array(Image.open(raster_path))
        
        # Run segmentation
        print(f"  Running segmentation on {image.shape}")
        masks = self.simulate_sam_segmentation(image)
        print(f"  Found {len(masks)} potential buildings")
        
        # Convert to polygons
        polygons = self.masks_to_polygons(masks, image.shape[:2])
        print(f"  {len(polygons)} buildings above confidence threshold")
        
        # Convert to geographic coordinates
        geo_polygons = []
        for poly_data in polygons:
            pixel_coords = list(poly_data['geometry'].exterior.coords)
            geo_coords = self.pixel_to_geo_coords(pixel_coords, metadata)
            
            try:
                geo_polygon = Polygon(geo_coords)
                if geo_polygon.is_valid:
                    geo_polygons.append({
                        'geometry': geo_polygon,
                        'confidence': poly_data['confidence'],
                        'area_m2': geo_polygon.area * 111000 * 111000,  # Rough conversion
                        'pixel_area': poly_data['area_pixels']
                    })
            except:
                continue
        
        # Create GeoDataFrame
        if geo_polygons:
            gdf = gpd.GeoDataFrame(geo_polygons, crs='EPSG:4326')
            return gdf
        else:
            return gpd.GeoDataFrame(columns=['geometry', 'confidence', 'area_m2', 'pixel_area'], 
                                   crs='EPSG:4326')

def main():
    """Process all raster files and create shapefiles"""
    # Create output directory
    os.makedirs('data/shapefiles', exist_ok=True)
    
    # Load metadata
    with open('data/metadata/raster_metadata.json', 'r') as f:
        raster_metadata = json.load(f)
    
    # Initialize segmenter
    segmenter = BuildingSegmenter(confidence_threshold=0.6)
    
    # Process statistics
    total_buildings = 0
    processed_files = 0
    
    # Process each raster
    for building_id, metadata in raster_metadata.items():
        raster_path = f'data/rasters/{building_id}.png'
        
        if not os.path.exists(raster_path):
            print(f"\nSkipping {building_id} - raster not found")
            continue
        
        print(f"\nProcessing {building_id}")
        
        try:
            # Process raster
            gdf = segmenter.process_raster(raster_path, metadata)
            
            if len(gdf) > 0:
                # Save shapefile
                shapefile_path = f'data/shapefiles/{building_id}.shp'
                gdf.to_file(shapefile_path)
                
                # Also save as GeoJSON for easier inspection
                geojson_path = f'data/shapefiles/{building_id}.geojson'
                gdf.to_file(geojson_path, driver='GeoJSON')
                
                print(f"  ✓ Saved {len(gdf)} buildings to shapefile")
                total_buildings += len(gdf)
            else:
                print(f"  ✗ No buildings found")
            
            processed_files += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Files processed: {processed_files}")
    print(f"Total buildings found: {total_buildings}")
    print(f"Average buildings per file: {total_buildings/processed_files:.1f}")

# Test cases
if __name__ == "__main__":
    # Test 1: Coordinate conversion
    print("Test 1: Pixel to geographic coordinate conversion")
    test_metadata = {
        'bounds': {
            'north': 37.7760,
            'south': 37.7740,
            'east': -122.4180,
            'west': -122.4200
        },
        'width': 512,
        'height': 512
    }
    
    segmenter = BuildingSegmenter()
    test_coords = [(0, 0), (256, 256), (512, 512)]
    for px, py in test_coords:
        geo = segmenter.pixel_to_geo_coords([(px, py)], test_metadata)[0]
        print(f"  Pixel ({px}, {py}) -> Geo ({geo[0]:.6f}, {geo[1]:.6f})")
    
    # Test 2: Simulate segmentation on test image
    print("\nTest 2: Simulate segmentation")
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    # Add some rectangular shapes
    cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(test_image, (200, 200), (400, 350), (255, 255, 255), -1)
    
    masks = segmenter.simulate_sam_segmentation(test_image)
    print(f"  Found {len(masks)} segments in test image")
    
    # Test 3: Process single test raster if it exists
    if os.path.exists('data/test/test_raster.png'):
        print("\nTest 3: Process test raster")
        with open('data/metadata/raster_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        if 'building_000' in metadata:
            test_gdf = segmenter.process_raster(
                'data/rasters/building_000.png', 
                metadata['building_000']
            )
            print(f"  Processed test raster: {len(test_gdf)} buildings found")
    
    # Run main pipeline
    print("\n" + "="*50)
    print("Running main pipeline...")
    print("="*50)
    main()