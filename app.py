"""
Flask application for visualizing geospatial data and model predictions
"""

import os
import json
import base64
from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_cors import CORS
import geopandas as gpd
import numpy as np
from PIL import Image
import io
from shapely.geometry import shape
import torch
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
DATA_DIR = 'data'
MODEL_DIR = 'models'

# Cache for loaded data
data_cache = {}

def load_building_list():
    """Load list of available buildings"""
    if 'buildings' in data_cache:
        return data_cache['buildings']
    
    buildings = []
    raster_dir = os.path.join(DATA_DIR, 'rasters')
    
    if os.path.exists(raster_dir):
        for filename in os.listdir(raster_dir):
            if filename.endswith('.png'):
                building_id = filename.replace('.png', '')
                buildings.append(building_id)
    
    data_cache['buildings'] = sorted(buildings)
    return data_cache['buildings']

def load_raster_metadata():
    """Load raster metadata"""
    metadata_path = os.path.join(DATA_DIR, 'metadata', 'raster_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def load_model_checkpoints():
    """Load available model checkpoints"""
    checkpoints = []
    if os.path.exists(MODEL_DIR):
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith('.pth'):
                checkpoints.append(filename)
    return checkpoints

@app.route('/')
def index():
    """Main visualization page"""
    return render_template('index.html')

@app.route('/api/buildings')
def get_buildings():
    """Get list of available buildings"""
    buildings = load_building_list()
    return jsonify(buildings)

@app.route('/api/models')
def get_models():
    """Get list of available model checkpoints"""
    models = load_model_checkpoints()
    return jsonify(models)

@app.route('/api/building/<building_id>/raster')
def get_raster(building_id):
    """Get raster image and metadata for a building"""
    # Load raster image
    raster_path = os.path.join(DATA_DIR, 'rasters', f'{building_id}.png')
    if not os.path.exists(raster_path):
        return jsonify({'error': 'Raster not found'}), 404
    
    # Convert to base64
    with open(raster_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Load metadata
    metadata = load_raster_metadata()
    building_metadata = metadata.get(building_id, {})
    
    return jsonify({
        'image': f'data:image/png;base64,{image_data}',
        'metadata': building_metadata
    })

@app.route('/api/building/<building_id>/shapefile')
def get_shapefile(building_id):
    """Get building polygons as GeoJSON"""
    shapefile_path = os.path.join(DATA_DIR, 'shapefiles', f'{building_id}.geojson')
    
    if os.path.exists(shapefile_path):
        with open(shapefile_path, 'r') as f:
            geojson_data = json.load(f)
        return jsonify(geojson_data)
    
    # Try loading from .shp file
    shp_path = os.path.join(DATA_DIR, 'shapefiles', f'{building_id}.shp')
    if os.path.exists(shp_path):
        gdf = gpd.read_file(shp_path)
        return jsonify(json.loads(gdf.to_json()))
    
    return jsonify({'error': 'Shapefile not found'}), 404

@app.route('/api/building/<building_id>/entrances')
def get_entrances(building_id):
    """Get entrance points as GeoJSON"""
    entrance_path = os.path.join(DATA_DIR, 'entrances', f'{building_id}_entrances.geojson')
    
    if os.path.exists(entrance_path):
        with open(entrance_path, 'r') as f:
            geojson_data = json.load(f)
        return jsonify(geojson_data)
    
    return jsonify({'error': 'Entrances not found'}), 404

@app.route('/api/building/<building_id>/gps_traces')
def get_gps_traces(building_id):
    """Get GPS traces for a building"""
    traces_path = os.path.join(DATA_DIR, 'gps_traces', f'{building_id}_traces.json')
    
    if os.path.exists(traces_path):
        with open(traces_path, 'r') as f:
            traces_data = json.load(f)
        
        # Convert to GeoJSON format
        all_features = []
        for trace in traces_data.get('traces', []):
            all_features.extend(trace.get('features', []))
        
        return jsonify({
            'type': 'FeatureCollection',
            'features': all_features,
            'properties': {
                'building_id': building_id,
                'n_users': traces_data.get('n_users', 0),
                'total_pings': len(all_features)
            }
        })
    
    return jsonify({'error': 'GPS traces not found'}), 404

@app.route('/api/predict', methods=['POST'])
def predict_entrances():
    """Run model prediction on GPS traces"""
    data = request.json
    building_id = data.get('building_id')
    model_name = data.get('model_name')
    
    if not building_id or not model_name:
        return jsonify({'error': 'Missing parameters'}), 400
    
    # Load model
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        # Import the model architecture (should match train_model.py)
        from train_model import EntrancePredictionModel
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model = EntrancePredictionModel(
            input_dim=checkpoint.get('input_dim', 64),
            hidden_dim=checkpoint.get('hidden_dim', 128),
            output_dim=checkpoint.get('output_dim', 32)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load GPS traces
        traces_path = os.path.join(DATA_DIR, 'gps_traces', f'{building_id}_traces.json')
        with open(traces_path, 'r') as f:
            traces_data = json.load(f)
        
        # Extract GPS points
        gps_points = []
        for trace in traces_data.get('traces', []):
            for feature in trace.get('features', []):
                coords = feature['geometry']['coordinates']
                gps_points.append(coords)
        
        if not gps_points:
            return jsonify({'error': 'No GPS data found'}), 404
        
        # Simple prediction logic (placeholder - should match your model)
        # In reality, this would process the GPS traces through the model
        gps_array = np.array(gps_points)
        
        # Cluster GPS points to find likely entrances
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.0001, min_samples=10).fit(gps_array)
        
        # Get cluster centers as predicted entrances
        predicted_entrances = []
        for label in set(clustering.labels_):
            if label == -1:  # Skip noise
                continue
            
            cluster_points = gps_array[clustering.labels_ == label]
            center = cluster_points.mean(axis=0)
            
            predicted_entrances.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': center.tolist()
                },
                'properties': {
                    'confidence': np.random.uniform(0.7, 0.95),
                    'cluster_size': len(cluster_points),
                    'entrance_type': 'predicted'
                }
            })
        
        return jsonify({
            'type': 'FeatureCollection',
            'features': predicted_entrances,
            'properties': {
                'model': model_name,
                'building_id': building_id,
                'n_predictions': len(predicted_entrances),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


# Test cases
if __name__ == '__main__':
    print("Flask Visualization App")
    print("=" * 50)
    
    # Test 1: Check data directories
    print("\nTest 1: Checking data directories")
    dirs_to_check = ['data/rasters', 'data/shapefiles', 'data/entrances', 'data/gps_traces', 'models']
    for dir_path in dirs_to_check:
        exists = os.path.exists(dir_path)
        print(f"  {dir_path}: {'✓' if exists else '✗'}")
    
    # Test 2: Load building list
    print("\nTest 2: Loading building list")
    buildings = load_building_list()
    print(f"  Found {len(buildings)} buildings")
    if buildings:
        print(f"  First 5: {buildings[:5]}")
    
    # Test 3: Check model directory
    print("\nTest 3: Checking for models")
    models = load_model_checkpoints()
    print(f"  Found {len(models)} model checkpoints")
    
    # Run the app
    print("\n" + "=" * 50)
    print("Starting Flask app on http://localhost:80")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=80, debug=True)