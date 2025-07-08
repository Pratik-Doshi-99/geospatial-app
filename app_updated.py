"""
Enhanced Flask application with model prediction capabilities
"""

import os
import json
import base64
import numpy as np
from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_cors import CORS
import geopandas as gpd
from PIL import Image
import io
import torch
import joblib
from shapely.geometry import Point
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import model architecture
from train_model import EntrancePredictionModel, extract_gps_features

app = Flask(__name__, template_folder='enhanced/templates', static_folder='enhanced/static')
CORS(app)

# Configuration
DATA_DIR = 'data'
MODEL_DIR = 'models'

# Cache for loaded models and data
cache = {
    'models': {},
    'scaler': None,
    'buildings': None
}

def load_model(model_name: str):
    """Load a trained model into memory"""
    if model_name in cache['models']:
        return cache['models'][model_name]
    
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize model
    model = EntrancePredictionModel(
        input_dim=checkpoint.get('input_dim', 64),
        hidden_dim=checkpoint.get('hidden_dim', 128),
        output_dim=checkpoint.get('output_dim', 32)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Cache the model
    cache['models'][model_name] = model
    
    return model

def load_scaler():
    """Load the feature scaler"""
    if cache['scaler'] is None:
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            cache['scaler'] = joblib.load(scaler_path)
        else:
            # Create dummy scaler if not found
            from sklearn.preprocessing import StandardScaler
            cache['scaler'] = StandardScaler()
    
    return cache['scaler']

def predict_entrances_from_gps(model, gps_traces, building_polygon):
    """Run model prediction on GPS traces"""
    # Extract features
    grid_size = int(np.sqrt(model.output_dim))
    features = extract_gps_features(gps_traces, grid_size=grid_size)
    
    # Normalize features
    scaler = load_scaler()
    if hasattr(scaler, 'mean_'):
        features = scaler.transform(features.reshape(1, -1))
    else:
        features = features.reshape(1, -1)
    
    # Run prediction
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features)
        predictions = model(features_tensor).numpy()[0]
    
    # Convert predictions to entrance locations
    predictions = predictions.reshape(grid_size, grid_size)
    
    # Get GPS bounds
    all_points = []
    for trace in gps_traces.get('traces', []):
        for feature in trace.get('features', []):
            coords = feature['geometry']['coordinates']
            all_points.append(coords)
    
    if not all_points:
        return []
    
    all_points = np.array(all_points)
    min_lon, min_lat = all_points.min(axis=0)
    max_lon, max_lat = all_points.max(axis=0)
    
    # Find peaks in prediction grid
    threshold = 0.5
    entrance_locations = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            if predictions[i, j] > threshold:
                # Convert grid position to coordinates
                lon = min_lon + (j + 0.5) / grid_size * (max_lon - min_lon)
                lat = min_lat + (i + 0.5) / grid_size * (max_lat - min_lat)
                
                entrance_locations.append({
                    'coordinates': [lon, lat],
                    'confidence': float(predictions[i, j]),
                    'grid_position': [i, j]
                })
    
    # Post-process: merge nearby predictions
    merged_entrances = []
    merge_distance = 0.0001  # degrees
    
    for entrance in sorted(entrance_locations, key=lambda x: x['confidence'], reverse=True):
        # Check if too close to existing entrance
        too_close = False
        for existing in merged_entrances:
            dist = np.sqrt(
                (entrance['coordinates'][0] - existing['coordinates'][0])**2 +
                (entrance['coordinates'][1] - existing['coordinates'][1])**2
            )
            if dist < merge_distance:
                too_close = True
                break
        
        if not too_close:
            merged_entrances.append(entrance)
    
    return merged_entrances

@app.route('/')
def index():
    """Enhanced visualization page"""
    return render_template('index_enhanced.html')

@app.route('/api/buildings')
def get_buildings():
    """Get list of available buildings"""
    if cache['buildings'] is None:
        buildings = []
        raster_dir = os.path.join(DATA_DIR, 'rasters')
        
        if os.path.exists(raster_dir):
            for filename in os.listdir(raster_dir):
                if filename.endswith('.png'):
                    building_id = filename.replace('.png', '')
                    buildings.append(building_id)
        
        cache['buildings'] = sorted(buildings)
    
    return jsonify(cache['buildings'])

@app.route('/api/models')
def get_models():
    """Get list of available model checkpoints"""
    models = []
    if os.path.exists(MODEL_DIR):
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith('.pth'):
                models.append(filename)
    return jsonify(models)

@app.route('/api/building/<building_id>/raster')
def get_raster(building_id):
    """Get raster image and metadata for a building"""
    raster_path = os.path.join(DATA_DIR, 'rasters', f'{building_id}.png')
    if not os.path.exists(raster_path):
        return jsonify({'error': 'Raster not found'}), 404
    
    with open(raster_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Load metadata
    metadata_path = os.path.join(DATA_DIR, 'metadata', 'raster_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
            building_metadata = all_metadata.get(building_id, {})
    else:
        building_metadata = {}
    
    return jsonify({
        'image': f'data:image/png;base64,{image_data}',
        'metadata': building_metadata
    })

@app.route('/api/building/<building_id>/shapefile')
def get_shapefile(building_id):
    """Get building polygons as GeoJSON"""
    geojson_path = os.path.join(DATA_DIR, 'shapefiles', f'{building_id}.geojson')
    
    if os.path.exists(geojson_path):
        with open(geojson_path, 'r') as f:
            return jsonify(json.load(f))
    
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
            return jsonify(json.load(f))
    
    return jsonify({'error': 'Entrances not found'}), 404

@app.route('/api/building/<building_id>/gps_traces')
def get_gps_traces(building_id):
    """Get GPS traces for a building"""
    traces_path = os.path.join(DATA_DIR, 'gps_traces', f'{building_id}_traces.json')
    
    if os.path.exists(traces_path):
        with open(traces_path, 'r') as f:
            traces_data = json.load(f)
        
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
    
    try:
        # Load model
        model = load_model(model_name)
        
        # Load GPS traces
        traces_path = os.path.join(DATA_DIR, 'gps_traces', f'{building_id}_traces.json')
        with open(traces_path, 'r') as f:
            traces_data = json.load(f)
        
        # Load building polygon
        shp_path = os.path.join(DATA_DIR, 'shapefiles', f'{building_id}.shp')
        if os.path.exists(shp_path):
            building_gdf = gpd.read_file(shp_path)
            building_polygon = building_gdf.iloc[0]['geometry']
        else:
            building_polygon = None
        
        # Run prediction
        predicted_entrances = predict_entrances_from_gps(model, traces_data, building_polygon)
        
        # Convert to GeoJSON features
        features = []
        for i, entrance in enumerate(predicted_entrances):
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': entrance['coordinates']
                },
                'properties': {
                    'confidence': entrance['confidence'],
                    'entrance_id': f'pred_{i}',
                    'type': 'predicted',
                    'model': model_name
                }
            })
        
        return jsonify({
            'type': 'FeatureCollection',
            'features': features,
            'properties': {
                'model': model_name,
                'building_id': building_id,
                'n_predictions': len(features),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/model/<model_name>/info')
def get_model_info(model_name):
    """Get information about a model"""
    model_path = os.path.join(MODEL_DIR, model_name)
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'name': model_name,
            'input_dim': checkpoint.get('input_dim', 'Unknown'),
            'hidden_dim': checkpoint.get('hidden_dim', 'Unknown'),
            'output_dim': checkpoint.get('output_dim', 'Unknown'),
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'train_loss': checkpoint.get('train_loss', 'Unknown'),
            'val_loss': checkpoint.get('val_loss', 'Unknown'),
            'timestamp': checkpoint.get('timestamp', 'Unknown')
        }
        
        if 'metrics' in checkpoint:
            info['metrics'] = checkpoint['metrics']
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': f'Failed to load model info: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


# Main execution
if __name__ == '__main__':
    print("Enhanced Flask Visualization App")
    print("=" * 50)
    
    # Check directories
    print("\nChecking directories:")
    for dir_name in ['data', 'models', 'templates', 'static']:
        exists = os.path.exists(dir_name)
        print(f"  {dir_name}: {'✓' if exists else '✗'}")
    
    # Check for trained models
    print("\nAvailable models:")
    models = []
    if os.path.exists(MODEL_DIR):
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith('.pth'):
                models.append(filename)
    if models:
        for model in models:
            print(f"  - {model}")
    else:
        print("  No models found. Run train_model.py first.")
    
    print("\n" + "=" * 50)
    print("Starting Flask app on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)