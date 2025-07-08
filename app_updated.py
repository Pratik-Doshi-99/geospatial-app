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

app = Flask(__name__)
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

# Enhanced HTML template
enhanced_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Geospatial ML Pipeline - Advanced Visualizer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="/static/style_enhanced.css">
</head>
<body>
    <div id="app">
        <div id="sidebar">
            <h1>🗺️ Geospatial ML Visualizer</h1>
            
            <div class="control-group">
                <label for="building-select">Select Building:</label>
                <select id="building-select">
                    <option value="">Loading...</option>
                </select>
            </div>
            
            <div class="control-group">
                <h3>📍 Visualization Layers</h3>
                <label class="checkbox-label">
                    <input type="checkbox" id="show-raster" checked>
                    <span>Satellite Imagery</span>
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" id="show-buildings" checked>
                    <span>Building Footprints</span>
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" id="show-entrances" checked>
                    <span>Ground Truth Entrances</span>
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" id="show-gps" checked>
                    <span>GPS Traces</span>
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" id="show-predictions" disabled>
                    <span>Model Predictions</span>
                </label>
            </div>
            
            <div class="control-group">
                <h3>🤖 Model Prediction</h3>
                <select id="model-select" disabled>
                    <option value="">No models available</option>
                </select>
                <button id="run-prediction" class="primary-button" disabled>
                    Run Prediction
                </button>
                <div id="model-info" class="info-box"></div>
            </div>
            
            <div id="info-panel" class="info-panel">
                <h3>📊 Statistics</h3>
                <div id="info-content">
                    Select a building to view details
                </div>
            </div>
            
            <div id="legend" class="legend">
                <h3>Legend</h3>
                <div class="legend-item">
                    <span class="legend-color" style="background: #3388ff;"></span>
                    Building Footprint
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: #ff7800;"></span>
                    Ground Truth Entrance
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: #00ff00;"></span>
                    Outdoor GPS
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: #ff0000;"></span>
                    Indoor GPS
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: #ff00ff;"></span>
                    Predicted Entrance
                </div>
            </div>
        </div>
        
        <div id="map-container">
            <div id="map"></div>
            <div id="loading-overlay" class="loading-overlay hidden">
                <div class="spinner"></div>
                <p>Processing...</p>
            </div>
        </div>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="/static/app_enhanced.js"></script>
</body>
</html>'''

# Enhanced CSS
enhanced_css = '''body {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f0f2f5;
}

#app {
    display: flex;
    height: 100vh;
}

#sidebar {
    width: 320px;
    background: white;
    padding: 20px;
    overflow-y: auto;
    box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    z-index: 1000;
}

#sidebar h1 {
    margin: 0 0 20px 0;
    color: #1a1a1a;
    font-size: 1.5em;
    display: flex;
    align-items: center;
    gap: 10px;
}

#sidebar h3 {
    color: #333;
    margin: 20px 0 10px 0;
    font-size: 1.1em;
    font-weight: 600;
}

#map-container {
    flex: 1;
    position: relative;
}

#map {
    width: 100%;
    height: 100%;
}

.control-group {
    margin-bottom: 25px;
}

.control-group label {
    display: block;
    margin-bottom: 8px;
    color: #555;
    font-size: 0.9em;
    font-weight: 500;
}

.control-group select {
    width: 100%;
    padding: 10px;
    border: 2px solid #e1e4e8;
    border-radius: 6px;
    background: white;
    font-size: 0.95em;
    transition: border-color 0.2s;
}

.control-group select:focus {
    outline: none;
    border-color: #007bff;
}

.primary-button {
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: 6px;
    background: #007bff;
    color: white;
    font-size: 1em;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 10px;
}

.primary-button:hover:not(:disabled) {
    background: #0056b3;
    transform: translateY(-1px);
}

.primary-button:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
}

.checkbox-label {
    display: flex;
    align-items: center;
    padding: 8px 0;
    cursor: pointer;
    transition: background 0.2s;
}

.checkbox-label:hover {
    background: #f8f9fa;
}

.checkbox-label input[type="checkbox"] {
    margin-right: 10px;
    width: 18px;
    height: 18px;
    cursor: pointer;
}

.checkbox-label span {
    color: #333;
    font-size: 0.95em;
}

.info-panel {
    background: #f8f9fa;
    border: 1px solid #e1e4e8;
    border-radius: 8px;
    padding: 15px;
    margin-top: 20px;
}

.info-box {
    margin-top: 10px;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 6px;
    font-size: 0.85em;
    color: #666;
}

#info-content {
    color: #555;
    font-size: 0.9em;
    line-height: 1.6;
}

.legend {
    margin-top: 20px;
    padding: 15px;
    background: #f8f9fa;
    border-radius: 8px;
}

.legend h3 {
    margin-top: 0;
    margin-bottom: 10px;
}

.legend-item {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    font-size: 0.85em;
    color: #555;
}

.legend-color {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    margin-right: 10px;
    border: 1px solid rgba(0,0,0,0.2);
}

.loading-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255,255,255,0.9);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 2000;
}

.loading-overlay.hidden {
    display: none;
}

.spinner {
    width: 40px;
    height: 40px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #007bff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.leaflet-overlay-pane img {
    opacity: 0.7;
}

/* Popup styles */
.leaflet-popup-content {
    margin: 12px;
    line-height: 1.5;
}

.leaflet-popup-content b {
    color: #333;
}'''

# Enhanced JavaScript
enhanced_js = '''// Global variables
let map;
let layers = {
    raster: null,
    buildings: null,
    entrances: null,
    gpsTraces: null,
    predictions: null
};
let currentBuilding = null;
let currentModel = null;

// Initialize map
function initMap() {
    map = L.map('map').setView([37.7749, -122.4194], 13);
    
    // Add dark tile layer for better contrast
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap contributors © CARTO',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);
}

// Show/hide loading overlay
function setLoading(loading) {
    const overlay = document.getElementById('loading-overlay');
    if (loading) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }
}

// Load building list
async function loadBuildings() {
    try {
        const response = await fetch('/api/buildings');
        const buildings = await response.json();
        
        const select = document.getElementById('building-select');
        select.innerHTML = '<option value="">Select a building...</option>';
        
        buildings.forEach(building => {
            const option = document.createElement('option');
            option.value = building;
            option.textContent = building.replace('building_', 'Building ');
            select.appendChild(option);
        });
        
        select.addEventListener('change', (e) => {
            if (e.target.value) {
                loadBuildingData(e.target.value);
            }
        });
    } catch (error) {
        console.error('Error loading buildings:', error);
    }
}

// Load model list
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const models = await response.json();
        
        const select = document.getElementById('model-select');
        const button = document.getElementById('run-prediction');
        const checkbox = document.getElementById('show-predictions');
        
        if (models.length > 0) {
            select.innerHTML = '<option value="">Select a model...</option>';
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model.replace('.pth', '').replace('_', ' ');
                select.appendChild(option);
            });
            
            select.disabled = false;
            select.addEventListener('change', async (e) => {
                currentModel = e.target.value;
                if (currentModel) {
                    button.disabled = false;
                    await loadModelInfo(currentModel);
                } else {
                    button.disabled = true;
                    document.getElementById('model-info').innerHTML = '';
                }
            });
            
            checkbox.disabled = false;
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Load model information
async function loadModelInfo(modelName) {
    try {
        const response = await fetch(`/api/model/${modelName}/info`);
        const info = await response.json();
        
        const infoHtml = `
            <strong>Model: ${info.name}</strong><br>
            Architecture: ${info.input_dim}→${info.hidden_dim}→${info.output_dim}<br>
            ${info.metrics ? `F1 Score: ${(info.metrics.f1_score * 100).toFixed(1)}%` : ''}
        `;
        
        document.getElementById('model-info').innerHTML = infoHtml;
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Load building data
async function loadBuildingData(buildingId) {
    currentBuilding = buildingId;
    clearLayers();
    setLoading(true);
    
    // Update info panel
    updateInfo(`Loading data for ${buildingId.replace('building_', 'Building ')}...`);
    
    try {
        // Load all data in parallel
        const promises = [];
        
        if (document.getElementById('show-raster').checked) {
            promises.push(loadRaster(buildingId));
        }
        
        if (document.getElementById('show-buildings').checked) {
            promises.push(loadShapefile(buildingId));
        }
        
        if (document.getElementById('show-entrances').checked) {
            promises.push(loadEntrances(buildingId));
        }
        
        if (document.getElementById('show-gps').checked) {
            promises.push(loadGPSTraces(buildingId));
        }
        
        await Promise.all(promises);
    } finally {
        setLoading(false);
    }
}

// Load raster image
async function loadRaster(buildingId) {
    try {
        const response = await fetch(`/api/building/${buildingId}/raster`);
        const data = await response.json();
        
        if (data.metadata && data.metadata.bounds) {
            const bounds = data.metadata.bounds;
            const imageBounds = [
                [bounds.south, bounds.west],
                [bounds.north, bounds.east]
            ];
            
            layers.raster = L.imageOverlay(data.image, imageBounds, {
                opacity: 0.7
            }).addTo(map);
            
            map.fitBounds(imageBounds, { padding: [50, 50] });
        }
    } catch (error) {
        console.error('Error loading raster:', error);
    }
}

// Load building shapefile
async function loadShapefile(buildingId) {
    try {
        const response = await fetch(`/api/building/${buildingId}/shapefile`);
        const geojson = await response.json();
        
        layers.buildings = L.geoJSON(geojson, {
            style: {
                fillColor: '#3388ff',
                fillOpacity: 0.3,
                color: '#0066ff',
                weight: 3
            },
            onEachFeature: (feature, layer) => {
                const props = feature.properties;
                layer.bindPopup(`
                    <b>Building</b><br>
                    Confidence: ${(props.confidence * 100).toFixed(1)}%<br>
                    Area: ${props.area_m2?.toFixed(0)} m²
                `);
            }
        }).addTo(map);
        
        // Update info
        const nBuildings = geojson.features ? geojson.features.length : 0;
        const totalArea = geojson.features?.reduce((sum, f) => 
            sum + (f.properties.area_m2 || 0), 0) || 0;
        
        updateInfo(`
            <strong>${buildingId.replace('building_', 'Building ')}</strong><br>
            Buildings detected: ${nBuildings}<br>
            Total area: ${totalArea.toFixed(0)} m²
        `);
    } catch (error) {
        console.error('Error loading shapefile:', error);
    }
}

// Load entrances
async function loadEntrances(buildingId) {
    try {
        const response = await fetch(`/api/building/${buildingId}/entrances`);
        const geojson = await response.json();
        
        layers.entrances = L.geoJSON(geojson, {
            pointToLayer: (feature, latlng) => {
                return L.circleMarker(latlng, {
                    radius: 10,
                    fillColor: '#ff7800',
                    color: '#000',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                });
            },
            onEachFeature: (feature, layer) => {
                const props = feature.properties;
                layer.bindPopup(`
                    <b>Ground Truth Entrance ${props.entrance_id}</b><br>
                    Type: ${props.type}<br>
                    Orientation: ${props.orientation?.toFixed(1)}°
                `);
            }
        }).addTo(map);
        
        const nEntrances = geojson.features?.length || 0;
        const currentInfo = document.getElementById('info-content').innerHTML;
        updateInfo(currentInfo + `<br>Ground truth entrances: ${nEntrances}`);
    } catch (error) {
        console.error('Error loading entrances:', error);
    }
}

// Load GPS traces
async function loadGPSTraces(buildingId) {
    try {
        const response = await fetch(`/api/building/${buildingId}/gps_traces`);
        const geojson = await response.json();
        
        layers.gpsTraces = L.geoJSON(geojson, {
            pointToLayer: (feature, latlng) => {
                const isIndoor = feature.properties.is_indoor;
                return L.circleMarker(latlng, {
                    radius: 3,
                    fillColor: isIndoor ? '#ff0000' : '#00ff00',
                    color: 'none',
                    fillOpacity: 0.4
                });
            }
        }).addTo(map);
        
        // Update info
        const props = geojson.properties;
        const currentInfo = document.getElementById('info-content').innerHTML;
        updateInfo(currentInfo + `<br>GPS traces: ${props.n_users} users, ${props.total_pings} pings`);
    } catch (error) {
        console.error('Error loading GPS traces:', error);
    }
}

// Run model prediction
async function runPrediction() {
    if (!currentModel || !currentBuilding) return;
    
    setLoading(true);
    updateInfo(document.getElementById('info-content').innerHTML + '<br><br>Running prediction...');
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                building_id: currentBuilding,
                model_name: currentModel
            })
        });
        
        if (!response.ok) {
            throw new Error(`Prediction failed: ${response.statusText}`);
        }
        
        const predictions = await response.json();
        
        // Remove existing predictions
        if (layers.predictions) {
            map.removeLayer(layers.predictions);
        }
        
        // Add new predictions
        layers.predictions = L.geoJSON(predictions, {
            pointToLayer: (feature, latlng) => {
                return L.circleMarker(latlng, {
                    radius: 12,
                    fillColor: '#ff00ff',
                    color: '#000',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.7
                });
            },
            onEachFeature: (feature, layer) => {
                const props = feature.properties;
                layer.bindPopup(`
                    <b>Predicted Entrance</b><br>
                    Confidence: ${(props.confidence * 100).toFixed(1)}%<br>
                    Model: ${props.model}
                `);
            }
        }).addTo(map);
        
        // Update UI
        document.getElementById('show-predictions').checked = true;
        document.getElementById('show-predictions').disabled = false;
        
        const currentInfo = document.getElementById('info-content').innerHTML;
        updateInfo(currentInfo + `<br>Model predictions: ${predictions.features.length} entrances`);
        
    } catch (error) {
        console.error('Error running prediction:', error);
        alert('Prediction failed: ' + error.message);
    } finally {
        setLoading(false);
    }
}

// Clear all layers
function clearLayers() {
    Object.values(layers).forEach(layer => {
        if (layer && map.hasLayer(layer)) {
            map.removeLayer(layer);
        }
    });
}

// Update info panel
function updateInfo(content) {
    document.getElementById('info-content').innerHTML = content;
}

// Setup event listeners
function setupEventListeners() {
    // Layer visibility toggles
    const layerControls = [
        { id: 'show-raster', layer: 'raster', loader: loadRaster },
        { id: 'show-buildings', layer: 'buildings', loader: loadShapefile },
        { id: 'show-entrances', layer: 'entrances', loader: loadEntrances },
        { id: 'show-gps', layer: 'gpsTraces', loader: loadGPSTraces }
    ];
    
    layerControls.forEach(control => {
        document.getElementById(control.id).addEventListener('change', async (e) => {
            if (currentBuilding) {
                if (e.target.checked) {
                    setLoading(true);
                    await control.loader(currentBuilding);
                    setLoading(false);
                } else if (layers[control.layer]) {
                    map.removeLayer(layers[control.layer]);
                }
            }
        });
    });
    
    // Predictions toggle
    document.getElementById('show-predictions').addEventListener('change', (e) => {
        if (layers.predictions) {
            if (e.target.checked) {
                layers.predictions.addTo(map);
            } else {
                map.removeLayer(layers.predictions);
            }
        }
    });
    
    // Prediction button
    document.getElementById('run-prediction').addEventListener('click', runPrediction);
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    loadBuildings();
    loadModels();
    setupEventListeners();
});'''

# Write enhanced files
with open('templates/index_enhanced.html', 'w') as f:
    f.write(enhanced_html)

with open('static/style_enhanced.css', 'w') as f:
    f.write(enhanced_css)

with open('static/app_enhanced.js', 'w') as f:
    f.write(enhanced_js)

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
    models = get_models().json if hasattr(get_models(), 'json') else []
    if models:
        for model in models:
            print(f"  - {model}")
    else:
        print("  No models found. Run train_model.py first.")
    
    print("\n" + "=" * 50)
    print("Starting Flask app on http://localhost:80")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=80, debug=True)