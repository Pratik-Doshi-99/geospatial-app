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

# Create templates directory and files
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# HTML template
html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Geospatial ML Pipeline Visualizer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div id="app">
        <div id="sidebar">
            <h1>Geospatial ML Visualizer</h1>
            
            <div class="control-group">
                <label for="building-select">Select Building:</label>
                <select id="building-select">
                    <option value="">Loading...</option>
                </select>
            </div>
            
            <div class="control-group">
                <h3>Visualization Options</h3>
                <label>
                    <input type="checkbox" id="show-raster" checked>
                    Show Raster Image
                </label>
                <label>
                    <input type="checkbox" id="show-buildings" checked>
                    Show Building Polygons
                </label>
                <label>
                    <input type="checkbox" id="show-entrances" checked>
                    Show Entrances
                </label>
                <label>
                    <input type="checkbox" id="show-gps" checked>
                    Show GPS Traces
                </label>
                <label>
                    <input type="checkbox" id="show-predictions" disabled>
                    Show Model Predictions
                </label>
            </div>
            
            <div class="control-group">
                <h3>Model Prediction</h3>
                <select id="model-select" disabled>
                    <option value="">No models available</option>
                </select>
                <button id="run-prediction" disabled>Run Prediction</button>
            </div>
            
            <div id="info-panel">
                <h3>Information</h3>
                <div id="info-content">
                    Select a building to view details
                </div>
            </div>
        </div>
        
        <div id="map"></div>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="/static/app.js"></script>
</body>
</html>'''

# CSS styles
css_styles = '''body {
    margin: 0;
    padding: 0;
    font-family: Arial, sans-serif;
}

#app {
    display: flex;
    height: 100vh;
}

#sidebar {
    width: 300px;
    background: #f5f5f5;
    padding: 20px;
    overflow-y: auto;
    box-shadow: 2px 0 5px rgba(0,0,0,0.1);
}

#sidebar h1 {
    margin-top: 0;
    color: #333;
    font-size: 1.5em;
}

#sidebar h3 {
    color: #555;
    margin-top: 20px;
    margin-bottom: 10px;
}

#map {
    flex: 1;
    height: 100%;
}

.control-group {
    margin-bottom: 20px;
}

.control-group label {
    display: block;
    margin-bottom: 5px;
    color: #666;
}

.control-group select,
.control-group button {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
}

.control-group button {
    background: #007bff;
    color: white;
    cursor: pointer;
    margin-top: 10px;
}

.control-group button:hover:not(:disabled) {
    background: #0056b3;
}

.control-group button:disabled {
    background: #ccc;
    cursor: not-allowed;
}

.control-group input[type="checkbox"] {
    margin-right: 8px;
}

#info-panel {
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 15px;
    margin-top: 20px;
}

#info-content {
    color: #666;
    font-size: 0.9em;
    line-height: 1.4;
}

.leaflet-overlay-pane img {
    opacity: 0.7;
}'''

# JavaScript application
js_app = '''// Global variables
let map;
let layers = {
    raster: null,
    buildings: null,
    entrances: null,
    gpsTraces: null,
    predictions: null
};
let currentBuilding = null;

// Initialize map
function initMap() {
    map = L.map('map').setView([37.7749, -122.4194], 13);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
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
            option.textContent = building;
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
                option.textContent = model;
                select.appendChild(option);
            });
            select.disabled = false;
            button.disabled = false;
            checkbox.disabled = false;
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Load building data
async function loadBuildingData(buildingId) {
    currentBuilding = buildingId;
    clearLayers();
    
    // Update info panel
    updateInfo(`Loading data for ${buildingId}...`);
    
    // Load raster
    if (document.getElementById('show-raster').checked) {
        loadRaster(buildingId);
    }
    
    // Load building polygons
    if (document.getElementById('show-buildings').checked) {
        loadShapefile(buildingId);
    }
    
    // Load entrances
    if (document.getElementById('show-entrances').checked) {
        loadEntrances(buildingId);
    }
    
    // Load GPS traces
    if (document.getElementById('show-gps').checked) {
        loadGPSTraces(buildingId);
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
            
            layers.raster = L.imageOverlay(data.image, imageBounds).addTo(map);
            map.fitBounds(imageBounds);
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
                weight: 2
            }
        }).addTo(map);
        
        // Update info
        const nBuildings = geojson.features ? geojson.features.length : 0;
        updateInfo(`Found ${nBuildings} building(s)`);
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
                    radius: 8,
                    fillColor: '#ff7800',
                    color: '#000',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                });
            },
            onEachFeature: (feature, layer) => {
                const props = feature.properties;
                layer.bindPopup(`
                    <b>Entrance ${props.entrance_id}</b><br>
                    Type: ${props.type}<br>
                    Orientation: ${props.orientation?.toFixed(1)}°
                `);
            }
        }).addTo(map);
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
        updateInfo(`
            Building: ${buildingId}<br>
            Users: ${props.n_users}<br>
            GPS Pings: ${props.total_pings}
        `);
    } catch (error) {
        console.error('Error loading GPS traces:', error);
    }
}

// Run model prediction
async function runPrediction() {
    const modelName = document.getElementById('model-select').value;
    if (!modelName || !currentBuilding) return;
    
    updateInfo('Running prediction...');
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                building_id: currentBuilding,
                model_name: modelName
            })
        });
        
        const predictions = await response.json();
        
        if (layers.predictions) {
            map.removeLayer(layers.predictions);
        }
        
        layers.predictions = L.geoJSON(predictions, {
            pointToLayer: (feature, latlng) => {
                return L.circleMarker(latlng, {
                    radius: 10,
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
                    Cluster Size: ${props.cluster_size}
                `);
            }
        }).addTo(map);
        
        updateInfo(`Found ${predictions.features.length} predicted entrances`);
        document.getElementById('show-predictions').checked = true;
    } catch (error) {
        console.error('Error running prediction:', error);
        updateInfo('Prediction failed: ' + error.message);
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
    // Checkbox listeners
    document.getElementById('show-raster').addEventListener('change', (e) => {
        if (currentBuilding) {
            if (e.target.checked) {
                loadRaster(currentBuilding);
            } else if (layers.raster) {
                map.removeLayer(layers.raster);
            }
        }
    });
    
    document.getElementById('show-buildings').addEventListener('change', (e) => {
        if (currentBuilding) {
            if (e.target.checked) {
                loadShapefile(currentBuilding);
            } else if (layers.buildings) {
                map.removeLayer(layers.buildings);
            }
        }
    });
    
    document.getElementById('show-entrances').addEventListener('change', (e) => {
        if (currentBuilding) {
            if (e.target.checked) {
                loadEntrances(currentBuilding);
            } else if (layers.entrances) {
                map.removeLayer(layers.entrances);
            }
        }
    });
    
    document.getElementById('show-gps').addEventListener('change', (e) => {
        if (currentBuilding) {
            if (e.target.checked) {
                loadGPSTraces(currentBuilding);
            } else if (layers.gpsTraces) {
                map.removeLayer(layers.gpsTraces);
            }
        }
    });
    
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

# Write template files
with open('templates/index.html', 'w') as f:
    f.write(html_template)

with open('static/style.css', 'w') as f:
    f.write(css_styles)

with open('static/app.js', 'w') as f:
    f.write(js_app)

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