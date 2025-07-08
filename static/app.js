// Global variables
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
});