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
});