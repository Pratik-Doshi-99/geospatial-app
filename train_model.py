"""
Train a deep learning model to predict building entrances from GPS traces
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class GPSTraceDataset(Dataset):
    """Dataset for GPS traces and entrance labels"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 building_ids: List[str], transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.building_ids = building_ids
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label, self.building_ids[idx]

class EntrancePredictionModel(nn.Module):
    """Neural network for entrance prediction from GPS traces"""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 32):
        super(EntrancePredictionModel, self).__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # Output layer for entrance heatmap
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output probability for each spatial bin
        )
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return output

def extract_gps_features(gps_traces: Dict, grid_size: int = 8) -> np.ndarray:
    """Extract features from GPS traces using spatial binning"""
    # Get all GPS points
    all_points = []
    for trace in gps_traces.get('traces', []):
        for feature in trace.get('features', []):
            coords = feature['geometry']['coordinates']
            all_points.append(coords)
    
    if not all_points:
        return np.zeros(grid_size * grid_size)
    
    all_points = np.array(all_points)
    
    # Get bounding box
    min_lon, min_lat = all_points.min(axis=0)
    max_lon, max_lat = all_points.max(axis=0)
    
    # Create spatial histogram
    hist, _, _ = np.histogram2d(
        all_points[:, 0], all_points[:, 1],
        bins=grid_size,
        range=[[min_lon, max_lon], [min_lat, max_lat]]
    )
    
    # Normalize
    features = hist.flatten()
    if features.sum() > 0:
        features = features / features.sum()
    
    return features

def create_entrance_labels(entrances_gdf: gpd.GeoDataFrame, 
                         gps_bounds: Tuple[float, float, float, float],
                         grid_size: int = 8) -> np.ndarray:
    """Create label grid for entrance locations"""
    min_lon, min_lat, max_lon, max_lat = gps_bounds
    
    # Initialize label grid
    labels = np.zeros((grid_size, grid_size))
    
    # Mark entrance locations
    for _, entrance in entrances_gdf.iterrows():
        point = entrance.geometry
        
        # Convert to grid coordinates
        x = int((point.x - min_lon) / (max_lon - min_lon) * (grid_size - 1))
        y = int((point.y - min_lat) / (max_lat - min_lat) * (grid_size - 1))
        
        # Ensure within bounds
        x = np.clip(x, 0, grid_size - 1)
        y = np.clip(y, 0, grid_size - 1)
        
        # Mark entrance with Gaussian blob
        for i in range(max(0, x-1), min(grid_size, x+2)):
            for j in range(max(0, y-1), min(grid_size, y+2)):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                labels[j, i] = max(labels[j, i], np.exp(-dist**2 / 0.5))
    
    return labels.flatten()

def prepare_dataset(data_dir: str = 'data') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and prepare the complete dataset"""
    features_list = []
    labels_list = []
    building_ids = []
    
    # Load GPS traces
    gps_dir = os.path.join(data_dir, 'gps_traces')
    entrance_dir = os.path.join(data_dir, 'entrances')
    
    trace_files = [f for f in os.listdir(gps_dir) if f.endswith('_traces.json')]
    
    print(f"Found {len(trace_files)} buildings with GPS traces")
    
    for trace_file in tqdm(trace_files, desc="Processing buildings"):
        building_id = trace_file.replace('_traces.json', '')
        
        try:
            # Load GPS traces
            with open(os.path.join(gps_dir, trace_file), 'r') as f:
                gps_data = json.load(f)
            
            # Extract features
            features = extract_gps_features(gps_data)
            
            # Load entrances
            entrance_file = f'{building_id}_entrances.geojson'
            entrance_path = os.path.join(entrance_dir, entrance_file)
            
            if os.path.exists(entrance_path):
                entrances_gdf = gpd.read_file(entrance_path)
                
                # Get GPS bounds
                all_points = []
                for trace in gps_data.get('traces', []):
                    for feature in trace.get('features', []):
                        coords = feature['geometry']['coordinates']
                        all_points.append(coords)
                
                if all_points:
                    all_points = np.array(all_points)
                    bounds = (
                        all_points[:, 0].min(),
                        all_points[:, 1].min(),
                        all_points[:, 0].max(),
                        all_points[:, 1].max()
                    )
                    
                    # Create labels
                    labels = create_entrance_labels(entrances_gdf, bounds)
                    
                    features_list.append(features)
                    labels_list.append(labels)
                    building_ids.append(building_id)
            
        except Exception as e:
            print(f"Error processing {building_id}: {e}")
    
    if not features_list:
        raise ValueError("No valid data found")
    
    return np.array(features_list), np.array(labels_list), building_ids

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 100, learning_rate: float = 0.001) -> Dict:
    """Train the entrance prediction model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, labels, _ in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels, _ in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
                'output_dim': model.output_dim
            }, 'models/best_model.pth')
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')
    
    return history

def visualize_training_history(history: Dict):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/training_history.png')
    plt.close()

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict:
    """Evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    ground_truths = []
    building_ids = []
    
    with torch.no_grad():
        for features, labels, bids in test_loader:
            features = features.to(device)
            outputs = model(features)
            
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(labels.numpy())
            building_ids.extend(bids)
    
    # Calculate metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Threshold predictions
    pred_binary = (predictions > 0.5).astype(float)
    
    # Calculate precision and recall
    true_positives = np.sum(pred_binary * ground_truths)
    false_positives = np.sum(pred_binary * (1 - ground_truths))
    false_negatives = np.sum((1 - pred_binary) * ground_truths)
    
    precision = true_positives / (true_positives + false_positives + 1e-7)
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    f1_score = 2 * precision * recall / (precision + recall + 1e-7)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'predictions': predictions,
        'ground_truths': ground_truths,
        'building_ids': building_ids
    }

def main():
    """Main training pipeline"""
    # Create model directory
    os.makedirs('models', exist_ok=True)
    
    print("Entrance Prediction Model Training")
    print("=" * 50)
    
    # Load and prepare dataset
    print("\nLoading dataset...")
    try:
        features, labels, building_ids = prepare_dataset()
        print(f"Loaded {len(features)} samples")
        print(f"Feature shape: {features.shape}")
        print(f"Label shape: {labels.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Split dataset
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        features, labels, building_ids, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train, y_train, ids_train, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Create datasets and loaders
    train_dataset = GPSTraceDataset(X_train, y_train, ids_train)
    val_dataset = GPSTraceDataset(X_val, y_val, ids_val)
    test_dataset = GPSTraceDataset(X_test, y_test, ids_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_dim = features.shape[1]
    output_dim = labels.shape[1]
    model = EntrancePredictionModel(input_dim=input_dim, output_dim=output_dim)
    
    print(f"\nModel architecture:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, train_loader, val_loader, epochs=100)
    
    # Visualize training
    visualize_training_history(history)
    
    # Load best model
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader)
    
    print(f"\nTest Results:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1 Score: {results['f1_score']:.4f}")
    
    # Save final model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = f'models/entrance_model_{timestamp}.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': model.hidden_dim,
        'output_dim': output_dim,
        'metrics': results,
        'timestamp': timestamp
    }, final_model_path)
    
    print(f"\nModel saved to: {final_model_path}")

# Test cases
if __name__ == "__main__":
    # Test 1: Feature extraction
    print("Test 1: Feature extraction")
    test_traces = {
        'traces': [{
            'features': [
                {'geometry': {'coordinates': [-122.4190, 37.7750]}},
                {'geometry': {'coordinates': [-122.4191, 37.7751]}},
                {'geometry': {'coordinates': [-122.4189, 37.7749]}}
            ]
        }]
    }
    features = extract_gps_features(test_traces, grid_size=4)
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature sum: {features.sum():.4f}")
    
    # Test 2: Model initialization
    print("\nTest 2: Model initialization")
    test_model = EntrancePredictionModel(input_dim=16, output_dim=16)
    test_input = torch.randn(5, 16)
    test_output = test_model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {test_output.shape}")
    print(f"  Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    # Test 3: Label creation
    print("\nTest 3: Label creation")
    from shapely.geometry import Point
    test_entrances = gpd.GeoDataFrame([
        {'geometry': Point(-122.4190, 37.7750)},
        {'geometry': Point(-122.4185, 37.7755)}
    ])
    test_bounds = (-122.4195, 37.7745, -122.4180, 37.7760)
    labels = create_entrance_labels(test_entrances, test_bounds, grid_size=4)
    print(f"  Label shape: {labels.shape}")
    print(f"  Non-zero labels: {(labels > 0).sum()}")
    
    # Run main training pipeline
    print("\n" + "="*50)
    print("Running main training pipeline...")
    print("="*50)
    main()