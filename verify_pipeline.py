"""
Verify that the entire geospatial ML pipeline is working correctly
"""

import os
import json
import sys
from datetime import datetime

def check_directory(path, expected_files=None, min_files=1):
    """Check if directory exists and contains expected files"""
    if not os.path.exists(path):
        return False, f"Directory {path} does not exist"
    
    files = os.listdir(path)
    if len(files) < min_files:
        return False, f"Directory {path} contains only {len(files)} files (expected at least {min_files})"
    
    if expected_files:
        missing = [f for f in expected_files if f not in files]
        if missing:
            return False, f"Missing files in {path}: {missing}"
    
    return True, f"✓ {path} ({len(files)} files)"

def check_file(path):
    """Check if file exists and is not empty"""
    if not os.path.exists(path):
        return False, f"File {path} does not exist"
    
    size = os.path.getsize(path)
    if size == 0:
        return False, f"File {path} is empty"
    
    return True, f"✓ {path} ({size:,} bytes)"

def verify_pipeline():
    """Run comprehensive pipeline verification"""
    print("="*60)
    print("Geospatial ML Pipeline Verification")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    all_good = True
    
    # 1. Check Python files
    print("\n1. Checking Python Scripts:")
    python_files = [
        'generate_raster.py',
        'segment_buildings.py',
        'generate_entrances.py',
        'simulate_gps.py',
        'train_model.py',
        'app.py',
        'app_updated.py'
    ]
    
    for py_file in python_files:
        status, msg = check_file(py_file)
        print(f"   {msg}")
        if not status:
            all_good = False
    
    # 2. Check data directories
    print("\n2. Checking Data Directories:")
    data_dirs = [
        ('data/rasters', None, 5),
        ('data/shapefiles', None, 5),
        ('data/entrances', None, 5),
        ('data/gps_traces', None, 5),
        ('data/metadata', ['localities.json', 'raster_metadata.json'], 2)
    ]
    
    for dir_info in data_dirs:
        status, msg = check_directory(*dir_info)
        print(f"   {msg}")
        if not status:
            all_good = False
    
    # 3. Check models
    print("\n3. Checking Trained Models:")
    status, msg = check_directory('models', None, 1)
    print(f"   {msg}")
    if not status:
        all_good = False
    else:
        # Check for specific model files
        model_files = ['best_model.pth', 'scaler.pkl', 'training_history.png']
        for model_file in model_files:
            path = os.path.join('models', model_file)
            if os.path.exists(path):
                print(f"   ✓ {model_file} found")
            else:
                print(f"   ✗ {model_file} not found")
    
    # 4. Check web app files
    print("\n4. Checking Web Application Files:")
    web_files = [
        'templates/index.html',
        'templates/index_enhanced.html',
        'static/style.css',
        'static/style_enhanced.css',
        'static/app.js',
        'static/app_enhanced.js'
    ]
    
    for web_file in web_files:
        status, msg = check_file(web_file)
        print(f"   {msg}")
        if not status:
            all_good = False
    
    # 5. Sample data validation
    print("\n5. Validating Sample Data:")
    
    # Check raster metadata
    try:
        with open('data/metadata/raster_metadata.json', 'r') as f:
            metadata = json.load(f)
        print(f"   ✓ Raster metadata: {len(metadata)} buildings")
        
        # Sample first building
        if metadata:
            first_building = list(metadata.keys())[0]
            bounds = metadata[first_building]['bounds']
            print(f"   ✓ Sample bounds: N={bounds['north']:.4f}, S={bounds['south']:.4f}")
    except Exception as e:
        print(f"   ✗ Error reading metadata: {e}")
        all_good = False
    
    # Check GPS traces
    try:
        trace_files = [f for f in os.listdir('data/gps_traces') if f.endswith('.json')]
        if trace_files:
            with open(os.path.join('data/gps_traces', trace_files[0]), 'r') as f:
                trace_data = json.load(f)
            n_users = trace_data.get('n_users', 0)
            n_traces = len(trace_data.get('traces', []))
            print(f"   ✓ Sample GPS data: {n_users} users, {n_traces} traces")
    except Exception as e:
        print(f"   ✗ Error reading GPS traces: {e}")
        all_good = False
    
    # 6. Dependencies check
    print("\n6. Checking Python Dependencies:")
    required_packages = [
        'numpy', 'torch', 'flask', 'geopandas', 'shapely', 
        'sklearn', 'PIL', 'cv2', 'requests'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package} installed")
        except ImportError:
            print(f"   ✗ {package} not installed")
            all_good = False
    
    # 7. Final summary
    print("\n" + "="*60)
    if all_good:
        print("✅ All checks passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python app_updated.py' to start the visualization")
        print("2. Open http://localhost:80 in your browser")
        print("3. Select a building and explore the data")
    else:
        print("❌ Some checks failed. Please address the issues above.")
        print("\nTroubleshooting:")
        print("1. Run each pipeline script in order")
        print("2. Check error messages in individual scripts")
        print("3. Ensure all dependencies are installed")
    print("="*60)
    
    return all_good

def quick_stats():
    """Print quick statistics about the pipeline data"""
    print("\n📊 Pipeline Statistics:")
    
    try:
        # Count files
        n_rasters = len([f for f in os.listdir('data/rasters') if f.endswith('.png')])
        n_shapefiles = len([f for f in os.listdir('data/shapefiles') if f.endswith('.shp')])
        n_traces = len([f for f in os.listdir('data/gps_traces') if f.endswith('.json')])
        
        print(f"   Raster images: {n_rasters}")
        print(f"   Building shapefiles: {n_shapefiles}")
        print(f"   GPS trace files: {n_traces}")
        
        # Load entrance statistics if available
        stats_path = 'data/entrances/entrance_statistics.json'
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            print(f"   Total entrances: {stats['total_entrances']}")
        
        # Check model performance
        model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
        if model_files and 'best_model.pth' in model_files:
            print(f"   Trained models: {len(model_files)}")
            
    except Exception as e:
        print(f"   Error collecting statistics: {e}")

if __name__ == "__main__":
    # Run verification
    success = verify_pipeline()
    
    # Show statistics if successful
    if success:
        quick_stats()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)