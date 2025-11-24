"""
Setup script for IMDB Sentiment Analysis project.
Downloads required NLTK data and verifies project structure.
"""
import os
import sys

def download_nltk_data():
    """Download required NLTK datasets."""
    print("Downloading NLTK data...")
    try:
        import nltk
        
        # Download required datasets
        datasets = ['stopwords', 'punkt']
        for dataset in datasets:
            print(f"  Downloading {dataset}...")
            nltk.download(dataset, quiet=True)
        
        print("✓ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False


def verify_structure():
    """Verify project directory structure."""
    print("\nVerifying project structure...")
    
    required_dirs = [
        'data',
        'notebooks',
        'src',
        'models',
        'results',
        'results/confusion_matrices',
        'results/feature_importance'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ (missing)")
            all_exist = False
    
    return all_exist


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'sklearn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'nltk',
        'jupyter',
        'ipywidgets'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (not installed)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_data():
    """Check if dataset exists."""
    print("\nChecking dataset...")
    
    data_path = 'data/IMDB Dataset.csv'
    if os.path.exists(data_path):
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"  ✓ Dataset found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ✗ Dataset not found at {data_path}")
        print("  Please download from Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        return False


def main():
    """Run setup checks."""
    print("="*60)
    print("IMDB Sentiment Analysis - Setup")
    print("="*60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Verify structure
    structure_ok = verify_structure()
    
    # Check data
    data_ok = check_data()
    
    # Download NLTK data
    nltk_ok = download_nltk_data()
    
    # Summary
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    print(f"Dependencies: {'✓' if deps_ok else '✗'}")
    print(f"Project Structure: {'✓' if structure_ok else '✗'}")
    print(f"Dataset: {'✓' if data_ok else '✗'}")
    print(f"NLTK Data: {'✓' if nltk_ok else '✗'}")
    
    if all([deps_ok, structure_ok, data_ok, nltk_ok]):
        print("\n✓ Setup complete! You can now run the notebooks.")
        print("\nNext steps:")
        print("1. jupyter notebook")
        print("2. Open notebooks/01_data_exploration.ipynb")
        print("3. Run notebooks in order (01 → 02 → 03 → 04 → 05)")
        return 0
    else:
        print("\n✗ Setup incomplete. Please resolve the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
