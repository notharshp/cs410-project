"""
Run the complete sentiment analysis pipeline.
Executes all notebooks in sequence to train models and generate results.
"""
import subprocess
import sys
import os
from pathlib import Path

def run_notebook(notebook_path):
    """
    Execute a Jupyter notebook using nbconvert.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running: {notebook_path}")
    print('='*80)
    
    try:
        result = subprocess.run(
            [
                'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                '--ExecutePreprocessor.timeout=600',
                notebook_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ {notebook_path} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {notebook_path}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    """Run all notebooks in sequence."""
    print("="*80)
    print("IMDB Sentiment Analysis - Pipeline Execution")
    print("="*80)
    
    # Define notebook execution order
    notebooks = [
        'notebooks/01_data_exploration.ipynb',
        'notebooks/02_preprocessing.ipynb',
        'notebooks/03_model_training.ipynb',
        'notebooks/04_evaluation_comparison.ipynb',
        'notebooks/05_interactive_predictor.ipynb'
    ]
    
    # Check if all notebooks exist
    for notebook in notebooks:
        if not Path(notebook).exists():
            print(f"✗ Notebook not found: {notebook}")
            return 1
    
    print(f"\nFound {len(notebooks)} notebooks to execute")
    print("This may take 10-15 minutes depending on your system...\n")
    
    # Execute notebooks
    results = {}
    for notebook in notebooks:
        success = run_notebook(notebook)
        results[notebook] = success
        
        if not success:
            print(f"\n✗ Pipeline stopped due to error in {notebook}")
            print("Please check the error messages above and fix any issues.")
            return 1
    
    # Summary
    print("\n" + "="*80)
    print("Pipeline Execution Summary")
    print("="*80)
    
    for notebook, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {notebook}")
    
    if all(results.values()):
        print("\n✓ All notebooks executed successfully!")
        print("\nGenerated outputs:")
        print("  - models/naive_bayes_model.pkl")
        print("  - models/logistic_regression_model.pkl")
        print("  - models/tfidf_vectorizer.pkl")
        print("  - results/metrics_comparison.csv")
        print("  - results/confusion_matrices/*.png")
        print("  - results/feature_importance/*.png")
        print("\nYou can now:")
        print("  1. Review the executed notebooks for detailed results")
        print("  2. Use the interactive predictor (notebook 05)")
        print("  3. Check the results/ directory for visualizations")
        return 0
    else:
        print("\n✗ Some notebooks failed to execute")
        return 1


if __name__ == '__main__':
    sys.exit(main())
