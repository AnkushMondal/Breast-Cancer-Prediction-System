"""Convenience entry-point for running the full project pipeline."""

from __future__ import annotations
import sys
import os
from pathlib import Path

# 1. PATH SETUP: Ensure the system can find your source code
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 2. TKINTER FIX: Force Matplotlib to use 'Agg' backend
# This MUST be done before importing breast_cancer_prediction.main
try:
    import matplotlib
    matplotlib.use('Agg') 
    print(" Matplotlib backend set to 'Agg' for headless execution.")
except ImportError:
    print(" Matplotlib not found; skipping backend configuration.")

# 3. IMPORT MAIN PIPELINE
try:
    from breast_cancer_prediction.main import main
except ImportError as e:
    print(f" Error importing main pipeline: {e}")
    print("Make sure your project structure matches: src/breast_cancer_prediction/main.py")
    sys.exit(1)

def run():
    """Executes the pipeline with error handling."""
    print("\n" + "="*50)
    print("Starting the Breast Cancer Prediction Pipeline")
    print("="*50)
    
    try:
        main()
        print("\n" + "="*50)
        print(" Pipeline completed successfully!")
        print("Check the 'outputs/' folder for your model and reports.")
        print("="*50)
    except Exception as e:
        print(f"\n PIPELINE CRASHED: {str(e)}")
        # Print the full error for debugging in VS Code
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run()