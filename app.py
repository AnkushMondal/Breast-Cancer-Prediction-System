from flask import Flask, render_template, request, jsonify
import joblib
import os
import pandas as pd
import sys
from pathlib import Path

app = Flask(__name__)

# 1. PATH SETUP
# Ensures the app can find custom modules in the 'src' directory
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 2. MODEL LOADING
# Absolute path to the joblib file to prevent file-not-found errors
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "prediction_system.joblib"

try:
    # This loads the full pipeline: Scaling + Dimensionality Reduction + Model
    model = joblib.load(MODEL_PATH)
    print(f"Model successfully loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model: {e}")
    model = None

# 3. FEATURE SCHEMA
# These names MUST match the columns used during model training exactly
FEATURE_COLUMNS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

@app.route('/')
def home():
    """Renders the landing page."""
    return render_template('home.html')

@app.route('/summary')
def summary():
    """Renders the project summary page."""
    return render_template('summary.html')

@app.route('/how-it-works')
def how_it_works():
    """Renders the how-it-works page."""
    return render_template('how_it_works.html')

@app.route('/favicon.ico')
def favicon():
    """Prevents 404 errors in the terminal from browser favicon requests."""
    return '', 204

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    if model is None:
        return jsonify({'error': 'Server Error: ML model not found.'}), 500
        
    data = request.json
    if not data:
        return jsonify({'error': 'No data received from frontend.'}), 400

    try:
        df = pd.DataFrame([data])
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            print(f"❌ DEBUG: Missing fields -> {missing}")
            return jsonify({'error': f'Missing clinical field: {missing[0]}'}), 400
            
        df = df[FEATURE_COLUMNS]
        df = df.apply(pd.to_numeric, errors='coerce')
        if df.isnull().values.any():
            return jsonify({'error': 'One or more fields contain non-numeric data.'}), 400

        prediction_df = model.predict(df)
        prediction = prediction_df['prediction'].iloc[0]
        return jsonify({'result': prediction})

    except Exception as e:
        print(f"❌ DEBUG ERROR: {type(e).__name__} -> {str(e)}")
        return jsonify({'error': f"Processing Error: {str(e)}"}), 400

if __name__ == '__main__':
    # Running on port 5000
    app.run(debug=True, port=5000)