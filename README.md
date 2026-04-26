# 🏥 BreastCare AI - Breast Cancer Prediction System

A complete end-to-end machine learning pipeline for predicting breast cancer diagnosis (benign vs. malignant) using clinical measurements. This project implements a modular workflow with advanced explainability features, comprehensive model evaluation, and a professional multipage web interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Web Interface](#web-interface)
- [Module Pipeline](#module-pipeline)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Architecture](#project-architecture)
- [Model Evaluation](#model-evaluation)
- [Explainability & Interpretability](#explainability--interpretability)
- [Output Artifacts](#output-artifacts)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for breast cancer classification, transforming raw clinical measurements into actionable diagnostic predictions. The system combines state-of-the-art classification algorithms with advanced explainability techniques to provide transparent, trustworthy predictions suitable for medical applications.

**Dataset**: Wisconsin Breast Cancer (Diagnostic) Dataset  
**Target Variable**: Diagnosis (B=Benign, M=Malignant)  
**Features**: 30 computed clinical measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension, and their standard errors and worst values)

---

## 🌐 Web Interface

An interactive multipage web dashboard built with Flask, HTML, CSS, and JavaScript for real-time breast cancer predictions.

### Features
- **Multipage Design**: Home, Predict, Summary, and How It Works pages
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Dynamic Form Generation**: Automatically creates input fields for all 30 clinical features
- **Real-time Validation**: Client-side validation with visual feedback
- **Prediction Dashboard**: Displays results with status indicators and recommendations
- **Prediction History**: Tracks recent predictions with timestamps
- **Modern UI**: Dark theme with smooth animations and professional styling

### Usage
```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

### API Endpoint
- **POST** `/predict`: Accepts JSON with 30 feature values, returns diagnosis prediction

### Web Structure
```
├── app.py                                 # Flask application with prediction API
├── templates/
│   ├── base.html                         # Shared layout template
│   ├── home.html                         # Landing page
│   ├── predict.html                      # Prediction form and dashboard
│   ├── summary.html                      # Project summary page
│   └── how_it_works.html                 # Workflow explanation page
├── static/
│   ├── css/
│   │   └── style.css                     # Responsive styling and animations
│   └── js/
│       └── script.js                     # Frontend JavaScript logic
```

---

## 🔄 Module Pipeline

The system executes the following stages in sequence:

### 1️⃣ **Data Preprocessing**
   - Load and validate raw data
   - Remove irrelevant columns (id, unnamed columns)
   - Handle missing values
   - Feature scaling using StandardScaler

### 2️⃣ **Feature Grouping & Analysis**
   - Group features by semantic meaning
   - Generate correlation matrices and summaries
   - Analyze feature-target relationships
   - Visualize grouped feature patterns

### 3️⃣ **Dimensionality Reduction**
   - **PCA**: Principal Component Analysis for variance analysis
   - **t-SNE**: t-Distributed Stochastic Neighbor Embedding for 2D visualization
   - Generate explained variance plots and scatter visualizations

### 4️⃣ **Data Splitting**
   - Stratified train/test split (80/20)
   - Maintains class balance distribution

### 5️⃣ **Model Training**
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - Gradient Boosting
   - Neural Networks (MLPClassifier)

### 6️⃣ **Model Evaluation**
   - Cross-validation scoring
   - Confusion matrices for each model
   - ROC-AUC curves comparison
   - Performance metrics comparison

### 7️⃣ **Best Model Selection**
   - Automatic selection based on validation performance
   - Model serialization and checkpointing

### 8️⃣ **Explainability Analysis**
   - SHAP values for global and local explanations
   - LIME for local interpretable model-agnostic explanations
   - Feature importance visualization

### 9️⃣ **Prediction System**
   - Bundled preprocessor + model for end-to-end inference
   - Batch prediction capabilities
   - Confidence scores (malignancy probability)

---

## 📁 Project Structure

```
Riya/
├── data.csv                                 # Wisconsin Breast Cancer dataset
├── requirements.txt                          # Python dependencies
├── run_pipeline.py                          # Entry point for pipeline execution
├── app.py                                   # Flask web application
├── README.md                                 # Project documentation
│
├── src/
│   └── breast_cancer_prediction/
│       ├── __init__.py                      # Package initialization
│       ├── config.py                        # Configuration parameters & paths
│       ├── utils.py                         # Utility functions
│       ├── data_preprocessing.py            # Data loading, cleaning, scaling
│       ├── feature_grouping.py              # Feature grouping & correlation analysis
│       ├── dimensionality_reduction.py      # PCA & t-SNE implementation
│       ├── models.py                        # Model definitions & initialization
│       ├── evaluation.py                    # Training, evaluation, best model selection
│       ├── explainability.py                # SHAP & LIME explanations
│       ├── prediction_system.py             # Reusable prediction class
│       └── main.py                          # Main orchestration pipeline
│
├── templates/
│   ├── base.html                            # Shared layout template
│   ├── home.html                            # Landing page
│   ├── predict.html                         # Prediction form and dashboard
│   ├── summary.html                         # Project summary page
│   └── how_it_works.html                    # Workflow explanation page
│
├── static/
│   ├── css/
│   │   └── style.css                        # Responsive web styling
│   └── js/
│       └── script.js                        # Frontend JavaScript logic
│
└── outputs/
    ├── run_summary.json                     # Pipeline execution summary
    ├── sample_predictions.csv               # Example predictions
    │
    ├── metrics/
    │   ├── model_comparison_metrics.csv     # Cross-model performance comparison
    │   ├── feature_group_summary.csv        # Feature grouping analysis
    │   └── best_model_selection.json        # Best model metadata
    │
    ├── models/
    │   ├── preprocessor.joblib              # Fitted data preprocessor
    │   ├── best_model.joblib                 # Best trained classifier
    │   └── prediction_system.joblib         # Complete prediction pipeline
    │
    ├── plots/
    │   ├── pca_explained_variance.png       # PCA variance analysis
    │   ├── pca_scatter_plot.png              # PCA 2D visualization
    │   ├── tsne_2d_visualization.png        # t-SNE 2D visualization
    │   ├── roc_curves.png                   # ROC curves comparison
    │   ├── grouped_feature_correlation_heatmap.png
    │   ├── top_feature_target_correlations.png
    │   └── confusion_matrices/
    │       ├── Logistic_Regression_cm.png
    │       ├── Random_Forest_cm.png
    │       ├── SVM_cm.png
    │       └── ...
    │
    └── explainability/
        ├── lime_local_explanation_idx_0.html
        ├── lime_local_explanation_idx_56.html
        └── lime_local_explanation_idx_113.html
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Step 1: Clone or Download the Repository
```bash
cd path/to/Riya
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Pipeline (Optional)
To train models and generate outputs:
```bash
python run_pipeline.py
```

---

## 💻 Usage

### Running the Web Application
```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your browser.

### Making Predictions via API
```python
import requests

data = {
    "radius_mean": 14.127,
    "texture_mean": 19.26,
    # ... include all 30 features
}

response = requests.post('http://127.0.0.1:5000/predict', json=data)
print(response.json())  # {'result': 'Malignant'}
```

### Running the Pipeline
```bash
python run_pipeline.py
```

---

## 🏗️ Project Architecture

### Core Components
- **Data Layer**: Handles data loading, preprocessing, and feature engineering
- **Model Layer**: Implements multiple ML algorithms with evaluation
- **Explainability Layer**: Provides SHAP and LIME explanations
- **Web Layer**: Flask-based API and multipage interface
- **Output Layer**: Manages artifacts, metrics, and visualizations

### Design Patterns
- Modular architecture with separation of concerns
- Factory pattern for model instantiation
- Strategy pattern for evaluation metrics
- Template method for pipeline orchestration

---

## 📊 Model Evaluation

The system evaluates multiple models using comprehensive metrics:

### Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Cross-validation scores

### Best Model Selection
- Automatic selection based on validation performance
- Model persistence using joblib
- Performance comparison across all models

---

## 🔍 Explainability & Interpretability

### SHAP Analysis
- Global feature importance
- Local explanations for individual predictions
- Feature interaction analysis

### LIME Explanations
- Local interpretable model-agnostic explanations
- HTML visualizations for individual cases
- Feature contribution analysis

---

## 📦 Output Artifacts

### Models
- `prediction_system.joblib`: Complete pipeline (preprocessor + model)
- `best_model.joblib`: Trained classifier
- `preprocessor.joblib`: Fitted data preprocessor

### Metrics
- `model_comparison_metrics.csv`: Performance comparison
- `best_model_selection.json`: Model metadata
- `feature_group_summary.csv`: Feature analysis

### Visualizations
- ROC curves, confusion matrices
- PCA and t-SNE plots
- Feature correlation heatmaps
- LIME explanation HTML files

---

## ⚙️ Configuration

Configuration is managed through `src/breast_cancer_prediction/config.py`:

- Data paths and file locations
- Model hyperparameters
- Feature engineering settings
- Evaluation parameters

---

## 🔧 Technical Details

### Dependencies
- Flask: Web framework
- scikit-learn: ML algorithms and preprocessing
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib/seaborn: Visualization
- shap: Explainability
- lime: Local explanations
- joblib: Model serialization

### Performance
- Fast inference with pre-trained models
- Efficient data preprocessing pipeline
- Responsive web interface

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Contact

For questions or support, please open an issue on GitHub.
