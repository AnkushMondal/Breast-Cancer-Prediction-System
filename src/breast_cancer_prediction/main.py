"""Main pipeline for Breast Cancer Prediction System."""

from __future__ import annotations

import json
import matplotlib
matplotlib.use('Agg')  # Prevents interactive window creation

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DATA_PATH,
    EXPLAINABILITY_DIR,
    ID_COLUMNS,
    METRICS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .data_preprocessing import DataPreprocessor
from .dimensionality_reduction import apply_pca, apply_tsne
from .evaluation import select_best_model, train_and_evaluate_models
from .explainability import run_lime_explanations, run_shap_explanations
from .feature_grouping import generate_group_summaries, group_features, plot_group_correlation_insights
from .models import get_models
from .prediction_system import BreastCancerPredictionSystem
from .utils import ensure_directories, save_json, save_model


def main() -> None:
    """Run full research-grade machine learning workflow."""
    np.random.seed(RANDOM_STATE)

    ensure_directories([OUTPUT_DIR, PLOTS_DIR, METRICS_DIR, MODELS_DIR, EXPLAINABILITY_DIR])

    # (i) Data preprocessing
    preprocessor = DataPreprocessor(
        target_column=TARGET_COLUMN,
        id_columns=ID_COLUMNS,
        random_state=RANDOM_STATE,
    )
    raw_df = preprocessor.load_data(DATA_PATH)
    clean_df, x_scaled_df, y = preprocessor.fit_transform(raw_df)

    # (ii) Feature grouping and correlation insights
    grouped = group_features(preprocessor.feature_columns)
    group_summary_df = generate_group_summaries(
        dataframe=clean_df,
        grouped_features=grouped,
        output_path=METRICS_DIR / "feature_group_summary.csv",
    )
    plot_group_correlation_insights(
        dataframe=clean_df,
        grouped_features=grouped,
        target_column=TARGET_COLUMN,
        heatmap_path=PLOTS_DIR / "grouped_feature_correlation_heatmap.png",
        target_corr_path=PLOTS_DIR / "top_feature_target_correlations.png",
    )

    # (iii) Dimensionality reduction (PCA and t-SNE)
    _, x_pca_df, optimal_components = apply_pca(
        x_scaled=x_scaled_df,
        y=pd.Series(y),
        random_state=RANDOM_STATE,
        explained_variance_path=PLOTS_DIR / "pca_explained_variance.png",
        pca_scatter_path=PLOTS_DIR / "pca_scatter_plot.png",
    )
    tsne_df = apply_tsne(
        x_scaled=x_scaled_df,
        y=pd.Series(y),
        random_state=RANDOM_STATE,
        output_path=PLOTS_DIR / "tsne_2d_visualization.png",
    )

    # (iv) Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled_df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # (v) Train machine learning models
    model_candidates = get_models(random_state=RANDOM_STATE)

    # (vi) Model evaluation
    metrics_df, trained_models = train_and_evaluate_models(
        models=model_candidates,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metrics_output_path=METRICS_DIR / "model_comparison_metrics.csv",
        confusion_matrices_dir=PLOTS_DIR / "confusion_matrices",
        roc_plot_path=PLOTS_DIR / "roc_curves.png",
    )

    # (vii) Best model selection
    best_model_name, best_model, rationale = select_best_model(metrics_df, trained_models)
    selection_record = {
        "best_model": best_model_name,
        "selection_criteria": "Highest ROC-AUC, then F1-Score and Accuracy",
        "metrics": rationale,
    }
    save_json(selection_record, METRICS_DIR / "best_model_selection.json")

    # (viii) Explainable AI with SHAP and LIME
    shap_plot = run_shap_explanations(
        model=best_model,
        x_train=x_train,
        x_test=x_test,
        output_dir=EXPLAINABILITY_DIR,
    )
    lime_artifacts = run_lime_explanations(
        model=best_model,
        x_train=x_train,
        x_test=x_test,
        output_dir=EXPLAINABILITY_DIR,
        class_names=["Benign", "Malignant"],
    )

    # (ix) Final reusable prediction system
    prediction_system = BreastCancerPredictionSystem(preprocessor=preprocessor, model=best_model)
    sample_new_patients = clean_df.drop(columns=[TARGET_COLUMN]).head(3)
    prediction_output = prediction_system.predict(sample_new_patients)
    prediction_output.to_csv(OUTPUT_DIR / "sample_predictions.csv", index=False)

    save_model(preprocessor, MODELS_DIR / "preprocessor.joblib")
    save_model(best_model, MODELS_DIR / "best_model.joblib")
    save_model(prediction_system, MODELS_DIR / "prediction_system.joblib")

    run_summary = {
        "dataset_path": str(DATA_PATH),
        "n_samples": int(len(clean_df)),
        "n_features": int(len(preprocessor.feature_columns)),
        "optimal_pca_components": int(optimal_components),
        "best_model": best_model_name,
        "group_summary_rows": int(len(group_summary_df)),
        "tsne_points": int(len(tsne_df)),
        "shap_plot": str(shap_plot),
        "lime_artifacts": [str(path) for path in lime_artifacts],
    }

    with (OUTPUT_DIR / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(run_summary, file, indent=2)


if __name__ == "__main__":
    main()
