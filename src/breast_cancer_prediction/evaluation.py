"""Model training and evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .utils import save_figure


def _get_probability_scores(model: object, x_test: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores for ROC-AUC and ROC plotting."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores
    return model.predict(x_test)


def train_and_evaluate_models(
    models: dict[str, object],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    metrics_output_path: Path,
    confusion_matrices_dir: Path,
    roc_plot_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train all models, compute metrics, and save evaluation artifacts."""
    confusion_matrices_dir.mkdir(parents=True, exist_ok=True)
    trained_models: dict[str, object] = {}
    metric_rows: list[dict[str, float | str]] = []

    # Convert DataFrames to numpy arrays for sklearn compatibility
    x_train_array = x_train.values
    x_test_array = x_test.values

    plt.figure(figsize=(9, 7))
    for model_name, model in models.items():
        estimator = clone(model)
        estimator.fit(x_train_array, y_train)
        trained_models[model_name] = estimator

        predictions = estimator.predict(x_test_array)
        scores = _get_probability_scores(estimator, x_test_array)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, scores)

        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
            }
        )

        matrix = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
        )
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        save_figure(confusion_matrices_dir / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")

        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

    plt.figure(1)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curves for All Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_figure(roc_plot_path)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df.sort_values(by=["ROC-AUC", "F1-Score", "Accuracy"], ascending=False)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_output_path, index=False)

    return metrics_df, trained_models


def select_best_model(metrics_df: pd.DataFrame, trained_models: dict[str, object]) -> tuple[str, object, dict[str, float]]:
    """Select best model based on sorted metric table."""
    best_row = metrics_df.iloc[0]
    best_model_name = str(best_row["Model"])
    best_model = trained_models[best_model_name]
    rationale = {
        "accuracy": float(best_row["Accuracy"]),
        "precision": float(best_row["Precision"]),
        "recall": float(best_row["Recall"]),
        "f1_score": float(best_row["F1-Score"]),
        "roc_auc": float(best_row["ROC-AUC"]),
    }
    return best_model_name, best_model, rationale
