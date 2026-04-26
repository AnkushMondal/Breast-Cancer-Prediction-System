"""Explainability with SHAP and LIME."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import save_figure


def run_shap_explanations(
    model: object,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    output_dir: Path,
    max_samples: int = 200,
) -> Path:
    """Generate SHAP global feature importance plot."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is required. Install with: pip install shap") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    background = x_train.sample(min(max_samples, len(x_train)), random_state=42)
    explain_data = x_test.sample(min(max_samples, len(x_test)), random_state=42)

    explainer = shap.Explainer(model, background)
    shap_values = explainer(explain_data)

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    shap_path = output_dir / "shap_global_beeswarm.png"
    save_figure(shap_path)
    return shap_path


def run_lime_explanations(
    model: object,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    output_dir: Path,
    class_names: list[str],
    n_examples: int = 3,
) -> list[Path]:
    """Generate LIME local explanations for selected test instances."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError("LIME is required. Install with: pip install lime") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = LimeTabularExplainer(
        training_data=x_train.to_numpy(),
        feature_names=list(x_train.columns),
        class_names=class_names,
        mode="classification",
        random_state=42,
    )

    saved_paths: list[Path] = []
    sample_indices = np.linspace(0, len(x_test) - 1, num=min(n_examples, len(x_test)), dtype=int)

    for idx in sample_indices:
        explanation = explainer.explain_instance(
            data_row=x_test.iloc[idx].to_numpy(),
            predict_fn=model.predict_proba,
            num_features=10,
        )

        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(8, 5)
        plot_path = output_dir / f"lime_local_explanation_idx_{idx}.png"
        plt.title(f"LIME Local Explanation (Test Index {idx})")
        save_figure(plot_path)
        saved_paths.append(plot_path)

        html_path = output_dir / f"lime_local_explanation_idx_{idx}.html"
        explanation.save_to_file(str(html_path))
        saved_paths.append(html_path)

    return saved_paths
