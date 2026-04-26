"""Feature grouping and correlation analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import save_figure


def group_features(feature_columns: list[str]) -> dict[str, list[str]]:
    """Group features into mean, standard error, and worst categories."""
    grouped = {
        "mean_features": [column for column in feature_columns if column.endswith("_mean")],
        "se_features": [column for column in feature_columns if column.endswith("_se")],
        "worst_features": [column for column in feature_columns if column.endswith("_worst")],
    }
    return grouped


def generate_group_summaries(
    dataframe: pd.DataFrame,
    grouped_features: dict[str, list[str]],
    output_path: Path,
) -> pd.DataFrame:
    """Create grouped descriptive summary statistics and save to CSV."""
    rows: list[dict[str, float | str]] = []
    for group_name, features in grouped_features.items():
        subset = dataframe[features]
        rows.append(
            {
                "group": group_name,
                "feature_count": float(len(features)),
                "mean_of_means": float(subset.mean().mean()),
                "mean_std": float(subset.std().mean()),
                "min_value": float(subset.min().min()),
                "max_value": float(subset.max().max()),
            }
        )

    summary_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    return summary_df


def plot_group_correlation_insights(
    dataframe: pd.DataFrame,
    grouped_features: dict[str, list[str]],
    target_column: str,
    heatmap_path: Path,
    target_corr_path: Path,
) -> None:
    """Generate global and target-focused correlation visualizations."""
    sns.set_theme(style="whitegrid")

    grouped_means = {
        group_name: dataframe[features].mean(axis=1)
        for group_name, features in grouped_features.items()
    }
    grouped_df = pd.DataFrame(grouped_means)
    grouped_df[target_column] = dataframe[target_column]

    plt.figure(figsize=(8, 6))
    corr_matrix = grouped_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Grouped Feature Correlation Heatmap")
    save_figure(heatmap_path)

    feature_target_corr = dataframe.drop(columns=[target_column]).corrwith(dataframe[target_column])
    feature_target_corr = feature_target_corr.abs().sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_target_corr.values, y=feature_target_corr.index, hue=feature_target_corr.index, palette="viridis", legend=False)
    plt.title("Top 15 Absolute Correlations With Diagnosis")
    plt.xlabel("Absolute Correlation")
    plt.ylabel("Feature")
    save_figure(target_corr_path)
