"""Dimensionality reduction with PCA and t-SNE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import save_figure


def apply_pca(
    x_scaled: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    explained_variance_path: Path,
    pca_scatter_path: Path,
    variance_threshold: float = 0.95,
) -> tuple[PCA, pd.DataFrame, int]:
    """Fit PCA, plot explained variance, and return transformed data."""
    pca_full = PCA(random_state=random_state)
    pca_full.fit(x_scaled)
    cumulative_variance = pca_full.explained_variance_ratio_.cumsum()
    optimal_components = int((cumulative_variance >= variance_threshold).argmax() + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    plt.axhline(variance_threshold, color="red", linestyle="--", label=f"{variance_threshold:.0%} variance")
    plt.axvline(optimal_components, color="green", linestyle="--", label=f"n={optimal_components}")
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    save_figure(explained_variance_path)

    pca = PCA(n_components=optimal_components, random_state=random_state)
    x_pca = pca.fit_transform(x_scaled)
    x_pca_df = pd.DataFrame(x_pca, columns=[f"PC{i + 1}" for i in range(optimal_components)])

    if optimal_components >= 2:
        scatter_df = pd.DataFrame(
            {
                "PC1": x_pca_df["PC1"],
                "PC2": x_pca_df["PC2"],
                "Diagnosis": y.map({0: "Benign", 1: "Malignant"}),
            }
        )
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=scatter_df, x="PC1", y="PC2", hue="Diagnosis", alpha=0.8)
        plt.title("PCA Projection (First 2 Components)")
        save_figure(pca_scatter_path)

    return pca, x_pca_df, optimal_components


def apply_tsne(
    x_scaled: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    output_path: Path,
) -> pd.DataFrame:
    """Run t-SNE and save a 2D separability plot."""
    n_samples = len(x_scaled)
    perplexity = min(30, max(5, n_samples // 10))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    x_tsne = tsne.fit_transform(x_scaled)

    tsne_df = pd.DataFrame(
        {
            "TSNE1": x_tsne[:, 0],
            "TSNE2": x_tsne[:, 1],
            "Diagnosis": y.map({0: "Benign", 1: "Malignant"}),
        }
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="Diagnosis", alpha=0.8)
    plt.title("t-SNE 2D Visualization")
    save_figure(output_path)

    return tsne_df
