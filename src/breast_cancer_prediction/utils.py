"""Utility helpers for file IO and plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt


def ensure_directories(paths: list[Path]) -> None:
    """Create output directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_figure(output_path: Path) -> None:
    """Save the active Matplotlib figure and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Persist JSON data to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_model(model: Any, output_path: Path) -> None:
    """Serialize and save a model object."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
