"""Reusable prediction system for new patient inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_preprocessing import DataPreprocessor


@dataclass
class BreastCancerPredictionSystem:
    """Bundle fitted preprocessor and model for end-to-end predictions."""

    preprocessor: DataPreprocessor
    model: object

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Return diagnosis label and probability for new patient records."""
        x_scaled = self.preprocessor.transform_new_data(new_data)
        probabilities = self.model.predict_proba(x_scaled)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        labels = np.where(predictions == 1, "Malignant", "Benign")
        output = pd.DataFrame(
            {
                "prediction": labels,
                "malignancy_probability": probabilities,
            }
        )
        return output
