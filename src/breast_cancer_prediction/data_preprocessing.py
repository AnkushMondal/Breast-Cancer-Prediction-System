"""Data loading, cleaning, encoding, and scaling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class DataPreprocessor:
    """Reusable preprocessing pipeline for training and inference."""

    target_column: str
    id_columns: list[str] = field(default_factory=list)
    random_state: int = 42
    imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median"))
    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_columns: list[str] = field(default_factory=list)

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load CSV dataset into a DataFrame."""
        return pd.read_csv(data_path)

    def _drop_irrelevant_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Drop ID and unnamed columns from dataset."""
        df = dataframe.copy()
        unnamed_cols = [column for column in df.columns if column.lower().startswith("unnamed")]
        removable = [column for column in self.id_columns + unnamed_cols if column in df.columns]
        if removable:
            df = df.drop(columns=removable)
        return df

    def _encode_target(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Encode diagnosis target as binary values."""
        df = dataframe.copy()
        if self.target_column in df.columns:
            mapping = {"B": 0, "M": 1, "0": 0, "1": 1}
            series = df[self.target_column]
            if series.dtype.kind in {"O", "U", "S"} or str(series.dtype).startswith("string"):
                series = series.astype(str).str.strip().map(mapping)
            else:
                series = pd.to_numeric(series, errors="coerce")

            if series.isna().any():
                invalid_values = df.loc[series.isna(), self.target_column].unique().tolist()
                raise ValueError(
                    f"Target column '{self.target_column}' contains unsupported labels: {invalid_values}"
                )

            df[self.target_column] = series.astype(int)
        return df

    def fit_transform(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Fit preprocessing steps and transform feature matrix."""
        np.random.seed(self.random_state)
        df = self._drop_irrelevant_columns(dataframe)
        df = self._encode_target(df)

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")

        self.feature_columns = [column for column in df.columns if column != self.target_column]
        x = df[self.feature_columns]
        y = df[self.target_column].astype(int).to_numpy()

        x_imputed = self.imputer.fit_transform(x)
        x_scaled = self.scaler.fit_transform(x_imputed)
        x_scaled_df = pd.DataFrame(x_scaled, columns=self.feature_columns)
        return df, x_scaled_df, y

    def transform_new_data(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessing to new data for inference."""
        if not self.feature_columns:
            raise ValueError("Preprocessor is not fitted. Run fit_transform first.")

        df = self._drop_irrelevant_columns(dataframe)
        if self.target_column in df.columns:
            df = df.drop(columns=[self.target_column])

        missing_columns = [column for column in self.feature_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")

        df = df[self.feature_columns]
        x_imputed = self.imputer.transform(df)
        x_scaled = self.scaler.transform(x_imputed)
        return x_scaled
