"""Feature encoding, alignment, and standardization pipeline."""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dispatch_iq.config import (
    MODEL_FEATURE_NAMES,
    ONE_HOT_COLUMNS,
    TRAFFIC_DENSITY_MAPPING,
)


class FeatureTransformer:
    """Transforms raw cleaned features into scaled, aligned 21-dimensional input vectors."""

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or MODEL_FEATURE_NAMES
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _encode_and_align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies ordinal encoding, one-hot encoding, and aligns columns to expected features."""
        data = df.copy()

        # Ordinal encoding for traffic density
        if "Road_traffic_density" in data.columns:
            data["Road_traffic_density"] = (
                data["Road_traffic_density"].map(TRAFFIC_DENSITY_MAPPING).fillna(1).astype(int)
            )

        # One-hot encode nominal categories
        available_ohe_cols = [c for c in ONE_HOT_COLUMNS if c in data.columns]
        if available_ohe_cols:
            data = pd.get_dummies(data, columns=available_ohe_cols, drop_first=True, dtype=int)

        # Column alignment: ensure every expected feature is present with default 0, drop unneeded
        for feat in self.feature_names:
            if feat not in data.columns:
                data[feat] = 0

        # Preserve strict order
        return data[self.feature_names].copy()

    def fit(self, X: pd.DataFrame) -> "FeatureTransformer":
        """Fits the underlying StandardScaler strictly on the training feature distribution."""
        aligned_df = self._encode_and_align(X)
        self.scaler.fit(aligned_df)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms and standardizes features for evaluation or real-time inference."""
        if not self.is_fitted:
            raise ValueError("FeatureTransformer must be fitted before transforming data.")
        aligned_df = self._encode_and_align(X)
        return self.scaler.transform(aligned_df)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits scaler and returns standardized array."""
        return self.fit(X).transform(X)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes transformer and fitted scaler to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FeatureTransformer":
        """Loads serialized FeatureTransformer instance."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Transformer not found at {path}")
        return joblib.load(path)
