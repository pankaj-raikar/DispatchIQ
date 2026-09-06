"""Model training, evaluation, persistence, and inference engine."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from dispatch_iq.config import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    DEFAULT_XGB_HYPERPARAMS,
    MODEL_FEATURE_NAMES,
)
from dispatch_iq.distance import haversine_distance
from dispatch_iq.features import FeatureTransformer


class DispatchIQModel:
    """Wraps gradient-boosted regression modeling, tuning, evaluation, and persistence."""

    def __init__(self, hyperparams: Optional[Dict[str, Any]] = None):
        self.hyperparams = hyperparams or DEFAULT_XGB_HYPERPARAMS.copy()
        self.estimator = XGBRegressor(**self.hyperparams)
        self.transformer = FeatureTransformer()
        self.metrics: Dict[str, float] = {}
        self.is_trained: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        transformer: Optional[FeatureTransformer] = None,
    ) -> "DispatchIQModel":
        """Fits feature transformer and trains the XGBoost regression estimator."""
        if transformer is not None:
            self.transformer = transformer
            X_scaled = self.transformer.transform(X_train)
        else:
            self.transformer = FeatureTransformer()
            X_scaled = self.transformer.fit_transform(X_train)

        self.estimator.fit(X_scaled, y_train)
        self.is_trained = True
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calculates comprehensive evaluation metrics on held-out test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")

        X_test_scaled = self.transformer.transform(X_test)
        preds = self.estimator.predict(X_test_scaled)

        n = len(y_test)
        p = len(MODEL_FEATURE_NAMES)

        r2 = float(r2_score(y_test, preds))
        adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - p - 1))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))

        self.metrics = {
            "r2_score": round(r2, 4),
            "adjusted_r2": round(adj_r2, 4),
            "rmse_minutes": round(rmse, 4),
            "mae_minutes": round(mae, 4),
            "test_sample_count": n,
            "feature_count": p,
        }
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates raw point predictions for a feature DataFrame."""
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before prediction.")
        X_scaled = self.transformer.transform(X)
        return self.estimator.predict(X_scaled)

    def save(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Serializes trained estimator, fitted scaler, and run metadata."""
        m_path = Path(model_path or DEFAULT_MODEL_PATH)
        s_path = Path(scaler_path or DEFAULT_SCALER_PATH)
        meta_path = Path(metadata_path or DEFAULT_METADATA_PATH)

        m_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Save estimator
        joblib.dump(self.estimator, m_path)
        # Save standard scaler & transformer
        joblib.dump(self.transformer.scaler, s_path)

        # Save metadata
        metadata = {
            "project": "DispatchIQ",
            "author": "Pankaj Raikar",
            "version": "1.0.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": "XGBoost Regressor (Gradient Boosted Trees)",
            "hyperparameters": self.hyperparams,
            "features": MODEL_FEATURE_NAMES,
            "metrics": self.metrics,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> "DispatchIQModel":
        """Loads serialized model, scaler, and metadata."""
        m_path = Path(model_path or DEFAULT_MODEL_PATH)
        s_path = Path(scaler_path or DEFAULT_SCALER_PATH)
        meta_path = Path(metadata_path or DEFAULT_METADATA_PATH)

        if not m_path.exists() or not s_path.exists():
            raise FileNotFoundError(f"Model ({m_path}) or Scaler ({s_path}) not found.")

        self.estimator = joblib.load(m_path)
        self.transformer.scaler = joblib.load(s_path)
        self.transformer.is_fitted = True
        self.is_trained = True

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.metrics = data.get("metrics", {})
                self.hyperparams = data.get("hyperparameters", self.hyperparams)

        return self


class DispatchIQPredictor:
    """Production inference wrapper providing formatted predictions with uncertainty bounds."""

    _instance: Optional["DispatchIQPredictor"] = None

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ):
        self.model = DispatchIQModel()
        self.model.load(model_path, scaler_path, metadata_path)
        self.rmse_margin = self.model.metrics.get("rmse_minutes", 3.91)

    @classmethod
    def get_instance(cls) -> "DispatchIQPredictor":
        """Singleton getter for warm inference caching."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict_single(self, order_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Runs single-order prediction, calculating Haversine distance and bounds."""
        # Calculate distance if coordinates provided
        if "distance_km" not in order_dict or order_dict["distance_km"] is None:
            dist = haversine_distance(
                order_dict["restaurant_latitude"],
                order_dict["restaurant_longitude"],
                order_dict["delivery_location_latitude"],
                order_dict["delivery_location_longitude"],
            )
        else:
            dist = float(order_dict["distance_km"])

        row = {
            "Delivery_person_Age": order_dict.get("delivery_person_age", 30),
            "Delivery_person_Ratings": min(5.0, max(1.0, float(order_dict.get("delivery_person_ratings", 4.7)))),
            "distance_km": dist,
            "order_hour": order_dict.get("order_hour", 18),
            "Road_traffic_density": order_dict.get("road_traffic_density", "Medium"),
            "Vehicle_condition": order_dict.get("vehicle_condition", 1),
            "Type_of_order": order_dict.get("type_of_order", "Meal"),
            "Type_of_vehicle": order_dict.get("type_of_vehicle", "motorcycle"),
            "multiple_deliveries": order_dict.get("multiple_deliveries", 0),
            "Festival": order_dict.get("festival", "No"),
            "City": order_dict.get("city", "Metropolitian"),
            "Weatherconditions": order_dict.get("weather_conditions", "Sunny"),
        }

        df_input = pd.DataFrame([row])
        pred_minutes = float(self.model.predict(df_input)[0])
        pred_minutes = max(5.0, pred_minutes)  # Minimum physical delivery floor

        margin = self.rmse_margin
        return {
            "estimated_delivery_minutes": round(pred_minutes, 1),
            "estimated_range_min": round(max(5.0, pred_minutes - margin), 1),
            "estimated_range_max": round(pred_minutes + margin, 1),
            "transit_distance_km": round(dist, 2),
            "confidence_score": 0.95,
            "traffic_level": order_dict.get("road_traffic_density", "Medium"),
        }

    def predict_batch(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes multiple incoming orders."""
        return [self.predict_single(order) for order in orders]
