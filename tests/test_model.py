"""Unit tests for DispatchIQ model loading and prediction."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from dispatch_iq.config import DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH
from dispatch_iq.model import DispatchIQModel, DispatchIQPredictor


class TestDispatchIQModel(unittest.TestCase):
    """Test suite for trained estimator persistence and prediction validity."""

    def test_saved_artifacts_exist(self):
        """Trained model and scaler files must exist on disk."""
        self.assertTrue(DEFAULT_MODEL_PATH.exists(), f"Model missing at {DEFAULT_MODEL_PATH}")
        self.assertTrue(DEFAULT_SCALER_PATH.exists(), f"Scaler missing at {DEFAULT_SCALER_PATH}")

    def test_model_loading_and_prediction(self):
        """Loaded model must predict realistic positive delivery minutes."""
        model = DispatchIQModel()
        model.load()
        self.assertTrue(model.is_trained)

        sample = pd.DataFrame([
            {
                "Delivery_person_Age": 28,
                "Delivery_person_Ratings": 4.8,
                "distance_km": 4.5,
                "order_hour": 19,
                "Road_traffic_density": "High",
                "Vehicle_condition": 1,
                "Type_of_order": "Meal",
                "Type_of_vehicle": "motorcycle",
                "multiple_deliveries": 0,
                "Festival": "No",
                "City": "Metropolitian",
                "Weatherconditions": "Sunny",
            }
        ])

        pred = model.predict(sample)
        self.assertEqual(len(pred), 1)
        self.assertTrue(10.0 <= pred[0] <= 60.0, f"Unrealistic delivery ETA prediction: {pred[0]}")

    def test_predictor_wrapper_format(self):
        """DispatchIQPredictor returns formatted payload with lower and upper bounds."""
        predictor = DispatchIQPredictor.get_instance()
        res = predictor.predict_single({
            "delivery_person_age": 25,
            "delivery_person_ratings": 4.9,
            "restaurant_latitude": 12.9352,
            "restaurant_longitude": 77.6245,
            "delivery_location_latitude": 12.9716,
            "delivery_location_longitude": 77.6412,
            "order_hour": 20,
            "road_traffic_density": "Medium",
            "weather_conditions": "Sunny",
        })

        self.assertIn("estimated_delivery_minutes", res)
        self.assertIn("estimated_range_min", res)
        self.assertIn("estimated_range_max", res)
        self.assertIn("transit_distance_km", res)
        self.assertTrue(res["estimated_range_min"] <= res["estimated_delivery_minutes"] <= res["estimated_range_max"])


if __name__ == "__main__":
    unittest.main()
