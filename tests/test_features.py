"""Unit tests for feature transformation, encoding, and alignment."""

import unittest
import numpy as np
import pandas as pd
from dispatch_iq.config import MODEL_FEATURE_NAMES
from dispatch_iq.features import FeatureTransformer


class TestFeatureTransformer(unittest.TestCase):
    """Test suite for feature alignment, encoding, and scaling."""

    def setUp(self):
        self.sample_df = pd.DataFrame([
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
            },
            {
                "Delivery_person_Age": 35,
                "Delivery_person_Ratings": 4.2,
                "distance_km": 12.0,
                "order_hour": 13,
                "Road_traffic_density": "Low",
                "Vehicle_condition": 2,
                "Type_of_order": "Snack",
                "Type_of_vehicle": "scooter",
                "multiple_deliveries": 1,
                "Festival": "Yes",
                "City": "Urban",
                "Weatherconditions": "Fog",
            },
        ])

    def test_fit_and_transform_dimensions(self):
        """Output transformed array must have shape (n_samples, 21)."""
        transformer = FeatureTransformer()
        transformed = transformer.fit_transform(self.sample_df)

        self.assertEqual(transformed.shape, (2, 21))
        self.assertEqual(len(transformer.feature_names), 21)
        self.assertEqual(transformer.feature_names, MODEL_FEATURE_NAMES)

    def test_single_sample_alignment(self):
        """A single row must align to exact 21 features with zero unhandled keys."""
        transformer = FeatureTransformer()
        transformer.fit(self.sample_df)

        single_row = pd.DataFrame([
            {
                "Delivery_person_Age": 24,
                "Delivery_person_Ratings": 4.9,
                "distance_km": 2.1,
                "order_hour": 20,
                "Road_traffic_density": "Jam",
                "Vehicle_condition": 0,
                "Type_of_order": "Drinks",
                "Type_of_vehicle": "electric_scooter",
                "multiple_deliveries": 2,
                "Festival": "No",
                "City": "Semi-Urban",
                "Weatherconditions": "Windy",
            }
        ])

        transformed_single = transformer.transform(single_row)
        self.assertEqual(transformed_single.shape, (1, 21))
        self.assertFalse(np.isnan(transformed_single).any())


if __name__ == "__main__":
    unittest.main()
