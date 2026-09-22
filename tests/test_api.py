"""Unit tests for FastAPI endpoints and Pydantic schemas."""

import unittest
from pydantic import ValidationError
from dispatch_iq.api import (
    get_model_metrics,
    health_check,
    predict_delivery_time,
    predict_delivery_time_batch,
    root,
)
from dispatch_iq.schemas import BatchDeliveryOrderInput, DeliveryOrderInput


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for DispatchIQ REST API route handlers."""

    def test_root_endpoint(self):
        """Root endpoint returns service descriptor."""
        res = root()
        self.assertEqual(res["service"], "DispatchIQ")
        self.assertEqual(res["version"], "1.0.0")

    def test_health_endpoint(self):
        """Health check returns healthy status and model_loaded=True."""
        res = health_check()
        self.assertEqual(res.status, "healthy")
        self.assertTrue(res.model_loaded)

    def test_metrics_endpoint(self):
        """Metrics endpoint returns model performance statistics."""
        metrics = get_model_metrics()
        self.assertEqual(metrics.project, "DispatchIQ")
        self.assertEqual(metrics.author, "Pankaj Raikar")
        self.assertEqual(metrics.feature_count, 21)
        self.assertTrue(metrics.r2_score > 0.80)
        self.assertTrue(metrics.rmse_minutes < 4.5)

    def test_predict_single_order(self):
        """Predict route processes order correctly."""
        order_payload = DeliveryOrderInput(
            delivery_person_age=26,
            delivery_person_ratings=4.7,
            restaurant_latitude=12.9352,
            restaurant_longitude=77.6245,
            delivery_location_latitude=12.9716,
            delivery_location_longitude=77.6412,
            order_hour=19,
            road_traffic_density="High",
            weather_conditions="Sunny",
            type_of_order="Meal",
            type_of_vehicle="motorcycle",
            multiple_deliveries=0,
            festival="No",
            city="Metropolitian",
            vehicle_condition=1,
        )

        res = predict_delivery_time(order=order_payload)
        self.assertTrue(10.0 <= res.estimated_delivery_minutes <= 60.0)
        self.assertTrue(res.transit_distance_km > 0)
        self.assertEqual(res.traffic_level, "High")

    def test_predict_batch_orders(self):
        """Batch predict route processes multiple orders."""
        order1 = DeliveryOrderInput(
            delivery_person_age=25,
            delivery_person_ratings=4.9,
            restaurant_latitude=12.9352,
            restaurant_longitude=77.6245,
            delivery_location_latitude=12.9716,
            delivery_location_longitude=77.6412,
            order_hour=14,
            road_traffic_density="Low",
            weather_conditions="Sunny",
        )
        order2 = DeliveryOrderInput(
            delivery_person_age=32,
            delivery_person_ratings=4.5,
            restaurant_latitude=12.9352,
            restaurant_longitude=77.6245,
            delivery_location_latitude=12.9816,
            delivery_location_longitude=77.6512,
            order_hour=20,
            road_traffic_density="Jam",
            weather_conditions="Stormy",
        )

        batch_input = BatchDeliveryOrderInput(orders=[order1, order2])
        res = predict_delivery_time_batch(batch_input=batch_input)
        self.assertEqual(res.total_orders_processed, 2)
        self.assertEqual(len(res.predictions), 2)
        # Order with storm + traffic jam should predict longer delivery time than order with sunny + low traffic
        self.assertTrue(res.predictions[1].estimated_delivery_minutes > res.predictions[0].estimated_delivery_minutes)

    def test_pydantic_validation_guards(self):
        """Invalid inputs must raise ValidationError."""
        # Age under 18
        with self.assertRaises(ValidationError):
            DeliveryOrderInput(
                delivery_person_age=16,
                delivery_person_ratings=4.8,
                restaurant_latitude=12.9352,
                restaurant_longitude=77.6245,
                delivery_location_latitude=12.9716,
                delivery_location_longitude=77.6412,
                order_hour=19,
                road_traffic_density="High",
                weather_conditions="Sunny",
            )

        # Rating above 5.0
        with self.assertRaises(ValidationError):
            DeliveryOrderInput(
                delivery_person_age=25,
                delivery_person_ratings=6.5,
                restaurant_latitude=12.9352,
                restaurant_longitude=77.6245,
                delivery_location_latitude=12.9716,
                delivery_location_longitude=77.6412,
                order_hour=19,
                road_traffic_density="High",
                weather_conditions="Sunny",
            )


if __name__ == "__main__":
    unittest.main()
