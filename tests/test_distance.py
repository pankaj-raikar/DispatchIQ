"""Unit tests for geodesic distance computation and coordinate validation."""

import math
import unittest
import numpy as np
from dispatch_iq.distance import (
    calculate_geodesic_distance,
    haversine_distance,
    validate_coordinates,
)


class TestDistanceCalculations(unittest.TestCase):
    """Test suite for Haversine geodesic distance logic."""

    def test_same_coordinate_zero_distance(self):
        """Distance between identical coordinates must be precisely 0.0."""
        dist = calculate_geodesic_distance(12.9352, 77.6245, 12.9352, 77.6245)
        self.assertAlmostEqual(dist, 0.0, places=4)

    def test_known_city_distance(self):
        """Great-circle distance between Bengaluru and Mumbai (~845 km)."""
        # Bengaluru: 12.9716 N, 77.5946 E
        # Mumbai: 19.0760 N, 72.8777 E
        dist = calculate_geodesic_distance(12.9716, 77.5946, 19.0760, 72.8777)
        self.assertTrue(840.0 < dist < 855.0, f"Calculated distance {dist} outside expected range.")

    def test_vectorized_haversine_matches_scalar(self):
        """Vectorized array input must yield identical results to scalar function."""
        lats1 = np.array([12.9352, 28.6139])
        lons1 = np.array([77.6245, 77.2090])
        lats2 = np.array([12.9716, 28.7041])
        lons2 = np.array([77.6412, 77.1025])

        vec_res = haversine_distance(lats1, lons1, lats2, lons2)
        s0 = calculate_geodesic_distance(lats1[0], lons1[0], lats2[0], lons2[0])
        s1 = calculate_geodesic_distance(lats1[1], lons1[1], lats2[1], lons2[1])

        self.assertAlmostEqual(vec_res[0], s0, places=4)
        self.assertAlmostEqual(vec_res[1], s1, places=4)

    def test_coordinate_validation(self):
        """Valid coordinates pass; invalid out-of-bound coords fail."""
        self.assertTrue(validate_coordinates(12.9716, 77.5946))
        self.assertTrue(validate_coordinates(0.0, 0.0))
        self.assertTrue(validate_coordinates(-90.0, 180.0))

        self.assertFalse(validate_coordinates(91.0, 77.0))
        self.assertFalse(validate_coordinates(-95.0, 0.0))
        self.assertFalse(validate_coordinates(12.0, 185.0))
        self.assertFalse(validate_coordinates(None, 77.0))
        self.assertFalse(validate_coordinates(float("nan"), 77.0))


if __name__ == "__main__":
    unittest.main()
