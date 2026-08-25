"""Geodesic distance computation using the Haversine formula."""

import math
from typing import Union
import numpy as np
import pandas as pd
from dispatch_iq.config import EARTH_RADIUS_KM, EARTH_RADIUS_MILES


def validate_coordinates(lat: float, lon: float) -> bool:
    """Validates if given latitude and longitude fall within physical Earth bounds."""
    if lat is None or lon is None:
        return False
    if math.isnan(lat) or math.isnan(lon):
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def calculate_geodesic_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    miles: bool = False,
) -> float:
    """Calculates great-circle distance between two scalar coordinate points.

    Uses the spherical Haversine formula:
    a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
    c = 2 * arcsin(√a)
    d = R * c
    """
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    radius = EARTH_RADIUS_MILES if miles else EARTH_RADIUS_KM

    phi1, lambda1, phi2, lambda2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    # Guard against float rounding exceeding 1.0
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(radius * c)


def haversine_distance(
    lat1: Union[float, np.ndarray, pd.Series],
    lon1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lon2: Union[float, np.ndarray, pd.Series],
    miles: bool = False,
) -> Union[float, np.ndarray, pd.Series]:
    """Vectorized Haversine distance calculator for Series, Arrays, or Scalars."""
    radius = EARTH_RADIUS_MILES if miles else EARTH_RADIUS_KM

    if isinstance(lat1, (int, float)) and isinstance(lat2, (int, float)):
        return calculate_geodesic_distance(float(lat1), float(lon1), float(lat2), float(lon2), miles=miles)

    phi1, lambda1, phi2, lambda2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return c * radius
