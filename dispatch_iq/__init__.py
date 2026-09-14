"""DispatchIQ: Intelligent Last-Mile Delivery ETA Prediction Platform.

Author: Pankaj Raikar <pankajraikar6@gmail.com>
GitHub: https://github.com/pankaj-raikar
"""

from dispatch_iq.distance import haversine_distance, calculate_geodesic_distance
from dispatch_iq.model import DispatchIQModel, DispatchIQPredictor

__version__ = "1.0.0"
__author__ = "Pankaj Raikar"
__email__ = "pankajraikar6@gmail.com"

__all__ = [
    "DispatchIQModel",
    "DispatchIQPredictor",
    "haversine_distance",
    "calculate_geodesic_distance",
    "__version__",
]
