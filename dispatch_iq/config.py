"""Configuration constants and default parameters for DispatchIQ."""

from pathlib import Path

# Project root paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DEFAULT_DATA_PATH = DATA_DIR / "food_delivery.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "xgb_best.pkl"
DEFAULT_SCALER_PATH = MODELS_DIR / "scaler.pkl"
DEFAULT_METADATA_PATH = MODELS_DIR / "metadata.json"

# Core raw features
INPUT_NUMERIC_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries",
    "Vehicle_condition",
]

RAW_CATEGORICAL_COLUMNS = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

TARGET_COLUMN = "Time_taken(min)"

# Ordinal traffic mapping
TRAFFIC_DENSITY_MAPPING = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Jam": 3,
}

# Categorical columns to One-Hot Encode (with drop_first=True)
ONE_HOT_COLUMNS = [
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
    "Weatherconditions",
]

# The precise 21 aligned features expected by the trained estimator
MODEL_FEATURE_NAMES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "distance_km",
    "order_hour",
    "Road_traffic_density",
    "Vehicle_condition",
    "multiple_deliveries",
    "Type_of_order_Drinks",
    "Type_of_order_Meal",
    "Type_of_order_Snack",
    "Type_of_vehicle_electric_scooter",
    "Type_of_vehicle_motorcycle",
    "Type_of_vehicle_scooter",
    "Festival_Yes",
    "City_Semi-Urban",
    "City_Urban",
    "Weatherconditions_Fog",
    "Weatherconditions_Sandstorms",
    "Weatherconditions_Stormy",
    "Weatherconditions_Sunny",
    "Weatherconditions_Windy",
]

# Tuned XGBoost hyperparameters from RandomizedSearchCV
DEFAULT_XGB_HYPERPARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.01,
    "subsample": 0.9,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
}

# Earth radius in kilometers for Haversine geodesic computation
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_MILES = 3956.0
MAX_VALID_DISTANCE_KM = 100.0
