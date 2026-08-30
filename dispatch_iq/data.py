"""Data loading, cleaning, imputation, and splitting pipeline."""

from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dispatch_iq.config import (
    DEFAULT_DATA_PATH,
    INPUT_NUMERIC_COLUMNS,
    RAW_CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    MAX_VALID_DISTANCE_KM,
)
from dispatch_iq.distance import haversine_distance


def clean_raw_dataframe(
    raw_df: pd.DataFrame,
    is_training: bool = True,
) -> pd.DataFrame:
    """Performs rigorous data sanitization on raw food delivery records.

    Steps:
    1. Strips leading/trailing whitespace across text features.
    2. Replaces literal 'NaN' strings with genuine np.nan.
    3. Cleans prefixed categorical strings (e.g., 'conditions Sunny' -> 'Sunny').
    4. Parses dirty target column (e.g., '(min) 24' -> 24 integer).
    5. Coerces numeric attributes with errors='coerce'.
    6. Clips invalid ratings (> 5.0).
    7. Computes geodesic distance in km using the Haversine formula.
    8. Filters physical coordinate outliers (> 100 km).
    9. Extracts continuous order_hour from order timestamps.
    10. Imputes missing values (median for numerics, mode for categoricals).
    """
    df = raw_df.copy()

    # 1 & 2: Clean object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"NaN": np.nan, "nan": np.nan, "None": np.nan})

    # Clean Weatherconditions prefix if present ('conditions Sunny' -> 'Sunny')
    if "Weatherconditions" in df.columns:
        df["Weatherconditions"] = df["Weatherconditions"].apply(
            lambda x: x.split(" ")[-1] if pd.notnull(x) and " " in str(x) else x
        )

    # 4: Clean target column if present
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
            lambda x: int(str(x).split(" ")[-1]) if pd.notnull(x) and any(c.isdigit() for c in str(x)) else np.nan
        )
        if is_training:
            df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    # 5: Numeric coercion
    numeric_cols = [c for c in INPUT_NUMERIC_COLUMNS if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6: Sanity-check ratings (ratings cannot exceed 5.0)
    if "Delivery_person_Ratings" in df.columns:
        df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].clip(lower=1.0, upper=5.0)

    # 7: Geodesic distance calculation
    coord_cols = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    if all(c in df.columns for c in coord_cols):
        for c in coord_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["distance_km"] = haversine_distance(
            df["Restaurant_latitude"],
            df["Restaurant_longitude"],
            df["Delivery_location_latitude"],
            df["Delivery_location_longitude"],
        )

        # 8: Remove coordinate corruptions/outliers during training
        if is_training:
            df = df[
                (df["distance_km"] >= 0.0) & (df["distance_km"] <= MAX_VALID_DISTANCE_KM)
            ].reset_index(drop=True)

    # 9: Order hour extraction
    if "Time_Orderd" in df.columns:
        parsed_time = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
        df["order_hour"] = parsed_time.dt.hour
    elif "order_hour" not in df.columns:
        df["order_hour"] = 18.0  # default evening rush hour if completely omitted

    # 10: Impute missing numerics with median
    impute_numerics = [c for c in INPUT_NUMERIC_COLUMNS + ["order_hour", "distance_km"] if c in df.columns]
    for col in impute_numerics:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)

    # Impute missing categoricals with mode
    impute_categoricals = [c for c in RAW_CATEGORICAL_COLUMNS if c in df.columns]
    for col in impute_categoricals:
        mode_series = df[col].dropna().mode()
        fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        df[col] = df[col].fillna(fill_val)

    return df


def load_dataset(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads dataset from path with fallback detection."""
    if filepath is None:
        filepath = DEFAULT_DATA_PATH

    path = Path(filepath)
    if not path.exists():
        # Check alternative common locations
        candidates = [
            Path("data/food_delivery.csv"),
            Path("../data/food_delivery.csv"),
            Path("food_delivery.csv"),
            Path("../food_delivery.csv"),
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath} or candidate locations.")

    return pd.read_csv(path)


def prepare_training_data(
    filepath: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads and preprocesses the full dataset into feature matrix X and target y."""
    raw_df = load_dataset(filepath)
    clean_df = clean_raw_dataframe(raw_df, is_training=True)

    feature_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "distance_km",
        "order_hour",
        "Weatherconditions",
        "Road_traffic_density",
        "Vehicle_condition",
        "Type_of_order",
        "Type_of_vehicle",
        "multiple_deliveries",
        "Festival",
        "City",
    ]

    X = clean_df[feature_cols].copy()
    y = clean_df[TARGET_COLUMN].astype(float)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Performs reproducible train-test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
