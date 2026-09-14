"""Pydantic request and response schemas for DispatchIQ API and CLI."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class DeliveryOrderInput(BaseModel):
    """Schema for a single delivery order prediction request."""

    delivery_person_age: int = Field(
        ...,
        ge=18,
        le=65,
        description="Age of the delivery rider in years (18-65)",
        examples=[29],
    )
    delivery_person_ratings: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Delivery rider rating score (1.0 - 5.0)",
        examples=[4.8],
    )
    restaurant_latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude of restaurant pickup location",
        examples=[12.9352],
    )
    restaurant_longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude of restaurant pickup location",
        examples=[77.6245],
    )
    delivery_location_latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude of customer drop-off destination",
        examples=[12.9716],
    )
    delivery_location_longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude of customer drop-off destination",
        examples=[77.6412],
    )
    order_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of order placement (0-23 in local time)",
        examples=[19],
    )
    road_traffic_density: Literal["Low", "Medium", "High", "Jam"] = Field(
        ...,
        description="Current traffic congestion density",
        examples=["High"],
    )
    weather_conditions: Literal["Sunny", "Stormy", "Sandstorms", "Windy", "Fog", "Cloudy"] = Field(
        ...,
        description="Atmospheric weather condition at transit time",
        examples=["Sunny"],
    )
    type_of_order: Literal["Snack", "Meal", "Drinks", "Buffet"] = Field(
        "Meal",
        description="Category of the food order",
        examples=["Meal"],
    )
    type_of_vehicle: Literal["motorcycle", "scooter", "electric_scooter", "bicycle"] = Field(
        "motorcycle",
        description="Type of delivery transit vehicle",
        examples=["motorcycle"],
    )
    multiple_deliveries: int = Field(
        0,
        ge=0,
        le=5,
        description="Number of concurrent multi-drop packages assigned",
        examples=[1],
    )
    festival: Literal["No", "Yes"] = Field(
        "No",
        description="Whether current day is a festival/holiday surge event",
        examples=["No"],
    )
    city: Literal["Metropolitian", "Urban", "Semi-Urban"] = Field(
        "Metropolitian",
        description="Density classification of delivery municipality",
        examples=["Metropolitian"],
    )
    vehicle_condition: int = Field(
        1,
        ge=0,
        le=3,
        description="Condition status index of delivery vehicle (0-3)",
        examples=[1],
    )


class DeliveryPredictionResponse(BaseModel):
    """Schema for estimated delivery duration output."""

    estimated_delivery_minutes: float = Field(
        ...,
        description="Predicted delivery arrival time in minutes",
        examples=[26.4],
    )
    estimated_range_min: float = Field(
        ...,
        description="Lower bound ETA estimate in minutes (estimate - 1 RMSE)",
        examples=[22.5],
    )
    estimated_range_max: float = Field(
        ...,
        description="Upper bound ETA estimate in minutes (estimate + 1 RMSE)",
        examples=[30.3],
    )
    transit_distance_km: float = Field(
        ...,
        description="Calculated geodesic transit distance via Haversine formula",
        examples=[4.52],
    )
    confidence_score: float = Field(
        0.95,
        description="Statistical confidence interval level (e.g. 95%)",
        examples=[0.95],
    )
    traffic_level: str = Field(
        ...,
        description="Traffic density tier utilized during estimation",
        examples=["High"],
    )


class BatchDeliveryOrderInput(BaseModel):
    """Batch input schema containing a list of delivery requests."""

    orders: List[DeliveryOrderInput]


class BatchDeliveryPredictionResponse(BaseModel):
    """Batch response schema returning list of estimated delivery predictions."""

    predictions: List[DeliveryPredictionResponse]
    total_orders_processed: int


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_version: str
    model_loaded: bool
    service: str
    timestamp: str


class ModelMetricsResponse(BaseModel):
    """Schema displaying validation metrics and model architecture info."""

    project: str
    author: str
    version: str
    algorithm: str
    r2_score: float
    adjusted_r2: float
    rmse_minutes: float
    mae_minutes: float
    feature_count: int
    features: List[str]
