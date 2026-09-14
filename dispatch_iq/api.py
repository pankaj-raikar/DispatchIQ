"""FastAPI REST API service for DispatchIQ delivery time inference."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dispatch_iq.model import DispatchIQPredictor
from dispatch_iq.schemas import (
    BatchDeliveryOrderInput,
    BatchDeliveryPredictionResponse,
    DeliveryOrderInput,
    DeliveryPredictionResponse,
    HealthResponse,
    ModelMetricsResponse,
)

app = FastAPI(
    title="DispatchIQ ETA Prediction Service",
    description=(
        "Production-grade microservice predicting food delivery duration and ETA using "
        "geodesic routing signals, operational parameters, and a tuned XGBoost regressor."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for cross-origin dashboard/frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Validates optional API key security header if configured in environment."""
    configured_key = os.getenv("DISPATCH_IQ_API_KEY")
    if configured_key:
        if not x_api_key or x_api_key != configured_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key authorization header.",
            )


@app.get("/", summary="Root index overview", tags=["System"])
def root():
    """Returns basic service descriptor and documentation link."""
    return {
        "service": "DispatchIQ",
        "description": "Intelligent Last-Mile Delivery ETA Prediction Service",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, summary="Service health check", tags=["System"])
def health_check():
    """Liveness probe verifying model availability."""
    try:
        predictor = DispatchIQPredictor.get_instance()
        loaded = predictor.model.is_trained
    except Exception:
        loaded = False

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_version="1.0.0",
        model_loaded=loaded,
        service="DispatchIQ",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/api/v1/model/metrics",
    response_model=ModelMetricsResponse,
    summary="Model performance metrics and architecture details",
    tags=["Model"],
)
def get_model_metrics():
    """Returns validated test metrics (R2, Adjusted R2, RMSE) and feature names."""
    predictor = DispatchIQPredictor.get_instance()
    m = predictor.model.metrics
    return ModelMetricsResponse(
        project="DispatchIQ",
        author="Pankaj Raikar",
        version="1.0.0",
        algorithm="XGBoost Regressor (Gradient Boosted Trees)",
        r2_score=m.get("r2_score", 0.829),
        adjusted_r2=m.get("adjusted_r2", 0.8288),
        rmse_minutes=m.get("rmse_minutes", 3.91),
        mae_minutes=m.get("mae_minutes", 3.12),
        feature_count=m.get("feature_count", 21),
        features=predictor.model.transformer.feature_names,
    )


@app.post(
    "/api/v1/predict",
    response_model=DeliveryPredictionResponse,
    summary="Predict delivery duration for a single order",
    tags=["Inference"],
)
def predict_delivery_time(
    order: DeliveryOrderInput,
    x_api_key: Optional[str] = Header(None),
):
    """Calculates geodesic distance via Haversine and computes ETA in minutes."""
    verify_api_key(x_api_key)
    try:
        predictor = DispatchIQPredictor.get_instance()
        result = predictor.predict_single(order.model_dump())
        return DeliveryPredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(exc)}",
        )


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchDeliveryPredictionResponse,
    summary="Predict delivery duration for a batch of orders",
    tags=["Inference"],
)
def predict_delivery_time_batch(
    batch_input: BatchDeliveryOrderInput,
    x_api_key: Optional[str] = Header(None),
):
    """Batch inference endpoint for high-throughput dispatch simulations."""
    verify_api_key(x_api_key)
    try:
        predictor = DispatchIQPredictor.get_instance()
        orders_data = [o.model_dump() for o in batch_input.orders]
        results = predictor.predict_batch(orders_data)
        preds = [DeliveryPredictionResponse(**r) for r in results]
        return BatchDeliveryPredictionResponse(
            predictions=preds,
            total_orders_processed=len(preds),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failure: {str(exc)}",
        )
