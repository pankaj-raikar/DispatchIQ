# DispatchIQ: Intelligent Last-Mile Delivery ETA Prediction

> High-throughput machine learning microservice for real-time delivery duration estimation and dispatch routing intelligence, powered by geodesic spatial calculations, gradient-boosted regression (XGBoost), and an asynchronous FastAPI serving layer.

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 1. Architectural Overview

DispatchIQ addresses the high-variance problem in urban last-mile food delivery operations. Estimating arrival time (ETA) requires synthesizing nonlinear geographical, temporal, and atmospheric signals:

```
[ Ingest Delivery Order ]
         │
         ├── Geodesic Distance Engine (Haversine Formula: R=6371km)
         ├── Temporal Discretization & Cyclical Feature Alignment
         ├── Ordinal Congestion Encoding (Low → Medium → High → Jam)
         └── Nominal One-Hot Projection (Drop-First Dummy Alignment)
         │
         ▼
[ Standardized Feature Matrix (21 Features) ]
         │
         ▼
[ Regularized Gradient Boosted Regressor (XGBoost) ]
         │
         ▼
[ ETA Prediction (Minutes) ± Uncertainty Bounds (RMSE Margin) ]
         │
         ├── FastAPI Asynchronous REST Endpoint (/api/v1/predict)
         └── Batch Dispatch Queue Simulator (/api/v1/predict/batch)
```

### Benchmark Comparison

Four candidate regression architectures were benchmarked on a stratified held-out test split (20% of 45,162 sanitized historical delivery records):

| Model Architecture | Test $R^2$ | Test RMSE (min) | Train $R^2$ | Architectural Diagnosis & Generalization Gap |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression (OLS Baseline)** | 0.5700 | 6.21 | 0.5800 | High bias; unable to capture complex road congestion interactions |
| **Decision Tree Regressor** | 0.6700 | 5.46 | 1.0000 | Extreme variance / catastrophic memorization of training leaves |
| **Random Forest (100 Trees)** | 0.8200 | 4.03 | 0.9700 | Significant variance reduction via bootstrap bagging; slight overfit |
| **XGBoost (Tuned & Regularized)** | **0.8287** | **3.9176** | **0.8450** | **Optimal variance-bias tradeoff; tight train-test gap (<0.02)** |

*Final RMSE is ~3.9 minutes, meaning ETA forecasts deviate by less than 4 minutes on real-world test trips.*

---

## 2. Key Engineering Highlights

1. **Haversine Geodesic Distance Matrix**:
   Turns raw restaurant/customer latitude-longitude coordinate pairs into true spherical great-circle distances ($R = 6371\text{ km}$):
   $$\Delta \sigma = 2 \arcsin \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}$$
   Physical anomaly filtering trims spurious GPS noise and invalid coordinates ($>100\text{ km}$).

2. **Leakage-Free Preprocessing Pipeline**:
   Numerical features (`distance_km`, `order_hour`, rider characteristics) are standardized with `StandardScaler` fitted strictly on training slices. Ordinal congestion encoding preserves monotonic ordering (`Low: 0`, `Medium: 1`, `High: 2`, `Jam: 3`).

3. **Controlled Feature Complexity Testing**:
   Tested synthetic `order_period` bucketing (Lunch, Dinner, Normal). Validated via **Adjusted $R^2$** penalization:
   $$R^2_{\text{adj}} = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}$$
   The marginal delta ($0.82877 \to 0.82878$) revealed minimal predictive utility, confirming the decision to preserve model parsimony and lower inference latency.

4. **Production Microservice Architecture**:
   Packaged as a modular Python package with asynchronous FastAPI routing, Pydantic v2 data contract validation, Swagger/OpenAPI documentation, health probes, and CLI tooling.

---

## 3. Tech Stack

- **Core Runtime**: Python 3.9+
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib
- **API Framework**: FastAPI, Pydantic v2, Uvicorn, Starlette
- **Testing & Verification**: Python Unittest, Pytest
- **Packaging**: Setuptools, Wheel, `pyproject.toml`

---

## 4. REST API Route Reference

DispatchIQ provides self-documenting REST endpoints accessible at `http://127.0.0.1:8000`. Interactive Swagger UI is available at `/docs`.

| Method | Endpoint | Description | Auth Level | Request Body | Response Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Service root descriptor and route index | Public | None | Service info JSON |
| `GET` | `/health` | Liveness and model loading status probe | Public | None | `HealthResponse` |
| `GET` | `/api/v1/model/metrics` | Model performance statistics ($R^2$, RMSE, MAE) | Public | None | `ModelMetricsResponse` |
| `POST` | `/api/v1/predict` | Predicts delivery duration for a single order | Optional API Key (`X-API-Key`) | `DeliveryOrderInput` | `DeliveryPredictionResponse` |
| `POST` | `/api/v1/predict/batch` | High-throughput batch order prediction | Optional API Key (`X-API-Key`) | `BatchDeliveryOrderInput` | `BatchDeliveryPredictionResponse` |

### Sample Inference Request (`POST /api/v1/predict`)

```json
{
  "delivery_person_age": 28,
  "delivery_person_ratings": 4.8,
  "restaurant_latitude": 12.9352,
  "restaurant_longitude": 77.6245,
  "delivery_location_latitude": 12.9716,
  "delivery_location_longitude": 77.6412,
  "order_hour": 19,
  "road_traffic_density": "High",
  "weather_conditions": "Sunny",
  "type_of_order": "Meal",
  "type_of_vehicle": "motorcycle",
  "multiple_deliveries": 0,
  "festival": "No",
  "city": "Metropolitian",
  "vehicle_condition": 1
}
```

### Sample Inference Response

```json
{
  "estimated_delivery_minutes": 22.5,
  "estimated_range_min": 18.6,
  "estimated_range_max": 26.4,
  "transit_distance_km": 4.43,
  "confidence_score": 0.95,
  "traffic_level": "High"
}
```

---

## 5. Repository Directory Structure

```
DispatchIQ/
├── .env.example                     # Environment configuration template
├── .gitignore                       # Production Git ignore rules
├── LICENSE                          # Apache 2.0 License
├── README.md                        # Production documentation
├── pyproject.toml                   # Modern PEP 517/518 build specification
├── requirements.txt                 # Pinned dependencies
├── data/
│   └── food_delivery.csv            # 45,593-record historical delivery dataset
├── dispatch_iq/                     # Core Python production package
│   ├── __init__.py                  # Package exports & version
│   ├── api.py                       # FastAPI serving application
│   ├── cli.py                       # Command line interface (CLI)
│   ├── config.py                    # Constants, hyperparams, feature schema
│   ├── data.py                      # Ingestion, cleaning & imputation
│   ├── distance.py                  # Geodesic Haversine calculation
│   ├── features.py                  # Feature encoding & alignment
│   ├── model.py                     # XGBoost wrapper & DispatchIQPredictor
│   └── schemas.py                   # Pydantic v2 validation contracts
├── models/
│   ├── metadata.json                # Model lineage, params, & test metrics
│   ├── scaler.pkl                   # Serialized StandardScaler
│   └── xgb_best.pkl                 # Serialized tuned XGBoost estimator
├── notebooks/
│   └── dispatch_iq_analysis.ipynb   # Reproducible exploratory & training notebook
├── scripts/
│   └── train.py                     # Standalone CLI training script
└── tests/
    ├── __init__.py                  # Test package
    ├── test_api.py                  # FastAPI route & validation tests
    ├── test_distance.py             # Haversine & coordinate boundary tests
    ├── test_features.py             # One-hot alignment & transformer tests
    └── test_model.py                # Model inference & artifact tests
```

---

## 6. Installation & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/pankaj-raikar/DispatchIQ.git
cd DispatchIQ
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# Or install in editable developer mode:
pip install -e .
```

### 4. Run the Test Suite
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 5. Train the Model & Persist Artifacts
```bash
python3 scripts/train.py
# Or via CLI:
python3 dispatch_iq/cli.py train
```

### 6. Launch the FastAPI Microservice
```bash
uvicorn dispatch_iq.api:app --host 127.0.0.1 --port 8000 --reload
# Or via CLI:
python3 dispatch_iq/cli.py serve --port 8000
```
Visit `http://127.0.0.1:8000/docs` to test endpoints via interactive Swagger UI.

### 7. Run CLI Inference Directly
```bash
python3 dispatch_iq/cli.py predict --age 28 --rating 4.8 --lat1 12.9352 --lon1 77.6245 --lat2 12.9716 --lon2 77.6412 --hour 19 --traffic High
```

---

## 7. Author

**Pankaj Raikar**  
- **Email**: [pankajraikar6@gmail.com](mailto:pankajraikar6@gmail.com)  
- **GitHub**: [@pankaj-raikar](https://github.com/pankaj-raikar)  
- **LinkedIn**: [Pankaj Raikar](https://linkedin.com)
