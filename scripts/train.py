"""End-to-end model training script for DispatchIQ."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dispatch_iq.data import prepare_training_data, split_data
from dispatch_iq.features import FeatureTransformer
from dispatch_iq.model import DispatchIQModel


def run_training(dataset_path=None, output_dir="models"):
    """Executes the complete training workflow and persists production artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  DispatchIQ - Last-Mile Delivery ETA Training Pipeline")
    print("=" * 65)

    start_time = time.time()
    print("[1/4] Loading and cleaning raw dataset...")
    X, y = prepare_training_data(dataset_path)
    print(f"      Sanitized sample count: {len(X):,} orders")

    print("[2/4] Splitting train (80%) and held-out test (20%) sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"      Training samples : {len(X_train):,}")
    print(f"      Test samples     : {len(X_test):,}")

    print("[3/4] Fitting FeatureTransformer (ordinal + one-hot + scaling)...")
    transformer = FeatureTransformer()
    transformer.fit(X_train)

    print("[4/4] Fitting tuned XGBoost Regressor estimator...")
    model = DispatchIQModel()
    model.fit(X_train, y_train, transformer=transformer)

    print("Evaluating held-out test set...")
    metrics = model.evaluate(X_test, y_test)

    model_file = output_path / "xgb_best.pkl"
    scaler_file = output_path / "scaler.pkl"
    meta_file = output_path / "metadata.json"

    print("Persisting production artifacts...")
    model.save(model_file, scaler_file, meta_file)

    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"Training completed successfully in {elapsed:.2f} seconds!")
    print(f"  Test R² Score         : {metrics['r2_score']}")
    print(f"  Adjusted R² Score    : {metrics['adjusted_r2']}")
    print(f"  Test RMSE (minutes)   : {metrics['rmse_minutes']}")
    print(f"  Test MAE (minutes)    : {metrics['mae_minutes']}")
    print(f"  Saved Model           : {model_file}")
    print(f"  Saved Scaler          : {scaler_file}")
    print(f"  Saved Metadata        : {meta_file}")
    print("=" * 65)

    return metrics


if __name__ == "__main__":
    run_training()
