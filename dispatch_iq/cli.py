"""Command-line interface (CLI) for DispatchIQ."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path for direct CLI execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        prog="dispatch-iq",
        description="DispatchIQ: Intelligent Last-Mile Delivery ETA Prediction Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train model and persist artifacts")
    train_parser.add_argument("--data", type=str, default=None, help="Path to food delivery CSV dataset")
    train_parser.add_argument("--output-dir", type=str, default="models", help="Directory to save models")

    # Command: predict
    predict_parser = subparsers.add_parser("predict", help="Predict delivery duration for an order")
    predict_parser.add_argument("--age", type=int, default=28, help="Rider age")
    predict_parser.add_argument("--rating", type=float, default=4.8, help="Rider rating (1.0-5.0)")
    predict_parser.add_argument("--lat1", type=float, default=12.9352, help="Restaurant latitude")
    predict_parser.add_argument("--lon1", type=float, default=77.6245, help="Restaurant longitude")
    predict_parser.add_argument("--lat2", type=float, default=12.9716, help="Delivery latitude")
    predict_parser.add_argument("--lon2", type=float, default=77.6412, help="Delivery longitude")
    predict_parser.add_argument("--hour", type=int, default=19, help="Order placement hour (0-23)")
    predict_parser.add_argument("--traffic", type=str, default="High", choices=["Low", "Medium", "High", "Jam"])
    predict_parser.add_argument("--weather", type=str, default="Sunny")
    predict_parser.add_argument("--vehicle", type=str, default="motorcycle")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI REST server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_parser.add_argument("--reload", action="store_true", help="Enable live auto-reload")

    args = parser.parse_args()

    if args.command == "train":
        from scripts.train import run_training
        print("Starting DispatchIQ model training pipeline...")
        metrics = run_training(dataset_path=args.data, output_dir=args.output_dir)
        print("Training complete! Evaluation metrics:")
        print(json.dumps(metrics, indent=2))

    elif args.command == "predict":
        from dispatch_iq.model import DispatchIQPredictor
        predictor = DispatchIQPredictor.get_instance()
        sample_order = {
            "delivery_person_age": args.age,
            "delivery_person_ratings": args.rating,
            "restaurant_latitude": args.lat1,
            "restaurant_longitude": args.lon1,
            "delivery_location_latitude": args.lat2,
            "delivery_location_longitude": args.lon2,
            "order_hour": args.hour,
            "road_traffic_density": args.traffic,
            "weather_conditions": args.weather,
            "type_of_order": "Meal",
            "type_of_vehicle": args.vehicle,
            "multiple_deliveries": 0,
            "festival": "No",
            "city": "Metropolitian",
            "vehicle_condition": 1,
        }
        res = predictor.predict_single(sample_order)
        print("\n--- DispatchIQ Prediction Result ---")
        print(f"Calculated Geodesic Distance : {res['transit_distance_km']} km")
        print(f"Predicted Delivery Time      : {res['estimated_delivery_minutes']} minutes")
        print(f"Confidence Range (95%)       : [{res['estimated_range_min']} - {res['estimated_range_max']}] min")
        print(f"Traffic Density Level        : {res['traffic_level']}")

    elif args.command == "serve":
        import uvicorn
        print(f"Starting DispatchIQ API server at http://{args.host}:{args.port} (Swagger docs: /docs)")
        uvicorn.run("dispatch_iq.api:app", host=args.host, port=args.port, reload=args.reload)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
