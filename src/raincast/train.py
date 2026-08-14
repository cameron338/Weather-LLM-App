from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from raincast import __version__
from raincast.data import build_training_dataset
from raincast.demo_data import generate_demo_dataset
from raincast.experiment import run_experiment
from raincast.features import FEATURES
from raincast.open_meteo import Location, OpenMeteoClient

DEFAULT_LOCATION = Location("London", latitude=51.5074, longitude=-0.1278)


def train_model(
    data: pd.DataFrame,
    output_path: Path,
    *,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Run model selection and persist the selected estimator with provenance."""
    ordered = data.sort_index()
    model, experiment = run_experiment(ordered)
    development_end = experiment["rows"]["train"] + experiment["rows"]["validation"]
    reference_values = {
        feature: float(ordered.iloc[:development_end][feature].median()) for feature in FEATURES
    }
    bundle = {
        "model_version": __version__,
        "model": model,
        "features": FEATURES,
        "metrics": experiment["test"],
        "experiment": experiment,
        "reference_values": reference_values,
        "source": source_metadata,
        "training_period": {
            "start": str(ordered.index.min()),
            "end": str(ordered.index.max()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return experiment


def load_open_meteo_dataset(
    location: Location, start_date: date, end_date: date
) -> pd.DataFrame:
    client = OpenMeteoClient()
    try:
        hourly = client.fetch_hourly(location, start_date, end_date)
    finally:
        client.close()
    return build_training_dataset(hourly)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare and train RainCast classifiers")
    parser.add_argument("--source", choices=("open-meteo", "demo"), default="open-meteo")
    parser.add_argument("--output", type=Path, default=Path("artifacts/rain_model.joblib"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--location", default=DEFAULT_LOCATION.name)
    parser.add_argument("--latitude", type=float, default=DEFAULT_LOCATION.latitude)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LOCATION.longitude)
    parser.add_argument("--rows", type=int, default=5000, help="Rows used only by --source demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "demo":
        dataset = generate_demo_dataset(rows=args.rows)
        source_metadata = {"type": "demo", "rows": args.rows}
    else:
        location = Location(args.location, args.latitude, args.longitude)
        dataset = load_open_meteo_dataset(location, args.start_date, args.end_date)
        source_metadata = {
            "type": "open-meteo",
            "location": location.name,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start_date": args.start_date.isoformat(),
            "end_date": args.end_date.isoformat(),
        }
    metrics = train_model(dataset, args.output, source_metadata=source_metadata)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
