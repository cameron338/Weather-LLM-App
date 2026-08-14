from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from raincast.data import build_training_dataset
from raincast.demo_data import generate_demo_dataset
from raincast.features import FEATURES
from raincast.open_meteo import Location, OpenMeteoClient

DEFAULT_LOCATION = Location("London", latitude=51.5074, longitude=-0.1278)


def train_model(
    data: pd.DataFrame,
    output_path: Path,
    *,
    source_metadata: dict[str, Any],
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    """Fit and evaluate the baseline using a chronological holdout."""
    if len(data) < 30:
        raise ValueError("training requires at least 30 complete examples")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    ordered = data.sort_index()
    split_at = int(len(ordered) * (1 - test_fraction))
    train = ordered.iloc[:split_at]
    test = ordered.iloc[split_at:]
    if train["will_rain"].nunique() < 2 or test["will_rain"].nunique() < 2:
        raise ValueError("both chronological splits must contain rainy and dry examples")

    model = Pipeline(
        [("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))]
    )
    model.fit(train[FEATURES], train["will_rain"])
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(test["will_rain"], probabilities)), 4),
        "brier_score": round(float(brier_score_loss(test["will_rain"], probabilities)), 4),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_positive_rate": round(float(test["will_rain"].mean()), 4),
    }
    bundle = {
        "model": model,
        "features": FEATURES,
        "metrics": metrics,
        "source": source_metadata,
        "training_period": {
            "start": str(ordered.index.min()),
            "end": str(ordered.index.max()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return metrics


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
    parser = argparse.ArgumentParser(description="Train the RainCast baseline classifier")
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
