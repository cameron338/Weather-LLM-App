from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from raincast.data import generate_weather_data
from raincast.features import FEATURES


def train_model(output_path: Path, rows: int = 5000) -> dict[str, float]:
    data = generate_weather_data(rows=rows)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data["will_rain"], test_size=0.2, random_state=42, stratify=data["will_rain"]
    )
    model = Pipeline(
        [("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
        "test_rows": len(y_test),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, output_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/rain_model.joblib"))
    parser.add_argument("--rows", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(train_model(args.output, args.rows), indent=2))


if __name__ == "__main__":
    main()

