from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from raincast import __version__
from raincast.explain import explain_prediction
from raincast.features import FEATURES


class WeatherFeatures(BaseModel):
    avg_temp_c: float = Field(ge=-60, le=60)
    avg_humidity_pct: float = Field(ge=0, le=100)
    avg_pressure_hpa: float = Field(ge=850, le=1100)
    pressure_change_hpa: float = Field(ge=-100, le=100)
    rainfall_7d_mm: float = Field(ge=0, le=1000)
    avg_wind_kph: float = Field(ge=0, le=300)


class Contribution(BaseModel):
    feature: str
    probability_change: float
    direction: Literal["increases", "decreases"]


class Prediction(BaseModel):
    will_rain: bool
    rain_probability: float
    explanation: list[Contribution]
    model_version: str


def model_path() -> Path:
    return Path(os.getenv("RAINCAST_MODEL_PATH", "artifacts/rain_model.joblib"))


@lru_cache
def load_bundle():
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run: python -m raincast.train")
    return joblib.load(path)


app = FastAPI(title="RainCast API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: WeatherFeatures) -> Prediction:
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    values = features.model_dump() if hasattr(features, "model_dump") else features.dict()
    frame = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
    probability = float(bundle["model"].predict_proba(frame)[0, 1])
    contributions = explain_prediction(bundle["model"], frame, bundle["reference_values"])
    return Prediction(
        will_rain=probability >= 0.5,
        rain_probability=round(probability, 4),
        explanation=[Contribution(**vars(item)) for item in contributions],
        model_version=bundle["model_version"],
    )
