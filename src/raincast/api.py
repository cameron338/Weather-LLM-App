from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from raincast.features import FEATURES


class WeatherFeatures(BaseModel):
    avg_temp_c: float = Field(ge=-60, le=60)
    avg_humidity_pct: float = Field(ge=0, le=100)
    avg_pressure_hpa: float = Field(ge=850, le=1100)
    pressure_change_hpa: float = Field(ge=-100, le=100)
    rainfall_7d_mm: float = Field(ge=0, le=1000)
    avg_wind_kph: float = Field(ge=0, le=300)


class Prediction(BaseModel):
    will_rain: bool
    rain_probability: float
    model_version: str = "0.1.0"


def model_path() -> Path:
    return Path(os.getenv("RAINCAST_MODEL_PATH", "artifacts/rain_model.joblib"))


@lru_cache
def load_bundle():
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run: python -m raincast.train")
    return joblib.load(path)


app = FastAPI(title="RainCast API", version="0.1.0")


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
    return Prediction(will_rain=probability >= 0.5, rain_probability=round(probability, 4))

