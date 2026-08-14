# RainCast

RainCast is an end-to-end machine-learning service that estimates the probability of rain tomorrow from a seven-day weather summary. It demonstrates data generation, model training, evaluation, API serving, validation, testing, and CI.

> **MVP status:** the pipeline currently uses deterministic synthetic weather observations so the project is reproducible and runnable without API keys. The next milestone replaces this source with historical public weather data and time-aware validation.

## Architecture

```text
weather data -> feature pipeline -> logistic regression -> saved artifact -> FastAPI
```

The initial logistic-regression model is intentionally interpretable and serves as a baseline for comparing tree-based models later. ROC AUC measures ranking quality; Brier score measures probability calibration.

## Run locally

Requires Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m raincast.train
uvicorn raincast.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

Example prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"avg_temp_c":12,"avg_humidity_pct":91,"avg_pressure_hpa":1004,"pressure_change_hpa":-8,"rainfall_7d_mm":28,"avg_wind_kph":19}'
```

## Test

```bash
ruff check .
pytest
```

## Roadmap

- Ingest historical observations from a public weather API
- Use chronological train/validation/test splits to prevent leakage
- Compare logistic regression with gradient-boosted trees
- Explain predictions with feature contributions
- Add a React/TypeScript interface with city search
- Containerize and deploy the API; monitor data and prediction drift

