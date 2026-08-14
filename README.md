# RainCast

RainCast is an end-to-end machine-learning service that estimates the probability of rain tomorrow from the previous seven days of weather. It demonstrates real-data ingestion, leakage-safe feature engineering, chronological evaluation, API serving, validation, testing, and CI.

The production training path uses historical hourly observations from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), which requires no API key. A separate deterministic demo source keeps tests and offline development reproducible.

## Architecture

```text
Open-Meteo -> hourly observations -> 7-day features -> classifier -> artifact -> FastAPI
```

The initial logistic-regression model is intentionally interpretable and serves as a baseline for comparing tree-based models later. Older observations are used for training and the newest 20% for testing, matching how the model would encounter future weather. ROC AUC measures ranking quality; Brier score measures probability calibration.

See the [model card](docs/model-card.md) for provenance, evaluation results, and limitations.

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

By default, training downloads London observations from 2020 through 2024. Choose another location or period with explicit coordinates:

```bash
python -m raincast.train \
  --location Manchester \
  --latitude 53.4808 \
  --longitude -2.2426 \
  --start-date 2020-01-01 \
  --end-date 2024-12-31
```

For offline development only:

```bash
python -m raincast.train --source demo --rows 5000
```

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

- Compare logistic regression with gradient-boosted trees
- Explain predictions with feature contributions
- Add a React/TypeScript interface with city search
- Containerize and deploy the API; monitor data and prediction drift
