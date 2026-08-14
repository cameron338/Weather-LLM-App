# RainCast

RainCast is an end-to-end machine-learning service that estimates the probability of rain tomorrow from the previous seven days of weather. It demonstrates real-data ingestion, leakage-safe feature engineering, chronological evaluation, local prediction explanations, API serving, validation, testing, and CI.

The production training path uses historical hourly observations from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), which requires no API key. A separate deterministic demo source keeps tests and offline development reproducible.

## Architecture

```text
Open-Meteo -> hourly observations -> 7-day features -> classifier -> artifact -> FastAPI
```

Training compares a class-prior baseline, standardized logistic regression, and histogram gradient boosting. The oldest 60% of observations trains each candidate, the next 20% selects the lowest validation Brier score, and the newest 20% evaluates only the winner. This mirrors future deployment while keeping the final test period untouched during model selection.

See the [model card](docs/model-card.md) for provenance and limitations, and the [experiment report](reports/model-comparison.json) for the machine-readable results.

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

The response contains a ranked explanation. Each `probability_change` measures how the prediction changes when that feature alone is replaced with its training-period median:

```json
{
  "will_rain": true,
  "rain_probability": 0.8902,
  "explanation": [
    {
      "feature": "avg_pressure_hpa",
      "probability_change": 0.0996,
      "direction": "increases"
    }
  ],
  "model_version": "0.2.0"
}
```

These are local sensitivity estimates, not causal effects, and independent contributions are not additive.

## Test

```bash
ruff check .
pytest
```

## Roadmap

- Measure calibration by season and probability bucket
- Add a React/TypeScript interface with city search
- Containerize and deploy the API; monitor data and prediction drift
