# RainCast baseline model card

## Summary

RainCast estimates whether measurable precipitation will occur tomorrow from a seven-day summary of recent weather. Version 0.2 selects among a class-prior baseline, standardized logistic regression, and histogram gradient boosting. Logistic regression was selected as the best-calibrated validation candidate.

## Intended use

The model is an educational portfolio project and supports low-stakes planning demonstrations. It is not suitable for emergency warnings, aviation, agriculture, flood response, or other safety-critical decisions. Operational weather services use physical forecasting systems and substantially richer observations.

## Data

The production pipeline retrieves hourly observations from the Open-Meteo Historical Weather API. The baseline artifact was trained on London data from 1 January 2020 through 31 December 2024.

Inputs are seven-day aggregates of:

- Mean temperature
- Mean relative humidity
- Mean surface pressure
- Surface-pressure change
- Total precipitation
- Mean wind speed

A day is discarded when fewer than 18 hourly temperature observations are present. A positive label means the following day recorded at least 0.1 mm of precipitation.

The generated artifact records source coordinates, date range, feature names, training period, and evaluation metrics.

## Evaluation

Data is ordered by date. The oldest 60% trains candidate models, the next 20% selects the lowest validation Brier score, and the newest 20% is an untouched final test. The selected model is refitted on the combined train and validation periods before final evaluation.

Validation comparison:

| Model | ROC AUC | Brier score |
|---|---:|---:|
| Class-prior baseline | 0.5000 | 0.2367 |
| Logistic regression | **0.6960** | **0.2123** |
| Histogram gradient boosting | 0.6628 | 0.2312 |

Selected logistic-regression results on the untouched 2024 test period:

| Metric | Value |
|---|---:|
| ROC AUC | 0.6836 |
| Brier score | 0.2057 |
| Holdout observations | 364 |
| Holdout positive rate | 0.6538 |

ROC AUC measures ranking quality, where 0.5 is random and 1.0 is perfect. Brier score is the mean squared error of predicted probabilities, where lower is better.

## Limitations

- The model is trained for one geographic point and should not be assumed to generalize elsewhere.
- A seven-day statistical summary omits radar, satellite, atmospheric-profile, and numerical forecast data.
- Historical reanalysis observations can differ from measurements available in real time.
- Weather patterns shift by season and over longer climate periods.
- The fixed 0.1 mm threshold treats trace and heavy rainfall identically.

## Next evaluation steps

- Compare against persistence and seasonal baselines.
- Measure calibration by probability bucket and season.
- Test geographic generalization across several UK cities.
