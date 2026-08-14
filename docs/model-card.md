# RainCast baseline model card

## Summary

RainCast estimates whether measurable precipitation will occur tomorrow from a seven-day summary of recent weather. Version 0.1 uses standardized logistic regression as an interpretable baseline.

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

Data is ordered by date. The oldest 80% trains the model and the newest 20% forms an untouched holdout, preventing future observations from leaking into training.

London baseline results on the 2024 holdout:

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

- Add a validation period for model selection while preserving the final holdout.
- Compare against naive persistence and seasonal baselines.
- Measure calibration by probability bucket and season.
- Evaluate gradient-boosted trees using the same chronological splits.
- Test geographic generalization across several UK cities.
