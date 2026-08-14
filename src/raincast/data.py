from __future__ import annotations

import pandas as pd

from raincast.features import FEATURES

MIN_HOURS_PER_DAY = 18
RAIN_THRESHOLD_MM = 0.1


def build_training_dataset(hourly: pd.DataFrame) -> pd.DataFrame:
    """Convert hourly observations into leakage-safe next-day examples.

    Every row contains aggregates from a seven-day window ending at the index
    date. The label describes precipitation on the following calendar day.
    """
    required = {
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "precipitation",
        "wind_speed_10m",
    }
    missing = sorted(required.difference(hourly.columns))
    if missing:
        raise ValueError(f"hourly data missing columns: {', '.join(missing)}")
    if not isinstance(hourly.index, pd.DatetimeIndex):
        raise TypeError("hourly data must use a DatetimeIndex")

    daily = hourly.resample("1D").agg(
        temperature_mean=("temperature_2m", "mean"),
        humidity_mean=("relative_humidity_2m", "mean"),
        pressure_mean=("surface_pressure", "mean"),
        precipitation_sum=("precipitation", lambda values: values.sum(min_count=1)),
        wind_mean=("wind_speed_10m", "mean"),
        observed_hours=("temperature_2m", "count"),
    )

    # Do not let partial API days masquerade as representative daily weather.
    incomplete = daily["observed_hours"] < MIN_HOURS_PER_DAY
    measurement_columns = [column for column in daily.columns if column != "observed_hours"]
    daily.loc[incomplete, measurement_columns] = pd.NA

    next_day_rainfall = daily["precipitation_sum"].shift(-1)
    target = next_day_rainfall.ge(RAIN_THRESHOLD_MM).where(next_day_rainfall.notna())
    window = daily.rolling(window=7, min_periods=7)
    dataset = pd.DataFrame(
        {
            "avg_temp_c": window["temperature_mean"].mean(),
            "avg_humidity_pct": window["humidity_mean"].mean(),
            "avg_pressure_hpa": window["pressure_mean"].mean(),
            "pressure_change_hpa": daily["pressure_mean"]
            - daily["pressure_mean"].shift(6),
            "rainfall_7d_mm": window["precipitation_sum"].sum(),
            "avg_wind_kph": window["wind_mean"].mean(),
            "will_rain": target,
        }
    )
    dataset.index.name = "feature_window_end"
    dataset = dataset.dropna(subset=FEATURES + ["will_rain"])
    dataset["will_rain"] = dataset["will_rain"].astype(int)
    return dataset[FEATURES + ["will_rain"]]
