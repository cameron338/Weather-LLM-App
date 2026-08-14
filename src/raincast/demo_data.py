from __future__ import annotations

import numpy as np
import pandas as pd

from raincast.features import FEATURES


def generate_demo_dataset(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create reproducible examples for offline development and CI only."""
    rng = np.random.default_rng(seed)
    temperature = rng.normal(13, 7, rows).clip(-8, 35)
    humidity = (78 - 0.7 * (temperature - 13) + rng.normal(0, 10, rows)).clip(30, 100)
    pressure = rng.normal(1013, 10, rows)
    pressure_change = rng.normal(0, 5, rows)
    recent_rain = rng.gamma(1.3, 5, rows)
    wind = rng.gamma(2.5, 4, rows)
    log_odds = (
        -2.4
        + 0.045 * (humidity - 65)
        - 0.09 * pressure_change
        + 0.055 * recent_rain
        + 0.025 * wind
        - 0.018 * (pressure - 1013)
    )
    rain_probability = 1 / (1 + np.exp(-log_odds))
    target = rng.binomial(1, rain_probability)
    index = pd.date_range("2000-01-01", periods=rows, freq="D", name="feature_window_end")

    return pd.DataFrame(
        {
            "avg_temp_c": temperature,
            "avg_humidity_pct": humidity,
            "avg_pressure_hpa": pressure,
            "pressure_change_hpa": pressure_change,
            "rainfall_7d_mm": recent_rain,
            "avg_wind_kph": wind,
            "will_rain": target,
        },
        index=index,
    )[FEATURES + ["will_rain"]]
