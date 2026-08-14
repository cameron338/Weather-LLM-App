from __future__ import annotations

import numpy as np
import pandas as pd

from raincast.features import FEATURES


def generate_weather_data(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create deterministic weather observations for a runnable MVP.

    This is deliberately isolated so a real historical-data loader can replace it
    without changing training or serving code.
    """
    rng = np.random.default_rng(seed)
    temp = rng.normal(13, 7, rows).clip(-8, 35)
    humidity = (78 - 0.7 * (temp - 13) + rng.normal(0, 10, rows)).clip(30, 100)
    pressure = rng.normal(1013, 10, rows)
    pressure_change = rng.normal(0, 5, rows)
    recent_rain = rng.gamma(1.3, 5, rows)
    wind = rng.gamma(2.5, 4, rows)

    # A hidden relationship gives the model a meaningful signal to learn.
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

    frame = pd.DataFrame(
        {
            "avg_temp_c": temp,
            "avg_humidity_pct": humidity,
            "avg_pressure_hpa": pressure,
            "pressure_change_hpa": pressure_change,
            "rainfall_7d_mm": recent_rain,
            "avg_wind_kph": wind,
            "will_rain": target,
        }
    )
    return frame[FEATURES + ["will_rain"]]

