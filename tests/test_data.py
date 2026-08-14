import numpy as np
import pandas as pd
import pytest

from raincast.data import build_training_dataset
from raincast.demo_data import generate_demo_dataset
from raincast.features import FEATURES


def test_generated_data_has_expected_shape_and_target():
    data = generate_demo_dataset(rows=100, seed=7)
    assert list(data.columns) == FEATURES + ["will_rain"]
    assert len(data) == 100
    assert set(data["will_rain"].unique()).issubset({0, 1})


def test_generated_data_is_reproducible():
    first = generate_demo_dataset(rows=20, seed=3)
    second = generate_demo_dataset(rows=20, seed=3)
    assert first.equals(second)


def _complete_hourly_days(days: int = 10) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    day_number = np.repeat(np.arange(days), 24)
    return pd.DataFrame(
        {
            "temperature_2m": 10 + day_number,
            "relative_humidity_2m": 70 + day_number,
            "surface_pressure": 1000 + day_number,
            "precipitation": np.where(day_number == 7, 0.2, 0.0),
            "wind_speed_10m": 12 + day_number,
        },
        index=index,
    )


def test_build_training_dataset_uses_past_seven_days_only():
    dataset = build_training_dataset(_complete_hourly_days())

    first = dataset.iloc[0]
    assert dataset.index[0] == pd.Timestamp("2024-01-07")
    assert first["avg_temp_c"] == pytest.approx(13.0)
    assert first["pressure_change_hpa"] == pytest.approx(6.0)
    assert first["rainfall_7d_mm"] == pytest.approx(0.0)
    assert first["will_rain"] == 1  # Rain belongs to Jan 8, the next-day label.


def test_build_training_dataset_drops_windows_with_incomplete_days():
    hourly = _complete_hourly_days()
    hourly = hourly.drop(hourly.loc["2024-01-04"].index[:7])

    dataset = build_training_dataset(hourly)

    assert pd.Timestamp("2024-01-07") not in dataset.index


def test_build_training_dataset_validates_schema():
    with pytest.raises(ValueError, match="wind_speed_10m"):
        build_training_dataset(_complete_hourly_days().drop(columns="wind_speed_10m"))
