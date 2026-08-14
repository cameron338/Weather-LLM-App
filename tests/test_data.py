from raincast.data import generate_weather_data
from raincast.features import FEATURES


def test_generated_data_has_expected_shape_and_target():
    data = generate_weather_data(rows=100, seed=7)
    assert list(data.columns) == FEATURES + ["will_rain"]
    assert len(data) == 100
    assert set(data["will_rain"].unique()).issubset({0, 1})


def test_generated_data_is_reproducible():
    first = generate_weather_data(rows=20, seed=3)
    second = generate_weather_data(rows=20, seed=3)
    assert first.equals(second)

