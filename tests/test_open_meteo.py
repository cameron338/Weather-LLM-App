from datetime import date

import httpx
import pytest

from raincast.open_meteo import Location, OpenMeteoClient, WeatherDataError


def _payload():
    return {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [8.1, 7.9],
            "relative_humidity_2m": [83, 85],
            "surface_pressure": [1004.2, 1004.0],
            "precipitation": [0.0, 0.2],
            "wind_speed_10m": [12.0, 13.5],
        }
    }


def test_fetch_hourly_builds_expected_request_and_frame():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["latitude"] == "51.5074"
        assert request.url.params["start_date"] == "2024-01-01"
        assert "surface_pressure" in request.url.params["hourly"]
        return httpx.Response(200, json=_payload())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenMeteoClient(http_client)
    frame = client.fetch_hourly(
        Location("London", 51.5074, -0.1278), date(2024, 1, 1), date(2024, 1, 1)
    )

    assert list(frame.columns) == [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "precipitation",
        "wind_speed_10m",
    ]
    assert len(frame) == 2


def test_fetch_hourly_wraps_api_errors():
    transport = httpx.MockTransport(lambda _: httpx.Response(503))
    client = OpenMeteoClient(httpx.Client(transport=transport))

    with pytest.raises(WeatherDataError, match="Unable to load"):
        client.fetch_hourly(
            Location("London", 51.5074, -0.1278), date(2024, 1, 1), date(2024, 1, 2)
        )


def test_fetch_hourly_rejects_reversed_dates():
    client = OpenMeteoClient(httpx.Client(transport=httpx.MockTransport(lambda _: None)))

    with pytest.raises(ValueError, match="start_date"):
        client.fetch_hourly(
            Location("London", 51.5074, -0.1278), date(2024, 1, 2), date(2024, 1, 1)
        )
