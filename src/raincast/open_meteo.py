from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

import httpx
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
)


class WeatherDataError(RuntimeError):
    """Raised when historical weather data cannot be retrieved or parsed."""


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    timezone: str = "auto"


class OpenMeteoClient:
    """Small, injectable client for Open-Meteo's historical weather API."""

    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        self._client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch_hourly(
        self, location: Location, start_date: date, end_date: date
    ) -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date")

        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": location.timezone,
        }
        try:
            response = self._client.get(ARCHIVE_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            return _parse_hourly(payload)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise WeatherDataError(f"Unable to load Open-Meteo observations: {exc}") from exc


def _parse_hourly(payload: Mapping[str, Any]) -> pd.DataFrame:
    hourly = payload["hourly"]
    expected = ("time",) + HOURLY_VARIABLES
    missing = [key for key in expected if key not in hourly]
    if missing:
        raise KeyError(f"response missing hourly fields: {', '.join(missing)}")

    frame = pd.DataFrame({key: hourly[key] for key in expected})
    if frame.empty:
        raise ValueError("response contains no hourly observations")
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    numeric_columns = list(HOURLY_VARIABLES)
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return frame.set_index("time").sort_index()
