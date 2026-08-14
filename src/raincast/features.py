from __future__ import annotations

FEATURES = [
    "avg_temp_c",
    "avg_humidity_pct",
    "avg_pressure_hpa",
    "pressure_change_hpa",
    "rainfall_7d_mm",
    "avg_wind_kph",
]


def validate_feature_order(values: dict[str, float]) -> list[float]:
    """Return model inputs in their canonical order."""
    return [float(values[name]) for name in FEATURES]

