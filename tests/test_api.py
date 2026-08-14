from pathlib import Path

from fastapi.testclient import TestClient

from raincast import api
from raincast.demo_data import generate_demo_dataset
from raincast.train import train_model


def test_health():
    assert TestClient(api.app).get("/health").json() == {"status": "ok"}


def test_prediction(tmp_path: Path, monkeypatch):
    path = tmp_path / "model.joblib"
    train_model(generate_demo_dataset(rows=800), path, source_metadata={"type": "test"})
    monkeypatch.setattr(api, "model_path", lambda: path)
    api.load_bundle.cache_clear()
    response = TestClient(api.app).post(
        "/predict",
        json={
            "avg_temp_c": 12,
            "avg_humidity_pct": 91,
            "avg_pressure_hpa": 1004,
            "pressure_change_hpa": -8,
            "rainfall_7d_mm": 28,
            "avg_wind_kph": 19,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["rain_probability"] <= 1
    assert isinstance(body["will_rain"], bool)
    assert len(body["explanation"]) == 6
    assert body["model_version"] == "0.2.0"
    assert [abs(item["probability_change"]) for item in body["explanation"]] == sorted(
        [abs(item["probability_change"]) for item in body["explanation"]], reverse=True
    )
