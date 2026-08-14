from pathlib import Path

import joblib
import pytest

from raincast.demo_data import generate_demo_dataset
from raincast.train import train_model


def test_train_model_records_metrics_and_provenance(tmp_path: Path):
    output = tmp_path / "model.joblib"
    metrics = train_model(
        generate_demo_dataset(rows=500), output, source_metadata={"type": "test-fixture"}
    )

    bundle = joblib.load(output)
    assert output.exists()
    assert metrics["train_rows"] == 400
    assert metrics["test_rows"] == 100
    assert bundle["source"] == {"type": "test-fixture"}
    assert bundle["training_period"]["start"] < bundle["training_period"]["end"]


def test_train_model_rejects_tiny_dataset(tmp_path: Path):
    with pytest.raises(ValueError, match="at least 30"):
        train_model(
            generate_demo_dataset(rows=20),
            tmp_path / "model.joblib",
            source_metadata={"type": "test"},
        )
