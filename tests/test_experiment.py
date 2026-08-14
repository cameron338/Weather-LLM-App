import pytest

from raincast.demo_data import generate_demo_dataset
from raincast.experiment import chronological_split, run_experiment


def test_chronological_split_keeps_future_data_out_of_training():
    splits = chronological_split(generate_demo_dataset(rows=100))

    assert len(splits.train) == 60
    assert len(splits.validation) == 20
    assert len(splits.test) == 20
    assert splits.train.index.max() < splits.validation.index.min()
    assert splits.validation.index.max() < splits.test.index.min()


def test_chronological_split_rejects_invalid_fractions():
    with pytest.raises(ValueError, match="leave observations"):
        chronological_split(
            generate_demo_dataset(rows=100), train_fraction=0.8, validation_fraction=0.2
        )


def test_experiment_compares_baseline_and_candidate_models():
    _, report = run_experiment(generate_demo_dataset(rows=500))

    assert set(report["validation"]) == {
        "class_prior",
        "logistic_regression",
        "hist_gradient_boosting",
    }
    selected_score = report["validation"][report["selected_model"]]["brier_score"]
    assert selected_score == min(
        result["brier_score"] for result in report["validation"].values()
    )
    assert 0 <= report["test"]["brier_score"] <= 1
    assert report["periods"]["train"][1] < report["periods"]["validation"][0]


def test_experiment_is_deterministic():
    data = generate_demo_dataset(rows=300)
    _, first = run_experiment(data)
    _, second = run_experiment(data)

    assert first == second
