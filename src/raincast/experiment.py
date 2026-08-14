from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from raincast.features import FEATURES


@dataclass(frozen=True)
class ChronologicalSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    data: pd.DataFrame, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> ChronologicalSplits:
    """Split ordered observations without allowing future rows into earlier sets."""
    if len(data) < 60:
        raise ValueError("model comparison requires at least 60 complete examples")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave observations for testing")

    ordered = data.sort_index()
    train_end = int(len(ordered) * train_fraction)
    validation_end = train_end + int(len(ordered) * validation_fraction)
    splits = ChronologicalSplits(
        train=ordered.iloc[:train_end],
        validation=ordered.iloc[train_end:validation_end],
        test=ordered.iloc[validation_end:],
    )
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        if frame["will_rain"].nunique() < 2:
            raise ValueError(f"{name} split must contain rainy and dry examples")
    return splits


def candidate_models() -> dict[str, Any]:
    """Return deterministic candidate estimators with intentionally modest complexity."""
    return {
        "class_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline(
            [("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def probability_metrics(target: pd.Series, probabilities) -> dict[str, float]:
    return {
        "roc_auc": round(float(roc_auc_score(target, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(target, probabilities)), 4),
    }


def run_experiment(data: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    """Select on validation Brier score and evaluate the winner once on test data."""
    splits = chronological_split(data)
    validation_results = {}
    models = candidate_models()
    for name, model in models.items():
        model.fit(splits.train[FEATURES], splits.train["will_rain"])
        probabilities = model.predict_proba(splits.validation[FEATURES])[:, 1]
        validation_results[name] = probability_metrics(
            splits.validation["will_rain"], probabilities
        )

    selected_name = min(
        validation_results,
        key=lambda name: (validation_results[name]["brier_score"], name),
    )
    selected_model = clone(models[selected_name])
    development = pd.concat([splits.train, splits.validation])
    selected_model.fit(development[FEATURES], development["will_rain"])
    test_probabilities = selected_model.predict_proba(splits.test[FEATURES])[:, 1]
    report = {
        "selection_metric": "brier_score",
        "selected_model": selected_name,
        "validation": validation_results,
        "test": probability_metrics(splits.test["will_rain"], test_probabilities),
        "rows": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "periods": {
            "train": [str(splits.train.index.min()), str(splits.train.index.max())],
            "validation": [
                str(splits.validation.index.min()),
                str(splits.validation.index.max()),
            ],
            "test": [str(splits.test.index.min()), str(splits.test.index.max())],
        },
    }
    return selected_model, report
