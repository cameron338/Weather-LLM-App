from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from raincast.features import FEATURES


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    probability_change: float
    direction: str


def explain_prediction(
    model: Any, observation: pd.DataFrame, reference_values: dict[str, float]
) -> list[FeatureContribution]:
    """Estimate local sensitivity by replacing each feature with its reference median.

    Contributions are independent perturbations and are not expected to sum to the
    prediction. Positive values mean the observed feature increased rain probability
    relative to its training-period median.
    """
    missing = [name for name in FEATURES if name not in reference_values]
    if missing:
        raise ValueError(f"reference values missing features: {', '.join(missing)}")
    if len(observation) != 1:
        raise ValueError("explanations require exactly one observation")

    predicted_probability = float(model.predict_proba(observation[FEATURES])[0, 1])
    contributions = []
    for feature in FEATURES:
        counterfactual = observation[FEATURES].astype(float).copy()
        counterfactual.loc[counterfactual.index[0], feature] = reference_values[feature]
        reference_probability = float(model.predict_proba(counterfactual)[0, 1])
        change = predicted_probability - reference_probability
        contributions.append(
            FeatureContribution(
                feature=feature,
                probability_change=round(change, 4),
                direction="increases" if change >= 0 else "decreases",
            )
        )
    return sorted(contributions, key=lambda item: abs(item.probability_change), reverse=True)
