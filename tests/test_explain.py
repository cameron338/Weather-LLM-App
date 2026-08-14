import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from raincast.explain import explain_prediction
from raincast.features import FEATURES


def test_explanation_ranks_absolute_probability_changes():
    training = pd.DataFrame(
        [
            [10, 40, 1010, 2, 0, 5],
            [10, 50, 1010, 1, 2, 6],
            [10, 80, 1010, -2, 20, 8],
            [10, 90, 1010, -3, 30, 9],
        ],
        columns=FEATURES,
    )
    target = [0, 0, 1, 1]
    model = LogisticRegression().fit(training, target)
    observation = pd.DataFrame([[10, 95, 1010, -4, 40, 10]], columns=FEATURES)
    references = {name: float(training[name].median()) for name in FEATURES}

    contributions = explain_prediction(model, observation, references)

    magnitudes = [abs(item.probability_change) for item in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert {item.feature for item in contributions} == set(FEATURES)


def test_explanation_validates_reference_schema():
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    observation = pd.DataFrame([[1] * len(FEATURES)], columns=FEATURES)

    with pytest.raises(ValueError, match="reference values missing"):
        explain_prediction(model, observation, {})
