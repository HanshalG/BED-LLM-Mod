import numpy as np
import pytest

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.chembench_mopen.raw_belief import GaussianParticleModel
from environments.chembench_mopen.raw_integration import predictive_expectation


def test_heteroscedastic_reference_and_close_choice():
    # Independent density-integral value banked in the previous diagnostic.
    model = GaussianParticleModel([[-0.2], [0.2]], [[0.3], [2]], [[0], [1]], [0.3, 0.7])
    estimate = predictive_expectation(
        model, model.initial_state, 0, model.risk, value_bound=0.25
    )
    assert estimate.value == pytest.approx(0.10822405486108717, abs=1e-7)
    assert estimate.quadrature_error + estimate.tail_bound <= 1e-6
    # Close Gaussian choices: a larger signal has strictly smaller Bayes risk.
    model = GaussianParticleModel(
        [[-0.5, -0.501], [0.5, 0.501]], 1, [[0], [1]], [0.5, 0.5]
    )
    values = [
        predictive_expectation(
            model, model.initial_state, a, model.risk, value_bound=0.25, tolerance=1e-8
        )
        for a in (0, 1)
    ]
    gap = values[0].value - values[1].value
    assert 0 < gap < 0.001
    assert gap > sum(v.quadrature_error + v.tail_bound for v in values)


def test_constant_continuation_and_tail():
    model = GaussianParticleModel([[-20], [3]], [[0.01], [20]], [[0], [1]], [0.4, 0.6])
    result = predictive_expectation(
        model, model.initial_state, 0, lambda _: 0.25, value_bound=0.25
    )
    assert result.value == pytest.approx(0.25, abs=1e-10)


def test_resource_and_semantic_failures():
    model = GaussianParticleModel([[-1], [1]], 1, [[0], [1]], [0.5, 0.5])
    with pytest.raises(SearchLimitExceeded):
        predictive_expectation(
            model,
            model.initial_state,
            0,
            model.risk,
            value_bound=0.25,
            max_evaluations=1,
        )
    for value in (-1, np.nan, 1):
        with pytest.raises(ValueError):
            predictive_expectation(
                model, model.initial_state, 0, lambda _: value, value_bound=0.25
            )
    with pytest.raises(ValueError):
        predictive_expectation(
            model, model.initial_state, 0, model.risk, value_bound=0.25, tolerance=0
        )
