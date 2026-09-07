import numpy as np
import pytest

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.chembench_mopen.raw_belief import GaussianParticleModel
from environments.chembench_mopen.raw_horizon import plan_raw_horizon
from environments.chembench_mopen.raw_integration import predictive_expectation


def test_two_measurements_match_independent_sufficient_statistic():
    model = GaussianParticleModel([[-0.3, -0.3], [0.3, 0.3]], 1, [[0], [1]], [0.5, 0.5])
    # Gaussian average is a sufficient statistic with sigma/sqrt(2).
    reference = GaussianParticleModel(
        [[-0.3], [0.3]], 1 / np.sqrt(2), [[0], [1]], [0.5, 0.5]
    )
    expected = predictive_expectation(
        reference,
        reference.initial_state,
        0,
        reference.risk,
        value_bound=0.25,
        tolerance=1e-8,
    )
    plan = plan_raw_horizon(
        model,
        model.initial_state,
        2,
        tolerance=1e-5,
        max_seconds=25,
        max_evaluations=1_000_000,
    )
    assert plan.value == pytest.approx(expected.value, abs=1e-5)
    assert plan.effective_horizon == 2
    assert plan.evaluations > expected.evaluations
    assert not plan.selection_resolved  # exchangeable designs


@pytest.mark.parametrize("mode", ["adaptive", "open_loop"])
def test_depth_three_with_uninformative_actions_matches_single_measurement(mode):
    model = GaussianParticleModel(
        [[-0.5, 0, 0], [0.5, 0, 0]], 1, [[0], [1]], [0.5, 0.5]
    )
    expected = predictive_expectation(
        model, model.initial_state, 0, model.risk, value_bound=0.25
    )
    plan = plan_raw_horizon(model, model.initial_state, 3, mode=mode)
    assert plan.effective_horizon == 3
    assert plan.value == pytest.approx(expected.value, abs=1e-5)
    assert (plan.fixed_sequence is not None) == (mode == "open_loop")


def test_global_budget_and_zero_horizon():
    model = GaussianParticleModel([[-0.3, -0.3], [0.3, 0.3]], 1, [[0], [1]], [0.5, 0.5])
    with pytest.raises(SearchLimitExceeded):
        plan_raw_horizon(model, model.initial_state, 2, max_evaluations=100)
    plan = plan_raw_horizon(model, model.initial_state, 0)
    assert plan.action is None and plan.evaluations == 0
    assert plan.value == 0.25
    with pytest.raises(ValueError):
        plan_raw_horizon(model, model.initial_state, 1, available=(0, 0))
