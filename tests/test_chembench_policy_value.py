import numpy as np
import pytest

from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel
from environments.chembench_mopen.horizon import HorizonPlanner, SearchLimitExceeded
from environments.chembench_mopen.policy_value import evaluate_myopic_policy


def model():
    return EnvelopeGaussianModel(
        [[-0.4, -0.2, 0.1], [0.6, 0.3, -0.5]],
        1,
        [[0, 1], [1, 0.2]],
        [0.4, 0.6],
        branch_count=4,
    )


def scalar_policy_value(m, state, menu, budget):
    if not budget:
        return m.risk(state)
    plan = HorizonPlanner(m).plan(state, 1, available=menu)
    action = plan.root.action
    return sum(
        b.probability
        * scalar_policy_value(
            m, b.state, tuple(a for a in menu if a != action), budget - 1
        )
        for b in m.branches(state, action)
    )


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
def test_matches_independent_scalar_policy_execution(budget):
    m = model()
    result = evaluate_myopic_policy(m, m.initial_state, budget)
    expected = scalar_policy_value(m, m.initial_state, (0, 1, 2), budget)
    assert result.value == pytest.approx(expected, abs=1e-12)
    assert result.measurement_budget == budget


def test_full_budget_is_not_immediate_root_value():
    m = model()
    one = evaluate_myopic_policy(m, m.initial_state, 1)
    three = evaluate_myopic_policy(m, m.initial_state, 3)
    assert one.root_action == three.root_action
    assert one.root_one_step_values == three.root_one_step_values
    assert three.value < one.value


def test_caps_and_batch_invariance():
    m = model()
    a = evaluate_myopic_policy(m, m.initial_state, 3, batch_size=1)
    b = evaluate_myopic_policy(m, m.initial_state, 3, batch_size=64)
    assert a.value == pytest.approx(b.value, abs=1e-12)
    for options in ({"max_states": 1}, {"max_workspace_bytes": 1}):
        with pytest.raises(SearchLimitExceeded):
            evaluate_myopic_policy(m, m.initial_state, 3, **options)
    with pytest.raises(ValueError):
        evaluate_myopic_policy(m, m.initial_state, 3, available=[0, 1])


def test_fixed_observations_commute_but_round_indexed_noise_can_change_them():
    m = model()
    a = m.condition(m.condition(m.initial_state, 0, 0.2), 1, -0.3)
    b = m.condition(m.condition(m.initial_state, 1, -0.3), 0, 0.2)
    np.testing.assert_allclose(a, b, atol=1e-12)
    changed = m.condition(m.condition(m.initial_state, 1, 0.8), 0, -0.7)
    assert not np.allclose(m.forecast(a), m.forecast(changed))


def test_three_step_value_of_receding_h2_is_h3_value_at_h2_root():
    m = model()
    planner = HorizonPlanner(m)

    def deployed(state, menu, remaining):
        if not remaining:
            return m.risk(state)
        action = planner.plan(state, min(2, remaining), available=menu).root.action
        return sum(
            b.probability
            * deployed(b.state, tuple(a for a in menu if a != action), remaining - 1)
            for b in m.branches(state, action)
        )

    h2 = planner.plan(m.initial_state, 2)
    h3 = planner.plan(m.initial_state, 3)
    assert deployed(m.initial_state, (0, 1, 2), 3) == pytest.approx(
        dict(h3.root_action_values)[h2.root.action], abs=1e-12
    )
