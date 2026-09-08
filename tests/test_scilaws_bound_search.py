import pytest

from environments.chembench_mopen.horizon import HorizonPlanner
from environments.scilaws.horizon_control_variate import HorizonControlVariateMixture
from environments.scilaws.regression_belief import RegressionBelief
from environments.scilaws.family_oracle_bound import family_oracle_bound


def make(mixed):
    beliefs = [RegressionBelief([0.0], [[1.0]], 3.0, 0.2)]
    features, targets, weights = [[[1.0], [2.0]]], [[[1.0]]], [1.0]
    if mixed:
        beliefs.append(RegressionBelief([0.5], [[2.0]], 3.0, 0.4))
        features.append([[2.0], [1.0]])
        targets.append([[1.0]])
        weights = [0.4, 0.6]
    return HorizonControlVariateMixture(
        features, targets, beliefs, weights, quadrature_order=4,
        include_observation_noise=True, target_weights=[1.0],
    )


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("repeats", [False, True])
def test_bounds_match_exhaustive(mixed, depth, repeats):
    m = make(mixed)
    state = m.condition(m.initial_state, 0, 0.3)
    full = HorizonPlanner(m).plan(state, depth, allow_repeats=repeats)
    bounded = HorizonPlanner(m).plan(
        state, depth, allow_repeats=repeats, use_action_bounds=True
    )

    def compare(a, b):
        assert a.action == b.action
        assert a.expected_risk == pytest.approx(b.expected_risk, abs=1e-12)
        assert len(a.branches) == len(b.branches)
        for x, y in zip(a.branches, b.branches, strict=True):
            assert x.observation == y.observation
            compare(x.child, y.child)

    compare(full.root, bounded.root)
    exact = dict(full.root_action_values)
    evaluated = dict(bounded.root_action_values)
    pruned = dict(bounded.root_pruned_lower_bounds)
    assert not evaluated.keys() & pruned.keys()
    assert evaluated.keys() | pruned.keys() == exact.keys()
    for a, value in evaluated.items():
        assert value == pytest.approx(exact[a], abs=1e-12)
    for a, lower in pruned.items():
        assert lower <= exact[a] + 1e-12
        assert lower > min(evaluated.values())
    if not mixed and repeats:
        assert bounded.pruned_actions > 0
        assert bounded.expanded_nodes < full.expanded_nodes
    for a in range(m.num_actions):
        assert m.action_risk_lower_bound(state, a, full.effective_horizon) == pytest.approx(
            family_oracle_bound(m, state, full.effective_horizon, first_action=a)["value"],
            abs=1e-12,
        )


def test_invalid_mode_and_bound_fail_closed():
    m = make(False)
    with pytest.raises(ValueError):
        HorizonPlanner(m).plan(m.initial_state, 2, mode="open_loop", use_action_bounds=True)
    m.action_risk_lower_bound = lambda *args: float("nan")
    with pytest.raises(ValueError, match="finite"):
        HorizonPlanner(m).plan(m.initial_state, 1, use_action_bounds=True)
    m.action_risk_lower_bound = lambda *args: 100.0
    with pytest.raises(ValueError, match="violates"):
        HorizonPlanner(m).plan(m.initial_state, 1, use_action_bounds=True)


def test_equal_bounds_preserve_ties_and_restricted_menu():
    m = make(False)
    m.action_risk_lower_bound = lambda *args: 0.0
    result = HorizonPlanner(m).plan(m.initial_state, 2, use_action_bounds=True)
    assert result.pruned_actions == 0
    assert len(result.root_action_values) == 2
    result = HorizonPlanner(m).plan(
        m.initial_state, 3, available=(0,), allow_repeats=True, use_action_bounds=True
    )
    assert result.root.action == 0
    assert not result.root_pruned_lower_bounds
