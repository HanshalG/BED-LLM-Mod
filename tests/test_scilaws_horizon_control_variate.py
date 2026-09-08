import pytest

from environments.chembench_mopen.horizon import HorizonPlanner
from environments.scilaws.family_oracle_bound import family_oracle_bound
from environments.scilaws.horizon_control_variate import HorizonControlVariateMixture
from environments.scilaws.regression_belief import RegressionBelief


def model():
    b = RegressionBelief([0.0], [[1.0]], 3.0, 0.2)
    return HorizonControlVariateMixture(
        [[[1.0], [2.0]]],
        [[[1.0], [3.0]]],
        [b],
        [1.0],
        target_weights=[0.25, 0.75],
        quadrature_order=4,
        include_observation_noise=True,
    )


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("mode", ["adaptive", "open_loop"])
def test_single_family_deep_analytic_risk_and_tree(depth, mode):
    m = model()
    p = HorizonPlanner(m).plan(m.initial_state, depth, mode=mode, allow_repeats=True)
    assert p.root.expected_risk == pytest.approx(
        0.1 * (1 + 7 / (1 + 4 * depth)), abs=1e-12
    )
    assert p.root.action == 1

    def verify(node):
        if node.branches:
            value = (
                sum(e.probability * verify(e.child) for e in node.branches)
                + node.quadrature_correction
            )
            assert value == pytest.approx(node.expected_risk, abs=1e-12)
        return node.expected_risk

    assert verify(p.root) == pytest.approx(
        min(v for _, v in p.root_action_values), abs=1e-12
    )


def test_mixture_correction_matches_scalar_family_potential():
    a, b = (
        RegressionBelief([-1.0], [[2.0]], 3.0, 1.0),
        RegressionBelief([1.0], [[1.0]], 3.0, 2.0),
    )
    m = HorizonControlVariateMixture(
        [[[1.0], [2.0]], [[2.0], [1.0]]],
        [[[1.0]], [[1.0]]],
        [a, b],
        [0.4, 0.6],
        target_weights=[1.0],
        quadrature_order=4,
    )
    state = m.condition(m.initial_state, 0, 0.7)
    for depth in (2, 3):
        for action in (0, 1):
            exact = family_oracle_bound(m, state, depth, first_action=action)["value"]
            sampled = sum(
                r.probability * family_oracle_bound(m, r.state, depth - 1)["value"]
                for r in m.branches(state, action)
            )
            assert m.horizon_chance_risk_correction(
                state, action, depth
            ) == pytest.approx(exact - sampled, abs=1e-12)
    assert m._coefficient.cache_info().maxsize == 1024
    with pytest.raises(ValueError):
        m.horizon_chance_risk_correction(state, 0, 4)
