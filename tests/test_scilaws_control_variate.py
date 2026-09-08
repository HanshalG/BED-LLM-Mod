import numpy as np
import pytest

from environments.chembench_mopen.horizon import HorizonPlanner
from environments.scilaws.control_variate import ControlVariateMixture
from environments.scilaws.regression_belief import RegressionBelief


def model():
    a = RegressionBelief([-1.0], [[2.0]], 3.0, 1.0)
    b = RegressionBelief([1.0], [[1.0]], 3.0, 2.0)
    return ControlVariateMixture(
        [[[1.0], [2.0]], [[2.0], [1.0]]],
        [[[1.0]], [[1.0]]],
        [a, b],
        [0.4, 0.6],
        target_weights=[1.0],
        quadrature_order=2,
    )


class ScalarCorrected:
    def __init__(self, m):
        self.m = m
        self.num_actions = m.num_actions
        self.risk = m.risk
        self.branches = m.branches

    def chance_risk_correction(self, state, action):
        m = self.m
        exact = 0.0
        for weight, b, x, target in zip(
            np.exp(state.log_weights),
            state.components,
            m.action_features,
            m.target_features,
        ):
            precision = np.asarray(b.precision) + np.outer(x[action], x[action])
            leverage = (
                np.diag(target @ np.linalg.inv(precision) @ target.T) @ m.target_weights
            )
            exact += (
                weight
                * b.noise_variance
                * (leverage + int(m.include_observation_noise))
            )
        sampled = 0.0
        for row in m.branches(state, action):
            within = 0.0
            for weight, b, target in zip(
                np.exp(row.state.log_weights), row.state.components, m.target_features
            ):
                within += weight * (
                    b.target_moments(target)[1] @ m.target_weights
                    + b.noise_variance * int(m.include_observation_noise)
                )
            sampled += row.probability * within
        return exact - sampled


@pytest.mark.parametrize("mode", ["adaptive", "open_loop"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_deep_scalar_equivalence_and_explicit_tree_reconstruction(mode, depth):
    m = model()
    fast = HorizonPlanner(m).plan(m.initial_state, depth, mode=mode, allow_repeats=True)
    slow = HorizonPlanner(ScalarCorrected(m)).plan(
        m.initial_state, depth, mode=mode, allow_repeats=True
    )
    np.testing.assert_allclose(
        fast.root_action_values, slow.root_action_values, atol=1e-12, rtol=1e-12
    )
    assert fast.root.action == slow.root.action

    def reconstruct(node):
        if not node.branches:
            assert node.quadrature_correction == 0
            return node.expected_risk
        value = (
            sum(edge.probability * reconstruct(edge.child) for edge in node.branches)
            + node.quadrature_correction
        )
        assert value == pytest.approx(node.expected_risk, abs=1e-12)
        return value

    assert reconstruct(fast.root) == pytest.approx(
        min(v for _, v in fast.root_action_values), abs=1e-12
    )


def test_single_component_exact_one_step_risk():
    b = RegressionBelief([0.0], [[1.0]], 3.0, 2.0)
    m = ControlVariateMixture(
        [[[2.0]]],
        [[[1.0], [3.0]]],
        [b],
        [1.0],
        target_weights=[0.25, 0.75],
        quadrature_order=2,
    )
    assert m.expected_terminal_risk(m.initial_state, 0)[0] == pytest.approx(
        7 / 5, abs=1e-12
    )
    assert m.chance_risk_correction(m.initial_state, 0) > 0


def test_invalid_chance_correction_fails():
    m = model()
    m.chance_risk_correction = lambda *args: float("nan")
    with pytest.raises(ValueError, match="correction"):
        HorizonPlanner(m).plan(m.initial_state, 1)
