import math

import numpy as np
import pytest
from scipy.integrate import quad

from environments.chembench_mopen.horizon import (
    HorizonPlanner,
    SearchLimits,
    SearchLimitExceeded,
)
from environments.chembench_mopen.raw_belief import GaussianParticleModel


def normal_density(y, means):
    return np.exp(-0.5 * (y - np.asarray(means)) ** 2) / math.sqrt(2 * math.pi)


def model(order=32):
    return GaussianParticleModel(
        [[-0.6, -0.2], [0.6, 0.2]],
        [[1, 1], [1, 1]],
        [[0], [1]],
        [0.5, 0.5],
        quadrature_order=order,
    )


def test_raw_update_matches_bayes_and_history_order():
    m = model()
    state = m.condition(m.initial_state, 0, 0.37)
    expected = normal_density(0.37, [-0.6, 0.6])
    np.testing.assert_allclose(np.exp(state), expected / expected.sum())
    a = m.condition(state, 1, -0.42)
    b = m.condition(m.condition(m.initial_state, 1, -0.42), 0, 0.37)
    np.testing.assert_allclose(a, b)


def test_tiny_posterior_recovers_without_support_loss():
    m = GaussianParticleModel([[-1], [1]], 1, [[0], [1]], [0.5, 0.5])
    state = m.condition(m.initial_state, 0, 1000)
    assert np.exp(state[0]) == 0
    state = m.condition(state, 0, -1000)
    np.testing.assert_allclose(np.exp(state), [0.5, 0.5], atol=1e-11)


def test_quadrature_against_independent_density_integral():
    m = model(64)
    for action in range(2):

        def integrand(y):
            masses = 0.5 * normal_density(
                y, [-0.6, 0.6] if action == 0 else [-0.2, 0.2]
            )
            return masses[0] * masses[1] / masses.sum() if masses.sum() else 0

        expected, error = quad(integrand, -12, 12, epsabs=1e-11)
        actual = math.fsum(
            b.probability * m.risk(b.state) for b in m.branches(m.initial_state, action)
        )
        assert error < 1e-9
        assert actual == pytest.approx(expected, abs=1e-9)
    assert HorizonPlanner(m).plan(m.initial_state, 1).root.action == 0


def test_quadrature_predictive_moments_and_determinism():
    m = GaussianParticleModel([[-4], [3]], [[0.5], [2]], [[0], [1]], [0.3, 0.7])
    branches = m.branches(m.initial_state, 0)
    assert branches == m.branches(m.initial_state, 0)
    assert sum(b.probability for b in branches) == pytest.approx(1)
    assert sum(b.probability * b.observation for b in branches) == pytest.approx(0.9)
    assert sum(b.probability * b.observation**2 for b in branches) == pytest.approx(
        0.3 * 16.25 + 0.7 * 13
    )


def test_uninformative_measurement_and_zero_prior():
    m = GaussianParticleModel([[2], [2], [9]], 1, [[0], [1], [50]], [0.5, 0.5, 0])
    for b in m.branches(m.initial_state, 0):
        assert b.state[-1] == -math.inf
        assert m.risk(b.state) == pytest.approx(0.25)
    assert len(m.branches(m.initial_state, 0)) == m.quadrature_order


def test_raw_model_obeys_planner_caps():
    m = model(3)
    with pytest.raises(SearchLimitExceeded):
        HorizonPlanner(m, limits=SearchLimits(max_nodes=2)).plan(m.initial_state, 2)
    plan = HorizonPlanner(m).plan(m.initial_state, 2)
    assert plan.root.remaining_depth == 2
    assert all(edge.child.remaining_depth == 1 for edge in plan.root.branches)


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), float("inf")])
def test_invalid_noise(sigma):
    with pytest.raises(ValueError):
        GaussianParticleModel([[0]], sigma, [[0]], [1])


@pytest.mark.parametrize("order", [0, -1, True, 1.5, 129])
def test_invalid_order(order):
    with pytest.raises(ValueError):
        model(order)


def test_invalid_observation_state_and_action():
    m = model()
    for observation in [float("nan"), float("inf"), 1e308]:
        with pytest.raises(ValueError):
            m.condition(m.initial_state, 0, observation)
    with pytest.raises(ValueError):
        m.risk((0.5, 0.5))
    with pytest.raises(ValueError):
        m.branches(m.initial_state, 2)
