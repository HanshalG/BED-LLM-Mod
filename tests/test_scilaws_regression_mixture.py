import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import t

from environments.chembench_mopen.horizon import HorizonPlanner, SearchLimits
from environments.scilaws.regression_belief import RegressionBelief
from environments.scilaws.regression_mixture import RegressionMixture


def model(order=8, duplicate=False, noisy=False):
    b = RegressionBelief([0.0], [[1.0]], 3.0, 2.0)
    n = 2 if duplicate else 1
    return RegressionMixture(
        [[[1.0], [2.0]]] * n,
        [[[1.0], [3.0]]] * n,
        [b] * n,
        [1 / n] * n,
        target_weights=[0.25, 0.75],
        quadrature_order=order,
        include_observation_noise=noisy,
    )


def test_within_component_uncertainty_and_noise_retained():
    m = model()
    assert m.risk(m.initial_state) == pytest.approx(7.0)
    noisy = model(noisy=True)
    assert noisy.risk(noisy.initial_state) == pytest.approx(8.0)
    duplicate = model(duplicate=True)
    assert duplicate.risk(duplicate.initial_state) == pytest.approx(7.0)
    original_rows = m.branches(m.initial_state, 0)
    duplicate_rows = duplicate.branches(duplicate.initial_state, 0)
    assert len(original_rows) == len(duplicate_rows)
    for a, b in zip(original_rows, duplicate_rows, strict=True):
        assert a.observation == b.observation
        assert a.probability == pytest.approx(b.probability)
        assert m.risk(a.state) == pytest.approx(duplicate.risk(b.state))


def test_mixture_bayes_update_and_total_variance():
    a = RegressionBelief([-2.0], [[2.0]], 3.0, 1.0)
    b = RegressionBelief([2.0], [[4.0]], 3.0, 2.0)
    m = RegressionMixture(
        [[[1.0]], [[1.0]]], [[[1.0]], [[1.0]]], [a, b], [0.3, 0.7], target_weights=[1.0]
    )
    assert m.forecast(m.initial_state)[0] == pytest.approx(0.8)
    expected_var = 0.3 * (0.25 + (-2 - 0.8) ** 2) + 0.7 * (0.25 + (2 - 0.8) ** 2)
    assert m.risk(m.initial_state) == pytest.approx(expected_var)
    updated = m.condition(m.initial_state, 0, 1.5)
    logs = np.log([0.3, 0.7]) + [
        a.log_predictive([1.0], 1.5),
        b.log_predictive([1.0], 1.5),
    ]
    np.testing.assert_allclose(updated.log_weights, logs - logsumexp(logs))
    assert updated.components[0] == a.condition([1.0], 1.5)[0]


def test_quadrature_refinement_against_analytic_expected_risk():
    # For a single linear model E[b_post/(a_post-1)] = E[sigma2].
    # Measuring feature 2 changes precision 1 -> 5, so risk becomes 7/5.
    errors = []
    for order in (8, 32, 64):
        m = model(order)
        branches = m.branches(m.initial_state, 1)
        assert sum(b.probability for b in branches) == pytest.approx(1.0)
        risk = sum(b.probability * m.risk(b.state) for b in branches)
        errors.append(abs(risk - 7 / 5))
        assert sum(
            b.probability * m.forecast(b.state)[0] for b in branches
        ) == pytest.approx(0.0, abs=1e-12)
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 1e-5


def test_horizons_retain_contingent_depth_and_repeat_choice():
    m = model(4)
    p = HorizonPlanner(m, limits=SearchLimits(max_seconds=20, max_nodes=100000))
    for depth in (1, 2, 3):
        r = p.plan(m.initial_state, depth, allow_repeats=True)
        assert r.effective_horizon == depth
        assert r.root.action == 1
        assert r.root.remaining_depth == depth
    # This control has no reason to branch actions, so it is not a depth-win fixture.


def test_invalid_weights_and_action_rejected():
    with pytest.raises(ValueError):
        model().condition(model().initial_state, True, 1.0)
    b = RegressionBelief([0.0], [[1.0]], 3.0, 2.0)
    with pytest.raises(ValueError):
        RegressionMixture([[[1.0]]], [[[1.0]]], [b], [0.9], target_weights=[1.0])


def test_mixture_quadrature_against_independent_density_integration():
    a = RegressionBelief([-1.0], [[2.0]], 3.0, 1.0)
    b = RegressionBelief([1.0], [[4.0]], 3.0, 2.0)
    m = RegressionMixture(
        [[[1.0]], [[1.0]]],
        [[[1.0]], [[1.0]]],
        [a, b],
        [0.3, 0.7],
        target_weights=[1.0],
        quadrature_order=64,
    )
    parameters = [component.predictive([1.0]) for component in (a, b)]

    def density(y):
        return sum(
            w * t.pdf(y, df, loc=loc, scale=np.sqrt(s2))
            for w, (df, loc, s2) in zip((0.3, 0.7), parameters)
        )

    def integrand(y):
        return density(y) * m.risk(m.condition(m.initial_state, 0, y))

    reference, error = quad(integrand, -np.inf, np.inf, epsabs=1e-9)
    assert error < 1e-7
    rows = m.branches(m.initial_state, 0)
    estimated = sum(r.probability * m.risk(r.state) for r in rows)
    assert abs(estimated - reference) < 1e-5
    mean_after = sum(r.probability * m.forecast(r.state)[0] for r in rows)
    assert abs(mean_after - m.forecast(m.initial_state)[0]) < 1e-5
