import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t

from environments.scilaws.moment_quadrature import MomentMatchedMixture, match_moments
from environments.scilaws.regression_belief import RegressionBelief
from environments.scilaws.regression_mixture import RegressionMixture


def model(cls=MomentMatchedMixture, order=8):
    a = RegressionBelief([-1.0], [[2.0]], 3.0, 1.0)
    b = RegressionBelief([1.0], [[4.0]], 3.0, 2.0)
    return cls(
        [[[1.0]], [[1.0]]],
        [[[1.0]], [[1.0]]],
        [a, b],
        [0.3, 0.7],
        target_weights=[1.0],
        quadrature_order=order,
        include_observation_noise=True,
    )


def test_conserves_component_weights_means_and_noise_variance():
    m = model()
    rows = m.branches(m.initial_state, 0)
    mass = np.array([r.probability for r in rows])
    weights = np.array([np.exp(r.state.log_weights) for r in rows])
    np.testing.assert_allclose(mass @ weights, [0.3, 0.7], atol=5e-9)
    for i, w in enumerate((0.3, 0.7)):
        means = np.array([r.state.components[i].mean[0] for r in rows])
        noise = np.array([r.state.components[i].noise_variance for r in rows])
        assert (mass * weights[:, i]) @ means == pytest.approx(
            w * m.components[i].mean[0], abs=5e-9
        )
        assert (mass * weights[:, i]) @ noise == pytest.approx(
            w * m.components[i].noise_variance, abs=5e-9
        )
    scalar = sum(r.probability * m.risk(r.state) for r in rows)
    fast, _ = m.expected_terminal_risk(m.initial_state, 0)
    assert fast == pytest.approx(scalar, abs=1e-12)


def test_no_change_to_likelihood_or_posterior_update():
    m, old = model(), model(RegressionMixture)
    assert m.condition(m.initial_state, 0, 1.7) == old.condition(
        old.initial_state, 0, 1.7
    )
    assert {y for y, _ in m._quadrature(m.initial_state, 0)} <= {
        y for y, _ in old._quadrature(old.initial_state, 0)
    }


def test_infeasible_moments_fail_without_fallback():
    with pytest.raises(ValueError, match="moment matching failed"):
        match_moments([-0.1, 0.1], [0.5, 0.5], [1.0], [[6.0, 0.0, 1.0]])


def test_single_component_expected_risk_identity():
    b = RegressionBelief([0.0], [[1.0]], 3.0, 2.0)
    m = MomentMatchedMixture(
        [[[2.0]]],
        [[[1.0], [3.0]]],
        [b],
        [1.0],
        target_weights=[0.25, 0.75],
        quadrature_order=8,
    )
    value, _ = m.expected_terminal_risk(m.initial_state, 0)
    assert value == pytest.approx(7 / 5, abs=5e-9)


def test_refined_risk_matches_independent_predictive_density_integral():
    m = model(order=64)
    parameters = [b.predictive([1.0]) for b in m.components]

    def integrand(y):
        density = sum(
            w * t.pdf(y, df, loc=loc, scale=np.sqrt(scale2))
            for w, (df, loc, scale2) in zip((0.3, 0.7), parameters)
        )
        return density * m.risk(m.condition(m.initial_state, 0, y))

    reference, error = quad(integrand, -np.inf, np.inf, epsabs=1e-9)
    assert error < 1e-7
    value, _ = m.expected_terminal_risk(m.initial_state, 0)
    assert abs(reference - value) < 1e-5
