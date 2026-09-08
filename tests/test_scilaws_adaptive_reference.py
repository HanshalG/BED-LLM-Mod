import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
import numpy as np

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.adaptive_reference import AdaptiveReference
from environments.scilaws.adaptive_reference import student_log_density
from scipy.stats import t
from scripts.scilaws_mixed_refinement_audit import fixture


def test_terminal_matches_direct_full_risk_integration():
    m, state = fixture(8, ((0, 0.7),))
    ref = AdaptiveReference(m)
    value, error = ref.terminal(state, 1)

    def integrand(y):
        logp = logsumexp([
            w + b.log_predictive(x[1], y)
            for w, b, x in zip(state.log_weights, state.components,
                                m.action_features, strict=True)
        ])
        return np.exp(logp) * m.risk(m.condition(state, 1, y))

    direct, _ = quad(integrand, -np.inf, np.inf, epsabs=1e-8, epsrel=1e-8)
    assert value == pytest.approx(direct, abs=1e-7)
    assert error < 1e-7


def test_caps_and_invalid_depth():
    m, state = fixture(4, ())
    with pytest.raises(SearchLimitExceeded):
        AdaptiveReference(m, max_evaluations=1).terminal(state, 0)
    with pytest.raises(ValueError):
        AdaptiveReference(m).action(state, 0, 3)


@pytest.mark.parametrize("y", [-1e150, -100.0, -0.5, 0.0, 0.7, 100.0, 1e150])
def test_precomputed_density_matches_scipy(y):
    df = np.array([2.1, 6.0, 30.0])
    loc = np.array([-0.5, 0.0, 3.0])
    scale2 = np.array([1e-4, 0.4, 100.0])
    value = student_log_density(df, loc, scale2)(y)
    np.testing.assert_allclose(value, t.logpdf(y, df, loc=loc, scale=np.sqrt(scale2)),
                               atol=1e-10, rtol=1e-13)


def test_invalid_density_parameters():
    with pytest.raises(ValueError):
        student_log_density([0.0], [0.0], [1.0])
