import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import t

from environments.scilaws.tail_risk_bound import student_tail_moments, tail_risk_bound
from scripts.scilaws_mixed_refinement_audit import fixture


@pytest.mark.parametrize('df,left,right', [(3., -2., 1.), (6., -4., 4.), (12., .2, 3.)])
def test_tail_moments_against_independent_integrals(df, left, right):
    moments = student_tail_moments(df, left, right)
    for j in range(3):
        expected = (quad(lambda z: z**j*t.pdf(z, df), -np.inf, left)[0]
                    + quad(lambda z: z**j*t.pdf(z, df), right, np.inf)[0])
        assert moments[j] == pytest.approx(expected, abs=1e-10)


def test_tail_bound_matches_full_conditional_second_moment():
    m, state = fixture(8, ((0, .7),))
    bound = tail_risk_bound(m, state, 1, -2, 3)

    def integrand(y):
        density = np.exp(logsumexp([
            w+b.log_predictive(x[1], y) for w,b,x in
            zip(state.log_weights, state.components, m.action_features, strict=True)]))
        child = m.condition(state, 1, y)
        mean = m.forecast(child)
        return density*(m.risk(child)+m.target_weights @ mean**2)

    exact = quad(integrand, -np.inf, -2, epsabs=1e-9)[0]+quad(integrand, 3, np.inf, epsabs=1e-9)[0]
    assert bound['value'] == pytest.approx(exact, abs=1e-8)
    assert tail_risk_bound(m, state, 1, -4, 6)['value'] < bound['value']


def test_invalid_variance_or_interval():
    with pytest.raises(ValueError):
        student_tail_moments(2, -1, 1)
    with pytest.raises(ValueError):
        student_tail_moments(6, 1, -1)
