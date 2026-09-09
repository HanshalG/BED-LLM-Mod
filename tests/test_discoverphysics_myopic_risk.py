import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from scripts.discoverphysics_myopic_risk import myopic_prediction_risk


def test_uninformative_observation_preserves_prior_risk():
    assert abs(myopic_prediction_risk([[0, 0], [0, 0]], [[0], [2]], [.5, .5], 1)-1) < 1e-12


def test_informative_observation_and_irrelevant_target():
    assert myopic_prediction_risk([[-5], [5]], [[0], [2]], [.5, .5], .1) < 1e-12
    assert myopic_prediction_risk([[-5], [5]], [[3], [3]], [.5, .5], 1) == 0


def test_binary_gaussian_against_independent_scalar_integral():
    def integrand(y):
        density = (np.exp(-.5*(y+1)**2)+np.exp(-.5*(y-1)**2))/(2*np.sqrt(2*np.pi))
        p = expit(2*y)
        return density*4*p*(1-p)
    expected = quad(integrand, -12, 12, epsabs=1e-11)[0]
    actual = myopic_prediction_risk([[-1], [1]], [[0], [2]], [.5, .5], 1, order=128)
    assert abs(actual-expected) < 1e-9


def test_zero_mass_and_permutation():
    a = myopic_prediction_risk([[0], [2], [99]], [[0], [1], [900]], [.4, .6, 0], 1)
    b = myopic_prediction_risk([[2], [0]], [[1], [0]], [.6, .4], 1)
    assert abs(a-b) < 1e-12


@pytest.mark.parametrize('prior', [[.4, .4], [-.1, 1.1], [np.nan, .5]])
def test_invalid_prior(prior):
    with pytest.raises(ValueError):
        myopic_prediction_risk([[0], [1]], [[0], [1]], prior, 1)
