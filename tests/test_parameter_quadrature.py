import math

import numpy as np
import pytest

from environments.chembench_mopen.parameter_quadrature import (
    IntegrationUnresolved, integrate_parameters,
)


def test_uniform_prior_mass_and_moments():
    result = integrate_parameters([-2, 1], [4, 3], lambda x: np.zeros(len(x)),
                                  lambda x: x, output_size=2)
    np.testing.assert_allclose(result.mean, [1, 2], atol=1e-12)
    np.testing.assert_allclose(result.variance, [3, 1/3], atol=1e-12)
    assert abs(result.log_evidence) < 1e-12
    assert result.orders == (32, 64, 128)
    assert not result.weights.flags.writeable


def test_log_uniform_prior_coordinates():
    result = integrate_parameters([0], [math.log(4)], lambda x: np.zeros(len(x)),
                                  np.exp, output_size=1)
    np.testing.assert_allclose(result.mean, [3/math.log(4)], atol=1e-12)
    assert abs(result.log_evidence) < 1e-12


def test_gaussian_evidence_and_posterior_against_closed_form():
    sigma = .3
    def likelihood(x):
        return -.5 * ((x[:, 0] - .4)/sigma)**2 - math.log(sigma*math.sqrt(2*math.pi))
    result = integrate_parameters([-4], [4], likelihood, lambda x: x, output_size=1)
    np.testing.assert_allclose(result.mean, [.4], atol=1e-10)
    np.testing.assert_allclose(result.variance, [sigma**2], atol=1e-10)
    assert abs(result.log_evidence + math.log(8)) < 1e-10


def test_limits_are_checked_before_callbacks():
    def bomb(x):
        raise AssertionError('callback opened')
    with pytest.raises(ValueError):
        integrate_parameters([0]*3, [1]*3, bomb, bomb, output_size=1)
    with pytest.raises(IntegrationUnresolved, match='cap'):
        integrate_parameters([0], [1], bomb, bomb, output_size=1, max_rows=1)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_invalid_or_zero_evidence_fails(bad):
    with pytest.raises(IntegrationUnresolved):
        integrate_parameters([0], [1], lambda x: np.full(len(x), bad),
                             lambda x: x, output_size=1)


def test_narrow_posterior_unresolved_not_silently_returned():
    with pytest.raises(IntegrationUnresolved, match='stabilize'):
        integrate_parameters([-1], [1], lambda x: -((x[:, 0]-.123)/.0001)**2,
                             lambda x: x, output_size=1, orders=(4, 8, 16))
