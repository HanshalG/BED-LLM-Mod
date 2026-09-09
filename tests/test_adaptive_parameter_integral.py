import sys

import numpy as np
import pytest

from environments.chembench_mopen.adaptive_parameter_integral import adaptive_parameter_integral


@pytest.fixture(autouse=True)
def remove_fake_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, 'torch', raising=False)


def test_uniform_box_prior():
    r = adaptive_parameter_integral([-2, 1], [4, 3], lambda x: np.zeros(len(x)),
                                    lambda x: x, output_size=2, log_likelihood_bound=0)
    assert r['status'] == 'agreement'
    for check in r['checks']:
        np.testing.assert_allclose(check['mean'], [1, 2], atol=1e-12)
        np.testing.assert_allclose(check['variance'], [3, 1/3], atol=1e-12)
        assert abs(check['log_evidence']) < 1e-12


def test_cap_before_dispatch_and_failure_diagnostics():
    def bomb(x):
        raise AssertionError('dispatch')
    r = adaptive_parameter_integral([0], [1], bomb, bomb, output_size=1,
                                    log_likelihood_bound=0, max_rows=1)
    assert r['status'] == 'unresolved'
    assert r['evaluated_rows'] == 0


@pytest.mark.parametrize('value', [1., np.nan, np.inf, -np.inf])
def test_invalid_bound_or_likelihood(value):
    r = adaptive_parameter_integral([0], [1], lambda x: np.full(len(x), value),
                                    lambda x: x, output_size=1, log_likelihood_bound=0)
    assert r['status'] == 'unresolved'


def test_likelihood_shift_keeps_conditional_moments():
    a = adaptive_parameter_integral([-1], [1], lambda x: -x[:, 0]**2,
                                    lambda x: x, output_size=1, log_likelihood_bound=0)
    b = adaptive_parameter_integral([-1], [1], lambda x: -10000-x[:, 0]**2,
                                    lambda x: x, output_size=1, log_likelihood_bound=-10000)
    assert a['status'] == b['status'] == 'agreement'
    np.testing.assert_allclose(a['checks'][-1]['variance'], b['checks'][-1]['variance'])
    assert abs(a['checks'][-1]['log_evidence']-b['checks'][-1]['log_evidence']-10000) < 1e-9
