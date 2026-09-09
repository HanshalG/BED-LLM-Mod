import sys

import numpy as np
import pytest
from scipy.special import roots_legendre

from environments.chembench_mopen.integrated_executable_belief import IntegratedExecutableBeliefPool
from environments.chembench_mopen.parameter_quadrature import IntegrationUnresolved
from environments.chembench_mopen.ir import RateLawError


@pytest.fixture(autouse=True)
def isolate_torch_stub(monkeypatch):
    monkeypatch.delitem(sys.modules, 'torch', raising=False)


def point(x):
    return [x, 0, 1, 0, 1, 310, 7]


def law(expr, name='law'):
    return {'name': name, 'expr': expr, 'params': [
        {'name': 'k', 'low': .5, 'high': 1.5, 'transform': 'identity'}]}


def reference(xs, ys, sigma):
    nodes, masses = roots_legendre(512)
    k = 1+nodes/2
    forecasts, log_mass = [], []
    for slope in (False, True):
        rates = k[:, None]*(np.asarray(xs) if slope else np.ones(len(xs)))
        ll = (-.5*((np.log1p(rates)-ys)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi))).sum(axis=1)
        log_mass.extend(np.log(masses/4)+ll)
        forecasts.extend(np.log1p(k*(2 if slope else 1)))
    log_mass, forecasts = np.array(log_mass), np.array(forecasts)
    z = np.logaddexp.reduce(log_mass)
    weights = np.exp(log_mass-z)
    mean = weights@forecasts
    return mean, weights@((forecasts-mean)**2), z


@pytest.mark.parametrize('xs', [[1], [1, 2]])
def test_two_structure_mixture_matches_independent_integral(xs):
    pool = IntegratedExecutableBeliefPool(max_scalar_nodes=10000000)
    pool.add(law('k'))
    pool.add(law('k*C_A'))
    ys = np.log1p(xs)
    result = pool.moment_snapshot(history_inputs=[point(x) for x in xs], observations=ys,
                                  targets=[point(2)], sigma=.1)
    mean, variance, evidence = reference(xs, ys, .1)
    np.testing.assert_allclose(result.mean, [mean], atol=1e-7)
    np.testing.assert_allclose(result.variance, [variance], atol=1e-7)
    assert abs(result.conditional_log_evidence-evidence) < 1e-7
    if len(xs) == 1:
        np.testing.assert_allclose(result.law_weights, [.5, .5], atol=1e-10)


def test_refresh_replays_full_history_and_deduplicates():
    pool = IntegratedExecutableBeliefPool(max_scalar_nodes=10000000)
    pool.add(law('k'))
    args = dict(history_inputs=[point(1), point(2)], observations=np.log1p([1, 2]),
                targets=[point(2)], sigma=.1)
    pool.moment_snapshot(**args)
    key = pool.add(law('k*C_A'))
    assert pool.add(law('k*C_A', 'renamed')) == key
    after = pool.moment_snapshot(**args)
    fresh = IntegratedExecutableBeliefPool(seed=99, max_scalar_nodes=10000000)
    fresh.add(law('k*C_A'))
    fresh.add(law('k'))
    expected = fresh.moment_snapshot(**args)
    assert after.mean == expected.mean
    assert after.law_weights == expected.law_weights
    assert after.history_sha256 == expected.history_sha256


def test_invalid_law_is_not_silently_removed():
    pool = IntegratedExecutableBeliefPool()
    pool.add(law('k'))
    pool.add(law('k-C_A'))
    with pytest.raises(RateLawError):
        pool.moment_snapshot(history_inputs=[], observations=[], targets=[point(2)], sigma=.1)
    assert len(pool._laws) == 2


def test_cap_and_unsupported_dimension_fail_before_fit(monkeypatch):
    pool = IntegratedExecutableBeliefPool(max_scalar_nodes=1)
    pool.add(law('k'))
    with pytest.raises(IntegrationUnresolved):
        pool.moment_snapshot(history_inputs=[point(1)], observations=[0], targets=[point(2)], sigma=.1)
    payload = law('k*a*b')
    payload['params'] += [dict(payload['params'][0], name=n) for n in ('a', 'b')]
    pool.add(payload)
    with pytest.raises(IntegrationUnresolved, match='two parameters'):
        pool.moment_snapshot(history_inputs=[], observations=[], targets=[point(2)], sigma=.1)
