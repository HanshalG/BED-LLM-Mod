import numpy as np
import pytest

from environments.chembench_mopen.batch_horizon import (
    plan_batched,
    posterior_branches_many,
)
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.chembench_mopen.horizon import HorizonPlanner, SearchLimitExceeded


def fixture():
    return QuantileGaussianModel(
        [[-0.4, -0.2, 0.1], [0.6, 0.3, -0.5]],
        [[0.7, 1, 0.9], [1.2, 1, 0.6]],
        [[0, 1], [1, 0.5]],
        [0.4, 0.6],
        branch_count=4,
    )


def test_branch_batch_matches_scalar_with_zero_and_tiny_support():
    m = fixture()
    states = [m.initial_state, m.condition(m.initial_state, 0, 30), (-np.inf, 0)]
    obs, logs = posterior_branches_many(m, states, 1)
    for i, state in enumerate(states):
        scalar = m.branches(state, 1)
        np.testing.assert_allclose(obs[i], [b.observation for b in scalar], atol=1e-11)
        np.testing.assert_allclose(logs[i], [b.state for b in scalar], atol=1e-10)


@pytest.mark.parametrize("mode", ["adaptive", "open_loop"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_full_policy_values_match_scalar(mode, depth):
    m = fixture()
    scalar = HorizonPlanner(m).plan(m.initial_state, depth, mode=mode)
    batch = plan_batched(m, m.initial_state, depth, mode=mode, batch_size=3)
    np.testing.assert_allclose(batch.root_values, scalar.root_action_values, atol=1e-9)
    assert batch.value == pytest.approx(scalar.root.expected_risk, abs=1e-9)
    assert batch.effective_horizon == depth


def test_batch_size_invariance_and_caps():
    m = fixture()
    a = plan_batched(m, m.initial_state, 2, batch_size=1)
    b = plan_batched(m, m.initial_state, 2, batch_size=64)
    assert a.value == pytest.approx(b.value, abs=1e-12)
    for kwargs in ({"max_states": 1}, {"max_workspace_bytes": 1}):
        with pytest.raises(SearchLimitExceeded):
            plan_batched(m, m.initial_state, 3, **kwargs)
    assert plan_batched(m, m.initial_state, 0).action is None


def test_pairwise_risk_matches_weighted_variance_with_large_target_offset():
    rng = np.random.default_rng(882)
    targets = 1e6 + rng.normal(size=(7, 23))
    weights = rng.dirichlet(np.ones(7))
    target_weights = rng.dirichlet(np.ones(23))
    model = QuantileGaussianModel(
        np.arange(7)[:, None], 1, targets, weights, target_weights=target_weights
    )
    actual = plan_batched(model, model.initial_state, 0).value
    wide = targets.astype(np.longdouble)
    mean = weights.astype(np.longdouble) @ wide
    expected = float(
        weights.astype(np.longdouble)
        @ ((wide - mean) ** 2 @ target_weights.astype(np.longdouble))
    )
    assert actual == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("seed", range(4))
def test_accelerated_quantiles_match_independent_bisection(seed):
    from scipy.special import ndtr, ndtri
    from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel

    rng = np.random.default_rng(seed)
    means = rng.normal(size=(16, 1)) * 4
    sigma = 0.15 if seed % 2 else np.exp(rng.normal(size=(16, 1)))
    prior = rng.dirichlet(np.full(16, 0.2))
    model_type = EnvelopeGaussianModel if seed % 2 else QuantileGaussianModel
    m = model_type(means, sigma, means, prior, branch_count=64)
    states = [m.initial_state, m.condition(m.initial_state, 0, 20)]
    actual, posterior = posterior_branches_many(m, states, 0)
    for index, state in enumerate(states):
        probabilities = (
            m.quadrature_rule(state, 0)[0]
            if hasattr(m, "quadrature_rule")
            else m._quantiles
        )
        probabilities = np.clip(
            probabilities, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
        )
        w = np.exp(state)
        mu, sd = m.means[:, 0], m.sigmas[:, 0]
        component = mu[:, None] + sd[:, None] * ndtri(probabilities)
        low = np.min(component[w > 0], axis=0)
        high = np.max(component[w > 0], axis=0)
        for _ in range(64):
            middle = low / 2 + high / 2
            cdf = np.sum(
                w[:, None] * ndtr((middle - mu[:, None]) / sd[:, None]), axis=0
            )
            low, high = (
                np.where(cdf < probabilities, middle, low),
                np.where(cdf < probabilities, high, middle),
            )
        reference = low / 2 + high / 2
        np.testing.assert_allclose(actual[index], reference, atol=1e-11, rtol=1e-12)
        reference_posteriors = [m.condition(state, 0, y) for y in reference]
        np.testing.assert_allclose(posterior[index], reference_posteriors, atol=1e-8)
