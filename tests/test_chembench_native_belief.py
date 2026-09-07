import numpy as np
import pytest

pytest.importorskip("Cython")

from environments.chembench_mopen.batch_horizon import (
    plan_batched,
    posterior_branches_many,
)
from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel
from environments.chembench_mopen.native_belief import NativeEnvelopeGaussianModel


@pytest.mark.parametrize("seed", range(4))
def test_compiled_branches_match_numpy_with_concentrated_beliefs(seed):
    rng = np.random.default_rng(seed)
    n = 16 if seed % 2 else 3
    means = rng.normal(size=(n, 2)) * 3
    sigma = 0.15 if seed % 2 else np.exp(rng.normal(size=(n, 2)))
    targets = rng.normal(size=(n, 7))
    prior = rng.dirichlet(np.full(n, 0.2))
    baseline = EnvelopeGaussianModel(means, sigma, targets, prior, branch_count=64)
    compiled = NativeEnvelopeGaussianModel(
        means, sigma, targets, prior, branch_count=64
    )
    states = [baseline.initial_state, baseline.condition(baseline.initial_state, 0, 20)]
    for action in range(2):
        old = posterior_branches_many(baseline, states, action, return_weights=True)
        new = posterior_branches_many(compiled, states, action, return_weights=True)
        np.testing.assert_allclose(new[0], old[0], atol=1e-10, rtol=1e-11)
        np.testing.assert_allclose(new[1], old[1], atol=1e-8, rtol=1e-10)
        np.testing.assert_array_equal(new[2], old[2])


@pytest.mark.parametrize("mode", ["adaptive", "open_loop"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_compiled_complete_policy_equivalence(mode, depth):
    args = ([[-0.4, -0.2, 0.1], [0.6, 0.3, -0.5]], 1, [[0], [1]], [0.4, 0.6])
    baseline = EnvelopeGaussianModel(*args, branch_count=8)
    compiled = NativeEnvelopeGaussianModel(*args, branch_count=8)
    old = plan_batched(baseline, baseline.initial_state, depth, mode=mode)
    new = plan_batched(compiled, compiled.initial_state, depth, mode=mode)
    np.testing.assert_allclose(new.root_values, old.root_values, atol=1e-10)
    assert new.processed_states == old.processed_states


def test_compiled_endpoint_tail_and_zero_support_are_finite():
    m = NativeEnvelopeGaussianModel(
        [[-5], [5]], 1, [[0], [1]], [1 - 2e-14, 2e-14], branch_count=64
    )
    for state in [m.initial_state, (0, -np.inf)]:
        branches = m.branches(state, 0)
        assert all(np.isfinite(b.observation) for b in branches)
        assert sum(b.probability for b in branches) == pytest.approx(1)


def test_native_shapes_fail_before_unchecked_memory_access():
    NativeEnvelopeGaussianModel([[0]], 1, [[0]], [1])
    from environments.chembench_mopen._native_quantiles import invert, envelope_starts

    with pytest.raises(ValueError, match="dimensions"):
        invert(np.ones((1, 2)), np.zeros(1), np.ones(1), np.full((1, 3), 0.5))
    with pytest.raises(ValueError, match="dimensions"):
        envelope_starts(np.zeros(2), 1, np.zeros(1))


def test_compiled_envelope_keeps_equal_mean_maximum_and_saturated_cdf():
    from scipy.special import ndtr

    assert ndtr(9.0) == 1.0 and ndtr(-40.0) == 0.0
    args = (
        [[-50], [-1], [-1], [1], [50]],
        0.15,
        [[0], [1], [2], [3], [4]],
        [0.1, 0.1, 0.2, 0.3, 0.3],
    )
    old = EnvelopeGaussianModel(*args, branch_count=64)
    new = NativeEnvelopeGaussianModel(*args, branch_count=64)
    for a, b in zip(
        old.quadrature_rule(old.initial_state, 0),
        new.quadrature_rule(new.initial_state, 0),
    ):
        np.testing.assert_array_equal(a, b)
    a = posterior_branches_many(old, [old.initial_state], 0)
    b = posterior_branches_many(new, [new.initial_state], 0)
    np.testing.assert_allclose(a[0], b[0], atol=1e-10)
