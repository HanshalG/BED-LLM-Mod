import numpy as np
import pytest
from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel
from environments.chembench_mopen.crossing_belief import CrossingGaussianModel
from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.horizon import HorizonPlanner


def test_sixteen_particles_preserved_with_bounded_knots():
    m = EnvelopeGaussianModel(
        np.arange(16)[:, None],
        0.3,
        np.arange(16)[:, None],
        np.full(16, 1 / 16),
        branch_count=64,
    )
    u, w = m.quadrature_rule(m.initial_state, 0)
    assert len(u) == len(w) == 64 and w.sum() == pytest.approx(1)
    assert m.num_particles == 16
    branches = m.branches(m.initial_state, 0)
    assert all(len(b.state) == 16 for b in branches)
    assert all(
        b.state == m.condition(m.initial_state, 0, b.observation) for b in branches
    )


def test_unequal_noise_matches_parent_rule():
    args = ([[-0.2], [0.2]], [[0.3], [2]], [[0], [1]], [0.3, 0.7])
    a = EnvelopeGaussianModel(*args, branch_count=32)
    b = CrossingGaussianModel(*args, branch_count=32)
    for x, y in zip(
        a.quadrature_rule(a.initial_state, 0), b.quadrature_rule(b.initial_state, 0)
    ):
        np.testing.assert_array_equal(x, y)


def test_scalar_batch_equivalence_with_dominated_density_lines():
    m = EnvelopeGaussianModel(
        [[-0.4, -0.2], [0, 0.1], [0.6, 0.3]],
        1,
        [[0], [0.2], [1]],
        [0.49, 0.02, 0.49],
        branch_count=12,
    )
    scalar = HorizonPlanner(m).plan(m.initial_state, 2)
    batch = plan_batched(m, m.initial_state, 2)
    assert batch.value == pytest.approx(scalar.root.expected_risk, abs=1e-10)


def test_tiny_tail_interval_does_not_round_to_infinite_observation():
    m = EnvelopeGaussianModel(
        [[-5], [5]], 1, [[0], [1]], [1 - 2e-14, 2e-14], branch_count=64
    )
    branches = m.branches(m.initial_state, 0)
    assert all(np.isfinite(b.observation) for b in branches)
    assert sum(b.probability for b in branches) == pytest.approx(1)
