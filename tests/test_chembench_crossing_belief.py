import numpy as np
import pytest
from environments.chembench_mopen.crossing_belief import CrossingGaussianModel
from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.horizon import HorizonPlanner, SearchLimitExceeded


def test_equal_density_split_and_exact_budget():
    m = CrossingGaussianModel([[-1], [1]], 1, [[0], [1]], [0.5, 0.5], branch_count=16)
    u, w = m.quadrature_rule(m.initial_state, 0)
    assert len(u) == 16 and len(w) == 16
    assert sum(u < 0.5) == 8 and sum(u > 0.5) == 8
    assert w.sum() == pytest.approx(1)
    assert np.all(w > 0)


def test_dynamic_rules_preserve_scalar_batch_values():
    m = CrossingGaussianModel(
        [[-0.4, -0.2], [0.6, 0.3]],
        [[0.7, 1], [1.2, 1]],
        [[0], [1]],
        [0.4, 0.6],
        branch_count=8,
    )
    for mode in ("adaptive", "open_loop"):
        scalar = HorizonPlanner(m).plan(m.initial_state, 2, mode=mode)
        batch = plan_batched(m, m.initial_state, 2, mode=mode)
        assert batch.value == pytest.approx(scalar.root.expected_risk, abs=1e-10)


def test_crossing_budget_fails_without_pruning():
    m = CrossingGaussianModel([[-1], [1]], 1, [[0], [1]], [0.5, 0.5], branch_count=2)
    with pytest.raises(SearchLimitExceeded):
        m.branches(m.initial_state, 0)
