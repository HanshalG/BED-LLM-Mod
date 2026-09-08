import json

import numpy as np
import pytest

from environments.chembench_mopen.horizon import (
    HorizonPlanner,
    SearchLimits,
    SearchLimitExceeded,
)
from environments.scilaws.reference_prior import make_model
from environments.scilaws.regression_belief import RegressionBelief
from environments.scilaws.regression_mixture import RegressionMixture


class Scalar:
    def __init__(self, model):
        self.num_actions = model.num_actions
        self.risk = model.risk
        self.branches = model.branches


def test_terminal_batch_full_designs_and_conditioned_states():
    for task in json.load(
        open("results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json")
    )["tasks"]:
        m = make_model(task, quadrature_order=4)
        state = m.condition(m.initial_state, 3, 1.7)
        for action in range(8):
            rows = m.branches(state, action)
            reference = sum(r.probability * m.risk(r.state) for r in rows)
            actual, leaves = m.expected_terminal_risk(state, action)
            assert leaves == len(rows)
            assert actual == pytest.approx(reference, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mode", ["adaptive", "open_loop"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_complete_root_vectors_and_materialized_tree_match(mode, depth):
    a = RegressionBelief([-1.0], [[2.0]], 3.0, 1.0)
    b = RegressionBelief([1.0], [[1.0]], 3.0, 2.0)
    m = RegressionMixture(
        [[[1.0], [2.0]], [[2.0], [1.0]]],
        [[[1.0]], [[1.0]]],
        [a, b],
        [0.4, 0.6],
        target_weights=[1.0],
        quadrature_order=2,
    )
    fast = HorizonPlanner(m).plan(m.initial_state, depth, mode=mode, allow_repeats=True)
    slow = HorizonPlanner(Scalar(m)).plan(
        m.initial_state, depth, mode=mode, allow_repeats=True
    )
    np.testing.assert_allclose(
        fast.root_action_values, slow.root_action_values, atol=1e-12, rtol=1e-12
    )
    assert fast.root.action == slow.root.action
    assert fast.root.expected_risk == pytest.approx(slow.root.expected_risk, abs=1e-12)
    assert fast.fixed_sequence == slow.fixed_sequence


def test_terminal_hook_preserves_node_limit_and_rejects_invalid_results():
    class Bad:
        num_actions = 1

        def expected_terminal_risk(self, state, action):
            return float("nan"), 1

    with pytest.raises(ValueError, match="terminal risk"):
        HorizonPlanner(Bad()).plan((), 1)

    class Many:
        num_actions = 1

        def expected_terminal_risk(self, state, action):
            return 1.0, 100

    with pytest.raises(SearchLimitExceeded):
        HorizonPlanner(Many(), limits=SearchLimits(max_nodes=10)).plan((), 1)
