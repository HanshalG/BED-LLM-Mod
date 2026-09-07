from fractions import Fraction
import json

import numpy as np
import pytest

from environments.chembench_mopen.horizon import FiniteBeliefModel, HorizonPlanner
from scripts.number_game_initial_horizon_audit import (
    ExactMembershipHorizon,
    initial_records,
    load_compiler,
)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_exact_dp_matches_independent_general_contingent_solver(seed):
    rng = np.random.default_rng(seed)
    extensions = sorted(set(tuple(row) for row in (rng.random((8, 6)) > 0.5).tolist()))
    e = np.array(extensions, dtype=float)
    likelihoods = np.stack((1 - e, e), axis=-1)
    model = FiniteBeliefModel(likelihoods, e, np.full(len(e), 1 / len(e)))
    reference = HorizonPlanner(model)
    exact = ExactMembershipHorizon(extensions)
    for h in [1, 2, 3]:
        plan = reference.plan(model.initial_state, h)
        assert float(exact.plan(exact.full, h)[0]) == pytest.approx(
            plan.root.expected_risk, abs=1e-14
        )

    def receding(state, available, budget, horizon):
        if not budget:
            return model.risk(state)
        values = reference.plan(
            state, min(budget, horizon), available=available
        ).root_action_values
        minimum = min(v for _, v in values)
        # The reference uses floats; reproduce the declared exact-tie rule when
        # values differ only by floating arithmetic, not by decision utility.
        q = min(q for q, v in values if abs(v - minimum) < 1e-13)
        return sum(
            b.probability
            * receding(
                b.state, tuple(a for a in available if a != q), budget - 1, horizon
            )
            for b in model.branches(state, q)
        )

    comparison = exact.compare()
    for index, horizon in enumerate([1, 2, 3]):
        value = receding(model.initial_state, tuple(range(6)), 3, horizon)
        assert comparison["full_budget_values"][index] == pytest.approx(
            value, abs=1e-14
        )


def test_queries_do_not_disappear_from_target_loss():
    solver = ExactMembershipHorizon([(False, False), (True, False)])
    assert solver.risk(solver.full) == Fraction(1, 8)
    assert solver.compare()["full_budget_values"] == [0.0, 0.0, 0.0]


def test_partition_equivalence_preserves_lowest_query_and_excludes_constants():
    solver = ExactMembershipHorizon([(False, True, False), (True, False, False)])
    assert len(list(solver.partitions(solver.full))) == 1
    assert solver.plan(solver.full, 1)[1] == 0


def test_root_projection_ignores_future_fields_without_converting_them():
    class Bomb:
        def __str__(self):
            raise AssertionError("future field inspected")

        def __int__(self):
            raise AssertionError("future field inspected")

    events = [
        ("trees.item", "start_map", None),
        ("trees.item.targets.anything", "string", Bomb()),
        ("trees.item.second_branches.anything", "number", Bomb()),
        ("trees.item.tree_seed", "number", 12),
        ("trees.item.initial.item", "start_map", None),
        ("trees.item.initial.item.expression", "string", "n < 20"),
        ("trees.item.initial.item", "end_map", None),
        ("trees.item", "end_map", None),
    ]
    assert initial_records(events) == [
        {"tree_seed": 12, "initial": [{"expression": "n < 20"}]}
    ]


def test_streaming_parser_returns_initial_fields_only():
    ijson = pytest.importorskip("ijson")
    import io

    data = {
        "trees": [
            {
                "tree_seed": 8,
                "initial": [{"expression": "n > 4"}],
                "targets": {"expression": "n < 3"},
                "first_branches": {"x": []},
            }
        ]
    }
    records = initial_records(ijson.parse(io.BytesIO(json.dumps(data).encode())))
    assert records == [{"tree_seed": 8, "initial": [{"expression": "n > 4"}]}]


def test_compiler_isolated_from_adapters_and_targets():
    compile_rule = load_compiler("scripts/number_game_generator_aware_bed.py")
    assert sum(compile_rule("n % 2 == 0")) == 51
    assert "TARGET_EXPRESSIONS" not in compile_rule.__globals__
    assert "DefaultRoutingStructuredAdapter" not in compile_rule.__globals__
    with pytest.raises(ValueError):
        compile_rule("__import__('os')")


def test_resource_limits_and_invalid_support():
    with pytest.raises(ValueError):
        ExactMembershipHorizon([(True,), (True,)])
    with pytest.raises(ValueError):
        ExactMembershipHorizon([(0,), (1,)])
    for kwargs, error in [
        ({"max_states": 1}, RuntimeError),
        ({"max_seconds": -1}, TimeoutError),
    ]:
        with pytest.raises(error):
            ExactMembershipHorizon([(False,), (True,)], **kwargs).compare()
