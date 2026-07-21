from scripts.nonmyopic_rock_branch_strategy_smoke import run_smoke
from scripts.nonmyopic_rock_strategy_prior import DeterministicStrategyModel


def test_dry_branch_strategy_serving_smoke_passes_all_ten_cells() -> None:
    summary = run_smoke(DeterministicStrategyModel(), concurrency=2)

    assert summary["passed"]
    assert summary["mechanics"] == {
        "requested_cells": 10,
        "passed_cells": 10,
        "terminal_failures": 0,
        "raw_rejected_attempts": 0,
        "parse_rate": 1.0,
        "all_cells_have_move_and_check_roots": True,
        "all_move_cells_include_move_then_check": True,
    }
    horizon_two = [row for row in summary["cells"] if row["horizon"] == 2]
    horizon_one = [row for row in summary["cells"] if row["horizon"] == 1]
    assert len(horizon_two) == 8
    assert len(horizon_one) == 2
    assert all(row["move_policy_count"] == 2 for row in horizon_two)
    assert all(row["check_policy_count"] == 2 for row in horizon_two)
    assert all(
        strategy["followups"] == {}
        for row in horizon_one
        for strategy in row["strategies"]
    )


def test_single_large_map_smoke_uses_ten_distinct_probe_states() -> None:
    summary = run_smoke(
        DeterministicStrategyModel(),
        concurrency=2,
        map_names=("7-8",),
        probe_states_per_map=10,
    )

    assert summary["passed"]
    assert summary["mechanics"]["requested_cells"] == 10
    assert {row["map_name"] for row in summary["cells"]} == {"7-8"}
    assert len({row["state_index"] for row in summary["cells"]}) == 10
    assert sum(row["horizon"] == 1 for row in summary["cells"]) == 2
