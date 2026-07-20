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
    assert all(row["move_policy_count"] == 2 for row in summary["cells"])
    assert all(row["check_policy_count"] == 2 for row in summary["cells"])
