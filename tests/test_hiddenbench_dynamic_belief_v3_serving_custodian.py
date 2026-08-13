from __future__ import annotations

from pathlib import Path

from scripts.hiddenbench_dynamic_belief_v3_serving_custodian import project, select_rows


SOURCE = Path("/tmp/bed-source-audits/hiddenbench/data/benchmark.json")


def test_serving_custodian_has_no_endpoint_surface() -> None:
    rows = select_rows(SOURCE)
    views = project(rows)
    assert set(views) == {"planner", "router"}
    assert len(views["planner"]) == len(views["router"]) == 4
    assert all(set(task) == {"slot", "description", "shared_facts", "options"} for task in views["planner"])
    assert all(set(task) == {"slot", "description", "private_facts"} for task in views["router"])
    assert all("correct_answer" not in task for task in views["planner"] + views["router"])


def test_module_contains_no_endpoint_function_or_mode() -> None:
    source = Path("scripts/hiddenbench_dynamic_belief_v3_serving_custodian.py").read_text()
    assert "correct_answer" not in source
    assert "endpoint_view" not in source
    assert "--view" not in source
