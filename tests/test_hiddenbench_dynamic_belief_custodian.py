from __future__ import annotations

import json
from pathlib import Path

from scripts.hiddenbench_dynamic_belief_custodian import (
    endpoint_view,
    planner_router_views,
    selected_rows,
)


SOURCE = Path("/tmp/bed-source-audits/hiddenbench/data/benchmark.json")


def test_real_source_planner_router_views_are_disjoint() -> None:
    rows = selected_rows(SOURCE)
    views = planner_router_views(rows)
    assert len(views["planner"]) == len(views["router"]) == 4
    assert all("options" in row and "private_facts" not in row for row in views["planner"])
    assert all("private_facts" in row and "options" not in row for row in views["router"])
    planner_text = json.dumps(views["planner"])
    for row in rows:
        assert all(value not in planner_text for value in row["hidden_information"])
    assert all(
        set(row) == {"slot", "description", "private_facts"}
        and all(set(fact) == {"id", "text"} for fact in row["private_facts"])
        for row in views["router"]
    )


def test_endpoint_projection_uses_synthetic_rows_only() -> None:
    rows = [
        {
            "possible_answers": ["synthetic red", "synthetic blue", "synthetic green"],
            "correct_answer": "synthetic blue",
        }
        for _ in range(4)
    ]
    endpoints = endpoint_view(rows)
    assert endpoints == {
        "endpoints": [
            {"slot": f"T{index + 1}", "correct_option_id": "O2"}
            for index in range(4)
        ]
    }
