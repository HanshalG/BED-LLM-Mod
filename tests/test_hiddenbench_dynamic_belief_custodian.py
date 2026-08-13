from __future__ import annotations

import json
from pathlib import Path

from scripts.hiddenbench_dynamic_belief_custodian import (
    endpoint_view,
    planner_router_views,
    selected_rows,
)


SOURCE = Path("/tmp/bed-source-audits/hiddenbench/data/benchmark.json")


def test_views_are_disjoint_and_endpoint_is_opaque() -> None:
    rows = selected_rows(SOURCE)
    views = planner_router_views(rows)
    endpoints = endpoint_view(rows)
    assert len(views["planner"]) == len(views["router"]) == 4
    assert all("options" in row and "private_facts" not in row for row in views["planner"])
    assert all("private_facts" in row and "options" not in row for row in views["router"])
    assert all(set(row) == {"slot", "correct_option_id"} for row in endpoints["endpoints"])
    planner_text = json.dumps(views["planner"])
    router_text = json.dumps(views["router"])
    source_by_slot = dict(zip((f"T{i + 1}" for i in range(4)), rows, strict=True))
    for slot, row in source_by_slot.items():
        assert row["correct_answer"] not in router_text
        assert all(value not in planner_text for value in row["hidden_information"])
        assert all(value not in router_text for value in row["possible_answers"])
        endpoint = next(item for item in endpoints["endpoints"] if item["slot"] == slot)
        assert endpoint["correct_option_id"].startswith("O")
