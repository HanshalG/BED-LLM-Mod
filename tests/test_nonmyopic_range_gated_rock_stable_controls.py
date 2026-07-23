from __future__ import annotations

import json
from pathlib import Path

from scripts.nonmyopic_range_gated_rock_stable_controls import (
    qualify_stable_controls,
    stable_best_index,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stable_best_index_uses_canonical_order_inside_tolerance() -> None:
    assert stable_best_index([1.0, 1.0 + 5e-13, 0.5]) == 0
    assert stable_best_index([1.0, 1.0 + 2e-12, 0.5]) == 1


def test_stable_control_qualification_passes_banked_proposal() -> None:
    path = (
        ROOT
        / "results"
        / "nonmyopic"
        / "range_gated_rock_json_prefix_cluster26b_proposal_20260723"
        / "PROPOSAL.json"
    )
    qualification = qualify_stable_controls(
        json.loads(path.read_text(encoding="utf-8"))
    )

    assert qualification["gate"]["passed"]
    assert len(qualification["canonical_digest"]) == 64
    assert len(qualification["records"]) == 16
    assert qualification["records"][0]["strong_d2_root"] == "check-0"
    assert qualification["records"][0]["exact_h3_plan"] == [
        "move-SOUTH",
        "move-SOUTH",
        "check-5",
    ]
