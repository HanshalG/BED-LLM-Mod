from __future__ import annotations

import json
from pathlib import Path

from scripts.longvid_bridge_path_opportunity_audit import _id_hash
from scripts.longvid_four_hop_tradeoff_opportunity import (
    CONFIRMATION_ID_HASH,
    OPPORTUNITY_ID_HASH,
    RESERVE_ID_HASH,
    _trajectory_key,
    summarize,
)


def test_frozen_four_hop_row_hashes_are_stable() -> None:
    opportunity = [
        1381, 2112, 955, 2545, 1047, 1802, 2366, 1985, 1504, 540,
        1148, 1695, 1233, 274, 392, 164, 479, 1358, 389, 1303,
        2462, 1722, 1332, 681, 969, 2695, 1068, 2501, 2156, 213,
        2250, 252, 2358, 2496, 594, 2062, 2421, 2671, 1689, 1648,
    ]
    confirmation = [
        236, 1482, 2958, 1522, 218, 1314, 1404, 1415, 749, 900,
        1422, 2703, 883, 1347, 1903, 1867, 2514, 485, 855, 1295,
        2923, 453, 1207, 2989, 134, 1103, 2728, 549, 1817, 1630,
        2433, 2642, 2345, 319, 120, 1476, 908, 561, 282, 2596,
    ]
    reserve = [
        154, 2286, 1826, 2935, 1791, 302, 720, 2093, 2263, 672,
        959, 525, 895, 520, 1578, 2146, 2977, 644, 341, 1193,
        1999, 150,
    ]
    assert _id_hash(opportunity) == OPPORTUNITY_ID_HASH
    assert _id_hash(confirmation) == CONFIRMATION_ID_HASH
    assert _id_hash(reserve) == RESERVE_ID_HASH
    assert len(set(opportunity + confirmation + reserve)) == 102


def test_trajectory_key_uses_candidate_order_for_equal_coverage() -> None:
    earlier = {"final_count": 3, "candidate_order": (0, 2, 1)}
    later = {"final_count": 3, "candidate_order": (1, 0, 0)}
    better = {"final_count": 4, "candidate_order": (7, 7, 7)}
    assert _trajectory_key(earlier) > _trajectory_key(later)
    assert _trajectory_key(better) > _trajectory_key(earlier)


def _passing_records() -> list[dict[str, object]]:
    records = []
    for index in range(40):
        strict = index < 6
        records.append(
            {
                "num_captions": 80,
                "num_answer_terms": 3,
                "num_roots": 8,
                "distinct_root_top1": 3 if index < 30 else 2,
                "depth_four_gain_count": 1 if index < 24 else 0,
                "ordered_chain_recovered": index < 8,
                "oracle_final_coverage": 0.75 if index < 24 else 0.25,
                "coverage_gain": 0.5 if index < 24 else 0.0,
                "strict_tradeoff": strict,
                "strict_gap_count": 1 if strict else 0,
                "strict_answer_sacrifice": 0.2 if strict else 0.0,
            }
        )
    return records


def test_four_hop_summary_passes_exact_thresholds() -> None:
    summary = summarize(_passing_records())
    assert summary["strict_tradeoff_count"] == 6
    assert summary["strict_total_gap"] == 6
    assert summary["gates"]["all_pass"]


def test_four_hop_summary_rejects_sparse_tradeoffs() -> None:
    records = _passing_records()
    records[5]["strict_tradeoff"] = False
    records[5]["strict_gap_count"] = 0
    records[5]["strict_answer_sacrifice"] = 0.0
    summary = summarize(records)
    assert summary["strict_tradeoff_count"] == 5
    assert not summary["gates"]["strict_tradeoffs_at_least_6"]
    assert not summary["gates"]["all_pass"]


def test_frozen_four_hop_artifact_fails_only_completeness() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "nonmyopic"
        / "longvid_four_hop_tradeoff_opportunity"
        / "AUDIT.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert payload["status"] == "gate_failed"
    assert summary["num_records"] == 40
    assert summary["complete_task_count"] == 39
    assert summary["strict_tradeoff_count"] == 10
    assert summary["strict_total_gap"] == 11
    assert not summary["gates"]["all_tasks_complete"]
    assert all(
        passed
        for name, passed in summary["gates"].items()
        if name not in {"all_tasks_complete", "all_pass"}
    )
    assert not summary["gates"]["all_pass"]
