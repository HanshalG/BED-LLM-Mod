from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.longvid_four_hop_ranking_mechanics import (
    TASK_LAYOUT_HASH,
    analyze_records,
    layout_hash,
    support_entropy,
)


def test_frozen_blinded_task_layout_hash_is_stable() -> None:
    assert layout_hash() == TASK_LAYOUT_HASH


def test_support_entropy_uses_normalized_integer_weights() -> None:
    uniform = [
        {"weight": 10} for _ in range(6)
    ]
    concentrated = [
        {"weight": 95},
        {"weight": 1},
        {"weight": 1},
        {"weight": 1},
        {"weight": 1},
        {"weight": 1},
    ]
    assert support_entropy(uniform) == pytest.approx(1.791759469228055)
    assert support_entropy(concentrated) < support_entropy(uniform)


def _candidate(
    root_index: int,
    immediate_score: float,
    final_score: float,
    coverage_count: int,
) -> dict[str, float | int]:
    return {
        "root_index": root_index,
        "immediate_score": immediate_score,
        "final_score": final_score,
        "coverage_count": coverage_count,
    }


def test_analysis_rewards_final_scores_that_rank_coverage() -> None:
    records = []
    for index in range(6):
        records.append(
            {
                "row_index": index,
                "category": "fixture",
                "candidates": [
                    _candidate(0, 0.4, 0.1, 1),
                    _candidate(1, 0.2, 0.6, 2),
                ],
            }
        )
    analysis = analyze_records(records)
    summary = analysis["summary"]
    assert summary["rankable_task_count"] == 6
    assert summary["final_pairwise_accuracy"] == 1.0
    assert summary["immediate_pairwise_accuracy"] == 0.0
    assert summary["final_score_spearman"] > 0.8
    assert summary["final_selected_coverage_total"] == 12
    assert summary["immediate_selected_coverage_total"] == 6


def test_analysis_marks_tied_coverage_tasks_unrankable() -> None:
    records = [
        {
            "row_index": index,
            "category": "fixture",
            "candidates": [
                _candidate(0, 0.1, 0.2, 2),
                _candidate(1, 0.3, 0.4, 2),
            ],
        }
        for index in range(6)
    ]
    summary = analyze_records(records)["summary"]
    assert summary["rankable_task_count"] == 0
    assert summary["final_pairwise_accuracy"] == 0.0


def test_frozen_live_mechanics_failure_has_no_endpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    failure_path = (
        root
        / "results"
        / "nonmyopic"
        / "longvid_four_hop_ranking_mechanics"
        / "longvid-four-hop-ranking-mechanics-20260727T150132Z"
        / "MECHANICS_FAILURE.json"
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "failed_closed"
    assert "H2 anchor must be one token" in failure["error"]
    assert failure["usage"]["physical_requests"] == 18
    assert failure["usage"]["http_attempts"] == 18
    assert failure["usage"]["reasoning_tokens"] == 0
    assert failure["usage"]["adapter_cost_usd"] == pytest.approx(0.1159575)
