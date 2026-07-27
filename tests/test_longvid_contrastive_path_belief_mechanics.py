from __future__ import annotations

import json

from scripts.longvid_contrastive_path_belief_mechanics import (
    TASK_LAYOUT_HASH,
    analyze_records,
    layout_hash,
    parse_rank,
    parse_support,
    support_response_format,
)


def _support_payload(anchor: str = "QUESTION") -> dict[str, object]:
    payload: dict[str, object] = {}
    for index in range(1, 7):
        payload[f"hypothesis_{index}"] = (
            f"Distinct evidence-chain hypothesis number {index}"
        )
        payload[f"weight_{index}"] = 10 + index
        payload[f"anchor_{index}"] = anchor
        payload[f"query_{index}"] = f"{anchor} distinct search {index}"
    return payload


def test_layout_hash_is_stable() -> None:
    assert layout_hash() == TASK_LAYOUT_HASH


def test_support_schema_is_flat_and_strict() -> None:
    schema = support_response_format()["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert len(schema["required"]) == 24


def test_support_parser_accepts_unicode_observation_anchor() -> None:
    parsed = parse_support(json.dumps(_support_payload("€4bn")))
    assert len(parsed) == 6
    assert parsed[0]["anchor"] == "€4bn"


def test_rank_parser_is_strict() -> None:
    parsed = parse_rank(
        json.dumps(
            {
                "choice": "B",
                "confidence": 79,
                "unresolved_need": "the intermediate event",
            }
        )
    )
    assert parsed["choice"] == "B"
    assert parsed["confidence"] == 79


def test_analysis_rewards_correct_nonmyopic_flips() -> None:
    records = []
    for row_index in range(4):
        records.append(
            {
                "row_index": row_index,
                "category": "fixture",
                "candidates": [
                    {"root_index": 0, "coverage_count": 1},
                    {"root_index": 1, "coverage_count": 2},
                ],
                "immediate_rank": {
                    "choice": "A",
                    "confidence": 70,
                    "unresolved_need": "x",
                },
                "final_rank": {
                    "choice": "B",
                    "confidence": 80,
                    "unresolved_need": "y",
                },
            }
        )
    summary = analyze_records(records)["summary"]
    assert summary["rankable_task_count"] == 4
    assert summary["final_pairwise_accuracy"] == 1.0
    assert summary["immediate_pairwise_accuracy"] == 0.0
    assert summary["policy_change_count"] == 4
    assert summary["correct_policy_change_count"] == 4
    assert summary["final_coverage_gain_over_immediate"] == 4
