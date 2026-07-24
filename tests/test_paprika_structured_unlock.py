from __future__ import annotations

import json

import pytest

from scripts.paprika_structured_unlock import (
    SEMANTIC_COVERAGE_THRESHOLD,
    diagnostic_candidate_messages,
    parse_coverage_response,
    semantic_coverage_messages,
    summarize,
)


def test_unlock_generation_prompt_excludes_private_solution() -> None:
    messages = diagnostic_candidate_messages(
        "The device fails.",
        ["A hidden cause and remedy."],
        4,
    )
    text = json.dumps(messages)
    assert "private_solution" not in text
    assert "hidden cause and remedy" in text
    assert "Do not suggest, perform, identify, or name a remedy" in text


def test_coverage_prompt_is_explicitly_measurement_only() -> None:
    messages = semantic_coverage_messages(
        "The device fails.",
        "Replace the broken controller.",
        [("initial", ["Repair the power cable."])],
    )
    text = json.dumps(messages)
    assert "private_solution" in text
    assert "hidden from every hypothesis generator" in text


def test_parse_coverage_response_preserves_ids_bounds_and_threshold() -> None:
    response = json.dumps(
        {
            "supports": [
                {
                    "id": "initial",
                    "best_match_score": SEMANTIC_COVERAGE_THRESHOLD,
                    "best_hypothesis_index": 1,
                    "reason": "same cause and remedy",
                },
                {
                    "id": "candidate_0",
                    "best_match_score": 0.3,
                    "best_hypothesis_index": 0,
                    "reason": "different cause",
                },
            ]
        }
    )
    parsed = parse_coverage_response(
        response,
        ["initial", "candidate_0"],
        [2, 1],
    )
    assert parsed[0]["covered"] is True
    assert parsed[1]["covered"] is False
    assert parsed[0]["best_hypothesis_index_valid"] is True
    with pytest.raises(ValueError, match="IDs or order"):
        parse_coverage_response(
            response,
            ["candidate_0", "initial"],
            [2, 1],
        )


def test_coverage_parser_records_invalid_explanatory_index_without_changing_score() -> None:
    response = json.dumps(
        {
            "supports": [
                {
                    "id": "initial",
                    "best_match_score": 0.9,
                    "best_hypothesis_index": 8,
                    "reason": "same cause and remedy",
                }
            ]
        }
    )
    parsed = parse_coverage_response(response, ["initial"], [8])
    assert parsed[0]["best_match_score"] == 0.9
    assert parsed[0]["best_hypothesis_index"] is None
    assert parsed[0]["reported_best_hypothesis_index"] == 8
    assert parsed[0]["best_hypothesis_index_valid"] is False


def test_unlock_summary_applies_frozen_gate() -> None:
    records = []
    for index in range(12):
        initial_score = 0.2 if index < 6 else 0.9
        candidate_scores = (
            [0.2, 0.4, 0.85, 0.3]
            if index < 4
            else [initial_score] * 4
        )
        records.append(
            {
                "initial": {
                    "best_match_score": initial_score,
                    "covered": initial_score >= SEMANTIC_COVERAGE_THRESHOLD,
                },
                "candidates": [
                    {
                        "coverage": {
                            "best_match_score": score,
                            "covered": score >= SEMANTIC_COVERAGE_THRESHOLD,
                        }
                    }
                    for score in candidate_scores
                ],
            }
        )
    summary = summarize(records)
    assert summary["initial_omitted"] == 6
    assert summary["initially_omitted_recovered_by_any_candidate"] == 4
    assert summary["gates"]["all_pass"] is True
