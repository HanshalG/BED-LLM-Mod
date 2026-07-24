from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import load_config
from scripts.clindiag_fixed_slot_pair_screen import (
    EXPECTED_REQUESTS,
    PAIR_IDS,
    PAIR_SCREEN_IDS,
    PREVALENCE_SHA256,
    load_prevalence_records,
    pair_support_id,
    parse_oracle_validation,
    parse_pair_support_id,
    summarize,
)


PREVALENCE_PATH = Path(
    "results/nonmyopic/clindiag_fixed_slot_opportunity_screen/"
    "prevalence_20260724/PREVALENCE_SCREEN.json"
)


def _validation_payload(
    original_score: float = 0.9,
    duplicate_score: float = 0.9,
    forward: float = 0.9,
    reverse: float = 0.8,
) -> dict[str, object]:
    return {
        "supports": [
            {
                "id": "oracle_original",
                "best_match_score": original_score,
                "reason": "Equivalent diagnosis.",
            },
            {
                "id": "oracle_duplicate",
                "best_match_score": duplicate_score,
                "reason": "Equivalent diagnosis.",
            },
        ],
        "original_to_duplicate_overlap": forward,
        "duplicate_to_original_overlap": reverse,
        "overlap_reason": "Most diagnoses match semantically.",
    }


def test_config_is_fail_closed_and_budgeted() -> None:
    config = load_config(
        "configs/config_clindiag_fixed_slot_pair_screen_openrouter.yaml"
    )
    assert config.openrouter_budget_usd == pytest.approx(70.38480269545715)
    assert config.openrouter_run_budget_usd == pytest.approx(3.0)
    assert config.openrouter_projected_cost_usd == pytest.approx(2.2)
    assert config.openrouter_concurrency == 24
    assert config.openrouter_max_output_tokens == 4096


def test_pair_grid_and_request_count_are_frozen() -> None:
    assert len(PAIR_SCREEN_IDS) == 7
    assert len(PAIR_IDS) == 56
    assert len(set(PAIR_IDS)) == 56
    assert EXPECTED_REQUESTS == 434
    for pair_id in PAIR_IDS:
        first, second = parse_pair_support_id(pair_id)
        assert first != second
        assert pair_support_id(first, second) == pair_id


def test_prevalence_artifact_is_pinned_and_selects_seven_cases() -> None:
    records = load_prevalence_records(PREVALENCE_PATH)
    assert [record["source_id"] for record in records] == list(PAIR_SCREEN_IDS)
    assert len(PREVALENCE_SHA256) == 64


def test_oracle_validation_parser_uses_worse_overlap_direction() -> None:
    parsed = parse_oracle_validation(json.dumps(_validation_payload()))
    assert parsed["duplicate_semantic_overlap"] == pytest.approx(0.8)
    assert [row["id"] for row in parsed["supports"]] == [
        "oracle_original",
        "oracle_duplicate",
    ]


def test_oracle_validation_rejects_bad_scores() -> None:
    payload = _validation_payload(original_score=1.1)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        parse_oracle_validation(json.dumps(payload))


def test_summary_requires_three_unlocks_and_mean_gain() -> None:
    records = [
        {
            "source_id": f"case{index}",
            "validated_unlock": index < 3,
            "validated_gain": 0.5 if index < 3 else 0.0,
        }
        for index in range(7)
    ]
    usage = {"physical_requests": 434, "reasoning_tokens": 0}
    result = summarize(
        records,
        usage,
        all_pair_supports_size_twelve=True,
        duplicate_prompts_exact=True,
        source_target_leaks=0,
    )
    assert result["gates"]["all_pass"]
    records[2]["validated_unlock"] = False
    result = summarize(
        records,
        usage,
        all_pair_supports_size_twelve=True,
        duplicate_prompts_exact=True,
        source_target_leaks=0,
    )
    assert not result["gates"]["all_pass"]
