from __future__ import annotations

import pytest

from helpers import load_config
from scripts.clindiag_filtered_retention_recovery_gate import (
    ACTION_ID,
    EXPECTED_REQUESTS,
    NUM_CANDIDATE_ROUNDS,
    SELECTION_SEED,
    SMOKE_IDS,
    _exact_overlap,
    _group_by_case,
    summarize,
)


def test_recovery_config_and_protocol_are_frozen() -> None:
    config = load_config(
        "configs/config_clindiag_filtered_retention_recovery_gate_openrouter.yaml"
    )
    assert config.openrouter_run_budget_usd == pytest.approx(0.50)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.15)
    assert config.generation_temperature_diverse == pytest.approx(0.5)
    assert config.openrouter_concurrency == 24
    assert SELECTION_SEED == 24300
    assert SMOKE_IDS == ("27223150", "rare130")
    assert ACTION_ID == "lab_2"
    assert NUM_CANDIDATE_ROUNDS == 3
    assert EXPECTED_REQUESTS == 30


def test_candidate_rows_group_by_case_without_reordering_rounds() -> None:
    assert _group_by_case(list(range(6)), num_cases=2) == [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(ValueError, match="cases times rounds"):
        _group_by_case(list(range(5)), num_cases=2)


def test_exact_overlap_is_descriptive_and_directional() -> None:
    overlap = _exact_overlap(["A", "B", "C"], ["a", "B", "D"])
    assert overlap == {
        "intersection": 2,
        "left_fraction": pytest.approx(2 / 3),
        "right_fraction": pytest.approx(2 / 3),
        "jaccard": pytest.approx(1 / 2),
    }


def _record(*, transition: bool = True) -> dict[str, object]:
    support = [f"D{index}" for index in range(12)]
    return {
        "source_id": "case",
        "supports": {
            "filtered_retention": support,
            "filtered_retention_duplicate": support,
        },
        "num_old_pruned": 4 if transition else 0,
        "num_new_introduced": 4 if transition else 0,
        "num_duplicate_new_introduced": 4 if transition else 0,
        "initial_truth_score": 0.0,
        "final_truth_score": 0.8,
        "duplicate_truth_score": 0.8,
        "duplicate_truth_score_gap": 0.0,
        "source_target_leak": False,
        "exact_overlap": {"jaccard": 1.0},
    }


def test_summary_requires_one_substantive_transition_not_both() -> None:
    usage = {"physical_requests": 30, "reasoning_tokens": 0}
    result = summarize(
        [_record(transition=True), _record(transition=False)],
        usage,
        candidate_generation_prompts_exact=True,
    )
    assert result["gates"]["all_pass"]

    result = summarize(
        [_record(transition=False), _record(transition=False)],
        usage,
        candidate_generation_prompts_exact=True,
    )
    assert not result["gates"]["all_pass"]
