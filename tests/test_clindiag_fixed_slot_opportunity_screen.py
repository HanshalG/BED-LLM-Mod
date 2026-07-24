from __future__ import annotations

import pytest

from helpers import load_config
from scripts.clindiag_fixed_slot_audit import ACTION_IDS
from scripts.clindiag_fixed_slot_opportunity_screen import (
    EXPECTED_REQUESTS,
    SCREEN_IDS,
    SUPPORT_IDS,
    source_evidence_contains_target,
    summarize,
)
from scripts.clindiag_staged_generator_gate import ClinDiagCase


def _case() -> ClinDiagCase:
    return ClinDiagCase(
        source_id="fresh",
        subset="challenging",
        initial_information="A patient has fatigue.",
        medical_history={},
        physical_examination={},
        diagnostic_test={},
        final_diagnosis="Target syndrome",
    )


def test_config_is_fail_closed_and_budgeted() -> None:
    config = load_config(
        "configs/config_clindiag_fixed_slot_opportunity_screen_openrouter.yaml"
    )
    assert config.openrouter_budget_usd == pytest.approx(70.38480269545715)
    assert config.openrouter_run_budget_usd == pytest.approx(0.5)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.25)
    assert config.openrouter_concurrency == 24
    assert config.openrouter_max_output_tokens == 4096


def test_case_action_and_request_sets_are_frozen() -> None:
    assert SCREEN_IDS == ("20220188", "11222813", "rare140", "rare122")
    assert len(SUPPORT_IDS) == 1 + len(ACTION_IDS)
    assert EXPECTED_REQUESTS == 40


def test_source_target_check_reads_every_slot() -> None:
    clean = {action_id: f"finding {index}" for index, action_id in enumerate(ACTION_IDS)}
    leaking = dict(clean)
    leaking["other_1"] = "Target syndrome was confirmed"
    assert not source_evidence_contains_target(_case(), clean)
    assert source_evidence_contains_target(_case(), leaking)


def test_summary_requires_two_cases_with_one_step_headroom() -> None:
    records = [
        {"source_id": "a", "two_step_room": True},
        {"source_id": "b", "two_step_room": True},
        {"source_id": "c", "two_step_room": False},
        {"source_id": "d", "two_step_room": False},
    ]
    usage = {"physical_requests": 40, "reasoning_tokens": 0}
    result = summarize(
        records,
        usage,
        all_supports_size_twelve=True,
        source_target_leaks=0,
    )
    assert result["gates"]["all_pass"]
    records[1]["two_step_room"] = False
    result = summarize(
        records,
        usage,
        all_supports_size_twelve=True,
        source_target_leaks=0,
    )
    assert not result["gates"]["all_pass"]
