from __future__ import annotations

import json

import pytest

from helpers import load_config
from scripts.clindiag_branch_opportunity_gate import (
    ACTIONS,
    EXPECTED_FORMAL_REQUESTS,
    OPPORTUNITY_IDS,
    SEQUENCES,
    SMOKE_IDS,
    action_evidence,
    analyze_record,
    refresh_differential_messages,
    summarize,
)
from scripts.clindiag_staged_generator_gate import (
    DEVELOPMENT_IDS,
    HOLDOUT_IDS,
    SMOKE_IDS as GENERATOR_SMOKE_IDS,
    ClinDiagCase,
)


def _case() -> ClinDiagCase:
    return ClinDiagCase(
        source_id="case",
        subset="challenging",
        initial_information="A patient has fatigue.",
        medical_history={"medical_history": {"history": "progressive symptoms"}},
        physical_examination={"physical_examinations": [{"finding": "rash"}]},
        diagnostic_test={
            "laboratory_examinations": [{"finding": "marker elevated"}],
            "radiographic_examinations": [{"finding": "mass"}],
            "other_examinations": [{"finding": "pathogenic variant"}],
        },
        final_diagnosis="Secret syndrome",
    )


def test_opportunity_split_is_fresh_balanced_and_action_complete() -> None:
    prior = {
        *DEVELOPMENT_IDS,
        *HOLDOUT_IDS,
        *GENERATOR_SMOKE_IDS,
    }
    assert len(OPPORTUNITY_IDS) == 12
    assert len(SMOKE_IDS) == 2
    assert not set(OPPORTUNITY_IDS).intersection(prior)
    assert not set(SMOKE_IDS).intersection(prior)
    assert not set(SMOKE_IDS).intersection(OPPORTUNITY_IDS)
    assert sum(value.startswith("rare") for value in OPPORTUNITY_IDS) == 6
    assert len(ACTIONS) == 5
    assert len(SEQUENCES) == 20
    assert EXPECTED_FORMAL_REQUESTS == 336


def test_action_evidence_requires_each_native_channel() -> None:
    case = _case()
    assert all(action_evidence(case, action) for action in ACTIONS)
    with pytest.raises(ValueError, match="unknown ClinDiag action"):
        action_evidence(case, "secret")


def test_refresh_prompt_preserves_order_and_hides_target() -> None:
    case = _case()
    messages = refresh_differential_messages(
        case,
        ("history", "imaging"),
        ["Prior diagnosis"],
    )
    text = json.dumps(messages)
    assert text.index("Medical history") < text.index(
        "Radiographic and imaging tests"
    )
    assert "Prior diagnosis" in text
    assert "Secret syndrome" not in text
    assert "final diagnosis" in text


def test_analyze_record_computes_oracle_nonmyopic_gap() -> None:
    one = {action: 0.1 for action in ACTIONS}
    one["history"] = 0.8
    seq = {f"{a}>{b}": 0.2 for a, b in SEQUENCES}
    seq["history>imaging"] = 0.7
    seq["laboratory_tests>other_tests"] = 1.0
    supports = {key: [key] for key in seq}
    record = {
        "one_step_scores": one,
        "sequence_scores": seq,
        "sequence_supports": supports,
        "duplicate_sequence": "history>physical_exam",
        "duplicate_score": seq["history>physical_exam"],
        "duplicate_support": supports["history>physical_exam"],
    }
    result = analyze_record(record)
    assert result["greedy_action"] == "history"
    assert result["oracle_sequence"] == "laboratory_tests>other_tests"
    assert result["greedy_continuation_score"] == pytest.approx(0.7)
    assert result["nonmyopic_gap_over_greedy_continuation"] == pytest.approx(0.3)
    assert result["duplicate_score_gap"] == 0.0
    assert result["duplicate_support_jaccard"] == 1.0


def test_summary_applies_frozen_opportunity_gate() -> None:
    records = []
    for index in range(12):
        one = {action: 0.1 for action in ACTIONS}
        one["history"] = 0.5
        seq = {f"{a}>{b}": 0.2 for a, b in SEQUENCES}
        seq["history>physical_exam"] = 0.6
        seq["laboratory_tests>other_tests"] = 0.9 if index < 8 else 0.6
        seq["other_tests>laboratory_tests"] = 0.4
        supports = {key: [key] for key in seq}
        records.append(
            {
                "source_id": str(index),
                "initial_score": 0.1,
                "one_step_scores": one,
                "sequence_scores": seq,
                "sequence_supports": supports,
                "duplicate_sequence": "history>physical_exam",
                "duplicate_score": 0.6,
                "duplicate_support": supports["history>physical_exam"],
            }
        )
    summary = summarize(records)
    assert summary["one_step_spread_at_least_0_20_cases"] == 12
    assert summary["reverse_order_gap_at_least_0_15_cases"] == 12
    assert summary["nonmyopic_gap_at_least_0_10_cases"] == 8
    assert summary["mean_duplicate_support_jaccard"] == 1.0
    assert "mean_duplicate_jaccard_at_least_0_75" not in summary["gates"]
    assert summary["gates"]["all_pass"] is True


def test_opportunity_config_is_deterministic_and_fail_closed() -> None:
    config = load_config(
        "configs/config_clindiag_branch_opportunity_openrouter.yaml"
    )
    assert config.generation_temperature_simple == 0.0
    assert config.openrouter_budget_usd == pytest.approx(70.38480269545715)
    assert config.openrouter_run_budget_usd == pytest.approx(3.0)
    assert config.openrouter_projected_cost_usd == pytest.approx(2.0)
    assert config.openrouter_concurrency == 128
