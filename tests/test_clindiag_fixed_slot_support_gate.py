from __future__ import annotations

import json

import pytest

from helpers import load_config
from scripts.clindiag_fixed_slot_support_gate import (
    EXPECTED_REQUESTS,
    REFRESH_ACTION_IDS,
    SMOKE_IDS,
    SUPPORT_IDS,
    deanchored_refresh_differential_messages,
    parse_stability_audit,
    refresh_differential_messages,
    source_evidence_contains_target,
    stability_audit_messages,
    summarize,
)
from scripts.clindiag_staged_generator_gate import ClinDiagCase


def _case() -> ClinDiagCase:
    return ClinDiagCase(
        source_id="fresh",
        subset="challenging",
        initial_information="A patient has fever and fatigue.",
        medical_history={},
        physical_examination={},
        diagnostic_test={},
        final_diagnosis="Target syndrome",
    )


def _audit_payload(
    lab_score: float = 0.8,
    duplicate_score: float = 0.8,
    forward: float = 0.9,
    reverse: float = 0.8,
) -> dict[str, object]:
    scores = (0.1, 0.2, lab_score, duplicate_score)
    return {
        "supports": [
            {
                "id": support_id,
                "best_match_score": score,
                "reason": "strict semantic comparison",
            }
            for support_id, score in zip(SUPPORT_IDS, scores, strict=True)
        ],
        "lab_to_duplicate_overlap": forward,
        "duplicate_to_lab_overlap": reverse,
        "overlap_reason": "Most diagnoses are standard synonyms.",
    }


def test_config_is_fail_closed_and_uses_small_budget() -> None:
    config = load_config(
        "configs/config_clindiag_fixed_slot_support_gate_openrouter.yaml"
    )
    assert config.openrouter_budget_usd == pytest.approx(70.38480269545715)
    assert config.openrouter_run_budget_usd == pytest.approx(0.5)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.15)
    assert config.openrouter_concurrency == 24
    assert config.openrouter_max_output_tokens == 2048


def test_smoke_split_and_refresh_actions_are_frozen() -> None:
    assert SMOKE_IDS == ("25992750", "rare167")
    assert REFRESH_ACTION_IDS == ("present_illness", "lab_1")
    assert EXPECTED_REQUESTS == 10


def test_refresh_prompt_is_ordered_deterministic_and_hides_truth() -> None:
    case = _case()
    observations = [
        ("present_illness", "Symptoms progressed for two weeks."),
        ("lab_1", {"procedure_name": "CBC", "findings": "Leukocytosis"}),
    ]
    support = ["Viral infection", "Bacterial infection"]
    first = refresh_differential_messages(case, observations, support)
    second = refresh_differential_messages(case, observations, support)
    assert first == second
    payload = first[1]["content"]
    assert payload.index("present_illness") < payload.index("lab_1")
    assert "Target syndrome" not in payload
    assert "Viral infection" in payload


def test_deanchored_refresh_rebuilds_without_prior_inclusion_constraint() -> None:
    messages = deanchored_refresh_differential_messages(
        _case(),
        [("present_illness", "Symptoms progressed.")],
        ["Viral infection"],
    )
    content = messages[1]["content"]
    assert "Rebuild the differential from scratch" in content
    assert "not as an inclusion constraint" in content
    assert "retain plausible earlier diagnoses" not in content
    assert "Target syndrome" not in content


def test_source_target_check_only_reads_visible_source_evidence() -> None:
    case = _case()
    clean = {"present_illness": "Fever", "lab_1": {"result": "Normal"}}
    leaking = {
        "present_illness": "Target syndrome was diagnosed",
        "lab_1": {"result": "Normal"},
    }
    assert not source_evidence_contains_target(case, clean)
    assert source_evidence_contains_target(case, leaking)


def test_stability_audit_parser_uses_worse_direction_as_overlap() -> None:
    parsed = parse_stability_audit(json.dumps(_audit_payload()))
    assert parsed["duplicate_semantic_overlap"] == pytest.approx(0.8)
    assert [row["id"] for row in parsed["supports"]] == list(SUPPORT_IDS)


def test_stability_audit_prompt_explicitly_lists_every_support_row() -> None:
    messages = stability_audit_messages(
        "Target syndrome",
        [(support_id, ["Diagnosis"]) for support_id in SUPPORT_IDS],
    )
    schema = messages[1]["content"].split("preserve support IDs", 1)[0]
    for support_id in SUPPORT_IDS:
        assert f'"id":"{support_id}"' in schema


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("lab_to_duplicate_overlap", 1.2, r"\[0,1\]"),
        ("duplicate_to_lab_overlap", "0.9", "numeric"),
    ),
)
def test_stability_audit_parser_rejects_invalid_overlap(
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _audit_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=match):
        parse_stability_audit(json.dumps(payload))


def test_summary_requires_both_semantic_stability_gates() -> None:
    records = [
        {
            "duplicate_semantic_overlap": 0.90,
            "duplicate_truth_score_gap": 0.01,
        },
        {
            "duplicate_semantic_overlap": 0.80,
            "duplicate_truth_score_gap": 0.05,
        },
    ]
    usage = {"physical_requests": 10, "reasoning_tokens": 0}
    result = summarize(
        records,
        usage,
        all_supports_size_twelve=True,
        duplicate_prompts_exact=True,
        source_target_leaks=0,
    )
    assert result["gates"]["all_pass"]

    records[1]["duplicate_semantic_overlap"] = 0.79
    result = summarize(
        records,
        usage,
        all_supports_size_twelve=True,
        duplicate_prompts_exact=True,
        source_target_leaks=0,
    )
    assert not result["gates"]["all_pass"]
