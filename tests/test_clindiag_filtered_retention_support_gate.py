from __future__ import annotations

import json

import pytest

from helpers import load_config
from scripts.clindiag_filtered_retention_support_gate import (
    ACTION_ID,
    EXPECTED_REQUESTS,
    FILTER_THRESHOLD,
    SELECTION_SEED,
    SMOKE_IDS,
    SUPPORT_IDS,
    compatibility_messages,
    generated_candidate_messages,
    merge_support,
    parse_compatibility,
    parse_semantic_audit,
    retain_by_threshold,
    semantic_audit_messages,
    summarize,
)
from scripts.clindiag_staged_generator_gate import ClinDiagCase


def _case() -> ClinDiagCase:
    return ClinDiagCase(
        source_id="fresh",
        subset="rare",
        initial_information="A patient has weakness and hypertension.",
        medical_history={},
        physical_examination={},
        diagnostic_test={},
        final_diagnosis="Hidden target syndrome",
    )


def _compatibility_payload(values: list[float]) -> str:
    return json.dumps(
        {
            "diagnoses": [
                {
                    "id": f"d{index:02d}",
                    "likelihoods": {"initial": value, ACTION_ID: value + 0.1},
                    "reason": "clinical compatibility",
                }
                for index, value in enumerate(values)
            ]
        }
    )


def _semantic_payload(
    forward: float = 0.9,
    reverse: float = 0.8,
) -> str:
    return json.dumps(
        {
            "supports": [
                {
                    "id": support_id,
                    "best_match_score": 0.8,
                    "reason": "same diagnosis",
                }
                for support_id in SUPPORT_IDS
            ],
            "final_to_duplicate_overlap": forward,
            "duplicate_to_final_overlap": reverse,
            "overlap_reason": "same diseases or synonyms",
        }
    )


def test_config_and_protocol_are_fail_closed() -> None:
    config = load_config(
        "configs/config_clindiag_filtered_retention_support_gate_openrouter.yaml"
    )
    assert config.openrouter_run_budget_usd == pytest.approx(0.50)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.15)
    assert config.openrouter_concurrency == 24
    assert config.openrouter_max_output_tokens == 4096
    assert SELECTION_SEED == 24299
    assert SMOKE_IDS == ("11388546", "rare203")
    assert ACTION_ID == "lab_1"
    assert FILTER_THRESHOLD == pytest.approx(0.20)
    assert EXPECTED_REQUESTS == 14


def test_generation_prompt_is_deanchored_and_hides_truth() -> None:
    messages = generated_candidate_messages(
        _case(),
        [(ACTION_ID, {"finding": "low potassium"})],
    )
    content = messages[1]["content"]
    assert "No previous differential" in content
    assert "Hidden target syndrome" not in content
    assert ACTION_ID in content


def test_compatibility_prompt_requests_likelihood_not_posterior() -> None:
    messages = compatibility_messages(
        _case(),
        ["Diagnosis A", "Diagnosis B"],
        [("initial", "weakness"), (ACTION_ID, {"finding": "low potassium"})],
    )
    content = messages[1]["content"]
    assert "p(evidence|diagnosis), not the posterior" in content
    assert '"id": "d00"' in content
    assert "Hidden target syndrome" not in content


def test_parse_filter_and_merge_use_minimum_history_likelihood() -> None:
    parsed = parse_compatibility(
        _compatibility_payload([0.1, 0.2, 0.7]),
        num_diagnoses=3,
        evidence_ids=("initial", ACTION_ID),
    )
    retained = retain_by_threshold(["A", "B", "C"], parsed)
    assert retained == ["B", "C"]
    assert merge_support(retained, ["C", "D", "E"], count=4) == [
        "B",
        "C",
        "D",
        "E",
    ]


@pytest.mark.parametrize(
    "payload, match",
    [
        (
            {
                "diagnoses": [
                    {
                        "id": "wrong",
                        "likelihoods": {"initial": 0.5},
                        "reason": "x",
                    }
                ]
            },
            "IDs or order",
        ),
        (
            {
                "diagnoses": [
                    {
                        "id": "d00",
                        "likelihoods": {"wrong": 0.5},
                        "reason": "x",
                    }
                ]
            },
            "evidence IDs",
        ),
    ],
)
def test_compatibility_parser_rejects_schema_drift(
    payload: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_compatibility(
            json.dumps(payload),
            num_diagnoses=1,
            evidence_ids=("initial",),
        )


def test_semantic_audit_schema_lists_all_supports_and_uses_worse_overlap() -> None:
    messages = semantic_audit_messages(
        "Hidden target syndrome",
        [(support_id, ["Diagnosis"]) for support_id in SUPPORT_IDS],
    )
    schema = messages[1]["content"].split("preserve support IDs", 1)[0]
    for support_id in SUPPORT_IDS:
        assert f'"id":"{support_id}"' in schema
    parsed = parse_semantic_audit(_semantic_payload())
    assert parsed["duplicate_semantic_overlap"] == pytest.approx(0.8)


def test_summary_requires_movement_and_replay_stability() -> None:
    support = [f"D{index}" for index in range(12)]
    records = [
        {
            "supports": {
                "filtered_retention": support,
                "filtered_retention_duplicate": support,
            },
            "num_old_pruned": 2,
            "num_new_introduced": 2,
            "num_duplicate_new_introduced": 2,
            "duplicate_semantic_overlap": 0.8,
            "duplicate_truth_score_gap": 0.05,
            "source_target_leak": False,
        }
        for _ in range(2)
    ]
    usage = {"physical_requests": 14, "reasoning_tokens": 0}
    assert summarize(
        records,
        usage,
        candidate_generation_prompts_exact=True,
    )["gates"]["all_pass"]
    records[1]["num_old_pruned"] = 0
    assert not summarize(
        records,
        usage,
        candidate_generation_prompts_exact=True,
    )["gates"]["all_pass"]
