from __future__ import annotations

import json

import pytest

from scripts.icraft_staged_diagnosis_unlock import (
    DEVELOPMENT_IDS,
    HOLDOUT_IDS,
    PRIOR_MODEL_CALL_IDS,
    initial_differential_messages,
    parse_diagnoses,
    parse_semantic_diagnosis,
    semantic_diagnosis_messages,
    summarize,
    workup_differential_messages,
)


def test_staged_icraft_splits_are_fixed_disjoint_and_unseen() -> None:
    assert len(DEVELOPMENT_IDS) == 20
    assert len(HOLDOUT_IDS) == 60
    assert not set(DEVELOPMENT_IDS).intersection(HOLDOUT_IDS)
    assert not set(DEVELOPMENT_IDS).intersection(PRIOR_MODEL_CALL_IDS)
    assert not set(HOLDOUT_IDS).intersection(PRIOR_MODEL_CALL_IDS)


def test_diagnosis_parser_normalizes_strings_and_named_objects() -> None:
    payload = {
        "diagnoses": [
            "Diagnosis A",
            {"diagnosis": "Diagnosis B"},
            {"name": "Diagnosis C"},
            {"label": "Diagnosis D"},
            "Diagnosis E",
            "Diagnosis F",
            "Diagnosis G",
            "Diagnosis H",
            "Extra diagnosis",
        ]
    }
    assert parse_diagnoses(json.dumps(payload)) == [
        "Diagnosis A",
        "Diagnosis B",
        "Diagnosis C",
        "Diagnosis D",
        "Diagnosis E",
        "Diagnosis F",
        "Diagnosis G",
        "Diagnosis H",
    ]
    with pytest.raises(ValueError, match="requires at least 8"):
        parse_diagnoses(json.dumps({"diagnoses": ["A"]}))


def test_generation_prompts_hide_options_and_true_diagnosis() -> None:
    initial = initial_differential_messages(["fact one", "fact two"])
    workup = workup_differential_messages(
        ["fact one", "fact two"],
        ["new fact"],
        ["Diagnosis A"],
    )
    text = json.dumps([initial, workup])
    assert "true_diagnosis" not in text
    assert "answer options" in text
    assert "Diagnosis A" in text


def test_semantic_measurement_prompt_is_explicitly_post_generation() -> None:
    messages = semantic_diagnosis_messages(
        "True diagnosis",
        (("initial", ["Other diagnosis"]),),
    )
    text = json.dumps(messages)
    assert "true_diagnosis" in text
    assert "hidden from every differential generator" in text


def test_semantic_parser_preserves_ids_and_threshold() -> None:
    response = json.dumps(
        {
            "supports": [
                {"id": "initial", "best_match_score": 0.2, "reason": "different"},
                {
                    "id": "workup_generated",
                    "best_match_score": 0.8,
                    "reason": "synonym",
                },
            ]
        }
    )
    parsed = parse_semantic_diagnosis(
        response, ("initial", "workup_generated")
    )
    assert parsed[0]["covered"] is False
    assert parsed[1]["covered"] is True
    with pytest.raises(ValueError, match="IDs or order"):
        parse_semantic_diagnosis(
            response, ("workup_generated", "initial")
        )


def test_staged_summary_applies_frozen_gate() -> None:
    records = []
    for index in range(20):
        initial_score = 0.9 if index < 8 else 0.1
        workup_score = 0.9 if index < 18 else 0.2
        records.append(
            {
                "initial_measurement": {
                    "covered": initial_score >= 0.8,
                    "best_match_score": initial_score,
                },
                "workup_measurement": {
                    "covered": workup_score >= 0.8,
                    "best_match_score": workup_score,
                },
            }
        )
    summary = summarize(records)
    assert summary["initial_covered"] == 8
    assert summary["initially_omitted_recovered_by_workup"] == 10
    assert summary["gates"]["all_pass"] is True
