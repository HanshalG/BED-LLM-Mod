from __future__ import annotations

import json

import pytest

from scripts.icae_bench_semantic_serving_smoke import (
    NUM_HYPOTHESES,
    NUM_QUESTIONS,
    answer_term_count,
    changed_question_count,
    parse_oracle,
    parse_support,
    select_mechanics_tasks,
)


def _support(prefix: str = "initial") -> dict:
    return {
        "hypotheses": [
            {"requirement": f"{prefix} requirement number {index}", "weight": 5}
            for index in range(NUM_HYPOTHESES)
        ],
        "questions": [
            {
                "question": f"What is the {prefix} behavior number {index}?",
                "rationale": f"This distinguishes {prefix} case number {index}.",
            }
            for index in range(NUM_QUESTIONS)
        ],
    }


def test_parse_support_requires_unique_exact_cardinality() -> None:
    parsed = parse_support(json.dumps(_support()), label="support")
    assert len(parsed["hypotheses"]) == NUM_HYPOTHESES
    assert len(parsed["questions"]) == NUM_QUESTIONS

    duplicate = _support()
    duplicate["questions"][1] = dict(duplicate["questions"][0])
    with pytest.raises(ValueError, match="duplicate questions"):
        parse_support(json.dumps(duplicate), label="support")


def test_parse_oracle_requires_strict_internal_log() -> None:
    value = {
        "_internal_log": {
            "triggers_hit": ["C001"],
            "api_alignment_triggered": False,
            "fallback_triggered": False,
            "cheating_attempt_detected": False,
            "score_adjustment": 0,
        },
        "reply": "The output must preserve insertion order.",
    }
    assert parse_oracle(json.dumps(value), label="oracle") == value
    value["_internal_log"]["fallback_triggered"] = "false"
    with pytest.raises(ValueError, match="fallback_triggered"):
        parse_oracle(json.dumps(value), label="oracle")


def test_selected_tasks_are_hash_deterministic() -> None:
    manifest = {
        "partitions": {
            "mechanics": [
                {"alias": f"realcode@{index:03d}", "language": "Python"}
                for index in range(1, 13)
            ]
        }
    }
    assert select_mechanics_tasks(manifest) == select_mechanics_tasks(manifest)
    assert len(select_mechanics_tasks(manifest)) == 2


def test_path_sensitivity_metrics_require_new_questions_and_answer_terms() -> None:
    initial = _support("initial")
    followup = _support("refreshed")
    followup["questions"][0]["question"] = (
        "How should multivalueheaders preserve duplicate values?"
    )
    followup["questions"][0]["rationale"] = (
        "The answer introduced multivalueheaders as a distinct contract."
    )
    assert changed_question_count(initial, followup) == NUM_QUESTIONS
    assert answer_term_count(
        fuzzy_prd="Build an event adapter.",
        initial=initial,
        answer="Use multivalueheaders for duplicate response values.",
        followup=followup,
    ) >= 1
