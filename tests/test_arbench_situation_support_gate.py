from __future__ import annotations

import hashlib
import json

import pytest

from helpers import load_config
from scripts.arbench_situation_support_gate import (
    AR_BENCH_COMMIT,
    AR_BENCH_TEST_SHA256,
    FORMAL_EXPECTED_REQUESTS,
    FORMAL_POSITIONS,
    NUM_CANDIDATE_QUESTIONS,
    NUM_INITIAL_EXPLANATIONS,
    NUM_REFRESHED_EXPLANATIONS,
    SELECTION_SEED,
    SEMANTIC_COVERAGE_THRESHOLD,
    SMOKE_EXPECTED_REQUESTS,
    answer_messages,
    candidate_question_messages,
    initial_explanation_messages,
    load_arbench_situation_puzzles,
    parse_answer,
    parse_coverage_response,
    parse_text_list,
    refreshed_explanation_messages,
    semantic_coverage_messages,
    summarize,
)


def test_frozen_config_and_protocol() -> None:
    config = load_config(
        "configs/config_arbench_situation_support_gate_openrouter.yaml"
    )
    assert config.openrouter_budget_usd == pytest.approx(70.38480269545715)
    assert config.openrouter_run_budget_usd == pytest.approx(0.50)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.25)
    assert config.openrouter_concurrency == 64
    assert config.openrouter_max_output_tokens == 2048
    assert AR_BENCH_COMMIT == "9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1"
    assert SELECTION_SEED == 24301
    assert FORMAL_POSITIONS == (10, 12, 19, 20, 28, 34, 47, 60, 62, 67, 91, 96)
    assert FORMAL_EXPECTED_REQUESTS == 132
    assert SMOKE_EXPECTED_REQUESTS == 10


def test_pinned_arbench_test_file_loads() -> None:
    path = "external/AR-Bench/data/sp/test.json"
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == AR_BENCH_TEST_SHA256
    rows = load_arbench_situation_puzzles(path)
    assert len(rows) == 100
    assert rows[FORMAL_POSITIONS[0]]["index"] == 11


def test_generation_prompts_hide_private_story_and_key_questions() -> None:
    surface = "A public puzzle."
    hidden = "The private causal story."
    initial = initial_explanation_messages(surface)
    questions = candidate_question_messages(surface, ["Theory A", "Theory B"])
    refreshed = refreshed_explanation_messages(
        surface,
        ["Theory A", "Theory B"],
        "Was a device involved?",
        "Yes",
    )
    generation_text = json.dumps([initial, questions, refreshed])
    assert hidden not in generation_text
    assert "key_question" not in generation_text
    assert surface in generation_text
    assert hidden in json.dumps(answer_messages(surface, hidden, "Question?"))
    assert hidden in json.dumps(
        semantic_coverage_messages(
            surface,
            hidden,
            [("initial", ["Theory A"])],
        )
    )


def test_text_and_answer_parsers_fail_closed() -> None:
    values = [f"Theory {index}" for index in range(NUM_INITIAL_EXPLANATIONS)]
    assert parse_text_list(
        json.dumps({"explanations": values}),
        "explanations",
        NUM_INITIAL_EXPLANATIONS,
    ) == values
    with pytest.raises(ValueError, match="exactly"):
        parse_text_list(
            json.dumps({"explanations": values[:-1]}),
            "explanations",
            NUM_INITIAL_EXPLANATIONS,
        )
    assert parse_answer('{"answer":"Unknown"}') == "Unknown"
    with pytest.raises(ValueError, match="exactly"):
        parse_answer('{"answer":"Maybe"}')


def test_coverage_parser_preserves_ids_and_threshold() -> None:
    response = json.dumps(
        {
            "supports": [
                {
                    "id": "initial",
                    "best_match_score": SEMANTIC_COVERAGE_THRESHOLD,
                    "best_explanation_index": 1,
                    "reason": "same causal mechanism",
                },
                {
                    "id": "candidate_0",
                    "best_match_score": 0.3,
                    "best_explanation_index": 0,
                    "reason": "different mechanism",
                },
            ]
        }
    )
    parsed = parse_coverage_response(
        response,
        ["initial", "candidate_0"],
        [NUM_INITIAL_EXPLANATIONS, NUM_REFRESHED_EXPLANATIONS],
    )
    assert parsed[0]["covered"] is True
    assert parsed[1]["covered"] is False
    with pytest.raises(ValueError, match="IDs or order"):
        parse_coverage_response(
            response,
            ["candidate_0", "initial"],
            [NUM_REFRESHED_EXPLANATIONS, NUM_INITIAL_EXPLANATIONS],
        )


def _record(
    *,
    initial_score: float,
    branch_scores: list[float],
) -> dict[str, object]:
    return {
        "initial": {
            "best_match_score": initial_score,
            "covered": initial_score >= SEMANTIC_COVERAGE_THRESHOLD,
        },
        "branches": [
            {
                "answer": "Yes",
                "coverage": {
                    "best_match_score": score,
                    "covered": score >= SEMANTIC_COVERAGE_THRESHOLD,
                },
            }
            for score in branch_scores
        ],
    }


def test_formal_summary_applies_frozen_mechanism_gate() -> None:
    records = [
        _record(initial_score=0.2, branch_scores=[0.2, 0.4, 0.85, 0.3])
        for _ in range(6)
    ] + [
        _record(initial_score=0.9, branch_scores=[0.9, 0.9, 0.9, 0.9])
        for _ in range(6)
    ]
    result = summarize(
        records,
        {"physical_requests": FORMAL_EXPECTED_REQUESTS, "reasoning_tokens": 0},
        stage="formal",
    )
    assert result["initial_omitted"] == 6
    assert result["initially_omitted_recovered_by_any_branch"] == 6
    assert result["gates"]["all_pass"] is True


def test_smoke_summary_requires_exact_ten_requests() -> None:
    records = [
        _record(initial_score=0.2, branch_scores=[0.8]),
        _record(initial_score=0.9, branch_scores=[0.9]),
    ]
    result = summarize(
        records,
        {"physical_requests": SMOKE_EXPECTED_REQUESTS, "reasoning_tokens": 0},
        stage="serving_smoke",
    )
    assert result["gates"]["all_pass"] is True
    result = summarize(
        records,
        {"physical_requests": SMOKE_EXPECTED_REQUESTS + 1, "reasoning_tokens": 0},
        stage="serving_smoke",
    )
    assert result["gates"]["all_pass"] is False
