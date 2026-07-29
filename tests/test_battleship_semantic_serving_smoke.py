import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "battleship_semantic_serving_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "battleship_semantic_serving_smoke",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_planner_parser_requires_direct_unique_yes_no_questions():
    response = json.dumps(
        {
            "questions": [
                "Is there a ship in row A?",
                "Does any ship touch column 8?",
                "Are two ships horizontal?",
                "Is the center occupied by a ship?",
            ]
        }
    )
    assert len(MODULE.parse_planner_response(response)) == 4

    duplicate = json.dumps(
        {
            "questions": [
                "Is there a ship in row A?",
                "Is there a ship in row A?",
                "Are two ships horizontal?",
                "Is the center occupied by a ship?",
            ]
        }
    )
    with pytest.raises(ValueError, match="not unique"):
        MODULE.parse_planner_response(duplicate)


def test_safe_expression_rejects_imports_and_unsafe_calls():
    with pytest.raises(ValueError):
        MODULE.compile_safe_expression("__import__('os').system('id')")
    with pytest.raises(ValueError):
        MODULE.compile_safe_expression("open('/tmp/x').read()")
    with pytest.raises(ValueError):
        MODULE.compile_safe_expression(
            "(lambda x: x)(bool(np.any(true_board > 0)))"
        )


def test_safe_expression_executes_boolean_numpy_predicate():
    compiled = MODULE.compile_safe_expression(
        "bool(np.any(true_board[0:4, 0:4] > 0))"
    )
    boards = np.zeros((2, 8, 8), dtype=int)
    boards[1, 2, 2] = 1
    partial = np.full((8, 8), -1, dtype=int)

    values = MODULE.evaluate_expression(compiled, boards, partial)

    assert values.tolist() == [0, 1]


def test_safe_expression_allows_bounded_comprehension():
    compiled = MODULE.compile_safe_expression(
        "bool(any(true_board[row, 0] > 0 for row in range(8)))"
    )
    boards = np.zeros((2, 8, 8), dtype=int)
    boards[1, 7, 0] = 2
    partial = np.full((8, 8), -1, dtype=int)

    values = MODULE.evaluate_expression(compiled, boards, partial)

    assert values.tolist() == [0, 1]
    with pytest.raises(ValueError, match="range bounds"):
        MODULE.compile_safe_expression(
            "bool(any(true_board[row, 0] > 0 for row in range(1000000)))"
        )


def test_safe_expression_rejects_non_boolean_result():
    compiled = MODULE.compile_safe_expression("np.sum(true_board > 0)")
    boards = np.zeros((1, 8, 8), dtype=int)
    partial = np.full((8, 8), -1, dtype=int)

    with pytest.raises(TypeError, match="not bool"):
        MODULE.evaluate_expression(compiled, boards, partial)


def test_aggregate_gates_requires_three_agreeing_distinct_questions():
    metrics = [
        {
            "safe_boolean_translations": 2,
            "all_translations_nontrivial_on_both_blocks": True,
            "translator_agreement": agreement,
            "reference_behavior_sha256": f"hash-{index}",
        }
        for index, agreement in enumerate((1.0, 0.99, 0.98, 0.8))
    ]
    usage = {
        "adapter_requests": 10,
        "http_attempts": 10,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.1,
    }

    gates = MODULE.aggregate_gates(
        question_metrics=metrics,
        usage=usage,
        selected_question_count=4,
    )

    assert gates["all_pass"]


def test_exact_ten_call_fixture_passes(tmp_path, monkeypatch):
    boards = np.zeros((16, 8, 8), dtype=int)
    for index in range(len(boards)):
        if index % 2 == 0:
            boards[index, 3, 3] = 1
            boards[index, 6, 0:5] = 4
        else:
            boards[index, 0:5, 6] = 4
        if index % 3 == 0:
            boards[index, 4, 4] = 1
        if index % 4 == 0:
            boards[index, 0, 0:3] = 2
    monkeypatch.setattr(
        MODULE,
        "sample_official_prior",
        lambda *args, **kwargs: [boards, boards.copy()],
    )
    monkeypatch.setattr(MODULE, "git_commit", lambda path: MODULE.EXPECTED_COMMIT)
    monkeypatch.setattr(
        MODULE,
        "sha256_file",
        lambda path: MODULE.EXPECTED_TRAJECTORY_SHA256,
    )
    planner = MODULE.DeterministicFixtureModel("planner")
    translators = (
        MODULE.DeterministicFixtureModel("translator", 0),
        MODULE.DeterministicFixtureModel("translator", 1),
    )

    result = MODULE.run_smoke(
        external_root=tmp_path,
        planner_model=planner,
        translator_models=translators,
        raw_path=tmp_path / "private.json",
    )

    assert result["status"] == "passed"
    assert result["usage"]["adapter_requests"] == 10
    assert result["usage"]["http_attempts"] == 10
    assert result["metrics"]["safe_boolean_translation_count"] == 8
    assert result["gates"]["all_pass"]
