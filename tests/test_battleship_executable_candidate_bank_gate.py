import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "battleship_executable_candidate_bank_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "battleship_executable_candidate_bank_gate",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_response_parser_accepts_exact_six_paired_experiments():
    response = json.dumps(
        {
            "candidates": [
                {
                    "question": f"Is cell A{index + 1} occupied by a ship?",
                    "expression": f"bool(true_board[0, {index}] > 0)",
                }
                for index in range(6)
            ]
        }
    )

    candidates = MODULE.parse_response(response)

    assert len(candidates) == 6
    assert candidates[0]["question"].endswith("?")


def test_candidate_evaluation_filters_unsafe_constant_and_extreme():
    boards = np.zeros((20, 8, 8), dtype=int)
    boards[::2, 0, 0] = 1
    boards[:, 0, 1] = 1
    batches = [
        [
            {
                "question": "Is cell A1 occupied by a ship?",
                "expression": "bool(true_board[0, 0] > 0)",
            },
            {
                "question": "Is cell A2 occupied by a ship?",
                "expression": "bool(true_board[0, 1] > 0)",
            },
            {
                "question": "Is this expression always true?",
                "expression": "bool(True)",
            },
            {
                "question": "Can this expression read a file?",
                "expression": "bool(open('/tmp/x'))",
            },
        ]
    ]

    valid, outcomes, per_call = MODULE.evaluate_candidates(
        batches,
        [boards, boards.copy()],
        released_keys=set(),
    )

    assert [item["question"] for item in valid] == [
        "Is cell A1 occupied by a ship?"
    ]
    assert len(outcomes) == 1
    assert per_call == [1]


def test_aggregate_gates_passes_complete_fixture_metrics():
    usage = {
        "adapter_requests": 10,
        "http_attempts": 10,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.2,
    }
    candidates = [
        {
            "novel_vs_released_bank": index < 8,
            "behavior_sha256": f"behavior-{index}",
        }
        for index in range(20)
    ]
    blocks = []
    for block_index in range(2):
        blocks.append(
            {
                "depth": {
                    "1": {"best_behavior_sha256": ["a"]},
                    "2": {"best_behavior_sha256": ["b"]},
                    "3": {"best_behavior_sha256": ["c"]},
                },
                "three_question_receding_hit_probability": {
                    "1": 0.60,
                    "2": 0.62,
                    "3": 0.62,
                },
                "greedy_eig": {"three_question_hit_probability": 0.59},
            }
        )

    gates = MODULE.aggregate_gates(
        usage=usage,
        candidate_batches=[[] for _ in range(10)],
        valid_per_call=[3] * 10,
        unique_candidates=candidates,
        block_results=blocks,
    )

    assert gates["all_pass"]
