import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "battleship_llm_native_opportunity_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "battleship_llm_native_opportunity_audit",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_extracts_only_stage_zero_question_programs():
    payload = {
        "games": [
            {
                "captain_llm": "planner",
                "captain_type": "method",
                "board_id": "B01",
                "events": [
                    {
                        "stage": 0,
                        "question": {"text": "Q0?"},
                        "fn_str": "def answer(true_board, partial_board): return True",
                    },
                    {
                        "stage": 1,
                        "question": {"text": "Q1?"},
                        "fn_str": "def answer(true_board, partial_board): return False",
                    },
                    {"stage": 0, "move": {"coords": [0, 0]}},
                ],
            }
        ]
    }

    programs = MODULE.extract_stage_zero_programs(payload)

    assert len(programs) == 1
    assert programs[0]["question"] == "Q0?"
    assert programs[0]["model"] == "planner"


def test_program_evaluation_rejects_non_boolean_values():
    answer = MODULE.compile_program(
        "def answer(true_board, partial_board): return 1"
    )
    boards = np.zeros((2, 2, 2), dtype=int)
    partial = np.full((2, 2), -1, dtype=int)

    try:
        MODULE.evaluate_program(answer, boards, partial)
    except TypeError as exc:
        assert "not bool" in str(exc)
    else:
        raise AssertionError("non-boolean program result was accepted")


def test_joint_behavior_deduplication_uses_every_block():
    programs = [{"question": "a"}, {"question": "b"}, {"question": "c"}]
    outcomes = [
        [
            np.array([0, 1], dtype=np.uint8),
            np.array([1, 0], dtype=np.uint8),
        ],
        [
            np.array([0, 1], dtype=np.uint8),
            np.array([1, 0], dtype=np.uint8),
        ],
        [
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
        ],
    ]

    kept, kept_outcomes = MODULE.dedupe_by_joint_behavior(
        programs, outcomes
    )

    assert [program["question"] for program in kept] == ["a", "c"]
    assert len(kept_outcomes) == 2


def _brute_terminal_value(outcomes, occupancy, epsilon, depth):
    outcomes = np.asarray(outcomes, dtype=np.uint8)
    occupancy = np.asarray(occupancy, dtype=float)
    initial = np.full(outcomes.shape[1], 1.0 / outcomes.shape[1])

    def terminal(weights):
        return float(np.max(weights @ occupancy))

    def recurse(weights, available, remaining):
        if remaining == 0:
            return terminal(weights)
        values = []
        for question in available:
            latent_true = float(outcomes[question] @ weights)
            probability_yes = epsilon + (1 - 2 * epsilon) * latent_true
            value = 0.0
            for answer, probability in (
                (1, probability_yes),
                (0, 1 - probability_yes),
            ):
                likelihood = np.where(
                    outcomes[question] == answer,
                    1 - epsilon,
                    epsilon,
                )
                posterior = weights * likelihood
                posterior /= posterior.sum()
                value += probability * recurse(
                    posterior,
                    tuple(x for x in available if x != question),
                    remaining - 1,
                )
            values.append(value)
        return max(values)

    return recurse(initial, tuple(range(len(outcomes))), depth)


def test_finite_horizon_planner_matches_brute_force_reference():
    outcomes = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    occupancy = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=float,
    )
    planner = MODULE.FiniteHorizonQuestionPlanner(
        outcomes,
        occupancy,
        epsilon=0.1,
    )

    for depth in (1, 2, 3):
        actual = planner.root_evaluation(depth).best_value
        expected = _brute_terminal_value(
            outcomes, occupancy, 0.1, depth
        )
        assert np.isclose(actual, expected)


def test_greedy_eig_root_uses_binary_channel_information():
    outcomes = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    occupancy = np.eye(4)
    planner = MODULE.FiniteHorizonQuestionPlanner(
        outcomes,
        occupancy,
        epsilon=0.1,
    )

    root, eig = planner.greedy_eig_root()

    assert root == 1
    assert eig > 0.5
