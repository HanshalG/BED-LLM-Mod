from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "discoverllm_source_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "discoverllm_source_audit",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _state(awareness: float):
    return [
        {
            "criterion": "example",
            "hierarchy": [
                {
                    "id": "1",
                    "aware": awareness,
                    "children": [
                        {
                            "id": "1.1",
                            "aware": awareness,
                            "children": [],
                        }
                    ],
                }
            ],
        }
    ]


def _row(
    *,
    turn: int,
    assistant_index: int,
    completion: str,
    score: float,
    history,
    prompt,
):
    return {
        "artifact_id": "artifact_1",
        "turn_id": turn,
        "assistant_index": assistant_index,
        "completion": completion,
        "score": score,
        "criteria_history": history,
        "prompt": prompt,
    }


def test_score_hierarchy_matches_parent_completion_rule():
    score, current, count = MODULE.score_hierarchy(
        [
            {
                "aware": 0,
                "children": [
                    {"aware": 1, "children": []},
                    {"aware": 1, "children": []},
                ],
            }
        ]
    )

    assert score == 3
    assert current == 1
    assert count == 3


def test_infer_committed_transition_from_next_prompt():
    initial = _state(0)
    updated = _state(1)
    rows = [
        _row(
            turn=1,
            assistant_index=0,
            completion="ignored",
            score=0.5,
            history=[initial],
            prompt=[{"role": "user", "content": "start"}],
        ),
        _row(
            turn=1,
            assistant_index=1,
            completion="chosen",
            score=1.75,
            history=[initial],
            prompt=[{"role": "user", "content": "start"}],
        ),
        _row(
            turn=2,
            assistant_index=0,
            completion="later a",
            score=0.1,
            history=[initial, updated],
            prompt=[
                {"role": "user", "content": "start"},
                {"role": "assistant", "content": "chosen"},
                {"role": "user", "content": "continue"},
            ],
        ),
        _row(
            turn=2,
            assistant_index=1,
            completion="later b",
            score=0.2,
            history=[initial, updated],
            prompt=[
                {"role": "user", "content": "start"},
                {"role": "assistant", "content": "chosen"},
                {"role": "user", "content": "continue"},
            ],
        ),
    ]

    transitions, diagnostics = MODULE.infer_committed_transitions(rows)

    assert diagnostics["recoverable_committed_transitions"] == 1
    assert diagnostics["committed_completion_match_failures"] == 0
    assert transitions[0]["assistant_index"] == 1
    assert transitions[0]["selected_is_max_score"]
    assert transitions[0]["immediate_awareness_gain"] == 2
    assert transitions[0]["implied_token_penalty"] == 0.25


def test_summarize_rows_identifies_one_step_reward_identity():
    initial = _state(0)
    updated = _state(1)
    prompt = [{"role": "user", "content": "start"}]
    next_prompt = [
        *prompt,
        {"role": "assistant", "content": "chosen"},
        {"role": "user", "content": "continue"},
    ]
    rows = [
        _row(
            turn=1,
            assistant_index=0,
            completion="ignored",
            score=0.4,
            history=[initial],
            prompt=prompt,
        ),
        _row(
            turn=1,
            assistant_index=1,
            completion="chosen",
            score=1.8,
            history=[initial],
            prompt=prompt,
        ),
        _row(
            turn=2,
            assistant_index=0,
            completion="later a",
            score=0.1,
            history=[initial, updated],
            prompt=next_prompt,
        ),
        _row(
            turn=2,
            assistant_index=1,
            completion="later b",
            score=0.2,
            history=[initial, updated],
            prompt=next_prompt,
        ),
    ]

    summary = MODULE.summarize_rows(rows)

    assert summary["candidate_count_histogram"] == {2: 2}
    assert summary["selected_is_max_score"] == 1
    assert (
        summary["implied_token_penalty"][
            "within_documented_zero_to_one_range"
        ]
        == 1
    )
    assert summary["initial_tree_structure"]["maximum_depth"]["median"] == 2


def test_pearson_handles_degenerate_vectors():
    assert MODULE.pearson([1.0], [1.0]) is None
    assert MODULE.pearson([1.0, 1.0], [2.0, 3.0]) is None
    assert MODULE.pearson([1.0, 2.0], [3.0, 5.0]) == 1.0
