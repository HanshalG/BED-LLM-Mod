import json

import pytest

from scripts.musique_answer_belief_bridge_gate import (
    BELIEF_SIZE,
    EXPECTED_REQUESTS,
    FIRST_ACTION_COUNT,
    FORMAL_IDS,
    SECOND_ACTION_COUNT,
    SMOKE_IDS,
    analyze_record,
    parse_belief_response,
    parse_equivalence,
    truth_probability,
)


def _belief(prefix="answer"):
    return [
        {"answer": f"{prefix} {index}", "probability": 1 / BELIEF_SIZE}
        for index in range(BELIEF_SIZE)
    ]


def test_request_counts_cover_full_two_step_tree_and_replay():
    per_case = (
        1
        + FIRST_ACTION_COUNT
        + FIRST_ACTION_COUNT * SECOND_ACTION_COUNT
        + 1
        + 1
    )
    assert EXPECTED_REQUESTS["serving_smoke"] == len(SMOKE_IDS) * per_case
    assert EXPECTED_REQUESTS["opportunity"] == len(FORMAL_IDS) * per_case


def test_parse_belief_response_normalizes_and_checks_actions():
    payload = {
        "belief": [
            {"answer": f"answer {index}", "probability": 1 / BELIEF_SIZE}
            for index in range(BELIEF_SIZE)
        ],
        "first_documents": [f"d{index + 1:02d}" for index in range(6)],
    }
    belief, actions = parse_belief_response(
        json.dumps(payload),
        available_doc_ids=[f"d{index + 1:02d}" for index in range(20)],
        action_key="first_documents",
        action_count=6,
    )
    assert sum(row["probability"] for row in belief) == pytest.approx(1.0)
    assert actions == [f"d{index + 1:02d}" for index in range(6)]


def test_parse_belief_response_rejects_duplicate_answers():
    payload = {"belief": _belief()}
    payload["belief"][-1]["answer"] = payload["belief"][0]["answer"]
    with pytest.raises(ValueError, match="distinct"):
        parse_belief_response(
            json.dumps(payload),
            available_doc_ids=["d01"],
        )


def test_parse_equivalence_preserves_state_ids_and_indices():
    states = [("initial", _belief()), ("d01", _belief("next"))]
    payload = {
        "matches": [
            {"state_id": "initial", "matching_indices": [2]},
            {"state_id": "d01", "matching_indices": [1, 3]},
        ]
    }
    result = parse_equivalence(json.dumps(payload), states)
    assert result == {"initial": [2], "d01": [1, 3]}
    assert truth_probability(states[1][1], result["d01"]) == pytest.approx(0.25)


def test_analysis_finds_nonmyopic_gold_pair():
    first_docs = [f"d{index + 1:02d}" for index in range(FIRST_ACTION_COUNT)]
    first_branches = []
    for first_doc in first_docs:
        seconds = [
            f"d{index + 7:02d}" for index in range(SECOND_ACTION_COUNT)
        ]
        second_branches = [
            {"second_doc_id": second, "truth_probability": 0.2}
            for second in seconds
        ]
        first_branches.append(
            {
                "first_doc_id": first_doc,
                "truth_probability": 0.6 if first_doc == "d01" else 0.2,
                "second_branches": second_branches,
            }
        )
    first_branches[1]["second_branches"][0]["truth_probability"] = 0.95
    record = {
        "gold_doc_ids": ["d02", "d07"],
        "first_branches": first_branches,
        "replay_pair": "d01>d07",
        "replay_truth_probability": 0.2,
    }
    result = analyze_record(record)
    assert result["greedy_first_doc_id"] == "d01"
    assert result["oracle_pair"] == "d02>d07"
    assert result["oracle_pair_is_gold_pair"]
    assert result["oracle_first_differs_from_greedy"]
    assert result["nonmyopic_probability_gap"] == pytest.approx(0.75)
