import json

import pytest

from scripts.musique_answer_belief_bridge_gate import BELIEF_SIZE
from scripts.paprika_murder_belief_opportunity import (
    EXPECTED_REQUESTS,
    FIRST_ACTION_COUNT,
    FORMAL_INDICES,
    SECOND_ACTION_COUNT,
    SMOKE_INDICES,
    analyze_record,
    culprit_reference,
    parse_investigator_response,
)


def _belief():
    return [
        {"answer": f"suspect {index}", "probability": 1 / BELIEF_SIZE}
        for index in range(BELIEF_SIZE)
    ]


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        (
            "The hidden culprit is Akiko, a janitor, who killed the curator. "
            "Key evidence follows.",
            "Akiko, a janitor",
        ),
        (
            "The murderer is Mrs. Laura Reed, who attacked Alan. "
            "The key clues follow.",
            "Mrs. Laura Reed",
        ),
        (
            "The hidden culprit is a smuggler, Javier Ortiz, who killed Ricardo.",
            "a smuggler, Javier Ortiz",
        ),
    ],
)
def test_culprit_reference_extracts_explicit_label(scenario, expected):
    assert culprit_reference(scenario) == expected


def test_parse_initial_actions_requires_direct_and_enabling_groups():
    payload = {
        "belief": _belief(),
        "direct_actions": [f"direct {index}" for index in range(3)],
        "enabling_actions": [f"enabling {index}" for index in range(3)],
    }
    belief, actions = parse_investigator_response(
        json.dumps(payload), stage="initial"
    )
    assert len(belief) == BELIEF_SIZE
    assert len(actions) == FIRST_ACTION_COUNT


def test_request_counts_cover_environment_and_belief_tree():
    per_case = (
        1
        + FIRST_ACTION_COUNT
        + FIRST_ACTION_COUNT
        + FIRST_ACTION_COUNT * SECOND_ACTION_COUNT
        + FIRST_ACTION_COUNT * SECOND_ACTION_COUNT
        + 1
        + 1
    )
    assert EXPECTED_REQUESTS["serving_smoke"] == len(SMOKE_INDICES) * per_case
    assert EXPECTED_REQUESTS["opportunity"] == len(FORMAL_INDICES) * per_case


def test_analysis_finds_nonmyopic_branch():
    first_branches = []
    for first_index in range(FIRST_ACTION_COUNT):
        seconds = [
            {
                "action": f"second {first_index} {second_index}",
                "response_sha256": f"s{first_index}-{second_index}",
                "truth_probability": 0.2,
            }
            for second_index in range(SECOND_ACTION_COUNT)
        ]
        first_branches.append(
            {
                "action": f"first {first_index}",
                "response_sha256": f"r{first_index}",
                "truth_probability": 0.6 if first_index == 0 else 0.2,
                "second_branches": seconds,
            }
        )
    first_branches[1]["second_branches"][2]["truth_probability"] = 0.95
    record = {
        "first_branches": first_branches,
        "replay_truth_probability": 0.2,
    }
    result = analyze_record(record)
    assert result["greedy_first_index"] == 0
    assert result["oracle_first_index"] == 1
    assert result["oracle_first_differs_from_greedy"]
    assert result["nonmyopic_probability_gap"] == pytest.approx(0.75)
