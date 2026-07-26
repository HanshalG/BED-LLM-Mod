from __future__ import annotations

import json
import math

import pytest

from helpers import load_config
from scripts.voi_medical_future_tree_mechanics import (
    DIAGNOSES,
    EXPECTED_REQUESTS,
    FOLLOWUPS_PER_BRANCH,
    MAX_COST_USD,
    MODEL_ID,
    OUTCOMES,
    ROOT_COUNT,
    branch_beliefs,
    count_path_dependent_roots,
    entropy,
    followup_question_messages,
    parse_answer_matrix,
    parse_questions,
    score_tree,
)


def test_question_parser_requires_exact_distinct_lines() -> None:
    assert parse_questions("Q1|Do you cough?\nQ2|Do you have a fever?", 2) == [
        "Do you cough?",
        "Do you have a fever?",
    ]
    with pytest.raises(ValueError, match="distinct"):
        parse_questions("Q1|Do you cough?\nQ2|  do you COUGH?  ", 2)
    with pytest.raises(ValueError, match="prefix"):
        parse_questions("1|Do you cough?\nQ2|Do you have a fever?", 2)


def test_answer_matrix_parser_requires_every_cell() -> None:
    questions = ["Question one?", "Question two?"]
    lines = [
        f"Q{qindex}|{diagnosis}|{OUTCOMES[(qindex + dindex) % len(OUTCOMES)]}"
        for qindex in (1, 2)
        for dindex, diagnosis in enumerate(DIAGNOSES)
    ]
    parsed = parse_answer_matrix("\n".join(lines), questions)
    assert set(parsed) == set(questions)
    assert all(set(values) == set(DIAGNOSES) for values in parsed.values())
    with pytest.raises(ValueError, match="exactly"):
        parse_answer_matrix("\n".join(lines[:-1]), questions)


def test_branch_beliefs_and_entropy_match_manual_partition() -> None:
    belief = {"a": 0.5, "b": 0.25, "c": 0.25}
    partitions = branch_beliefs(
        belief,
        {"a": "Yes", "b": "No", "c": "No"},
    )
    assert partitions["Yes"] == (0.5, {"a": 1.0})
    assert partitions["No"] == (0.5, {"b": 0.5, "c": 0.5})
    assert partitions["Maybe"] == (0.0, {})
    assert entropy(belief) == pytest.approx(
        -0.5 * math.log(0.5) - 2 * 0.25 * math.log(0.25)
    )


def test_path_dependence_ignores_impossible_outcomes() -> None:
    prior = {"a": 0.5, "b": 0.5}
    roots = ["Root?"]
    root_maps = {roots[0]: {"a": "Yes", "b": "No"}}
    followups = {
        (0, "Yes"): ["Shared one?", "Shared two?"],
        (0, "No"): ["Shared one?", "Shared two?"],
        (0, "Maybe"): ["Unused one?", "Unused two?"],
    }
    assert count_path_dependent_roots(prior, roots, root_maps, followups) == 0
    followups[(0, "No")] = ["Different one?", "Different two?"]
    assert count_path_dependent_roots(prior, roots, root_maps, followups) == 1


def test_tree_scoring_prefers_branch_adaptive_root() -> None:
    prior = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    roots = ["Root one?", "Root two?"]
    root_maps = {
        roots[0]: {"a": "Yes", "b": "Yes", "c": "No", "d": "No"},
        roots[1]: {"a": "Yes", "b": "No", "c": "No", "d": "No"},
    }
    followups = {
        (root, outcome): [f"R{root}{outcome}A?", f"R{root}{outcome}B?"]
        for root in range(2)
        for outcome in OUTCOMES
    }
    maps = {}
    for questions in followups.values():
        for question in questions:
            maps[question] = {"a": "Maybe", "b": "Maybe", "c": "Maybe", "d": "Maybe"}
    maps[followups[(1, "No")][0]] = {
        "a": "Maybe",
        "b": "Yes",
        "c": "No",
        "d": "Maybe",
    }
    scored = score_tree(prior, roots, root_maps, followups, maps)
    assert scored["myopic_root_index"] == 0
    assert scored["full_root_index"] == 1
    assert scored["full_minus_myopic_depth_two_eig_nats"] > 0.0


def test_followup_prompt_contains_complete_hypothetical_history() -> None:
    messages = followup_question_messages(
        root_question="Do you have abdominal pain?",
        root_outcome="No",
        posterior={diagnosis: 1 / len(DIAGNOSES) for diagnosis in DIAGNOSES},
    )
    text = json.dumps(messages)
    assert "Do you have abdominal pain?" in text
    assert "Hypothetical answer: No" in text
    assert DIAGNOSES[0] in text


def test_frozen_config_and_request_shape() -> None:
    config = load_config("configs/config_voi_medical_future_tree_openrouter.yaml")
    assert config.model_pairs[0].questioner.model == MODEL_ID
    assert config.openrouter_run_budget_usd == pytest.approx(MAX_COST_USD)
    assert config.openrouter_max_retries == 0
    assert ROOT_COUNT == 4
    assert FOLLOWUPS_PER_BRANCH == 2
    assert EXPECTED_REQUESTS == 18
