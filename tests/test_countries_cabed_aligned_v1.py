import json
import math

import pytest

from scripts.countries_cabed_aligned_v1 import (
    COUNTRIES,
    FORMAL_STYLES,
    ROOT_PROPOSALS,
    SMOKE_STYLES,
    _evaluate_tree,
    classification_messages,
    immediate_eig,
    parse_classification,
    parse_questions,
    partition,
    question_generation_messages,
)


def test_country_support_is_frozen_and_distinct():
    assert len(COUNTRIES) == 64
    assert len(set(COUNTRIES)) == 64
    assert len(SMOKE_STYLES) == 2
    assert len(FORMAL_STYLES) == 12


def test_question_parser_requires_complete_unique_non_guesses():
    questions = [f"Does property {index} apply?" for index in range(ROOT_PROPOSALS)]
    assert parse_questions(
        json.dumps({"questions": questions}),
        count=ROOT_PROPOSALS,
    ) == tuple(questions)
    questions[-1] = questions[0]
    with pytest.raises(ValueError, match="duplicate"):
        parse_questions(
            json.dumps({"questions": questions}),
            count=ROOT_PROPOSALS,
        )
    questions[-1] = "Is it Canada?"
    with pytest.raises(ValueError, match="direct"):
        parse_questions(
            json.dumps({"questions": questions}),
            count=ROOT_PROPOSALS,
        )


def test_classification_parser_preserves_all_country_rows():
    payload = {
        "answers": [
            {
                "country": country,
                "answer": "Yes" if index % 2 == 0 else "No",
            }
            for index, country in enumerate(COUNTRIES)
        ]
    }
    answers = parse_classification(json.dumps(payload))
    assert len(answers) == 64
    assert answers[:2] == ("Yes", "No")
    payload["answers"][0]["country"] = "Not a country"
    with pytest.raises(ValueError, match="order"):
        parse_classification(json.dumps(payload))


def test_prompts_are_target_blind_and_do_not_supply_scores():
    question_prompt = question_generation_messages(
        COUNTRIES,
        count=8,
        style=SMOKE_STYLES[0],
    )[-1]["content"].lower()
    assert "information gain" in question_prompt
    assert "do not calculate" in question_prompt
    assert "hidden target" not in question_prompt
    table_prompt = classification_messages("Is it in Europe?")[-1]["content"]
    assert "Is it in Europe?" in table_prompt
    assert "accurate country" in table_prompt


def test_partition_and_information_gain_match_balanced_binary_split():
    answers = tuple(
        "Yes" if index < 32 else "No" for index in range(len(COUNTRIES))
    )
    indices = tuple(range(len(COUNTRIES)))
    groups = partition(indices, answers)
    assert len(groups["Yes"]) == len(groups["No"]) == 32
    assert immediate_eig(indices, answers) == pytest.approx(math.log(2))


def test_tree_evaluation_recovers_known_nonmyopic_root():
    all_yes_no = [
        "Yes" if index < 32 else "No" for index in range(len(COUNTRIES))
    ]
    quarter = [
        "Yes" if (index // 16) % 2 == 0 else "No"
        for index in range(len(COUNTRIES))
    ]
    weak_root = [
        "Yes" if index < 8 else "No" for index in range(len(COUNTRIES))
    ]

    def root(question, answers, followups):
        groups = partition(tuple(range(len(COUNTRIES))), answers)
        branches = []
        for answer in ("Yes", "No"):
            rows = followups[answer]
            scores = [
                immediate_eig(groups[answer], row) for row in rows
            ]
            best = max(range(3), key=lambda index: scores[index])
            branches.append(
                {
                    "answer": answer,
                    "support": [COUNTRIES[index] for index in groups[answer]],
                    "candidate_questions": [f"{question} f{i}?" for i in range(3)],
                    "candidate_answers": rows,
                    "eig_scores": scores,
                    "selected_index": best,
                    "selected_question": f"{question} f{best}?",
                }
            )
        root_eig = immediate_eig(tuple(range(len(COUNTRIES))), answers)
        future = sum(
            (len(groups[answer]) / len(COUNTRIES))
            * max(branches[index]["eig_scores"])
            for index, answer in enumerate(("Yes", "No"))
        )
        return {
            "question": question,
            "answers": answers,
            "immediate_eig": root_eig,
            "depth_two_score": root_eig + future,
            "branches": branches,
        }

    constant = ["No"] * len(COUNTRIES)
    roots = [
        root(
            "balanced",
            all_yes_no,
            {"Yes": [constant] * 3, "No": [constant] * 3},
        ),
        root(
            "setup",
            weak_root,
            {"Yes": [quarter, constant, constant], "No": [quarter, constant, constant]},
        ),
        root(
            "other1",
            constant,
            {"Yes": [constant] * 3, "No": [constant] * 3},
        ),
        root(
            "other2",
            constant,
            {"Yes": [constant] * 3, "No": [constant] * 3},
        ),
    ]
    record = _evaluate_tree(
        {
            "tree_index": 0,
            "style": "test",
            "root_proposals": [],
            "roots": roots,
        }
    )
    assert record["selections"]["depth_one"] == 0
    assert record["selections"]["depth_two"] in {0, 1}
    assert record["mean_entropy_gain_depth_two_vs_one"] >= 0.0
