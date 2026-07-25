from __future__ import annotations

import math

import pytest

from scripts.clariq_dynamic_support_smoke import (
    Support,
    information_gain,
    parse_support,
    policy_scores,
)


QUESTIONS = [
    {
        "question_id": "Q01",
        "question": "first",
        "response_options": [
            {"code": "A", "answer": "alpha"},
            {"code": "B", "answer": "beta"},
        ],
    },
    {
        "question_id": "Q02",
        "question": "second",
        "response_options": [
            {"code": "A", "answer": "alpha"},
            {"code": "B", "answer": "beta"},
        ],
    },
]


def _response(
    *,
    masses: tuple[int, ...] = (25, 20, 15, 12, 10, 8, 6, 4),
    predictions: tuple[str, ...] = (
        "AA",
        "AA",
        "AB",
        "AB",
        "BA",
        "BA",
        "BB",
        "BB",
    ),
) -> str:
    return "\n".join(
        f"H{index:02d}|{mass}|{prediction}|intent {index}"
        for index, (mass, prediction) in enumerate(
            zip(masses, predictions, strict=True),
            start=1,
        )
    )


def test_parse_support_accepts_unnormalized_zero_masses() -> None:
    support = parse_support(
        _response(masses=(30, 20, 10, 5, 0, 0, 0, 0)),
        questions=QUESTIONS,
    )
    assert sum(support.probabilities) == pytest.approx(1.0)
    assert support.probabilities[-1] == 0.0


def test_parse_support_rejects_all_zero_or_invalid_codes() -> None:
    with pytest.raises(ValueError, match="positive total"):
        parse_support(
            _response(masses=(0, 0, 0, 0, 0, 0, 0, 0)),
            questions=QUESTIONS,
        )
    with pytest.raises(ValueError, match="outside its alphabet"):
        parse_support(
            _response(
                predictions=(
                    "CA",
                    "AA",
                    "AB",
                    "AB",
                    "BA",
                    "BA",
                    "BB",
                    "BB",
                )
            ),
            questions=QUESTIONS,
        )


def test_information_gain_matches_response_partition_entropy() -> None:
    support = Support(
        question_ids=("Q01", "Q02"),
        hypotheses=("h1", "h2", "h3"),
        probabilities=(0.5, 0.25, 0.25),
        predictions=("AA", "AB", "BA"),
    )
    assert information_gain(support, "Q01") == pytest.approx(
        -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    )
    assert information_gain(
        support,
        "Q02",
        condition=("Q01", "A"),
    ) == pytest.approx(
        -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    )


def test_dynamic_scores_use_branch_regenerated_support() -> None:
    task = {
        "roots": [
            {
                "question_id": "Q01",
                "branches": [
                    {
                        "response_code": "A",
                        "legal_followup_question_ids": ["Q02"],
                    },
                    {
                        "response_code": "B",
                        "legal_followup_question_ids": ["Q02"],
                    },
                ],
            },
            {
                "question_id": "Q02",
                "branches": [
                    {
                        "response_code": "A",
                        "legal_followup_question_ids": ["Q01"],
                    },
                    {
                        "response_code": "B",
                        "legal_followup_question_ids": ["Q01"],
                    },
                ],
            },
        ]
    }
    initial = Support(
        question_ids=("Q01", "Q02"),
        hypotheses=("h1", "h2", "h3", "h4"),
        probabilities=(0.4, 0.3, 0.2, 0.1),
        predictions=("AA", "AA", "BA", "BB"),
    )
    deterministic_q02 = Support(
        question_ids=("Q02",),
        hypotheses=("a", "b"),
        probabilities=(0.5, 0.5),
        predictions=("A", "A"),
    )
    diverse_q02 = Support(
        question_ids=("Q02",),
        hypotheses=("a", "b"),
        probabilities=(0.5, 0.5),
        predictions=("A", "B"),
    )
    deterministic_q01 = Support(
        question_ids=("Q01",),
        hypotheses=("a", "b"),
        probabilities=(0.5, 0.5),
        predictions=("A", "A"),
    )
    branch_supports = {
        ("Q01", "A"): diverse_q02,
        ("Q01", "B"): diverse_q02,
        ("Q02", "A"): deterministic_q01,
        ("Q02", "B"): deterministic_q01,
    }
    scores = policy_scores(task, initial, branch_supports)

    assert scores["dynamic_depth_two_question_id"] == "Q01"
    assert (
        scores["dynamic_depth_two_scores"]["Q01"]
        > scores["fixed_depth_two_scores"]["Q01"]
    )
    assert scores["branch_dynamic_gains"]["Q01"]["A"] == pytest.approx(
        math.log(2)
    )
