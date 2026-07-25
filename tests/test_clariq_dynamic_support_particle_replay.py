import pytest

from scripts.clariq_dynamic_support_particle_replay import (
    parse_particle_support,
)


QUESTIONS = [
    {
        "question_id": "Q01",
        "response_options": [
            {"code": "A", "answer": "alpha"},
            {"code": "B", "answer": "beta"},
        ],
    },
    {
        "question_id": "Q02",
        "response_options": [
            {"code": "A", "answer": "alpha"},
            {"code": "B", "answer": "beta"},
        ],
    },
]


def _response(last_prediction: str) -> str:
    lines = [
        "H01|30|A A|same intent",
        "H02|20|A B|second intent",
        "H03|15|B A|third intent",
        "H04|10|B B|fourth intent",
        "H05|8|A A|fifth intent",
        "H06|7|A B|sixth intent",
        "H07|5|B A|duplicate wording",
        f"H08|5|{last_prediction}|duplicate wording",
    ]
    return "\n".join(lines)


def test_particle_parser_allows_same_text_with_different_profiles() -> None:
    support = parse_particle_support(
        _response("B B"),
        questions=QUESTIONS,
    )
    assert support.hypotheses[-2:] == (
        "duplicate wording",
        "duplicate wording",
    )
    assert support.predictions[-2:] == ("BA", "BB")


def test_particle_parser_rejects_exact_joint_duplicate() -> None:
    with pytest.raises(ValueError, match="particles must be distinct"):
        parse_particle_support(
            _response("B A"),
            questions=QUESTIONS,
        )
