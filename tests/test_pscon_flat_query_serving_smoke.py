from __future__ import annotations

import json

import pytest

from scripts.pscon_flat_query_serving_smoke import parse_flat_query


def test_parse_flat_query_accepts_exact_three_line_grammar() -> None:
    parsed = parse_flat_query(
        "QUESTION: Which screen size do you prefer?\n"
        "OPTIONS: Under 60 inches || 60 to 80 inches || Over 80 inches\n"
        "ASSIGNMENTS: AABBCC",
        6,
    )
    assert parsed["question"] == "Which screen size do you prefer?"
    assert parsed["options"][2] == "Over 80 inches"
    assert parsed["assignments"] == "AABBCC"
    assert parsed["eig"] == pytest.approx(1.0986122886681098)


@pytest.mark.parametrize(
    "text,error",
    [
        (
            "QUESTION: Which size?\n"
            "OPTIONS: A || B || C\n"
            "ASSIGNMENTS: 112233",
            "A/B/C",
        ),
        (
            "QUESTION: Which size?\n"
            "OPTIONS: A || B || C\n"
            "ASSIGNMENTS: AAAAAA",
            "every assignment",
        ),
        (
            "QUESTION: Which size?\n"
            "OPTIONS: A || B\n"
            "ASSIGNMENTS: AABBCC",
            "three distinct",
        ),
    ],
)
def test_parse_flat_query_rejects_invalid_grammar(text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_flat_query(text, 6)


def test_flat_query_serializes_without_nonfinite_values() -> None:
    parsed = parse_flat_query(
        "QUESTION: Which feature matters most？\n"
        "OPTIONS: Picture || Sound || Design\n"
        "ASSIGNMENTS: ABCABC",
        6,
    )
    assert json.loads(json.dumps(parsed, allow_nan=False))["assignments"] == "ABCABC"
