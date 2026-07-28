from __future__ import annotations

import pytest

from scripts.swe_interact_mechanics_serving import JUDGE_KEYS
from scripts.swe_interact_mechanics_serving_v2 import (
    judge_messages,
    parse_judgement,
    parse_numeric_requirement_set,
)
from tests.test_swe_interact_mechanics_serving import _task


def test_numeric_requirement_parser_accepts_only_bare_canonical_indexes() -> None:
    assert parse_numeric_requirement_set("1,3,4", 4) == {
        "R1",
        "R3",
        "R4",
    }
    assert parse_numeric_requirement_set("NONE", 4) == set()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("R1", "noncanonical"),
        ("01", "noncanonical"),
        ("1,1", "repeats"),
        ("2,1", "not sorted"),
        ("5", "unknown"),
        ("1,", "noncanonical"),
    ],
)
def test_numeric_requirement_parser_fails_closed(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_numeric_requirement_set(value, 4)


def test_numeric_judgement_is_keyed_order_insensitive() -> None:
    values = {
        "INITIAL": "1",
        "GENERIC": "NONE",
        "ROOT_A_1": "2",
        "ROOT_A_2": "2",
        "ROOT_B_1": "4",
        "ROOT_B_2": "4",
        "REVIEW_A": "2",
        "REVIEW_B": "4",
    }
    text = "\n".join(f"{key}|{values[key]}" for key in reversed(JUDGE_KEYS))
    parsed = parse_judgement(text, 4)
    assert parsed["GENERIC"] == set()
    assert parsed["ROOT_A_1"] == {"R2"}
    assert parsed["ROOT_B_1"] == {"R4"}


def test_v2_judge_prompt_forbids_letter_prefix() -> None:
    task = _task()
    messages = judge_messages(
        task,
        "initial",
        {key: "reply" for key in JUDGE_KEYS if key != "INITIAL"},
    )
    system = messages[0]["content"]
    assert "bare decimal" not in system
    assert "Never prefix an index with a letter." in system
    assert "1|PRIVATE_REQ_ALPHA" in messages[1]["content"]
