from __future__ import annotations

import pytest

from scripts.swe_interact_mechanics_failure_diagnostic import (
    parse_diagnostic_ids,
    parse_diagnostic_judgement,
)
from scripts.swe_interact_mechanics_serving import JUDGE_KEYS


def test_diagnostic_accepts_only_optional_missing_prefix() -> None:
    valid = ("R1", "R2", "R3")
    assert parse_diagnostic_ids("1,R2,3", valid) == {
        "R1",
        "R2",
        "R3",
    }
    assert parse_diagnostic_ids("NONE", valid) == set()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("R0", "malformed"),
        ("R4", "unknown"),
        ("1,01", "malformed"),
        ("R2,2", "repeats"),
        ("R1;R2", "malformed"),
    ],
)
def test_diagnostic_still_fails_closed_on_ambiguity(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_diagnostic_ids(value, ("R1", "R2", "R3"))


def test_diagnostic_judgement_is_keyed_and_complete() -> None:
    text = "\n".join(
        f"{key}|{'NONE' if key == 'GENERIC' else '1'}"
        for key in reversed(JUDGE_KEYS)
    )
    result = parse_diagnostic_judgement(text, ("R1", "R2"))
    assert set(result) == set(JUDGE_KEYS)
    assert result["GENERIC"] == set()
    assert result["ROOT_A_1"] == {"R1"}
