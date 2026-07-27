import pytest

from core.strict_rows import parse_keyed_pipe_rows


def test_keyed_rows_accept_arbitrary_complete_order() -> None:
    assert parse_keyed_pipe_rows(
        "R1|2\nR4|3\nR2|1\nR3|4",
        expected_keys=["R1", "R2", "R3", "R4"],
        value_fields=1,
    ) == {
        "R1": ("2",),
        "R4": ("3",),
        "R2": ("1",),
        "R3": ("4",),
    }


@pytest.mark.parametrize(
    "text",
    [
        "R1|2\nR2|1\nR3|4",
        "R1|2\nR2|1\nR3|4\nR3|3",
        "R1|2\nR2|1\nR3|4\nR5|3",
        "R1|2\nR2|1\nR3|4\nR4|",
        "R1|2|extra\nR2|1\nR3|4\nR4|3",
    ],
)
def test_keyed_rows_reject_incomplete_or_ambiguous_responses(
    text: str,
) -> None:
    with pytest.raises(ValueError):
        parse_keyed_pipe_rows(
            text,
            expected_keys=["R1", "R2", "R3", "R4"],
            value_fields=1,
        )
