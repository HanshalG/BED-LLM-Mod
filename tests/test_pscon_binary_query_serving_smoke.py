from __future__ import annotations

import pytest

from scripts.pscon_binary_query_serving_smoke import parse_labels, parse_question


def test_binary_question_parser_accepts_ascii_and_fullwidth_question_marks() -> None:
    assert parse_question("Would you prefer an OLED display?").endswith("?")
    assert parse_question("Would you prefer an OLED display？").endswith("？")


def test_binary_question_parser_rejects_prose() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        parse_question("Here is a question:\nWould you prefer OLED?")


def test_binary_label_parser_is_exact() -> None:
    assert parse_labels("YNUYNU", 6) == "YNUYNU"
    with pytest.raises(ValueError, match="one Y/N/U"):
        parse_labels("Y N U Y N U", 6)
    with pytest.raises(ValueError, match="one Y/N/U"):
        parse_labels("YYNN", 6)
