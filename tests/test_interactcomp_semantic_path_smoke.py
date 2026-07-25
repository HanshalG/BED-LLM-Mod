import pytest

from scripts.interactcomp_semantic_path_smoke import (
    parse_classification_ascii_whitespace,
)


def test_ascii_whitespace_compaction_accepts_spaced_labels():
    assert parse_classification_ascii_whitespace("Y N U Y\n") == "YNUY"


def test_ascii_whitespace_compaction_rejects_other_characters():
    with pytest.raises(ValueError):
        parse_classification_ascii_whitespace("Y,N,U,Y")
    with pytest.raises(ValueError):
        parse_classification_ascii_whitespace("labels: YNUY")
