from __future__ import annotations

import pytest

from scripts.hotpot_directional_unlock_audit import (
    analyze_row,
    contains_phrase,
    mentioned_context_titles,
    normalize_text,
    title_aliases,
)


def _row(*, reverse_link: bool = False) -> dict:
    titles = [
        "Bridge Person",
        "Answer Place (city)",
        "Distractor One",
        "Distractor Two",
        "Distractor Three",
        "Distractor Four",
        "Distractor Five",
        "Distractor Six",
        "Distractor Seven",
        "Distractor Eight",
    ]
    answer_sentences = [
        "Answer Place is a city in the country.",
        "Its official founding year was 1901.",
    ]
    if reverse_link:
        answer_sentences.append("Bridge Person visited the city.")
    return {
        "id": "example",
        "question": "In what year was the city connected to the person founded?",
        "answer": "1901",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {
            "title": ["Bridge Person", "Answer Place (city)"],
            "sent_id": [0, 1],
        },
        "context": {
            "title": titles,
            "sentences": [
                ["Bridge Person was born near Answer Place."],
                answer_sentences,
                ["Unrelated text."],
                ["More unrelated text."],
                ["Another paragraph."],
                ["Nothing relevant."],
                ["Background material."],
                ["An unrelated biography."],
                ["A separate location."],
                ["A final distractor."],
            ],
        },
    }


def test_normalize_text_is_ascii_token_based() -> None:
    assert normalize_text("Die Rhöner_Säuwäntzt!") == "die rhoner sauwantzt"
    assert contains_phrase("the answer place is here", "answer place")
    assert not contains_phrase("the marketplace is here", "place")


def test_title_aliases_strip_final_disambiguator() -> None:
    assert title_aliases("Answer Place (city)") == (
        "answer place city",
        "answer place",
    )


def test_mentions_are_candidate_bounded_and_exclude_root() -> None:
    titles = ["Bridge Person", "Answer Place (city)", "Other"]
    assert mentioned_context_titles(
        paragraph_text="Bridge Person points to Answer Place.",
        context_titles=titles,
        root_title="Bridge Person",
    ) == ["Answer Place (city)"]


def test_analyze_row_detects_strict_directional_unlock() -> None:
    result = analyze_row(_row())
    assert result["strict_unlock"]
    assert result["support_coverage_gain"] == 1
    assert result["level"] == "hard"
    assert result["neither_support_title_in_question"]


def test_analyze_row_rejects_reverse_support_link() -> None:
    result = analyze_row(_row(reverse_link=True))
    assert not result["strict_unlock"]
    assert result["exclusion"] == "reverse_support_link_exists"


def test_analyze_row_rejects_ambiguous_answer_support() -> None:
    row = _row()
    row["context"]["sentences"][0][0] += " It was notable in 1901."
    result = analyze_row(row)
    assert not result["strict_unlock"]
    assert result["exclusion"] == "answer_support_not_unique"


def test_non_bridge_is_rejected() -> None:
    row = _row()
    row["type"] = "comparison"
    result = analyze_row(row)
    assert result["exclusion"] == "not_bridge"


def test_invalid_support_sentence_reference_raises() -> None:
    row = _row()
    row["supporting_facts"]["sent_id"][0] = 99
    with pytest.raises(ValueError, match="invalid sentence"):
        analyze_row(row)
