from __future__ import annotations

import json

import pytest

from scripts.twowiki_directional_chain_audit import (
    analyze_row,
    contains_phrase,
    iter_json_array,
    normalize_text,
    summarize,
    title_aliases,
)


def _row(*, reverse_link: bool = False) -> dict:
    child_sentences = [
        "Samuel Wood died in Hollywood in 1949.",
        "He was an American film director.",
    ]
    if reverse_link:
        child_sentences.append("He directed Kings Row.")
    return {
        "_id": "example",
        "type": "compositional",
        "question": "Where did the director of film Kings Row die?",
        "answer": "Hollywood",
        "supporting_facts": [["Kings Row", 1], ["Sam Wood", 0]],
        "evidences": [
            ["Kings Row", "director", "Sam Wood"],
            ["Sam Wood", "place of death", "Hollywood"],
        ],
        "context": [
            ["Distractor One", ["Unrelated text."]],
            ["Sam Wood", child_sentences],
            [
                "Kings Row",
                [
                    "Kings Row is a 1942 film.",
                    "The picture was directed by Sam Wood.",
                ],
            ],
            ["Distractor Two", ["Unrelated text."]],
            ["Distractor Three", ["Unrelated text."]],
            ["Distractor Four", ["Unrelated text."]],
            ["Distractor Five", ["Unrelated text."]],
            ["Distractor Six", ["Unrelated text."]],
            ["Distractor Seven", ["Unrelated text."]],
            ["Distractor Eight", ["Unrelated text."]],
        ],
    }


def test_normalization_and_alias_matching() -> None:
    assert normalize_text("Ian_Bárry (director)") == "ian barry director"
    assert title_aliases("Ian Barry (director)") == (
        "ian barry director",
        "ian barry",
    )
    assert contains_phrase("The film was by Sam Wood.", "sam wood")
    assert not contains_phrase("The woodsampler.", "sam wood")


def test_iter_json_array_streams_objects(tmp_path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(
        json.dumps([{"_id": "a", "value": "x" * 20}, {"_id": "b"}]),
        encoding="utf-8",
    )
    assert [row["_id"] for row in iter_json_array(path, chunk_size=7)] == [
        "a",
        "b",
    ]


def test_iter_json_array_rejects_non_array(tmp_path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('{"_id": "a"}', encoding="utf-8")
    with pytest.raises(ValueError, match="top-level JSON array"):
        list(iter_json_array(path, chunk_size=4))


def test_analyze_row_detects_strict_directional_chain() -> None:
    result = analyze_row(_row())
    assert result["exact_two_triple_chain"]
    assert result["strict_directional_chain"]
    assert result["support_coverage_gain"] == 1
    assert result["question_names_setup_root"]
    assert result["title_bm25_root_rank"] == 1


def test_analyze_row_rejects_reverse_link() -> None:
    result = analyze_row(_row(reverse_link=True))
    assert result["exact_two_triple_chain"]
    assert not result["strict_directional_chain"]
    assert result["exclusion"] == "reverse_child_to_root_link_exists"


def test_analyze_row_rejects_non_chaining_evidence() -> None:
    row = _row()
    row["evidences"][1][0] = "Another Person"
    result = analyze_row(row)
    assert not result["exact_two_triple_chain"]
    assert result["exclusion"] == "evidence_triples_do_not_chain"


def test_analyze_row_rejects_answer_in_root_support() -> None:
    row = _row()
    row["context"][2][1][1] += " He later visited Hollywood."
    result = analyze_row(row)
    assert result["exact_two_triple_chain"]
    assert result["exclusion"] == "answer_not_unique_to_child_support"


def test_invalid_support_sentence_reference_is_excluded() -> None:
    row = _row()
    row["supporting_facts"][0][1] = 99
    result = analyze_row(row)
    assert result["exact_two_triple_chain"]
    assert result["exclusion"] == "invalid_support_sentence_reference"


def test_summary_enforces_shallow_miss_gates() -> None:
    diagnostic = analyze_row(_row())
    rows = [dict(diagnostic) for _ in range(500)]
    for index, row in enumerate(rows):
        row["question_names_setup_root"] = index >= 50
        row["title_bm25_root_rank"] = 2 if index < 75 else 1
        row["paragraph_bm25_root_rank"] = 2 if index < 75 else 1
    summary = summarize(rows)
    assert summary["gates"]["all_pass"]
    rows[0]["support_coverage_gain"] = 0
    assert not summarize(rows)["gates"][
        "all_strict_chain_coverage_gains_equal_one"
    ]
