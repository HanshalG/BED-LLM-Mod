from __future__ import annotations

from scripts.musique_branching_unlock_audit import (
    EXPECTED_DEPENDENCIES,
    analyze_row,
    contains_token_phrase,
    dependency_sets,
    summarize,
)


def _row() -> dict:
    paragraphs = [
        {
            "title": f"Document {index}",
            "paragraph_text": f"unrelated paragraph {index}",
            "is_supporting": index in {1, 3, 5, 7},
        }
        for index in range(20)
    ]
    paragraphs[1] = {
        "title": "Starting Person",
        "paragraph_text": "Starting Person belongs to the Bridge Group.",
        "is_supporting": True,
    }
    paragraphs[3] = {
        "title": "Bridge Group",
        "paragraph_text": "The Bridge Group was controlled by Deep Country.",
        "is_supporting": True,
    }
    paragraphs[5] = {
        "title": "Separate Event",
        "paragraph_text": "The separate event involved Shallow Organization.",
        "is_supporting": True,
    }
    paragraphs[7] = {
        "title": "Final Comparison",
        "paragraph_text": "Deep Country joined Shallow Organization in 1990.",
        "is_supporting": True,
    }
    return {
        "id": "4hop3__1_2_3_4",
        "question": (
            "When did the country controlling the group of Starting Person "
            "join the organization involved in Separate Event?"
        ),
        "answer": "1990",
        "question_decomposition": [
            {
                "question": "Starting Person >> member of",
                "answer": "Bridge Group",
                "paragraph_support_idx": 1,
            },
            {
                "question": "Who controlled #1?",
                "answer": "Deep Country",
                "paragraph_support_idx": 3,
            },
            {
                "question": "Separate Event >> organization",
                "answer": "Shallow Organization",
                "paragraph_support_idx": 5,
            },
            {
                "question": "When did #2 join #3?",
                "answer": "1990",
                "paragraph_support_idx": 7,
            },
        ],
        "paragraphs": paragraphs,
    }


def test_contains_token_phrase_is_contiguous_and_casefolded() -> None:
    assert contains_token_phrase("The Deep Country joined.", "deep country")
    assert not contains_token_phrase("The deeper countryside.", "deep country")


def test_dependency_sets_extract_exact_branching_graph() -> None:
    assert dependency_sets(_row()) == EXPECTED_DEPENDENCIES


def test_analyze_row_detects_connected_prefix_unlock() -> None:
    result = analyze_row(_row())
    assert result["valid_branching_row"]
    assert result["deep_first_connected_prefix"] == 2
    assert result["shallow_first_connected_prefix"] == 1
    assert result["connected_prefix_gap"] == 1
    assert result["root_answers_hidden"]


def test_analyze_row_rejects_wrong_dependency_graph() -> None:
    row = _row()
    row["question_decomposition"][3]["question"] = "When did #2 join?"
    result = analyze_row(row)
    assert not result["valid_branching_row"]
    assert result["exclusion"] == "wrong_dependency_graph"


def test_analyze_row_rejects_duplicate_support_indices() -> None:
    row = _row()
    row["question_decomposition"][2]["paragraph_support_idx"] = 1
    result = analyze_row(row)
    assert not result["valid_branching_row"]
    assert result["exclusion"] == "invalid_or_duplicate_support_indices"


def test_summary_requires_every_structural_gap() -> None:
    diagnostic = analyze_row(_row())
    rows = [dict(diagnostic) for _ in range(120)]
    for row in rows:
        row["shallow_scores_above_deep"] = True
        row["deep_root_not_rank_one"] = True
        row["top_is_shallow_or_distractor"] = True
    summary = summarize(rows)
    assert summary["gates"]["all_pass"]
    rows[0]["connected_prefix_gap"] = 0
    assert not summarize(rows)["gates"]["all_connected_prefix_gaps_equal_one"]
