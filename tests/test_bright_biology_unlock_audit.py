from __future__ import annotations

import numpy as np

from scripts.bright_biology_unlock_audit import (
    BM25Corpus,
    followup_queries,
    root_queries,
    summarize_records,
    tokenize,
)


def test_root_queries_are_deduped_in_frozen_order() -> None:
    query = (
        "A short title\n"
        "Why does a signal appear? A substantially longer explanatory sentence "
        "with several distinct words."
    )
    roots = root_queries(query, "A bridge mechanism may explain the signal.")
    assert roots[0] == " ".join(query.split())
    assert roots[1] == "A short title"
    assert roots[2] == "A bridge mechanism may explain the signal."
    assert roots[3] == "Why does a signal appear?"
    assert roots[4].startswith("A substantially longer explanatory")
    assert len({tuple(tokenize(root)) for root in roots}) == len(roots)


def test_followups_use_only_observed_document_terms() -> None:
    documents = [
        {"id": "d1", "content": "alpha bridge kinase kinase zephyr"},
        {"id": "d2", "content": "alpha receptor quasar membrane"},
        {"id": "d3", "content": "alpha tissue xylophone pathway"},
        {"id": "d4", "content": "unrelated control words"},
    ]
    corpus = BM25Corpus(documents)
    followups = followup_queries("alpha question", documents[:3], corpus)
    assert len(followups) == 4
    assert all(followup.startswith("alpha question") for followup in followups)
    assert "alpha" not in tokenize(followups[0])[2:]
    assert any("zephyr" in followup for followup in followups)
    assert not any("unrelated" in followup for followup in followups)


def test_sparse_bm25_scores_match_reference_implementation() -> None:
    documents = [
        {"id": "d1", "content": "alpha beta beta"},
        {"id": "d2", "content": "alpha gamma"},
        {"id": "d3", "content": "delta epsilon"},
    ]
    corpus = BM25Corpus(documents)
    query = "alpha beta beta missing"
    expected = corpus.index.get_scores(tokenize(query))
    assert np.allclose(corpus.scores(query), expected, rtol=0.0, atol=1e-15)


def _record(
    *,
    root_count: int = 5,
    diversity: int = 3,
    pair_gain: int = 1,
    changed: bool = True,
    gap: int = 1,
    gold_count: int = 2,
    direct: int = 0,
) -> dict[str, int]:
    return {
        "root_count": root_count,
        "distinct_first_results": diversity,
        "pair_gain": pair_gain,
        "oracle_root_index": 1 if changed else 0,
        "immediate_root_index": 0,
        "nonmyopic_gap": gap,
        "gold_count": gold_count,
        "direct_query_utility": direct,
    }


def test_summary_passes_exact_frozen_thresholds() -> None:
    records = [
        _record(
            changed=index < 5,
            gap=1 if index < 4 else 0,
            pair_gain=1 if index < 10 else 0,
            root_count=3 if index < 16 else 2,
            diversity=3 if index < 10 else 2,
        )
        for index in range(20)
    ]
    summary = summarize_records(records)
    assert summary["analyzable_count"] == 16
    assert summary["mean_distinct_first_results"] == 2.5
    assert summary["pair_gain_count"] == 10
    assert summary["mean_pair_gain"] == 0.5
    assert summary["root_change_count"] == 5
    assert summary["nonmyopic_gap_count"] == 4
    assert summary["mean_nonmyopic_gap"] == 0.2
    assert summary["direct_query_coverage"] == 0.0
    assert summary["passed"] is True


def test_summary_fails_direct_retrieval_saturation() -> None:
    records = [_record(direct=2) for _ in range(20)]
    summary = summarize_records(records)
    assert summary["direct_query_coverage"] == 1.0
    assert summary["gates"]["direct_coverage_below_0_60"] is False
    assert summary["passed"] is False
