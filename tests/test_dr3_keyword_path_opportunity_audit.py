from __future__ import annotations

from scripts.bright_biology_unlock_audit import BM25Corpus
from scripts.dr3_keyword_path_opportunity_audit import (
    ALL_ORDER_HASH,
    DEVELOPMENT_ID_HASH,
    HOLDOUT_ID_HASH,
    OPPORTUNITY_ID_HASH,
    _id_hash,
    followup_queries,
    frozen_split_ids,
    query_observation,
    root_queries,
    summarize,
)


def _tiny_corpus() -> BM25Corpus:
    return BM25Corpus(
        [
            {
                "id": "a",
                "content": "salmon telemetry river migration barriers",
                "raw_source": "Telemetry reveals migration barriers in rivers.",
                "title": "Migration study",
                "keyword": "SECRET HABITAT LABEL",
            },
            {
                "id": "b",
                "content": "fish stock acoustic survey population estimate",
                "raw_source": "Acoustic surveys estimate a fish population.",
                "title": "Stock assessment",
                "keyword": "SECRET STOCK LABEL",
            },
            {
                "id": "c",
                "content": "marine reserve recovery bycatch fishing gear",
                "raw_source": "Selective gear reduces bycatch near reserves.",
                "title": "Reserve recovery",
                "keyword": "SECRET POLICY LABEL",
            },
            {
                "id": "d",
                "content": "coffee prices unrelated market",
                "raw_source": "Coffee prices changed.",
                "title": "Unrelated",
                "keyword": "SECRET NOISE LABEL",
            },
        ]
    )


def test_frozen_split_hashes_and_quarantine_reproduce() -> None:
    split = frozen_split_ids()
    assert [len(split[name]) for name in ("opportunity", "development", "holdout")] == [
        20,
        8,
        10,
    ]
    assert _id_hash(split["opportunity"]) == OPPORTUNITY_ID_HASH
    assert _id_hash(split["development"]) == DEVELOPMENT_ID_HASH
    assert _id_hash(split["holdout"]) == HOLDOUT_ID_HASH
    assert _id_hash([*split["opportunity"], *split["development"], *split["holdout"]]) == ALL_ORDER_HASH
    assert set().union(*map(set, split.values())).isdisjoint(
        {f"{index:03d}" for index in range(1, 13)}
    )


def test_roots_and_followups_are_visible_text_conditioned() -> None:
    corpus = _tiny_corpus()
    query = (
        "How do scientists measure fish stocks, and how do migration barriers "
        "and marine reserves affect recovery?"
    )
    roots = root_queries(query, corpus)
    assert len(roots) >= 5
    first = [corpus.documents[0]]
    observation = query_observation(first[0], roots[0], corpus)
    assert "SECRET HABITAT LABEL" not in observation
    followups = followup_queries(query, roots[0], first, corpus)
    assert followups
    assert all("SECRET" not in followup for followup in followups)


def test_summary_requires_strict_lower_immediate_tradeoffs() -> None:
    records = []
    for index in range(20):
        strict = index < 4
        records.append(
            {
                "task_id": f"{index + 13:03d}",
                "num_pages": 36,
                "num_keywords": 10,
                "num_roots": 12,
                "distinct_root_top1": 4 if index < 15 else 2,
                "best_immediate_coverage": 0.2,
                "oracle_pair_coverage": 0.4 if index < 16 else 0.3,
                "pair_gain_count": 1 if index < 16 else 0,
                "pair_coverage_gain": 0.1 if index < 16 else 0.0,
                "nonmyopic_gap_count": 1 if strict else 0,
                "nonmyopic_normalized_gap": 0.1 if strict else 0.0,
                "strict_opportunity": strict,
            }
        )
    summary = summarize(records)
    assert summary["strict_opportunity_count"] == 4
    assert summary["strict_total_gap"] == 4
    assert summary["gates"]["all_pass"]


def test_summary_fails_when_depth_two_is_only_order_commutative() -> None:
    records = []
    for index in range(20):
        records.append(
            {
                "task_id": f"{index + 13:03d}",
                "num_pages": 36,
                "num_keywords": 10,
                "num_roots": 12,
                "distinct_root_top1": 4,
                "best_immediate_coverage": 0.2,
                "oracle_pair_coverage": 0.4,
                "pair_gain_count": 2,
                "pair_coverage_gain": 0.2,
                "nonmyopic_gap_count": 0,
                "nonmyopic_normalized_gap": 0.0,
                "strict_opportunity": False,
            }
        )
    summary = summarize(records)
    assert summary["pair_gain_task_count"] == 20
    assert not summary["gates"]["strict_opportunities_at_least_4"]
    assert not summary["gates"]["all_pass"]
