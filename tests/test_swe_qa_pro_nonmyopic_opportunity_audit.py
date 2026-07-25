from __future__ import annotations

from scripts.bright_biology_unlock_audit import BM25Corpus
from scripts.swe_qa_pro_nonmyopic_opportunity_audit import (
    DEVELOPMENT_INDEX_HASH,
    HOLDOUT_INDEX_HASH,
    OPPORTUNITY_INDEX_HASH,
    _index_hash,
    followup_queries,
    frozen_split_indices,
    gold_evidence_paths,
    root_queries,
    summarize,
)


def test_frozen_split_hashes_and_sizes_reproduce() -> None:
    split = frozen_split_indices()
    assert [len(split[name]) for name in ("opportunity", "development", "holdout")] == [
        120,
        40,
        100,
    ]
    assert _index_hash(split["opportunity"]) == OPPORTUNITY_INDEX_HASH
    assert _index_hash(split["development"]) == DEVELOPMENT_INDEX_HASH
    assert _index_hash(split["holdout"]) == HOLDOUT_INDEX_HASH
    assert len(set().union(*map(set, split.values()))) == 260


def test_gold_evidence_paths_accept_exact_paths_and_unique_basenames() -> None:
    paths = [
        "pkg/core/engine.py",
        "pkg/io/reader.py",
        "tests/io/reader.py",
        "pkg/special/adapter_impl.py",
    ]
    answer = (
        "The flow begins in `pkg/core/engine.py` and is finalized by "
        "`adapter_impl.py`. The generic reader.py name is ambiguous."
    )
    assert gold_evidence_paths(answer, paths) == [
        "pkg/core/engine.py",
        "pkg/special/adapter_impl.py",
    ]


def _tiny_corpus() -> BM25Corpus:
    return BM25Corpus(
        [
            {
                "id": "parser.py",
                "content": "parser alias nested unexpected token handler",
                "raw_source": "def parse_alias(value):\n    return nested_handler(value)\n",
            },
            {
                "id": "nested.py",
                "content": "nested handler recursive alias",
                "raw_source": "def nested_handler(value):\n    return value\n",
            },
            {
                "id": "other.py",
                "content": "unrelated formatter output",
                "raw_source": "def format_output(value):\n    return value\n",
            },
        ]
    )


def test_root_and_followup_queries_are_target_blind_and_observation_conditioned() -> None:
    corpus = _tiny_corpus()
    question = "Why does `parse_alias` fail for nested input?"
    roots = root_queries(question, corpus)
    assert len(roots) >= 3
    assert len({root.casefold() for root in roots}) == len(roots)
    first = [corpus.documents[0]]
    followups = followup_queries(question, roots[0], first, corpus)
    assert followups
    assert any("handler" in followup for followup in followups)


def test_summary_requires_prevalent_strict_tradeoffs() -> None:
    records = []
    for index in range(120):
        strict = index < 12
        records.append(
            {
                "usable": True,
                "repo": f"org/repo{index % 8}",
                "num_roots": 5,
                "num_evidence_files": 3,
                "distinct_root_top1": 3,
                "best_immediate_evidence_count": 1,
                "pair_gain_count": int(index < 18),
                "strict_opportunity": strict,
                "nonmyopic_normalized_gap": 1 / 3 if strict else 0.0,
            }
        )
    summary = summarize(records)
    assert summary["gates"]["all_pass"]

