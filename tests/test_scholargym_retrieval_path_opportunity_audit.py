from __future__ import annotations

import json
from pathlib import Path

from scripts.scholargym_retrieval_path_opportunity_audit import (
    DEVELOPMENT_ID_HASH,
    HOLDOUT_ID_HASH,
    OPPORTUNITY_ID_HASH,
    ScholarCorpus,
    _id_hash,
    analyze_task,
    analyze_task_from_index,
    followup_queries,
    root_queries,
    stream_json_object,
    summarize,
)


def _build_tiny_index(path: Path) -> ScholarCorpus:
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE papers (
                rowid INTEGER PRIMARY KEY,
                arxiv_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                date TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE papers_fts USING fts5(
                title,
                abstract,
                content='papers',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        connection.executemany(
            "INSERT INTO papers(arxiv_id,title,abstract,date) VALUES(?,?,?,?)",
            [
                (
                    "a",
                    "Migration telemetry",
                    "Telemetry reveals hidden river barriers for salmon.",
                    "2024-01-01",
                ),
                (
                    "b",
                    "Acoustic stock assessment",
                    "Sonar estimates fish populations in marine reserves.",
                    "2023-01-01",
                ),
                (
                    "c",
                    "Selective fishing gear",
                    "Bycatch declines with selective gear near reserves.",
                    "2022-01-01",
                ),
                (
                    "d",
                    "Coffee markets",
                    "Prices changed in unrelated commodity markets.",
                    "2021-01-01",
                ),
            ],
        )
        connection.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
        connection.execute(
            "CREATE VIRTUAL TABLE papers_vocab USING fts5vocab(papers_fts, 'row')"
        )
    return ScholarCorpus(path)


def test_stream_json_object_handles_chunk_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "one": {"title": "A long title", "values": [1, 2, 3]},
                "two": {"title": "Second", "nested": {"ok": True}},
            }
        ),
        encoding="utf-8",
    )
    assert list(stream_json_object(source, chunk_size=7)) == [
        ("one", {"title": "A long title", "values": [1, 2, 3]}),
        ("two", {"title": "Second", "nested": {"ok": True}}),
    ]


def test_frozen_selected_id_hashes_are_stable() -> None:
    opportunity = (
        "AutoScholarQuery_dev_891,AutoScholarQuery_dev_716,"
        "AutoScholarQuery_dev_161,AutoScholarQuery_dev_703,"
        "AutoScholarQuery_dev_139,AutoScholarQuery_dev_569,"
        "AutoScholarQuery_dev_233,AutoScholarQuery_dev_948,"
        "AutoScholarQuery_dev_907,AutoScholarQuery_dev_772,"
        "AutoScholarQuery_dev_325,AutoScholarQuery_dev_925,"
        "AutoScholarQuery_dev_298,AutoScholarQuery_dev_379,"
        "AutoScholarQuery_dev_745,AutoScholarQuery_dev_849,"
        "AutoScholarQuery_dev_614,AutoScholarQuery_dev_771,"
        "AutoScholarQuery_dev_864,AutoScholarQuery_dev_632,"
        "AutoScholarQuery_dev_941,AutoScholarQuery_dev_215,"
        "AutoScholarQuery_dev_431,AutoScholarQuery_dev_111,"
        "AutoScholarQuery_dev_641,AutoScholarQuery_dev_295,"
        "AutoScholarQuery_dev_455,AutoScholarQuery_dev_625,"
        "AutoScholarQuery_dev_239,AutoScholarQuery_dev_258,"
        "AutoScholarQuery_dev_954,AutoScholarQuery_dev_44,"
        "AutoScholarQuery_dev_804,AutoScholarQuery_dev_453,"
        "AutoScholarQuery_dev_943,AutoScholarQuery_dev_743,"
        "AutoScholarQuery_dev_43,AutoScholarQuery_dev_136,"
        "AutoScholarQuery_dev_419,AutoScholarQuery_dev_231"
    ).split(",")
    development = (
        "AutoScholarQuery_test_891,AutoScholarQuery_test_668,"
        "AutoScholarQuery_test_559,AutoScholarQuery_test_606,"
        "AutoScholarQuery_test_688,AutoScholarQuery_test_853,"
        "AutoScholarQuery_test_684,AutoScholarQuery_test_793,"
        "AutoScholarQuery_test_835,AutoScholarQuery_test_448,"
        "AutoScholarQuery_test_899,AutoScholarQuery_test_860,"
        "AutoScholarQuery_test_156,AutoScholarQuery_test_322,"
        "AutoScholarQuery_test_502,AutoScholarQuery_test_717,"
        "AutoScholarQuery_test_262,AutoScholarQuery_test_567,"
        "AutoScholarQuery_test_656,AutoScholarQuery_test_327,"
        "AutoScholarQuery_test_224,AutoScholarQuery_test_961,"
        "AutoScholarQuery_test_741,AutoScholarQuery_test_693"
    ).split(",")
    holdout = (
        "RealScholarQuery_2,RealScholarQuery_44,RealScholarQuery_19,"
        "RealScholarQuery_6,RealScholarQuery_31,RealScholarQuery_22,"
        "RealScholarQuery_20,RealScholarQuery_18,RealScholarQuery_3,"
        "RealScholarQuery_10,RealScholarQuery_23,RealScholarQuery_38,"
        "RealScholarQuery_13,RealScholarQuery_27,RealScholarQuery_9,"
        "RealScholarQuery_46,RealScholarQuery_29,RealScholarQuery_42,"
        "RealScholarQuery_35,RealScholarQuery_49,RealScholarQuery_48,"
        "RealScholarQuery_40,RealScholarQuery_30,RealScholarQuery_15"
    ).split(",")
    assert _id_hash(opportunity) == OPPORTUNITY_ID_HASH
    assert _id_hash(development) == DEVELOPMENT_ID_HASH
    assert _id_hash(holdout) == HOLDOUT_ID_HASH


def test_roots_and_followups_are_observation_conditioned(tmp_path: Path) -> None:
    corpus = _build_tiny_index(tmp_path / "tiny.sqlite")
    try:
        query = (
            "How do researchers assess fish stocks and study migration "
            "barriers near marine reserves?"
        )
        roots = root_queries(query, corpus)
        assert len(roots) >= 5
        first = corpus.search(roots[0], top_k=2)
        followups = followup_queries(query, roots[0], first, corpus)
        assert followups
        initial_terms = set(query.lower().split())
        for followup, observed_terms in followups:
            assert observed_terms
            assert set(observed_terms).isdisjoint(initial_terms)
            visible = " ".join(
                f"{paper['title']} {paper['abstract']}" for paper in first
            ).lower()
            assert all(term in visible for term in observed_terms)
            assert all(term in followup.lower() for term in observed_terms)
    finally:
        corpus.close()


def test_default_rank_matches_scalar_bm25_order(tmp_path: Path) -> None:
    corpus = _build_tiny_index(tmp_path / "tiny.sqlite")
    try:
        match_query = '"fish" OR "reserves"'
        rank_rows = corpus.connection.execute(
            """
            SELECT papers.arxiv_id
            FROM papers_fts JOIN papers ON papers.rowid = papers_fts.rowid
            WHERE papers_fts MATCH ?
            ORDER BY papers_fts.rank ASC, papers.rowid ASC
            """,
            (match_query,),
        ).fetchall()
        scalar_rows = corpus.connection.execute(
            """
            SELECT papers.arxiv_id
            FROM papers_fts JOIN papers ON papers.rowid = papers_fts.rowid
            WHERE papers_fts MATCH ?
            ORDER BY bm25(papers_fts) ASC, papers.rowid ASC
            """,
            (match_query,),
        ).fetchall()
        assert rank_rows == scalar_rows
    finally:
        corpus.close()


def test_task_worker_reopens_index_without_changing_record(tmp_path: Path) -> None:
    index_path = tmp_path / "tiny.sqlite"
    corpus = _build_tiny_index(index_path)
    row = {
        "qid": "tiny",
        "query": (
            "How do researchers assess fish stocks and study migration "
            "barriers near marine reserves?"
        ),
        "cited_paper": [{"arxiv_id": "a"}, {"arxiv_id": "b"}],
        "gt_label": [1, 1],
        "date": "2024-12-31",
    }
    try:
        direct = analyze_task(row, corpus)
    finally:
        corpus.close()
    assert analyze_task_from_index(row, index_path) == direct


def test_summary_requires_strict_lower_immediate_tradeoffs() -> None:
    records = []
    for index in range(40):
        strict = index < 5
        records.append(
            {
                "task_id": f"task-{index}",
                "num_gt": 3,
                "num_roots": 8,
                "distinct_root_top1": 3 if index < 30 else 2,
                "best_immediate_recall": 1 / 3,
                "oracle_pair_recall": 2 / 3 if index < 20 else 1 / 3,
                "pair_gain_count": 1 if index < 20 else 0,
                "pair_recall_gain": 1 / 3 if index < 20 else 0.0,
                "nonmyopic_gap_count": 1 if strict else 0,
                "nonmyopic_normalized_gap": 1 / 3 if strict else 0.0,
                "strict_opportunity": strict,
            }
        )
    summary = summarize(records)
    assert summary["strict_opportunity_count"] == 5
    assert summary["strict_total_gap"] == 5
    assert summary["gates"]["all_pass"]


def test_summary_rejects_order_commutative_depth_gain() -> None:
    records = [
        {
            "task_id": f"task-{index}",
            "num_gt": 3,
            "num_roots": 8,
            "distinct_root_top1": 4,
            "best_immediate_recall": 1 / 3,
            "oracle_pair_recall": 2 / 3,
            "pair_gain_count": 1,
            "pair_recall_gain": 1 / 3,
            "nonmyopic_gap_count": 0,
            "nonmyopic_normalized_gap": 0.0,
            "strict_opportunity": False,
        }
        for index in range(40)
    ]
    summary = summarize(records)
    assert summary["pair_gain_task_count"] == 40
    assert not summary["gates"]["strict_opportunities_at_least_5"]
    assert not summary["gates"]["all_pass"]


def test_frozen_opportunity_artifact_closes_myopically_aligned_route() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "nonmyopic"
        / "scholargym_retrieval_path_opportunity"
        / "AUDIT.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert payload["status"] == "gate_failed"
    assert summary["num_records"] == 40
    assert summary["pair_gain_task_count"] == 9
    assert summary["strict_opportunity_count"] == 0
    assert summary["strict_total_gap"] == 0
    assert all(
        record["greedy_root_index"] == record["oracle_root_index"]
        for record in payload["records"]
    )
    assert not summary["gates"]["all_pass"]
