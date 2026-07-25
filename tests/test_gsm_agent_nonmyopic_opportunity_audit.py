from __future__ import annotations

from scripts.gsm_agent_nonmyopic_opportunity_audit import (
    BM25Index,
    CorpusDocument,
    analyze_task,
    continuation_queries,
    ordered_list_hash,
    root_queries,
    split_test_ids,
    summarize,
)


def _document(document_id: str, content: str) -> CorpusDocument:
    return CorpusDocument(document_id, content, {"identity": document_id})


def test_split_is_deterministic_and_complete() -> None:
    splits = split_test_ids(f"task-{index:04d}" for index in range(1073))
    assert len(splits["opportunity"]) == 500
    assert len(splits["development"]) == 100
    assert len(splits["holdout"]) == 473
    assert len(set(splits["all_test"])) == 1073
    assert ordered_list_hash(splits["all_test"]) == ordered_list_hash(
        splits["all_test"]
    )


def test_query_generation_is_bounded_and_deterministic() -> None:
    documents = [
        _document("alpha-ledger", "Alpha paid seven credits for cobalt."),
        _document("beta-note", "Beta received a cobalt shipment."),
    ]
    index = BM25Index(documents)
    question = "How many cobalt credits did Alpha pay?"
    roots = root_queries(question, index)
    assert roots == root_queries(question, index)
    assert 1 <= len(roots) <= 24
    continuations = continuation_queries(question, documents, index)
    assert continuations == continuation_queries(question, documents, index)
    assert 1 <= len(continuations) <= 30


def test_analyze_task_detects_strict_nonmyopic_tradeoff(monkeypatch) -> None:
    documents = [
        _document("greedy", "alpha immediate"),
        _document("setup", "beta bridgeword"),
        _document("child", "bridgeword hidden one"),
        _document("child2", "bridgeword hidden two"),
        _document("noise", "unrelated"),
        _document("noise2", "unrelated again"),
        _document("noise3", "other material"),
        _document("noise4", "more other material"),
    ]
    index = BM25Index(documents)
    entry = {
        "question_id": "task",
        "question": "alpha beta",
        "document_ids": ["greedy", "child", "child2"],
        "source_split": "test",
    }

    monkeypatch.setattr(
        "scripts.gsm_agent_nonmyopic_opportunity_audit.root_queries",
        lambda _question, _index: ["alpha", "beta"],
    )
    monkeypatch.setattr(
        "scripts.gsm_agent_nonmyopic_opportunity_audit.continuation_queries",
        lambda _question, page, _index: (
            ["bridgeword"] if page[0].document_id == "setup" else ["unrelated"]
        ),
    )
    pages = {
        "alpha": (0, 4, 5, 6, 1),
        "beta": (1, 4, 5, 6, 7),
        "bridgeword": (2, 3, 1, 4, 5),
        "unrelated": (4, 5, 6, 7, 1),
    }
    monkeypatch.setattr(index, "search", lambda query: pages[query])
    result = analyze_task(entry, index)
    assert result["strict_nonmyopic_opportunity"]
    assert result["immediate_coverage_sacrifice"] == 1
    assert result["two_step_coverage_gain"] == 1


def test_summary_requires_prevalence_and_power_gates() -> None:
    base = {
        "strict_nonmyopic_opportunity": True,
        "source_split": "test",
        "oracle_document_count": 4,
        "immediate_coverage_sacrifice": 1,
        "two_step_coverage_gain": 1,
    }
    rows = [dict(base) for _ in range(40)]
    rows.extend(
        {
            **base,
            "strict_nonmyopic_opportunity": False,
            "two_step_coverage_gain": 0,
        }
        for _ in range(460)
    )
    assert summarize(rows)["gates"]["all_pass"]
    rows[0]["immediate_coverage_sacrifice"] = 0
    assert not summarize(rows)["gates"]["all_pass"]
