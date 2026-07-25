from __future__ import annotations

from scripts import multihop_rag_semantic_bed_mechanics as mechanics


def _document(url: str, title: str, body: str) -> dict[str, str]:
    return {
        "url": url,
        "title": title,
        "body": body,
        "author": "",
        "source": "",
        "published_at": "",
        "category": "",
    }


def test_search_excludes_root_documents():
    index = mechanics.CorpusIndex(
        [
            _document("u0", "alpha", "alpha alpha"),
            _document("u1", "alpha beta", "alpha beta"),
            _document("u2", "beta", "beta beta"),
        ]
    )
    roots = index.search("alpha", limit=2)
    continuation = index.search("alpha", exclude=roots, limit=2)
    assert len(roots) == 2
    assert set(roots).isdisjoint(continuation)
    assert continuation == list({0, 1, 2} - set(roots))


def test_evaluate_task_detects_no_reversal_when_greedy_root_is_sufficient(
    monkeypatch,
):
    corpus = [
        _document("u0", "alpha", "alpha bridge"),
        _document("u1", "beta", "beta answer"),
        _document("u2", "noise", "noise"),
    ]
    index = mechanics.CorpusIndex(corpus)
    task = {
        "task_id": "t0",
        "question_type": "inference_query",
        "query": "alpha beta",
        "evidence_list": [{"url": "u0"}, {"url": "u1"}],
    }
    monkeypatch.setattr(mechanics, "ROOT_RETRIEVAL_SIZE", 1)
    monkeypatch.setattr(mechanics, "CONTINUATION_RETRIEVAL_SIZE", 1)
    monkeypatch.setattr(
        mechanics,
        "root_queries",
        lambda question, index: ["alpha", "noise"],
    )
    monkeypatch.setattr(
        mechanics,
        "continuation_queries",
        lambda question, document, index: ["beta"],
    )
    result = mechanics.evaluate_task(task, index)
    assert result["best_immediate_evidence"] == 1
    assert result["best_total_evidence"] == 2
    assert result["greedy_root_best_total_evidence"] == 2
    assert result["continuation_gain"] == 1
    assert result["strict_root_reversal"] is False


def test_public_task_result_contains_no_content():
    corpus = [
        _document("u0", "alpha", "alpha"),
        _document("u1", "beta", "beta"),
        _document("u2", "noise", "noise"),
        _document("u3", "other", "other"),
    ]
    task = {
        "task_id": "t0",
        "question_type": "comparison_query",
        "query": "alpha beta",
        "answer": "private answer",
        "evidence_list": [{"url": "u0"}, {"url": "u1"}],
    }
    result = mechanics.evaluate_task(task, mechanics.CorpusIndex(corpus))
    assert "query" not in result
    assert "answer" not in result
    assert "evidence_list" not in result
    assert "url" not in result
