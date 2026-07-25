import json

import pytest

from scripts.tau_knowledge_receding_continuation import (
    EXPECTED_REQUESTS,
    FRESH_CONFIRMATION_IDS,
    compact_continuation_input,
    continuation_messages,
    parse_continuation_scores,
    summarize,
)


def _result(document_id):
    return {
        "id": document_id,
        "title": f"Title {document_id}",
        "content": f"Content {document_id}",
        "bm25_score": 1.0,
    }


def _record():
    branches = []
    for root in range(5):
        branches.append(
            {
                "query": f"root query {root}",
                "first_results": [_result(f"first-{root}")],
                "refreshed_information_need_hypotheses": [
                    f"need {index}" for index in range(8)
                ],
                "followups": [
                    {
                        "query": f"followup {root} {followup}",
                        "results": [_result(f"result-{root}-{followup}")],
                    }
                    for followup in range(4)
                ],
            }
        )
    return {
        "task_id": "task_test",
        "opening": "I need help.",
        "required_documents": ["SECRET_REQUIRED"],
        "initial_information_need_hypotheses": [
            f"initial {index}" for index in range(8)
        ],
        "first_branches": branches,
    }


def _payload():
    payload = {}
    for index in range(1, 5):
        payload[f"followup_{index}_score"] = str(index * 10)
        payload[f"followup_{index}_rationale"] = f"reason {index}"
    return payload


def test_focused_input_hides_endpoint_and_other_roots():
    encoded = json.dumps(compact_continuation_input(_record(), 2))
    assert "SECRET_REQUIRED" not in encoded
    assert "root query 2" in encoded
    assert "root query 1" not in encoded


def test_focused_prompt_contains_all_four_continuations():
    encoded = json.dumps(continuation_messages(_record(), 1))
    for index in range(4):
        assert f"followup 1 {index}" in encoded
    assert "followup 0 0" not in encoded


def test_canonical_continuation_schema_parses():
    parsed = parse_continuation_scores(json.dumps(_payload()))
    assert parsed["scores"] == [10, 20, 30, 40]


def test_continuation_schema_rejects_json_number():
    payload = _payload()
    payload["followup_1_score"] = 10
    with pytest.raises(ValueError, match="canonical"):
        parse_continuation_scores(json.dumps(payload))


def test_request_counts_and_fresh_split_are_frozen():
    assert len(FRESH_CONFIRMATION_IDS) == 20
    assert EXPECTED_REQUESTS["serving_smoke"] == 10
    assert EXPECTED_REQUESTS["development"] == 100
    assert EXPECTED_REQUESTS["confirmation"] == 280


def test_smoke_summary_uses_all_root_scores(monkeypatch):
    records = [_record(), _record()]
    endpoint = {
        "pair_counts": [[1, 2, 3, 4] for _root in range(5)]
    }
    monkeypatch.setattr(
        "scripts.tau_knowledge_receding_continuation.analyze_record",
        lambda _record: endpoint,
    )
    scores = [
        [
            {
                "scores": [10, 20, 30, 40],
                "rationales": ["a", "b", "c", "d"],
            }
            for _root in range(5)
        ]
        for _record in records
    ]
    usage = {"physical_requests": 10, "reasoning_tokens": 0}
    summary = summarize(
        records,
        scores,
        usage,
        stage="serving_smoke",
        myopic_scores=None,
        nonmyopic_scores=None,
    )
    assert summary["focused_optimal_followup_count"] == 10
    assert summary["focused_total_regret"] == 0
    assert summary["gates"]["all_pass"]


def test_confirmation_summary_reports_root_ranking_metrics(monkeypatch):
    records = [_record() for _case in range(20)]
    endpoint = {
        "pair_counts": [[1, 2, 3, 4] for _root in range(5)]
    }
    monkeypatch.setattr(
        "scripts.tau_knowledge_receding_continuation.analyze_record",
        lambda _record: endpoint,
    )
    continuation = [
        [
            {
                "scores": [10, 20, 30, 40],
                "rationales": ["a", "b", "c", "d"],
            }
            for _root in range(5)
        ]
        for _record in records
    ]
    root_scores = [
        {
            "scores": [10, 20, 30, 40, 50],
            "best_followup_indices": [3, 3, 3, 3, 3],
        }
        for _record in records
    ]
    summary = summarize(
        records,
        continuation,
        {"physical_requests": 280, "reasoning_tokens": 0},
        stage="confirmation",
        myopic_scores=root_scores,
        nonmyopic_scores=root_scores,
    )
    assert summary["root_pairwise_comparable_count"] == 0
    assert summary["myopic_root_pairwise_accuracy"] == 0.0
    assert summary["nonmyopic_root_pairwise_accuracy"] == 0.0
    assert summary["root_pairwise_accuracy_gain"] == 0.0
