import json

from scripts.tau_knowledge_receding_continuation_v2 import (
    compact_evidence_input,
    evidence_continuation_messages,
)


def _result(document_id):
    return {
        "id": document_id,
        "title": f"Title {document_id}",
        "content": f"Evidence {document_id}",
        "bm25_score": 1.0,
    }


def _record():
    branches = []
    for root in range(5):
        branches.append(
            {
                "query": f"realized root {root}",
                "first_results": [_result(f"first-{root}")],
                "refreshed_information_need_hypotheses": ["need"],
                "followups": [
                    {
                        "query": f"SECRET QUERY {root} {followup}",
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
        "initial_information_need_hypotheses": ["initial need"],
        "first_branches": branches,
    }


def test_evidence_input_hides_candidate_queries_and_endpoint():
    encoded = json.dumps(compact_evidence_input(_record(), 2))
    assert "SECRET QUERY" not in encoded
    assert "SECRET_REQUIRED" not in encoded
    assert "realized root 2" in encoded
    for followup in range(4):
        assert f"Evidence result-2-{followup}" in encoded


def test_evidence_prompt_requires_document_only_support():
    encoded = json.dumps(evidence_continuation_messages(_record(), 1))
    assert "Candidate query wording is deliberately hidden" in encoded
    assert "do not explicitly support" in encoded
    assert "SECRET QUERY" not in encoded
