import json

import pytest

from scripts.tau_knowledge_receding_continuation_v3 import (
    document_count_messages,
    parse_document_count_scores,
)


def _result(document_id):
    return {
        "id": document_id,
        "title": f"Title {document_id}",
        "content": f"Evidence {document_id}",
        "bm25_score": 1.0,
    }


def _record():
    branch = {
        "query": "realized root",
        "first_results": [_result("first")],
        "refreshed_information_need_hypotheses": ["fallible need"],
        "followups": [
            {
                "query": f"SECRET QUERY {index}",
                "results": [_result(f"result-{index}")],
            }
            for index in range(4)
        ],
    }
    return {
        "task_id": "task_test",
        "opening": "Compare accounts.",
        "required_documents": ["SECRET_REQUIRED"],
        "initial_information_need_hypotheses": ["compare all accounts"],
        "first_branches": [branch] * 5,
    }


def _scores():
    return {
        "followup_1_score": "7",
        "followup_2_score": "34",
        "followup_3_score": "68",
        "followup_4_score": "95",
    }


def test_document_count_prompt_is_compact_target_blind_and_count_dominant():
    encoded = json.dumps(document_count_messages(_record(), 0))
    assert "SECRET QUERY" not in encoded
    assert "SECRET_REQUIRED" not in encoded
    assert "Refreshed information needs are fallible" in encoded
    assert "N=3 gives 90-99" in encoded
    assert "rationale" not in encoded


def test_document_count_scores_parse_valid_bands():
    parsed = parse_document_count_scores(json.dumps(_scores()))
    assert parsed["scores"] == [7, 34, 68, 95]


@pytest.mark.parametrize("bad", ["10", "29", "40", "70", "100"])
def test_document_count_scores_reject_values_between_bands(bad):
    payload = _scores()
    payload["followup_1_score"] = bad
    with pytest.raises(ValueError, match="valid band"):
        parse_document_count_scores(json.dumps(payload))


def test_document_count_scores_reject_extra_keys_and_json_numbers():
    payload = _scores()
    payload["explanation"] = "extra"
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_document_count_scores(json.dumps(payload))
    payload = _scores()
    payload["followup_1_score"] = 7
    with pytest.raises(ValueError, match="canonical"):
        parse_document_count_scores(json.dumps(payload))
