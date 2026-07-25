import json

import pytest

from scripts.tau_knowledge_first_link_scorer import (
    FIRST_QUERY_COUNT,
    compact_scorer_input,
    pairwise_ranking_points,
    parse_scores,
    scorer_messages,
)


def _result(document_id, title="Policy", content="Policy content"):
    return {
        "id": document_id,
        "title": title,
        "content": content,
        "bm25_score": 1.0,
    }


def _record():
    branches = []
    for root in range(FIRST_QUERY_COUNT):
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
                        "results": [_result(f"followup-{root}-{followup}")],
                    }
                    for followup in range(4)
                ],
            }
        )
    return {
        "task_id": "task_test",
        "opening": "I need help.",
        "required_documents": ["SECRET_REQUIRED_DOC"],
        "initial_information_need_hypotheses": [
            f"initial need {index}" for index in range(8)
        ],
        "first_branches": branches,
    }


def _score_payload(include_followups):
    payload = {}
    for index in range(1, FIRST_QUERY_COUNT + 1):
        payload[f"root_{index}_score"] = index * 10
        if include_followups:
            payload[f"root_{index}_best_followup"] = (index % 4) + 1
        payload[f"root_{index}_rationale"] = f"reason {index}"
    return payload


def test_compact_input_hides_required_document_labels():
    encoded = json.dumps(
        compact_scorer_input(_record(), include_followups=True)
    )
    assert "required_documents" not in encoded
    assert "SECRET_REQUIRED_DOC" not in encoded


def test_myopic_prompt_does_not_include_followups():
    encoded = json.dumps(scorer_messages(_record(), include_followups=False))
    assert "followup 0 0" not in encoded
    assert "followups" not in encoded


def test_nonmyopic_prompt_includes_full_tree():
    encoded = json.dumps(scorer_messages(_record(), include_followups=True))
    assert "followup 0 0" in encoded
    assert "refreshed_information_needs" in encoded


@pytest.mark.parametrize("include_followups", [False, True])
def test_flat_score_schema_parses(include_followups):
    parsed = parse_scores(
        json.dumps(_score_payload(include_followups)),
        include_followups=include_followups,
    )
    assert parsed["scores"] == [10, 20, 30, 40, 50]
    assert len(parsed["best_followup_indices"]) == (
        FIRST_QUERY_COUNT if include_followups else 0
    )


def test_score_parser_rejects_out_of_range_followup():
    payload = _score_payload(True)
    payload["root_1_best_followup"] = 5
    with pytest.raises(ValueError, match="followup"):
        parse_scores(json.dumps(payload), include_followups=True)


def test_pairwise_ranking_points_handle_ties():
    points, comparable = pairwise_ranking_points(
        [90, 80, 80, 10],
        [3, 2, 1, 1],
    )
    assert comparable == 5
    assert points == 4.5
