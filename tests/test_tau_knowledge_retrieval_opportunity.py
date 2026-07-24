import json

import pytest

from scripts.tau_knowledge_retrieval_opportunity import (
    EXPECTED_REQUESTS,
    FIRST_QUERY_COUNT,
    FOLLOWUP_QUERY_COUNT,
    OPPORTUNITY_IDS,
    SMOKE_IDS,
    analyze_record,
    initial_messages,
    parse_followup,
    parse_initial,
)


def test_parse_initial_requires_global_query_diversity():
    payload = {
        "information_need_hypotheses": [
            f"need {index}" for index in range(8)
        ],
        "direct_queries": ["direct one", "direct two"],
        "enabling_queries": ["enable one", "enable two", "enable three"],
    }
    hypotheses, queries = parse_initial(json.dumps(payload))
    assert len(hypotheses) == 8
    assert len(queries) == FIRST_QUERY_COUNT
    payload["enabling_queries"][0] = "direct one"
    with pytest.raises(ValueError, match="globally distinct"):
        parse_initial(json.dumps(payload))


def test_initial_prompt_contains_no_endpoint_fields():
    visible = json.dumps(initial_messages("I need help with my card."))
    assert "required_documents" not in visible
    assert "evaluation_criteria" not in visible


def test_parse_followup_rejects_repeated_first_query():
    payload = {
        "information_need_hypotheses": [
            f"updated need {index}" for index in range(8)
        ],
        "followup_queries": [
            "first query",
            "next two",
            "next three",
            "next four",
        ],
    }
    with pytest.raises(ValueError, match="repeats"):
        parse_followup(json.dumps(payload), first_query="first query")


def _results(*ids):
    return [
        {
            "id": value,
            "title": value,
            "content": value,
            "bm25_score": 1.0,
        }
        for value in ids
    ]


def test_analysis_finds_path_dependent_required_document_gain():
    branches = []
    for first_index in range(FIRST_QUERY_COUNT):
        first = _results(f"noise-{first_index}")
        if first_index == 0:
            first = _results("required-a", "noise-a", "noise-b")
        followups = [
            {
                "query": f"followup {first_index} {second_index}",
                "results": _results(f"noise-{first_index}-{second_index}"),
            }
            for second_index in range(FOLLOWUP_QUERY_COUNT)
        ]
        branches.append(
            {
                "query": f"first {first_index}",
                "first_results": first,
                "followups": followups,
            }
        )
    branches[1]["followups"][2]["results"] = _results(
        "required-a", "required-b", "noise"
    )
    result = analyze_record(
        {
            "required_documents": ["required-a", "required-b"],
            "first_branches": branches,
        }
    )
    assert result["greedy_first_index"] == 0
    assert result["oracle_first_index"] == 1
    assert result["pair_gain_over_best_one_step"] == 1
    assert result["nonmyopic_required_document_gap"] == 1


def test_request_counts_match_one_initial_plus_five_followups():
    per_case = 1 + FIRST_QUERY_COUNT
    assert EXPECTED_REQUESTS["serving_smoke"] == len(SMOKE_IDS) * per_case
    assert EXPECTED_REQUESTS["opportunity"] == (
        len(OPPORTUNITY_IDS) * per_case
    )
