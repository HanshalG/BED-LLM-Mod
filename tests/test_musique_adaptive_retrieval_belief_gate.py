import json

import pytest

from scripts.musique_adaptive_retrieval_belief_gate import (
    BM25Index,
    EXPECTED_REQUESTS,
    FIRST_ACTION_COUNT,
    FORMAL_IDS,
    SECOND_ACTION_COUNT,
    SMOKE_IDS,
    analyze_record,
    parse_query_response,
)
from scripts.musique_answer_belief_bridge_gate import BELIEF_SIZE


def _belief():
    return [
        {"answer": f"answer {index}", "probability": 1 / BELIEF_SIZE}
        for index in range(BELIEF_SIZE)
    ]


def test_bm25_retrieves_matching_document_and_honors_exclusion():
    documents = {
        "d01": {"title": "Apples", "paragraph_text": "red fruit orchard"},
        "d02": {"title": "Engines", "paragraph_text": "diesel motor vehicle"},
        "d03": {"title": "Pears", "paragraph_text": "green fruit tree"},
    }
    index = BM25Index(documents)
    assert index.retrieve("diesel engine motor") == "d02"
    assert index.retrieve("diesel engine motor", exclude=("d02",)) == "d01"


def test_parse_initial_queries_requires_three_direct_and_three_bridge():
    payload = {
        "belief": _belief(),
        "direct_queries": [f"direct {index}" for index in range(3)],
        "bridge_queries": [f"bridge {index}" for index in range(3)],
    }
    belief, queries = parse_query_response(
        json.dumps(payload), stage="initial"
    )
    assert len(belief) == BELIEF_SIZE
    assert queries == [
        "direct 0",
        "direct 1",
        "direct 2",
        "bridge 0",
        "bridge 1",
        "bridge 2",
    ]


def test_parse_followup_queries_rejects_duplicates():
    payload = {
        "belief": _belief(),
        "next_queries": ["same", "same", "third", "fourth"],
    }
    with pytest.raises(ValueError, match="distinct"):
        parse_query_response(json.dumps(payload), stage="followup")


def test_request_counts_cover_full_adaptive_tree():
    per_case = (
        1
        + FIRST_ACTION_COUNT
        + FIRST_ACTION_COUNT * SECOND_ACTION_COUNT
        + 1
        + 1
    )
    assert EXPECTED_REQUESTS["serving_smoke"] == len(SMOKE_IDS) * per_case
    assert EXPECTED_REQUESTS["opportunity"] == len(FORMAL_IDS) * per_case


def test_analysis_finds_gold_adaptive_pair_over_greedy():
    first_branches = []
    for first_index in range(FIRST_ACTION_COUNT):
        seconds = [
            {
                "query": f"followup {first_index} {second_index}",
                "retrieved_doc_id": f"d{10 + second_index:02d}",
                "truth_probability": 0.2,
            }
            for second_index in range(SECOND_ACTION_COUNT)
        ]
        first_branches.append(
            {
                "query": f"first {first_index}",
                "retrieved_doc_id": f"d{first_index + 1:02d}",
                "truth_probability": 0.6 if first_index == 0 else 0.2,
                "second_branches": seconds,
            }
        )
    first_branches[1]["retrieved_doc_id"] = "d07"
    first_branches[1]["second_branches"][2]["retrieved_doc_id"] = "d19"
    first_branches[1]["second_branches"][2]["truth_probability"] = 0.95
    record = {
        "gold_doc_ids": ["d07", "d19"],
        "first_branches": first_branches,
        "replay_indices": [0, 0],
        "replay_truth_probability": 0.2,
    }
    result = analyze_record(record)
    assert result["gold_root_retrieved"]
    assert result["gold_second_retrieved_after_gold_root"]
    assert result["gold_pair_is_oracle"]
    assert result["oracle_first_differs_from_greedy"]
    assert result["nonmyopic_probability_gap"] == pytest.approx(0.75)
