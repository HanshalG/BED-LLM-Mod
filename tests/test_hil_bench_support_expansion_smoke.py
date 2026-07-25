import json

import pytest

from scripts.hil_bench_support_expansion_smoke import (
    SUPPORT_SIZE,
    analyze,
    parse_initial,
    parse_judgment,
    parse_refresh,
)


def _support(prefix: str):
    return [
        {
            "hypothesis": f"{prefix} ambiguity {index} needs a database mapping",
            "question": f"Which {prefix} mapping {index} should be used?",
        }
        for index in range(SUPPORT_SIZE)
    ]


def test_parse_initial_requires_exact_support_and_queries():
    payload = {
        "hypotheses": _support("initial"),
        "business_search_queries": ["first business query", "second business query"],
    }
    parsed = parse_initial(json.dumps(payload))
    assert len(parsed["hypotheses"]) == SUPPORT_SIZE
    assert len(parsed["business_search_queries"]) == 2
    payload["hypotheses"][1]["question"] = payload["hypotheses"][0]["question"]
    with pytest.raises(ValueError, match="distinct"):
        parse_initial(json.dumps(payload))


def test_parse_refresh_rejects_multi_question_rows():
    payload = {"hypotheses": _support("refresh")}
    payload["hypotheses"][0]["question"] = "Which value? Which column?"
    with pytest.raises(ValueError, match="exactly one"):
        parse_refresh(json.dumps(payload))


def test_parse_judgment_requires_every_candidate_once():
    response = json.dumps(
        {
            "matches": [
                {"candidate_id": "a", "blocker_id": "blocker_a"},
                {"candidate_id": "b", "blocker_id": None},
            ]
        }
    )
    assert parse_judgment(response, {"a", "b"}, {"blocker_a"}) == {
        "a": "blocker_a",
        "b": None,
    }
    with pytest.raises(ValueError, match="every"):
        parse_judgment(
            json.dumps(
                {"matches": [{"candidate_id": "a", "blocker_id": "blocker_a"}]}
            ),
            {"a", "b"},
            {"blocker_a"},
        )


def test_analysis_requires_new_blocker_beyond_no_evidence_control():
    registry = [
        {"id": "question_a", "type": "question"},
        {"id": "business_b", "type": "business info"},
        {"id": "schema_c", "type": "schema"},
    ]
    matches = {
        **{f"initial_{index}": None for index in range(4)},
        **{f"control_{index}": None for index in range(4)},
        **{f"branch_0_{index}": None for index in range(4)},
        **{f"branch_1_{index}": None for index in range(4)},
    }
    matches["initial_0"] = "question_a"
    matches["control_0"] = "question_a"
    matches["branch_0_0"] = "question_a"
    matches["branch_0_1"] = "business_b"
    result = analyze(
        [
            {
                "task_id": "sql_fixture",
                "registry": registry,
                "matches": matches,
            }
        ]
    )
    assert result["efficacy"]["tasks_with_evidence_gain_over_initial_and_control"] == 1
    assert result["efficacy"]["recovered_business_blockers"] == 1
