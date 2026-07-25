import json

import pytest

from scripts.tau_knowledge_retrieval_opportunity_v2 import (
    EXPECTED_REQUESTS,
    OPPORTUNITY_IDS,
    SMOKE_IDS,
    initial_messages,
    opening_messages,
    parse_followup,
    parse_initial,
    parse_opening,
)


def _flat_initial():
    payload = {f"need_{index + 1}": f"need {index}" for index in range(8)}
    payload.update(
        {f"direct_query_{index + 1}": f"direct {index}" for index in range(2)}
    )
    payload.update(
        {
            f"enabling_query_{index + 1}": f"enabling {index}"
            for index in range(3)
        }
    )
    return payload


def test_flat_initial_schema_parses_without_arrays():
    hypotheses, queries = parse_initial(json.dumps(_flat_initial()))
    assert len(hypotheses) == 8
    assert len(queries) == 5


def test_flat_initial_rejects_extra_key():
    payload = _flat_initial()
    payload["information_need_hypotheses"] = []
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_initial(json.dumps(payload))


def test_flat_followup_rejects_repeated_first_query():
    payload = {f"need_{index + 1}": f"need {index}" for index in range(8)}
    payload.update(
        {
            f"followup_query_{index + 1}": (
                "first query" if index == 0 else f"followup {index}"
            )
            for index in range(4)
        }
    )
    with pytest.raises(ValueError, match="repeats"):
        parse_followup(json.dumps(payload), first_query="first query")


def test_opening_parser_is_bounded():
    assert parse_opening('{"opening":"I need help."}') == "I need help."
    with pytest.raises(ValueError, match="invalid length"):
        parse_opening(json.dumps({"opening": "x" * 1201}))


def test_v2_request_counts_include_opening_generation():
    assert EXPECTED_REQUESTS["serving_smoke"] == len(SMOKE_IDS) * 7
    assert EXPECTED_REQUESTS["opportunity"] == len(OPPORTUNITY_IDS) * 7


def test_private_script_is_confined_to_opening_generation():
    private_marker = "PRIVATE_LATER_STEP_9f8cb"
    opening_prompt = json.dumps(opening_messages(private_marker))
    policy_prompt = json.dumps(initial_messages("I need help with my card."))
    assert private_marker in opening_prompt
    assert private_marker not in policy_prompt
