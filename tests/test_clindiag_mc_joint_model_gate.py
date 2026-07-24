from __future__ import annotations

import json

import pytest

from scripts.clindiag_mc_joint_model_gate import (
    OUTCOME_IDS,
    QUERY_IDS,
    parse_candidate_queries,
    parse_gatekeeper_response,
    parse_likelihood_matrix,
    parse_response_audit,
)


def _queries() -> list[dict[str, object]]:
    return [
        {
            "id": query_id,
            "kind": "test",
            "request": f"Order test {query_id}",
            "outcomes": [
                {"id": outcome_id, "label": f"Finding {query_id}-{outcome_id}"}
                for outcome_id in OUTCOME_IDS
            ],
        }
        for query_id in QUERY_IDS
    ]


def test_parse_candidate_queries_requires_fixed_ids_and_options() -> None:
    parsed = parse_candidate_queries(json.dumps({"queries": _queries()}))

    assert [query["id"] for query in parsed] == list(QUERY_IDS)
    assert all(
        [outcome["id"] for outcome in query["outcomes"]] == list(OUTCOME_IDS)
        for query in parsed
    )

    invalid = _queries()
    invalid[0]["outcomes"] = invalid[0]["outcomes"][:3]
    with pytest.raises(ValueError, match="exactly four outcomes"):
        parse_candidate_queries(json.dumps({"queries": invalid}))


def test_parse_gatekeeper_response_checks_schema() -> None:
    query = _queries()[0]
    parsed = parse_gatekeeper_response(
        '{"outcome_id":"B","finding":"Objective result","source":"synthetic"}',
        query,
    )
    assert parsed["outcome_id"] == "B"

    with pytest.raises(ValueError, match="recorded or synthetic"):
        parse_gatekeeper_response(
            '{"outcome_id":"B","finding":"Objective result","source":"unknown"}',
            query,
        )


def test_parse_likelihood_matrix_requires_normalized_rows() -> None:
    hypotheses = [{"id": "h1", "name": "One"}, {"id": "hT", "name": "Truth"}]
    rows = [
        {
            "hypothesis_id": hypothesis["id"],
            "query_id": query_id,
            "probabilities": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
        }
        for hypothesis in hypotheses
        for query_id in QUERY_IDS
    ]
    parsed = parse_likelihood_matrix(json.dumps({"rows": rows}), hypotheses)
    assert len(parsed) == 8

    rows[0]["probabilities"]["D"] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        parse_likelihood_matrix(json.dumps({"rows": rows}), hypotheses)


def test_parse_response_audit_requires_boolean_rows() -> None:
    rows = [
        {
            "id": query_id,
            "relevant": True,
            "objective": True,
            "no_target_leak": True,
            "case_consistent": True,
            "duplicate_semantically_consistent": True,
        }
        for query_id in QUERY_IDS
    ]
    assert len(parse_response_audit(json.dumps({"queries": rows}))) == 4

    rows[0]["objective"] = "yes"
    with pytest.raises(ValueError, match="booleans"):
        parse_response_audit(json.dumps({"queries": rows}))
