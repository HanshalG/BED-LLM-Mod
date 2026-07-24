from __future__ import annotations

import json

import pytest

from scripts.clindiag_binary_joint_model_gate import (
    QUERY_IDS,
    parse_binary_answer,
    parse_binary_audit,
    parse_binary_likelihoods,
    parse_binary_queries,
)


def _queries() -> list[dict[str, str]]:
    return [
        {
            "id": query_id,
            "kind": "test",
            "request": f"Is marker {index} elevated?",
            "positive_finding": f"Marker {index} is elevated",
            "negative_finding": f"Marker {index} is not elevated",
        }
        for index, query_id in enumerate(QUERY_IDS, start=1)
    ]


def test_binary_query_parser_accepts_fixed_safe_grammar() -> None:
    parsed = parse_binary_queries(json.dumps({"queries": _queries()}))
    assert [item["id"] for item in parsed] == list(QUERY_IDS)


@pytest.mark.parametrize(
    "action_text",
    (
        "Obtain a skin biopsy",
        "Order genetic sequencing",
        "Perform surgical resection",
    ),
)
def test_binary_query_parser_rejects_confirmatory_actions(
    action_text: str,
) -> None:
    rows = _queries()
    rows[0]["request"] = action_text
    with pytest.raises(ValueError, match="blocked"):
        parse_binary_queries(json.dumps({"queries": rows}))


def test_binary_answer_parser_rejects_missingness() -> None:
    assert parse_binary_answer(
        '{"answer":"yes","finding":"Fever recurs monthly","source":"recorded"}'
    )["answer"] == "yes"
    with pytest.raises(ValueError, match="missingness"):
        parse_binary_answer(
            '{"answer":"no","finding":"Not recorded","source":"recorded"}'
        )


def test_binary_likelihood_parser_checks_ids_and_probability() -> None:
    hypotheses = [{"id": "h1", "name": "One"}, {"id": "hT", "name": "Truth"}]
    rows = [
        {
            "hypothesis_id": hypothesis["id"],
            "query_id": query_id,
            "p_yes": 0.5,
        }
        for hypothesis in hypotheses
        for query_id in QUERY_IDS
    ]
    assert len(
        parse_binary_likelihoods(json.dumps({"rows": rows}), hypotheses)
    ) == 12
    rows[0]["p_yes"] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        parse_binary_likelihoods(json.dumps({"rows": rows}), hypotheses)


def test_binary_audit_parser_requires_boolean_rows() -> None:
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
    assert len(parse_binary_audit(json.dumps({"queries": rows}))) == 6
    rows[0]["objective"] = "true"
    with pytest.raises(ValueError, match="booleans"):
        parse_binary_audit(json.dumps({"queries": rows}))
