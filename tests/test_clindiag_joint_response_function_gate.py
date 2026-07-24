from __future__ import annotations

import json

import pytest

from scripts.clindiag_joint_response_function_gate import (
    AUDIT_FIELDS,
    QUERY_IDS,
    parse_joint_audit,
    parse_joint_response,
)


def _answers() -> list[dict[str, str]]:
    return [
        {
            "query_id": query_id,
            "answer": "yes" if index % 2 else "no",
            "finding": f"Objective patient fact {index}",
            "source": "synthetic",
        }
        for index, query_id in enumerate(QUERY_IDS, start=1)
    ]


def test_joint_response_parser_accepts_complete_response_function() -> None:
    parsed = parse_joint_response(json.dumps({"answers": _answers()}))
    assert [row["query_id"] for row in parsed] == list(QUERY_IDS)


@pytest.mark.parametrize(
    "finding",
    (
        "This was not reported in the chart",
        "No documentation exists",
        "The result is absent from the record",
        "The assay was not measured",
    ),
)
def test_joint_response_parser_rejects_chart_missingness(finding: str) -> None:
    rows = _answers()
    rows[0]["finding"] = finding
    with pytest.raises(ValueError, match="missingness"):
        parse_joint_response(json.dumps({"answers": rows}))


def test_joint_response_parser_requires_fixed_query_order() -> None:
    rows = _answers()
    rows[0]["query_id"] = "q2"
    with pytest.raises(ValueError, match="IDs or order"):
        parse_joint_response(json.dumps({"answers": rows}))


def test_joint_audit_parser_requires_every_boolean() -> None:
    rows = [
        {"id": query_id, **{field: True for field in AUDIT_FIELDS}}
        for query_id in QUERY_IDS
    ]
    assert len(parse_joint_audit(json.dumps({"queries": rows}))) == 6
    rows[0]["provenance_consistent"] = "true"
    with pytest.raises(ValueError, match="booleans"):
        parse_joint_audit(json.dumps({"queries": rows}))
