from __future__ import annotations

import json

import pytest

from scripts import cupid_active_preference_serving_smoke as v1
from scripts import cupid_active_preference_serving_smoke_v2 as v2


def _v2_planner_response() -> str:
    fixture = v2.DeterministicFixtureModel("planner")
    return fixture.chat_complete_messages_batched_structured(
        [[{"role": "user", "content": '{"case_id":"fixture"}'}]],
        temperature=0.7,
        block_size=1,
        response_format=v2.planner_response_format(),
        max_new_tokens=6000,
    )[0]


def test_v2_parses_integer_arrays_into_canonical_signatures() -> None:
    planner = v2.parse_planner_response(_v2_planner_response())
    assert planner["hypotheses"][0]["answer_signature"] == "010101"
    assert (
        v2.parse_target_response('{"answer_signature":[0,1,0,1,0,1]}')
        == "010101"
    )


def test_v2_rejects_boolean_or_nonbinary_signature_items() -> None:
    with pytest.raises(ValueError, match="integer 0 or 1"):
        v2.parse_target_response(
            '{"answer_signature":[true,false,true,false,true,false]}'
        )
    with pytest.raises(ValueError, match="integer 0 or 1"):
        v2.parse_target_response('{"answer_signature":[0,1,2,1,0,1]}')


def test_v2_cases_are_disjoint_manifest_development_rows() -> None:
    rows = v2.v2_serving_rows()
    v1_ids = {v1.source.row_id(row) for row in v1.serving_rows()}

    assert [v1.source.row_id(row) for row in rows] == list(v2.V2_CASE_IDS)
    assert not (set(v2.V2_CASE_IDS) & v1_ids)
    assert [row["instance_type"] for row in rows] == [
        "consistent",
        "consistent",
        "contrastive",
        "contrastive",
        "changing",
    ]


def test_v2_fixture_passes_exact_ten_call_gate(tmp_path) -> None:
    payload = v2.run_smoke(
        rows=v2.v2_serving_rows(),
        planner_model=v2.DeterministicFixtureModel("planner"),
        target_model=v2.DeterministicFixtureModel("target"),
        raw_path=tmp_path / "private" / "RAW_RESPONSES.json",
    )

    assert payload["status"] == "passed"
    assert payload["usage"]["adapter_requests"] == 10
    assert payload["metrics"]["exact_target_coverage_cases"] == 5
    assert payload["gates"]["all_pass"]


def test_v2_schema_uses_integer_arrays_without_unique_items() -> None:
    planner = json.dumps(v2.planner_response_format(), sort_keys=True)
    target = json.dumps(v2.target_response_format(), sort_keys=True)

    assert '"type": "integer"' in planner
    assert '"type": "integer"' in target
    assert "uniqueItems" not in planner
    assert "uniqueItems" not in target
