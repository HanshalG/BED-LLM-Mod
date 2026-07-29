from __future__ import annotations

import json

import pytest

from scripts import cupid_active_preference_serving_smoke as smoke
from scripts import cupid_active_preference_source_audit as source


def _planner_payload() -> dict:
    patterns = (
        "010101",
        "101010",
        "001111",
        "110000",
        "011001",
        "100110",
        "000111",
        "111000",
        "010110",
        "101001",
        "011010",
        "100101",
    )
    return {
        "questions": [
            {"id": f"Q{index}", "question": f"Would feature {index} be desired?"}
            for index in range(1, 7)
        ],
        "hypotheses": [
            {
                "id": f"H{index}",
                "preference": f"Distinct preference hypothesis {index}",
                "answer_signature": patterns[index - 1],
            }
            for index in range(1, 13)
        ],
    }


def test_parse_planner_and_target_strictly() -> None:
    parsed = smoke.parse_planner_response(json.dumps(_planner_payload()))
    assert len(parsed["questions"]) == 6
    assert len(parsed["hypotheses"]) == 12
    assert smoke.parse_target_response('{"answer_signature":"010101"}') == "010101"

    duplicate = _planner_payload()
    duplicate["questions"][1]["question"] = duplicate["questions"][0]["question"]
    with pytest.raises(ValueError, match="not unique"):
        smoke.parse_planner_response(json.dumps(duplicate))
    with pytest.raises(ValueError, match="six bits"):
        smoke.parse_target_response('{"answer_signature":"yes"}')


def test_planner_payload_has_no_hidden_fields() -> None:
    row = source.load_rows()[0]
    request = json.loads(smoke.planner_messages(row)[-1]["content"])
    serialized = source.canonical_json(request)

    assert "hidden_contextual_preference" not in serialized
    assert "current_checklist" not in serialized
    assert "contextual_preference" not in serialized
    assert request["observation"] == source.candidate_payload(row)


def test_target_payload_uses_preference_but_not_checklist() -> None:
    row = source.load_rows()[0]
    questions = [f"Would preference feature {index} apply?" for index in range(6)]
    request = json.loads(smoke.target_messages(row, questions)[-1]["content"])

    assert request["hidden_contextual_preference"] == row[
        "current_contextual_preference"
    ]
    assert "current_checklist" not in request
    assert "prior_interactions" not in request


def test_case_metrics_match_manual_signature_distance() -> None:
    row = source.load_rows()[0]
    planner = smoke.parse_planner_response(json.dumps(_planner_payload()))
    metrics = smoke.case_metrics(
        row=row,
        planner=planner,
        target_signature="010100",
    )

    assert metrics["unique_hypothesis_signature_count"] == 12
    assert not metrics["target_signature_exactly_covered"]
    assert metrics["target_signature_nearest_hamming"] == 1
    assert min(metrics["partition_minority_counts"]) >= 2


def test_fixture_smoke_passes_exact_ten_call_gate(tmp_path) -> None:
    payload = smoke.run_smoke(
        rows=smoke.serving_rows(),
        planner_model=smoke.DeterministicFixtureModel("planner"),
        target_model=smoke.DeterministicFixtureModel("target"),
        raw_path=tmp_path / "private" / "RAW_RESPONSES.json",
    )

    assert payload["status"] == "passed"
    assert payload["usage"]["adapter_requests"] == 10
    assert payload["metrics"]["exact_target_coverage_cases"] == 5
    assert payload["gates"]["all_pass"]
    assert all(
        set(case)
        == {
            "id",
            "instance_type",
            "question_count",
            "unique_question_count",
            "hypothesis_count",
            "unique_hypothesis_count",
            "unique_hypothesis_signature_count",
            "partition_minority_counts",
            "partition_entropies_nats",
            "mean_partition_entropy_nats",
            "target_signature_exactly_covered",
            "target_signature_nearest_hamming",
        }
        for case in payload["cases"]
    )
