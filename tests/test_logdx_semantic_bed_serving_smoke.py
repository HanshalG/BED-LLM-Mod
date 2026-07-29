from __future__ import annotations

import json

from scripts import logdx_semantic_bed_serving_smoke as smoke


def _planner_response() -> str:
    return json.dumps(
        {
            "hypotheses": [
                {
                    "id": f"H{index}",
                    "description": f"Distinct root cause hypothesis number {index}",
                }
                for index in range(1, smoke.NUM_HYPOTHESES + 1)
            ],
            "candidate_greps": [
                {
                    "id": f"Q{index}",
                    "pattern": f"specific_signal_{index}",
                }
                for index in range(1, smoke.NUM_FIRST_QUERIES + 1)
            ],
        }
    )


def _updater_response() -> str:
    return json.dumps(
        {
            "likelihoods": [
                {"id": f"H{index}", "score": score}
                for index, score in enumerate(
                    (90, 70, 50, 30, 10, 0),
                    start=1,
                )
            ],
            "followups": [
                {
                    "id": "F1",
                    "tool": "view_log_lines",
                    "argument": "842",
                },
                {
                    "id": "F2",
                    "tool": "grep",
                    "argument": "specific_test_name",
                },
                {
                    "id": "F3",
                    "tool": "grep",
                    "argument": "specific_file.py",
                },
            ],
        }
    )


def test_planner_parser_accepts_exact_unique_support_and_queries() -> None:
    parsed = smoke.parse_planner_response(_planner_response())
    assert len(parsed["hypotheses"]) == smoke.NUM_HYPOTHESES
    assert len(parsed["candidate_greps"]) == smoke.NUM_FIRST_QUERIES


def test_planner_parser_rejects_generic_or_duplicate_query() -> None:
    value = json.loads(_planner_response())
    value["candidate_greps"][0]["pattern"] = "error|failed|traceback"
    try:
        smoke.parse_planner_response(json.dumps(value))
    except ValueError as exc:
        assert "generic" in str(exc)
    else:
        raise AssertionError("generic query should fail")


def test_updater_parser_and_followup_codec() -> None:
    parsed = smoke.parse_updater_response(_updater_response())
    assert [item["score"] for item in parsed["likelihoods"]] == [
        90,
        70,
        50,
        30,
        10,
        0,
    ]
    assert smoke.followup_to_call(parsed["followups"][0]) == {
        "tool": "view_log_lines",
        "args": {"center_line": 842, "radius": 30},
    }


def test_planner_messages_exclude_ground_truth() -> None:
    case = smoke.load_cases()[0]
    payload = json.loads(smoke.planner_messages(case)[-1]["content"])
    assert set(payload) == {
        "case_id",
        "safe_metadata",
        "reduced_context_method",
        "reduced_context",
    }
    assert "ground_truth" not in smoke.source.canonical_json(payload)
    assert "failure_category" not in smoke.source.canonical_json(payload)


def test_response_schemas_avoid_unique_items() -> None:
    assert "uniqueItems" not in smoke.source.canonical_json(
        smoke.planner_response_format()
    )
    assert "uniqueItems" not in smoke.source.canonical_json(
        smoke.updater_response_format()
    )


def test_exact_ten_call_fixture_passes(tmp_path) -> None:
    payload = smoke.run_smoke(
        cases=smoke.load_cases(),
        planner_model=smoke.DeterministicFixtureModel("planner"),
        updater_model=smoke.DeterministicFixtureModel("updater"),
        raw_path=tmp_path / "RAW_RESPONSES.json",
    )
    assert payload["status"] == "passed"
    assert payload["usage"]["adapter_requests"] == smoke.EXPECTED_REQUESTS
    assert payload["metrics"]["dependency_case_count"] == smoke.NUM_CASES
    assert payload["gates"]["all_pass"]
