from __future__ import annotations

import json

from scripts import guessinggame_path_bed_serving_smoke as smoke


def _response() -> str:
    return json.dumps(
        {
            "hypotheses": [
                {"object_id": index, "weight": 100 - 2 * index}
                for index in range(smoke.SUPPORT_SIZE)
            ]
        }
    )


def test_parser_accepts_exact_unique_weighted_support() -> None:
    hypotheses = smoke.parse_response(
        _response(),
        vocabulary_size=smoke.source.EXPECTED_OBJECTS,
    )
    assert len(hypotheses) == smoke.SUPPORT_SIZE
    assert hypotheses[0] == smoke.Hypothesis(object_id=0, weight=100)


def test_parser_rejects_duplicate_or_out_of_range_id() -> None:
    duplicate = json.loads(_response())
    duplicate["hypotheses"][1]["object_id"] = 0
    try:
        smoke.parse_response(
            json.dumps(duplicate),
            vocabulary_size=smoke.source.EXPECTED_OBJECTS,
        )
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate object id should fail")

    invalid = json.loads(_response())
    invalid["hypotheses"][0]["object_id"] = smoke.source.EXPECTED_OBJECTS
    try:
        smoke.parse_response(
            json.dumps(invalid),
            vocabulary_size=smoke.source.EXPECTED_OBJECTS,
        )
    except ValueError as exc:
        assert "invalid object_id" in str(exc)
    else:
        raise AssertionError("out-of-range object id should fail")


def test_schema_avoids_unique_items() -> None:
    assert "uniqueItems" not in json.dumps(smoke.response_format())


def test_model_message_hides_target_marker_and_unasked_answer() -> None:
    vocabulary, cases = smoke.load_cases()
    case = cases[0]
    messages = smoke.retrieval_messages(
        vocabulary=vocabulary,
        case=case,
        action="material",
    )
    text = messages[-1]["content"]
    request = json.loads(text)
    assert set(request) == {
        "task",
        "prior",
        "observation_history",
        "object_vocabulary",
        "requirements",
    }
    assert "target_object_id" not in text
    assert case.function_answer not in text
    assert request["observation_history"] == [
        {
            "question": case.material_question,
            "answer": case.material_answer,
        }
    ]


def test_exact_ten_call_fixture_passes(tmp_path) -> None:
    vocabulary, cases = smoke.load_cases()
    payload = smoke.run_smoke(
        vocabulary=vocabulary,
        cases=cases,
        model=smoke.fixture_model(cases),
        raw_path=tmp_path / "RAW_RESPONSES.json",
    )
    assert payload["status"] == "passed"
    assert payload["usage"]["adapter_requests"] == smoke.EXPECTED_REQUESTS
    assert payload["metrics"]["material_target_recall"] == smoke.NUM_CASES
    assert payload["metrics"]["function_target_recall"] == smoke.NUM_CASES
    assert payload["gates"]["all_pass"]
    assert "target_object_id" not in json.dumps(payload)


def test_source_manifest_and_cases_are_bound() -> None:
    assert (
        smoke.sha256_file(smoke.SOURCE_MANIFEST_PATH)
        == smoke.SOURCE_MANIFEST_SHA256
    )
    vocabulary, cases = smoke.load_cases()
    assert len(vocabulary) == smoke.source.EXPECTED_OBJECTS
    assert len(cases) == smoke.NUM_CASES
    assert len({case.case_id for case in cases}) == smoke.NUM_CASES
    assert all(
        0 <= case.target_object_id < len(vocabulary) for case in cases
    )
