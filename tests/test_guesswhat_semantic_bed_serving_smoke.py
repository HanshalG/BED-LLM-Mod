from __future__ import annotations

from io import BytesIO
import json

from PIL import Image

from scripts import guesswhat_semantic_bed_serving_smoke as smoke


def _image_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), (190, 200, 210))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _planner_response() -> str:
    return json.dumps(
        {
            "questions": [
                {"id": "Q1", "text": "Is it on the left side?"},
                {"id": "Q2", "text": "Is it on the right side?"},
                {"id": "Q3", "text": "Is it a person?"},
                {"id": "Q4", "text": "Does it touch the ground?"},
            ]
        }
    )


def test_planner_parser_accepts_semantic_yes_no_questions() -> None:
    parsed = smoke.parse_planner_response(_planner_response())
    assert [item["id"] for item in parsed] == ["Q1", "Q2", "Q3", "Q4"]


def test_planner_parser_rejects_overlay_reference_or_duplicate() -> None:
    value = json.loads(_planner_response())
    value["questions"][0]["text"] = "Is it candidate C1?"
    try:
        smoke.parse_planner_response(json.dumps(value))
    except ValueError as exc:
        assert "overlay" in str(exc)
    else:
        raise AssertionError("overlay reference should fail")

    value = json.loads(_planner_response())
    value["questions"][1]["text"] = value["questions"][0]["text"]
    try:
        smoke.parse_planner_response(json.dumps(value))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate question should fail")


def test_likelihood_parser_requires_exact_candidate_order() -> None:
    candidate_ids = ("C1", "C2", "C3", "C4", "C5")
    response = json.dumps(
        {
            "rows": [
                {
                    "question_id": f"Q{question_index}",
                    "candidates": [
                        {
                            "candidate_id": candidate_id,
                            "yes_probability": 10 * candidate_index,
                        }
                        for candidate_index, candidate_id in enumerate(
                            candidate_ids,
                            start=1,
                        )
                    ],
                }
                for question_index in range(1, 5)
            ]
        }
    )
    parsed = smoke.parse_likelihood_response(
        response,
        candidate_ids=candidate_ids,
    )
    assert parsed[0]["probabilities"] == [10, 20, 30, 40, 50]

    value = json.loads(response)
    value["rows"][0]["candidates"][0]["candidate_id"] = "C2"
    try:
        smoke.parse_likelihood_response(
            json.dumps(value),
            candidate_ids=candidate_ids,
        )
    except ValueError as exc:
        assert "ordered" in str(exc)
    else:
        raise AssertionError("misordered candidates should fail")


def test_oracle_parser_is_exact_binary() -> None:
    assert smoke.parse_oracle_response('{"answer":"Yes"}') == "Yes"
    try:
        smoke.parse_oracle_response('{"answer":"Maybe"}')
    except ValueError as exc:
        assert "Yes or No" in str(exc)
    else:
        raise AssertionError("nonbinary answer should fail")


def test_messages_do_not_expose_target_or_categories() -> None:
    case = smoke.load_cases()[0]
    image = Image.new(
        "RGB",
        (case.image_width, case.image_height),
        (255, 255, 255),
    )
    overlay = smoke.candidate_overlay(image, case)
    messages = smoke.planner_messages(case, overlay)
    text = next(
        item["text"]
        for item in messages[-1]["content"]
        if item["type"] == "text"
    )
    assert "target_index" not in text
    assert "category_id" not in text
    assert "object_id" not in text
    assert set(json.loads(text)) == {
        "task",
        "candidate_ids",
        "requirements",
    }


def test_response_schemas_avoid_unique_items() -> None:
    serialized = smoke.source.canonical_json(
        {
            "planner": smoke.planner_response_format(),
            "likelihood": smoke.likelihood_response_format(),
            "oracle": smoke.oracle_response_format(),
        }
    )
    assert "uniqueItems" not in serialized


def test_exact_ten_call_fixture_passes(tmp_path) -> None:
    cases = smoke.load_cases()
    planner, likelihood, oracle = smoke.fixture_models(cases)
    payload = smoke.run_smoke(
        cases=cases,
        planner_model=planner,
        likelihood_model=likelihood,
        oracle_model=oracle,
        raw_path=tmp_path / "RAW_RESPONSES.json",
        image_loader=lambda case: _image_bytes(
            case.image_width,
            case.image_height,
        ),
    )
    assert payload["status"] == "passed"
    assert payload["usage"]["adapter_requests"] == smoke.EXPECTED_REQUESTS
    assert payload["metrics"]["cross_model_consistent_count"] == 6
    assert payload["gates"]["all_pass"]
    assert all(
        "target_overlay_sha256" not in case for case in payload["cases"]
    )
    assert "target_index" not in json.dumps(payload)


def test_source_manifest_and_serving_cases_are_bound() -> None:
    assert (
        smoke.sha256_file(smoke.SOURCE_MANIFEST_PATH)
        == smoke.SOURCE_AUDIT_SHA256
    )
    cases = smoke.load_cases()
    assert len(cases) == smoke.NUM_CASES
    assert len({case.picture_id for case in cases}) == smoke.NUM_CASES
    assert all(
        smoke.source.MIN_OBJECTS
        <= len(case.objects)
        <= smoke.source.MAX_OBJECTS
        for case in cases
    )
