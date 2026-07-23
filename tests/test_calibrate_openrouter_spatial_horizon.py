import json

from scripts.calibrate_openrouter_spatial_horizon import (
    CASES,
    messages_for_case,
    parse_tail,
)


def test_spatial_calibration_is_endpoint_free_and_has_unique_reachable_target() -> None:
    for case in CASES:
        prompt = messages_for_case(case)[-1]["content"]

        assert "Rock" not in prompt
        assert "EIG" not in prompt
        assert "posterior" not in prompt
        assert case["fixed_root"] in prompt
        assert case["expected"][-1] in prompt


def test_spatial_calibration_parser_accepts_json_prefix_only() -> None:
    response = json.dumps({"tail": ["move-EAST", "move-EAST", "inspect-A"]})

    assert parse_tail(response + "\ntrailing") == [
        "move-EAST",
        "move-EAST",
        "inspect-A",
    ]
    assert parse_tail('{"tail":["move-EAST"]}') is None
    assert parse_tail("not json") is None
