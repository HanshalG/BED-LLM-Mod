import json

import pytest

from environments.gated_sensor import (
    GatedSensorModel,
    GatedStrategyParseError,
    parse_gated_strategy_cell,
    score_gated_strategy_exact,
)


def _cell(items: list[dict[str, object]]) -> str:
    return json.dumps({"strategies": items})


def _strategy(root: str, followups: dict[str, str], index: int = 0) -> dict[str, object]:
    return {
        "name": f"strategy-{index}",
        "description": "A complete branch policy used for exact mechanics testing.",
        "root_action": root,
        "followups": followups,
    }


def test_parser_enforces_assigned_activation_roots_and_precise_followups() -> None:
    model = GatedSensorModel()
    response = _cell(
        [
            _strategy("activate:A", {"none": "precise:bit-0"}, 0),
            _strategy("activate:B", {"none": "precise:bit-3"}, 1),
            _strategy(
                "screen:bit-0",
                {"positive": "activate:C", "negative": "screen:bit-1"},
                2,
            ),
        ]
    )
    parsed = parse_gated_strategy_cell(
        response,
        model=model,
        state=model.initial_state,
        horizon=2,
        expected_count=3,
        required_activation_roots=("activate:A", "activate:B"),
    )
    assert [strategy.root_action for strategy in parsed] == [
        "activate:A",
        "activate:B",
        "screen:bit-0",
    ]

    invalid = _cell([_strategy("activate:A", {"none": "screen:bit-0"})])
    with pytest.raises(GatedStrategyParseError, match="precise"):
        parse_gated_strategy_cell(
            invalid,
            model=model,
            state=model.initial_state,
            horizon=2,
            expected_count=1,
            required_activation_roots=("activate:A",),
        )


def test_exact_branch_score_matches_manual_two_step_eig() -> None:
    model = GatedSensorModel()
    parsed = parse_gated_strategy_cell(
        _cell([_strategy("activate:A", {"none": "precise:bit-0"})]),
        model=model,
        state=model.initial_state,
        horizon=2,
        expected_count=1,
        required_activation_roots=("activate:A",),
    )[0]
    score = score_gated_strategy_exact(
        model,
        parsed,
        state=model.initial_state,
        belief=model.initial_belief,
        horizon=2,
    )

    assert score.eig == pytest.approx(model.expected_information_gain(model.initial_belief, "precise:bit-0"))
    assert score.start_entropy - score.expected_final_entropy == pytest.approx(score.eig)
    assert score.expanded_decision_nodes == 2
    assert score.leaf_nodes == 2


def test_horizon_one_requires_empty_followups() -> None:
    model = GatedSensorModel()
    parsed = parse_gated_strategy_cell(
        _cell([_strategy("screen:bit-0", {})]),
        model=model,
        state=model.initial_state,
        horizon=1,
        expected_count=1,
    )
    assert parsed[0].followups == {}

    with pytest.raises(GatedStrategyParseError, match="exactly"):
        parse_gated_strategy_cell(
            _cell([_strategy("screen:bit-0", {"positive": "screen:bit-1"})]),
            model=model,
            state=model.initial_state,
            horizon=1,
            expected_count=1,
        )
