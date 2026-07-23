import json

import pytest

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_range_gated_rock_fixed_tail import (
    DeterministicFixedTailModel,
    FixedRootTailProvider,
    compile_fixed_tail_cell,
    fixed_roots,
    run_smoke,
)
from scripts.nonmyopic_range_gated_rock_strategy import RangeGatedStrategyConfig


def _model() -> RangeGatedRockDiagnosisModel:
    return RangeGatedRockDiagnosisModel(
        get_paper_map("7-8"), remote_accuracy=0.55, onsite_accuracy=0.95
    )


def test_fixed_roots_include_delayed_route_and_two_myopic_checks() -> None:
    model = _model()
    roots = fixed_roots(
        model,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
    )

    assert roots[0] == "move-SOUTH"
    assert sum(model.is_move(root) for root in roots) == 2
    assert len(roots) == len(set(roots)) == 4


def test_fixed_tail_compiler_preserves_roots_and_checks_dynamic_legality() -> None:
    model = _model()
    position = model.map_spec.start_position
    roots = fixed_roots(model, position=position, belief=model.initial_belief)
    response = json.dumps(
        {
            "r0": ["move-SOUTH", "check-5"],
            "r1": ["move-EAST", "check-2"],
            "r2": ["check-1", "check-2"],
            "r3": ["check-2", "check-3"],
        }
    )
    plans = compile_fixed_tail_cell(
        response,
        model=model,
        position=position,
        roots=roots,
        config=RangeGatedStrategyConfig(),
    )

    assert plans[0] == ("move-SOUTH", "move-SOUTH", "check-5")
    assert tuple(plan[0] for plan in plans) == roots

    invalid = json.dumps(
        {
            "r0": ["move-WEST", "check-5"],
            "r1": ["move-EAST", "check-2"],
            "r2": ["check-1", "check-2"],
            "r3": ["check-2", "check-3"],
        }
    )
    with pytest.raises(StrategyProposalError, match="illegal action"):
        compile_fixed_tail_cell(
            invalid,
            model=model,
            position=position,
            roots=roots,
            config=RangeGatedStrategyConfig(),
        )


def test_fixed_tail_prompt_exposes_geometry_but_no_scores_or_answer() -> None:
    model = _model()
    config = RangeGatedStrategyConfig()
    provider = FixedRootTailProvider(DeterministicFixedTailModel(), config)
    roots = fixed_roots(
        model,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
    )
    messages = provider._messages(
        model,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
        roots=roots,
    )
    prompt = messages[-1]["content"]

    assert "position_after_root" in prompt
    assert "remote_accuracy" in prompt
    assert "onsite_accuracy" in prompt
    assert '"fixed_root":"move-SOUTH"' in prompt
    assert "expected_information_gain" not in prompt
    assert '"score"' not in prompt
    assert "preferred" not in prompt
    assert "move-SOUTH\",\"check-5" not in prompt


def test_successor_grounding_lists_transitions_without_utility_or_preference() -> None:
    model = _model()
    config = RangeGatedStrategyConfig()
    provider = FixedRootTailProvider(
        DeterministicFixedTailModel(),
        config,
        include_successor_grounding=True,
    )
    roots = fixed_roots(
        model,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
    )
    messages = provider._messages(
        model,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
        roots=roots,
    )
    prompt = messages[-1]["content"]

    assert "second_action_successors" in prompt
    assert '"action":"move-SOUTH","position_after_action":[0,5]' in prompt
    assert "expected_information_gain" not in prompt
    assert '"score"' not in prompt
    assert "preferred" not in prompt
    assert "move-SOUTH\",\"check-5" not in prompt


def test_deterministic_fixed_tail_smoke_covers_delayed_route() -> None:
    config = RangeGatedStrategyConfig(seed=24_177)
    provider = FixedRootTailProvider(DeterministicFixedTailModel(), config)
    result = run_smoke(provider, config)

    assert all(result["mechanics"].values())
    assert result["delayed_onsite_route_count"] == 10
    assert result["provider"] == {"accepted_requests": 10, "invalid_responses": 0}
