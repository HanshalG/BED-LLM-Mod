import json

import pytest

from scripts.audit_nonmyopic_range_gated_rock_depth5_goal_compiler import (
    audit_result,
)
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_range_gated_rock_depth5_goal_compiler import (
    Depth5GoalCompilerConfig,
    Depth5GoalCompilerProvider,
    DeterministicDepth5GoalModel,
    RangeGatedDepth5ProposalConfig,
    build_depth5_model,
    compile_goal_cell,
    compile_target_plan,
    run_proposal_gate,
    run_smoke,
)
from scripts.nonmyopic_range_gated_rock_depth5_oracle import H5_ROUTE
from scripts.nonmyopic_range_gated_rock_fixed_tail import fixed_roots


def test_target_compiler_recovers_critical_h5_route() -> None:
    model = build_depth5_model()
    position = model.map_spec.start_position
    roots = fixed_roots(model, position=position, belief=model.initial_belief)

    plan = compile_target_plan(
        model,
        position=position,
        root=roots[0],
        target_rock=6,
    )

    assert plan == H5_ROUTE


def test_goal_response_requires_distinct_valid_targets() -> None:
    model = build_depth5_model()
    position = model.map_spec.start_position
    roots = fixed_roots(model, position=position, belief=model.initial_belief)
    config = Depth5GoalCompilerConfig()

    targets, plans = compile_goal_cell(
        json.dumps({"r0": 6, "r1": 7, "r2": 0, "r3": 1}),
        model=model,
        position=position,
        roots=roots,
        config=config,
    )

    assert targets == (6, 7, 0, 1)
    assert plans[0] == H5_ROUTE
    with pytest.raises(StrategyProposalError, match="distinct"):
        compile_goal_cell(
            json.dumps({"r0": 6, "r1": 6, "r2": 0, "r3": 1}),
            model=model,
            position=position,
            roots=roots,
            config=config,
        )
    with pytest.raises(StrategyProposalError, match="out of range"):
        compile_goal_cell(
            json.dumps({"r0": 8, "r1": 7, "r2": 0, "r3": 1}),
            model=model,
            position=position,
            roots=roots,
            config=config,
        )


def test_goal_prompt_exposes_geometry_but_no_route_or_utility_answer() -> None:
    model = build_depth5_model()
    position = model.map_spec.start_position
    provider = Depth5GoalCompilerProvider(
        DeterministicDepth5GoalModel(),
        Depth5GoalCompilerConfig(),
    )
    roots = fixed_roots(model, position=position, belief=model.initial_belief)
    prompt = provider._messages(
        model,
        position=position,
        belief=model.initial_belief,
        history=(),
        roots=roots,
    )[-1]["content"]

    assert '"fixed_root":"move-NORTH"' in prompt
    assert '"transit_slots_after_root":3' in prompt
    assert '"movement_distance_after_root":3' in prompt
    assert "shortest-path" in prompt
    assert "expected_information_gain" not in prompt
    assert '"score"' not in prompt
    assert "preferred target" not in prompt.lower()
    assert "move-NORTH,move-WEST,move-WEST,move-WEST,check-6" not in prompt


def test_goal_compiler_config_is_frozen() -> None:
    Depth5GoalCompilerConfig().validate()
    RangeGatedDepth5ProposalConfig().validate()

    with pytest.raises(ValueError, match="horizon five"):
        Depth5GoalCompilerConfig(horizon=4).validate()
    with pytest.raises(ValueError, match="128-token"):
        Depth5GoalCompilerConfig(max_new_tokens=256).validate()
    with pytest.raises(ValueError, match="16 cells"):
        RangeGatedDepth5ProposalConfig(num_cells=10).validate()


def test_deterministic_goal_smoke_passes() -> None:
    config = Depth5GoalCompilerConfig(seed=24_239)
    provider = Depth5GoalCompilerProvider(
        DeterministicDepth5GoalModel(), config
    )

    result = run_smoke(provider, config, cell_concurrency=4)

    assert all(result["mechanics"].values())
    assert result["critical_target_count"] == 10
    assert result["critical_route_count"] == 10
    assert result["exact_plan_match_count"] == 10


def test_deterministic_goal_proposal_and_independent_audit_pass() -> None:
    strategy_config = Depth5GoalCompilerConfig(seed=24_240)
    gate_config = RangeGatedDepth5ProposalConfig(
        seed=24_240,
        bootstrap_seed=24_241,
    )
    provider = Depth5GoalCompilerProvider(
        DeterministicDepth5GoalModel(), strategy_config
    )

    result = run_proposal_gate(
        provider,
        strategy_config,
        gate_config,
        cell_concurrency=4,
    )
    result["usage"] = {}
    result["mechanics"]["usage_accounted"] = True
    result["gate"] = {
        "passed": all(result["mechanics"].values())
        and all(result["endpoint_gate"].values())
    }
    audit = audit_result(result)

    assert result["gate"]["passed"]
    assert result["comparisons"]["exact_h5_route_selection_rate"] == 1.0
    assert result["comparisons"]["recovery_fraction"]["mean"] == pytest.approx(
        1.0
    )
    assert audit["gate"]["passed"]
