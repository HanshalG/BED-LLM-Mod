import json

import pytest

from scripts.audit_nonmyopic_range_gated_rock_depth4_goal_compiler import (
    audit_result,
)
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_range_gated_rock_depth4_fixed_tail import (
    CRITICAL_ROUTE,
    RangeGatedDepth4ProposalConfig,
)
from scripts.nonmyopic_range_gated_rock_depth4_goal_compiler import (
    Depth4GoalCompilerConfig,
    Depth4GoalCompilerProvider,
    DeterministicDepth4GoalModel,
    build_depth4_model,
    compile_goal_cell,
    compile_target_plan,
    run_proposal_gate,
    run_smoke,
)
from scripts.nonmyopic_range_gated_rock_fixed_tail import fixed_roots


def test_target_compiler_recovers_critical_route_without_llm_actions() -> None:
    model = build_depth4_model()
    position = model.map_spec.start_position
    roots = fixed_roots(model, position=position, belief=model.initial_belief)

    plan = compile_target_plan(
        model,
        position=position,
        root=roots[0],
        target_rock=4,
    )

    assert plan == CRITICAL_ROUTE


def test_goal_response_requires_distinct_valid_targets() -> None:
    model = build_depth4_model()
    position = model.map_spec.start_position
    roots = fixed_roots(model, position=position, belief=model.initial_belief)
    config = Depth4GoalCompilerConfig()

    targets, plans = compile_goal_cell(
        json.dumps({"r0": 4, "r1": 6, "r2": 0, "r3": 1}),
        model=model,
        position=position,
        roots=roots,
        config=config,
    )

    assert targets == (4, 6, 0, 1)
    assert plans[0] == CRITICAL_ROUTE
    with pytest.raises(StrategyProposalError, match="distinct"):
        compile_goal_cell(
            json.dumps({"r0": 4, "r1": 4, "r2": 0, "r3": 1}),
            model=model,
            position=position,
            roots=roots,
            config=config,
        )
    with pytest.raises(StrategyProposalError, match="out of range"):
        compile_goal_cell(
            json.dumps({"r0": 8, "r1": 6, "r2": 0, "r3": 1}),
            model=model,
            position=position,
            roots=roots,
            config=config,
        )


def test_goal_prompt_exposes_geometry_but_no_route_or_utility_answer() -> None:
    model = build_depth4_model()
    position = model.map_spec.start_position
    provider = Depth4GoalCompilerProvider(
        DeterministicDepth4GoalModel(),
        Depth4GoalCompilerConfig(),
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
    assert '"movement_distance_after_root":2' in prompt
    assert "shortest-path" in prompt
    assert "expected_information_gain" not in prompt
    assert '"score"' not in prompt
    assert "preferred target" not in prompt.lower()
    assert "move-NORTH,move-NORTH,move-NORTH,check-4" not in prompt


def test_goal_compiler_config_is_frozen() -> None:
    Depth4GoalCompilerConfig().validate()

    with pytest.raises(ValueError, match="horizon four"):
        Depth4GoalCompilerConfig(horizon=3).validate()
    with pytest.raises(ValueError, match="128-token"):
        Depth4GoalCompilerConfig(max_new_tokens=256).validate()


def test_deterministic_goal_smoke_passes() -> None:
    config = Depth4GoalCompilerConfig(seed=24_228)
    provider = Depth4GoalCompilerProvider(
        DeterministicDepth4GoalModel(), config
    )

    result = run_smoke(provider, config, cell_concurrency=4)

    assert all(result["mechanics"].values())
    assert result["critical_target_count"] == 10
    assert result["critical_route_count"] == 10
    assert result["exact_plan_match_count"] == 10


def test_deterministic_goal_proposal_and_independent_audit_pass() -> None:
    strategy_config = Depth4GoalCompilerConfig(seed=24_229)
    gate_config = RangeGatedDepth4ProposalConfig(
        seed=24_229,
        bootstrap_seed=24_230,
    )
    provider = Depth4GoalCompilerProvider(
        DeterministicDepth4GoalModel(), strategy_config
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
    assert result["comparisons"]["exact_h4_route_selection_rate"] == 1.0
    assert result["comparisons"]["recovery_fraction"]["mean"] == pytest.approx(1.0)
    assert audit["gate"]["passed"]
