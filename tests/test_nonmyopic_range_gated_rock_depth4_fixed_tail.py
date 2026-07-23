import json

import pytest

from scripts.audit_nonmyopic_range_gated_rock_depth4_fixed_tail import (
    audit_result,
)
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_range_gated_rock_depth4_fixed_tail import (
    CRITICAL_ROUTE,
    Depth4FixedRootTailProvider,
    DeterministicDepth4TailModel,
    RangeGatedDepth4ProposalConfig,
    RangeGatedDepth4StrategyConfig,
    build_depth4_model,
    compile_depth4_fixed_tail_cell,
    run_proposal_gate,
    run_smoke,
)
from scripts.nonmyopic_range_gated_rock_fixed_tail import fixed_roots


def test_depth4_fixed_tail_compiler_preserves_roots_and_dynamic_legality() -> None:
    model = build_depth4_model()
    position = model.map_spec.start_position
    roots = fixed_roots(model, position=position, belief=model.initial_belief)
    config = RangeGatedDepth4StrategyConfig()
    response = json.dumps(
        {
            "r0": ["move-NORTH", "move-NORTH", "check-4"],
            "r1": ["move-WEST", "move-WEST", "check-6"],
            "r2": ["check-1", "check-2", "check-3"],
            "r3": ["check-2", "check-3", "check-4"],
        }
    )

    plans = compile_depth4_fixed_tail_cell(
        response,
        model=model,
        position=position,
        roots=roots,
        config=config,
    )

    assert plans[0] == CRITICAL_ROUTE
    assert tuple(plan[0] for plan in plans) == roots

    invalid = json.dumps(
        {
            "r0": ["move-EAST", "move-NORTH", "check-4"],
            "r1": ["move-WEST", "move-WEST", "check-6"],
            "r2": ["check-1", "check-2", "check-3"],
            "r3": ["check-2", "check-3", "check-4"],
        }
    )
    with pytest.raises(StrategyProposalError, match="illegal action"):
        compile_depth4_fixed_tail_cell(
            invalid,
            model=model,
            position=position,
            roots=roots,
            config=config,
        )


def test_depth4_prompt_exposes_transition_graph_without_utility_answer() -> None:
    model = build_depth4_model()
    position = model.map_spec.start_position
    provider = Depth4FixedRootTailProvider(
        DeterministicDepth4TailModel(),
        RangeGatedDepth4StrategyConfig(),
    )
    roots = fixed_roots(model, position=position, belief=model.initial_belief)
    prompt = provider._messages(
        model,
        position=position,
        belief=model.initial_belief,
        history=(),
        roots=roots,
    )[-1]["content"]

    assert "reachable_position_graph" in prompt
    assert '"fixed_root":"move-NORTH"' in prompt
    assert '"position_after_root":[6,5]' in prompt
    assert "expected_information_gain" not in prompt
    assert '"score"' not in prompt
    assert "preferred route" not in prompt.lower()
    assert '["move-NORTH","move-NORTH","move-NORTH","check-4"]' not in prompt


def test_depth4_config_validation_is_frozen() -> None:
    RangeGatedDepth4StrategyConfig().validate()
    RangeGatedDepth4ProposalConfig().validate()

    with pytest.raises(ValueError, match="horizon four"):
        RangeGatedDepth4StrategyConfig(horizon=3).validate()
    with pytest.raises(ValueError, match="16 cells"):
        RangeGatedDepth4ProposalConfig(num_cells=8).validate()


def test_deterministic_depth4_smoke_recovers_critical_route() -> None:
    config = RangeGatedDepth4StrategyConfig(seed=24_208)
    provider = Depth4FixedRootTailProvider(
        DeterministicDepth4TailModel(), config
    )

    result = run_smoke(provider, config)

    assert all(result["mechanics"].values())
    assert result["critical_route_count"] == 10
    assert result["exact_root_match_count"] == 10


def test_deterministic_depth4_proposal_and_audit_pass() -> None:
    strategy_config = RangeGatedDepth4StrategyConfig(seed=24_209)
    gate_config = RangeGatedDepth4ProposalConfig(seed=24_209)
    provider = Depth4FixedRootTailProvider(
        DeterministicDepth4TailModel(), strategy_config
    )

    result = run_proposal_gate(provider, strategy_config, gate_config)
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
