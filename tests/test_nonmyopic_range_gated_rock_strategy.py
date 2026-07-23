import json

import pytest

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map
from scripts.nonmyopic_range_gated_rock_strategy import (
    DeterministicNamedPlanModel,
    NamedPlanProvider,
    RangeGatedStrategyConfig,
    compile_plan_cell,
    enumerate_legal_plans,
    plan_value,
    run_smoke,
)
from scripts.nonmyopic_range_gated_rock_proposal_gate import matched_random_plans
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError


def _model() -> RangeGatedRockDiagnosisModel:
    return RangeGatedRockDiagnosisModel(get_paper_map("7-8"))


def test_named_plan_parser_enforces_root_mix_and_legality() -> None:
    model = _model()
    position = model.map_spec.start_position
    response = json.dumps(
        {
            "plans": [
                ["move-SOUTH", "move-SOUTH", "check-5"],
                ["move-EAST", "move-EAST", "check-2"],
                ["check-0", "check-1", "check-2"],
                ["check-3", "check-4", "check-5"],
            ]
        }
    )
    plans = compile_plan_cell(
        response, model=model, position=position, config=RangeGatedStrategyConfig()
    )
    assert plans[0] == ("move-SOUTH", "move-SOUTH", "check-5")

    invalid = json.dumps({"plans": [["check-0", "check-1", "check-2"]] * 4})
    with pytest.raises(StrategyProposalError):
        compile_plan_cell(
            invalid, model=model, position=position, config=RangeGatedStrategyConfig()
        )


def test_on_site_plan_has_more_value_than_three_remote_checks() -> None:
    model = _model()
    position = model.map_spec.start_position
    belief = model.initial_belief
    travel = ("move-SOUTH", "move-SOUTH", "check-5")
    remote = ("check-0", "check-1", "check-2")
    assert plan_value(model, position=position, belief=belief, plan=travel) > plan_value(
        model, position=position, belief=belief, plan=remote
    )


def test_exhaustive_open_loop_plan_selects_delayed_inspection() -> None:
    model = _model()
    position = model.map_spec.start_position
    plans = enumerate_legal_plans(model, position=position, horizon=3)
    best = max(
        plans,
        key=lambda plan: plan_value(
            model, position=position, belief=model.initial_belief, plan=plan
        ),
    )
    assert best == ("move-SOUTH", "move-SOUTH", "check-5")


def test_deterministic_smoke_makes_one_call_per_cell() -> None:
    config = RangeGatedStrategyConfig()
    provider = NamedPlanProvider(DeterministicNamedPlanModel(), config)
    result = run_smoke(provider, config)
    assert all(result["mechanics"].values())
    assert result["provider"] == {"accepted_requests": 10, "invalid_responses": 0}


def test_matched_random_plans_preserve_registered_root_mix() -> None:
    model = _model()
    plans = matched_random_plans(
        model, position=model.map_spec.start_position, seed=17
    )
    assert len(plans) == len(set(plans)) == 4
    assert sum(model.is_move(plan[0]) for plan in plans) == 2
    assert all(len(plan) == 3 for plan in plans)
