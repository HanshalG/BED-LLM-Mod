import json
import math

import numpy as np
import pytest

from environments.rock_diagnosis import (
    RockDiagnosisModel,
    RockStrategyExecutionError,
    RockStrategyExecutor,
    RockStrategyParseError,
    get_paper_map,
    parse_rock_strategy,
    random_rock_strategy_text,
    score_rock_strategy_exact,
)
from scripts.nonmyopic_rock_strategy_l0_smoke import L0SmokeConfig, run_l0_smoke


def _strategy_text(rules: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "name": "test strategy",
            "description": "A complete compact policy used by a deterministic unit test.",
            "rules": rules,
        }
    )


def test_parser_and_executor_apply_ordered_posterior_rules() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    text = _strategy_text(
        [
            {
                "when": [{"kind": "good_probability_at_least", "rock_id": 0, "threshold": 0.501}],
                "action": {"kind": "check_rock", "rock_id": 1},
            },
            {"when": [], "action": {"kind": "check_rock", "rock_id": 0}},
        ]
    )
    strategy = parse_rock_strategy(text, model)
    executor = RockStrategyExecutor(model)
    position = model.map_spec.start_position

    assert executor.choose_action(
        strategy, position=position, belief=model.initial_belief, history=(), strategy_step=0
    ) == "check-0"
    posterior = model.posterior(position, model.initial_belief, "check-0", "good")
    assert executor.choose_action(
        strategy,
        position=position,
        belief=posterior,
        history=(("check-0", "good"),),
        strategy_step=1,
    ) == "check-1"


def test_target_rock_macro_is_legal_and_path_sensitive() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    position = model.map_spec.start_position
    executor = RockStrategyExecutor(model)
    x_first = parse_rock_strategy(
        _strategy_text([{"when": [], "action": {"kind": "target_rock", "rock_id": 1, "path": "x_first"}}]),
        model,
    )
    y_first = parse_rock_strategy(
        _strategy_text([{"when": [], "action": {"kind": "target_rock", "rock_id": 1, "path": "y_first"}}]),
        model,
    )

    assert executor.choose_action(
        x_first, position=position, belief=model.initial_belief, history=(), strategy_step=0
    ) == "move-EAST"
    assert executor.choose_action(
        y_first, position=position, belief=model.initial_belief, history=(), strategy_step=0
    ) == "move-SOUTH"


def test_strategy_parser_and_executor_fail_closed() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    with pytest.raises(RockStrategyParseError, match="final rule"):
        parse_rock_strategy(
            _strategy_text(
                [
                    {
                        "when": [{"kind": "step_at_least", "value": 1}],
                        "action": {"kind": "check_rock", "rock_id": 0},
                    }
                ]
            ),
            model,
        )
    with pytest.raises(RockStrategyParseError, match="out of range"):
        parse_rock_strategy(
            _strategy_text([{"when": [], "action": {"kind": "check_rock", "rock_id": 99}}]),
            model,
        )
    illegal = parse_rock_strategy(
        _strategy_text([{"when": [], "action": {"kind": "move", "direction": "WEST"}}]),
        model,
    )
    with pytest.raises(RockStrategyExecutionError, match="illegal action"):
        RockStrategyExecutor(model).choose_action(
            illegal,
            position=model.map_spec.start_position,
            belief=model.initial_belief,
            history=(),
            strategy_step=0,
        )


def test_one_step_exact_strategy_score_matches_action_eig() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    strategy = parse_rock_strategy(
        _strategy_text([{"when": [], "action": {"kind": "check_rock", "rock_id": 0}}]),
        model,
    )
    score = score_rock_strategy_exact(
        model,
        strategy,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        horizon=1,
    )

    expected = model.expected_information_gain(model.map_spec.start_position, model.initial_belief, "check-0")
    assert score.eig == pytest.approx(expected)
    assert score.start_entropy - score.expected_final_entropy == pytest.approx(score.eig)
    assert score.root_action == "check-0"
    assert score.expanded_decision_nodes == 1
    assert score.leaf_nodes == 2


def test_random_strategy_sampler_is_reproducible_and_parseable() -> None:
    model = RockDiagnosisModel(get_paper_map("5-7"))
    first = random_rock_strategy_text(model, np.random.default_rng(42), index=0)
    second = random_rock_strategy_text(model, np.random.default_rng(42), index=0)

    assert first == second
    assert parse_rock_strategy(first, model).name == "random-strategy-0"


def test_l0_zero_llm_smoke_passes_on_both_maps() -> None:
    summary = run_l0_smoke(
        L0SmokeConfig(num_trials_per_map=5, num_rounds=3, num_strategies=3, planning_horizon=3)
    )

    assert summary["gate_passed"]
    assert summary["num_trajectories"] == 10
    assert summary["mechanics"] == {
        "zero_llm_calls": True,
        "parse_failures": 0,
        "execution_failures": 0,
        "all_selected_actions_legal": True,
        "all_exact_scores_finite_nonnegative": True,
        "all_candidate_sets_complete": True,
        "scored_root_matches_executed_root": True,
    }
    assert all(math.isfinite(trace["final_entropy"]) for trace in summary["traces"])
