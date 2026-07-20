import copy

import pytest

from scripts.analyze_nonmyopic_rock_branch_result import ARMS, BASELINES, analyze


def _payload() -> dict:
    traces = {}
    maps = {}
    for map_name in ("3-6", "5-7"):
        arm_traces = {}
        for arm in ARMS:
            arm_traces[arm] = [
                {
                    "trial_index": trial_index,
                    "truth_index": trial_index % 4,
                    "steps": [
                        {
                            "action": "move-EAST" if arm == "strategy_eig" else "check-0",
                            "exhaustive_fraction": 0.9,
                        }
                        for _ in range(8)
                    ],
                }
                for trial_index in range(30)
            ]
        traces[map_name] = arm_traces
        maps[map_name] = {
            "summary": {arm: {"round_entropy_mean": [1.0] * 8} for arm in ARMS},
            "paired": {
                f"strategy_eig_minus_{baseline}": {
                    "entropy_auc_gain_mean": 0.2,
                    "entropy_auc_gain_ci95": [0.1, 0.3],
                    "truth_log_probability_auc_gain_mean": 0.15,
                    "truth_log_probability_auc_gain_ci95": [0.05, 0.25],
                    "final_entropy_gain_mean": 0.25,
                    "entropy_auc_wins_ties_losses": [25, 0, 5],
                }
                for baseline in BASELINES
            },
        }
    return {
        "config": {
            "map_names": ["3-6", "5-7"],
            "num_trials_per_map": 30,
            "num_rounds": 8,
            "num_strategies": 6,
            "planning_horizon": 2,
            "seed": 12041,
            "bootstrap_replicates": 10_000,
            "strategy_schema": "branch_policy_v2",
            "primary_endpoint": "entropy_auc",
        },
        "mechanics": {
            "terminal_cell_failures": 0,
            "rollout_scoring_llm_calls": 0,
            "all_selected_actions_legal": True,
            "initial_strategy_cells_shared_with_d1": True,
            "width_logical_llm_calls_match_strategy_eig": True,
            "width_exact_scorer_units_match_strategy_eig": True,
            "random_strategy_cells_have_k_candidates": True,
            "raw_rejected_responses": 0,
            "terminal_followup_repairs": 0,
        },
        "maps": maps,
        "traces": traces,
        "gate_passed": True,
        "usage": {"run_cost_usd": 0.1, "requests": 10},
    }


def test_analyzer_verifies_registered_gates_and_pairing() -> None:
    audit = analyze(_payload())

    assert audit["primary_gate_passed"]
    assert audit["truth_log_corroboration_passed"]
    assert audit["maps"]["3-6"]["arms"]["strategy_eig"]["move_rate"] == 1.0
    assert audit["maps"]["3-6"]["arms"]["shared_d1"]["move_rate"] == 0.0


def test_analyzer_rejects_unpaired_truths() -> None:
    payload = copy.deepcopy(_payload())
    payload["traces"]["5-7"]["width"][3]["truth_index"] = 999

    with pytest.raises(AssertionError):
        analyze(payload)
