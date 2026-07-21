import copy

import pytest

from scripts.analyze_nonmyopic_rock_branch_result import ARMS, BASELINES
from scripts.analyze_nonmyopic_rocksample_scale_result import analyze


def _qualification() -> dict:
    comparison = {
        "final_entropy_reduction_mean": 1.5,
        "final_entropy_reduction_ci95": [1.4, 1.6],
        "wins_ties_losses": [1000, 0, 0],
    }
    return {
        "config": {
            "map_name": "7-8",
            "num_trials": 1000,
            "num_rounds": 10,
            "candidate_widths": [12],
            "seed": 24071,
            "trial_offset": 0,
            "bootstrap_replicates": 10_000,
            "half_efficiency_distance": 0.6931471805599453,
        },
        "no_llm_calls": True,
        "source": {
            "map": "7-8",
            "page": 5,
            "map_spec": {
                "rock_positions": [
                    [1, 0],
                    [5, 1],
                    [2, 2],
                    [3, 2],
                    [6, 3],
                    [0, 5],
                    [3, 5],
                    [2, 6],
                ]
            },
        },
        "decision": {"confirmation_passes": True},
        "widths": {
            "12": {
                "passed_confirmation_gate": True,
                "mechanics": {"legal": True, "paired": True},
                "comparisons": {
                    "d2_minus_shared_d1": comparison,
                    "d2_minus_call_matched_width": comparison,
                },
            }
        },
    }


def _confirmation() -> dict:
    traces = {
        arm: [
            {
                "map_name": "7-8",
                "arm": arm,
                "trial_index": trial_index,
                "truth_index": trial_index,
                "steps": [
                    {
                        "action": "move-EAST" if arm == "strategy_eig" else "check-0",
                        "exhaustive_fraction": 0.8,
                    }
                    for _ in range(10)
                ],
            }
            for trial_index in range(30)
        ]
        for arm in ARMS
    }
    paired = {
        f"strategy_eig_minus_{baseline}": {
            "entropy_auc_gain_mean": 0.5,
            "entropy_auc_gain_ci95": [0.4, 0.6],
            "truth_log_probability_auc_gain_mean": 0.4,
            "truth_log_probability_auc_gain_ci95": [0.3, 0.5],
            "final_entropy_gain_mean": 1.0,
            "final_entropy_gain_ci95": [0.8, 1.2],
            "entropy_auc_wins_ties_losses": [30, 0, 0],
        }
        for baseline in BASELINES
    }
    mechanics = {
        "terminal_cell_failures": 0,
        "rollout_scoring_llm_calls": 0,
        "all_selected_actions_legal": True,
        "initial_strategy_cells_shared_with_d1": True,
        "width_logical_llm_calls_match_strategy_eig": True,
        "width_exact_scorer_units_match_strategy_eig": True,
        "random_strategy_cells_have_k_candidates": True,
        "accepted_llm_cells": 1,
        "raw_rejected_responses": 1,
        "physical_llm_requests": 2,
    }
    return {
        "schema_version": 1,
        "stage": "L1",
        "dry_run": False,
        "run_id": "nonmyopic-rocksample-7-8-scale-20260721",
        "config": {
            "map_names": ["7-8"],
            "num_trials_per_map": 30,
            "num_rounds": 10,
            "num_strategies": 6,
            "planning_horizon": 2,
            "seed": 24072,
            "bootstrap_replicates": 10_000,
            "temperature": 0.0,
            "validation_retries": 1,
            "trial_concurrency": 32,
            "strategy_schema": "branch_policy_v2",
            "primary_endpoint": "entropy_auc",
        },
        "mechanics": mechanics,
        "candidate_requests": [{}],
        "invalid_responses": [{}],
        "resume": {
            "accepted_cells_reused": 1,
            "rejected_responses_preserved": 1,
            "prior_error": "failed cell",
            "failure_artifact": "results/L1_FAILURE.json",
        },
        "usage": {"requests": 2, "run_cost_usd": 0.1},
        "traces": {"7-8": traces},
        "maps": {
            "7-8": {
                "gate_passed": True,
                "paired": paired,
                "summary": {
                    arm: {"round_entropy_mean": [5.0 - 0.1 * index for index in range(10)]}
                    for arm in ARMS
                },
            }
        },
        "gate_passed": True,
    }


def test_scale_analyzer_validates_both_registered_stages() -> None:
    audit = analyze(_qualification(), _confirmation())

    assert audit["qualification"]["passed"]
    assert audit["confirmation"]["primary_gate_passed"]
    assert audit["confirmation"]["truth_log_corroboration_passed"]
    assert audit["confirmation"]["arms"]["strategy_eig"]["move_rate"] == 1.0


def test_scale_analyzer_rejects_unpaired_confirmation_truths() -> None:
    confirmation = copy.deepcopy(_confirmation())
    confirmation["traces"]["7-8"]["width"][4]["truth_index"] = 999

    with pytest.raises(AssertionError):
        analyze(_qualification(), confirmation)


def test_scale_analyzer_rejects_failed_exact_qualification() -> None:
    qualification = copy.deepcopy(_qualification())
    qualification["widths"]["12"]["comparisons"]["d2_minus_shared_d1"][
        "final_entropy_reduction_ci95"
    ] = [-0.1, 0.2]

    with pytest.raises(AssertionError):
        analyze(qualification, _confirmation())
