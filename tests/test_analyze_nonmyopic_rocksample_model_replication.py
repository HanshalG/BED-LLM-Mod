import copy

import pytest

from scripts.analyze_nonmyopic_rock_branch_result import ARMS, BASELINES
from scripts.analyze_nonmyopic_rocksample_model_replication import EXPECTED_RUNS, analyze


def _result(run_key: str) -> dict:
    expected = EXPECTED_RUNS[run_key]
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
    paired["strategy_eig_minus_exhaustive_d2"] = {
        "entropy_auc_gain_mean": -0.2,
        "entropy_auc_gain_ci95": [-0.3, -0.1],
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
        "run_id": expected["run_id"],
        "config": {
            "map_names": ["7-8"],
            "num_trials_per_map": 30,
            "num_rounds": 10,
            "num_strategies": 6,
            "planning_horizon": 2,
            "seed": expected["seed"],
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
        "usage": {
            "backend": "openrouter",
            "model": expected["model"],
            "requests": 2,
            "reasoning_tokens": 0,
            "forced_exits": 0,
            "run_cost_usd": 0.1,
            "model_usage": {expected["model"]: {}},
        },
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


def test_model_replication_audits_both_families() -> None:
    audit = analyze(_result("gemma_26b"), _result("gpt54_mini"))

    assert audit["all_primary_gates_passed"]
    assert audit["all_truth_log_corroboration_passed"]
    assert set(audit["runs"]) == {"gemma_26b", "gpt54_mini"}


def test_model_replication_rejects_unpaired_truths() -> None:
    gpt = copy.deepcopy(_result("gpt54_mini"))
    gpt["traces"]["7-8"]["width"][4]["truth_index"] = 999

    with pytest.raises(AssertionError):
        analyze(_result("gemma_26b"), gpt)


def test_model_replication_rejects_reasoning_usage() -> None:
    gpt = copy.deepcopy(_result("gpt54_mini"))
    gpt["usage"]["reasoning_tokens"] = 1

    with pytest.raises(AssertionError):
        analyze(_result("gemma_26b"), gpt)
