import copy

import pytest

from scripts.analyze_nonmyopic_rock_branch_result import ARMS, BASELINES
from scripts.analyze_nonmyopic_rocksample_slot_replication import (
    EXPECTED_RUNS,
    analyze,
)


def _trace(arm: str, trial_index: int) -> dict:
    entropy = 4.0 if arm in {"strategy_eig", "exhaustive_d2"} else 4.5
    truth_log_probability = -entropy
    return {
        "map_name": "7-8",
        "arm": arm,
        "trial_index": trial_index,
        "truth_index": trial_index,
        "steps": [
            {
                "action": "move-EAST" if arm == "strategy_eig" else "check-0",
                "entropy_after": entropy,
                "truth_log_probability": truth_log_probability,
                "exhaustive_fraction": 0.8,
            }
            for _ in range(10)
        ],
    }


def _result(run_key: str, *, gain: float = 0.5) -> dict:
    expected = EXPECTED_RUNS[run_key]
    traces = {
        arm: [_trace(arm, trial_index) for trial_index in range(30)] for arm in ARMS
    }
    # Adjust the trace-derived values together with the synthetic stored result.
    for baseline in BASELINES:
        for trace in traces[baseline]:
            for step in trace["steps"]:
                step["entropy_after"] = 4.0 + gain
                step["truth_log_probability"] = -4.0 - gain
    paired = {
        f"strategy_eig_minus_{baseline}": {
            "entropy_auc_gain_mean": gain,
            "entropy_auc_gain_ci95": [gain - 0.1, gain + 0.1],
            "entropy_auc_paired_values": [gain] * 30,
            "truth_log_probability_auc_gain_mean": gain,
            "truth_log_probability_auc_gain_ci95": [gain - 0.1, gain + 0.1],
            "truth_log_probability_auc_paired_values": [gain] * 30,
            "entropy_auc_wins_ties_losses": [30, 0, 0],
        }
        for baseline in BASELINES
    }
    paired["strategy_eig_minus_exhaustive_d2"] = {
        "entropy_auc_gain_mean": -0.1,
        "entropy_auc_gain_ci95": [-0.2, 0.0],
    }
    resumed = expected["resumed"]
    mechanics = {
        "terminal_cell_failures": 0,
        "rollout_scoring_llm_calls": 0,
        "all_selected_actions_legal": True,
        "initial_strategy_cells_shared_with_d1": True,
        "width_logical_llm_calls_match_strategy_eig": True,
        "width_exact_scorer_units_match_strategy_eig": True,
        "random_strategy_cells_have_k_candidates": True,
        "accepted_llm_cells": 1,
        "raw_rejected_responses": int(resumed),
        "physical_llm_requests": 1 + int(resumed),
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
        "invalid_responses": [{}] if resumed else [],
        "resume": (
            {
                "accepted_cells_reused": 1,
                "prior_error": "failed cell",
                "failure_artifact": "results/L1_FAILURE.json",
            }
            if resumed
            else None
        ),
        "usage": {
            "backend": "openrouter",
            "model": expected["model"],
            "requests": 1 + int(resumed),
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
                    arm: {"round_entropy_mean": [4.0] * 10} for arm in ARMS
                },
            }
        },
        "gate_passed": True,
    }


def test_slot_replication_audits_trace_derived_values() -> None:
    audit = analyze(
        _result("gemma_semantic", gain=0.4),
        _result("gemma_slots", gain=0.6),
        _result("gpt54_mini_slots", gain=0.5),
    )

    assert audit["same_interface_primary_gates_passed"]
    assert audit["same_interface_truth_log_gates_passed"]
    assert audit["descriptive_gemma_slot_minus_semantic_gain"]["shared_d1"] == pytest.approx(
        0.2
    )


def test_slot_replication_rejects_trace_metric_mismatch() -> None:
    gemma_slots = _result("gemma_slots")
    gemma_slots["traces"]["7-8"]["width"][3]["steps"][0]["entropy_after"] += 0.2

    with pytest.raises(AssertionError):
        analyze(
            _result("gemma_semantic"),
            gemma_slots,
            _result("gpt54_mini_slots"),
        )


def test_slot_replication_rejects_unexpected_resume() -> None:
    gemma_slots = copy.deepcopy(_result("gemma_slots"))
    gemma_slots["resume"] = {
        "accepted_cells_reused": 1,
        "prior_error": "unexpected",
        "failure_artifact": "results/L1_FAILURE.json",
    }

    with pytest.raises(AssertionError):
        analyze(
            _result("gemma_semantic"),
            gemma_slots,
            _result("gpt54_mini_slots"),
        )
