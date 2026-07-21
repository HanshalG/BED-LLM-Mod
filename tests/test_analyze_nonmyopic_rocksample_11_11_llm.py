import copy

import pytest

from scripts.analyze_nonmyopic_rock_branch_result import ARMS, BASELINES
from scripts.analyze_nonmyopic_rocksample_11_11_llm import (
    EXPECTED_CONFIG,
    EXPECTED_MODEL,
    EXPECTED_RUN_ID,
    analyze,
)


def _trace(arm: str, trial_index: int) -> dict:
    entropy = 4.0 if arm in {"strategy_eig", "exhaustive_d2"} else 4.5
    return {
        "map_name": "11-11",
        "arm": arm,
        "trial_index": trial_index,
        "truth_index": trial_index,
        "steps": [
            {
                "action": "move-EAST" if arm == "strategy_eig" else "check-0",
                "entropy_after": entropy,
                "truth_log_probability": -entropy,
                "exhaustive_fraction": 0.8,
            }
            for _ in range(12)
        ],
    }


def _result(*, gain: float = 0.5) -> dict:
    traces = {
        arm: [_trace(arm, trial_index) for trial_index in range(30)] for arm in ARMS
    }
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
        "run_id": EXPECTED_RUN_ID,
        "config": EXPECTED_CONFIG,
        "mechanics": mechanics,
        "candidate_requests": [{}],
        "invalid_responses": [{}],
        "resume": None,
        "usage": {
            "backend": "openrouter",
            "model": EXPECTED_MODEL,
            "requests": 2,
            "reasoning_tokens": 0,
            "forced_exits": 0,
            "run_cost_usd": 0.1,
            "model_usage": {EXPECTED_MODEL: {}},
        },
        "traces": {"11-11": traces},
        "maps": {
            "11-11": {
                "gate_passed": True,
                "paired": paired,
                "summary": {
                    arm: {"round_entropy_mean": [4.0] * 12} for arm in ARMS
                },
            }
        },
        "gate_passed": True,
    }


def test_auditor_reconstructs_trace_derived_values() -> None:
    audit = analyze(_result(gain=0.6))

    assert audit["primary_gate_passed"]
    assert audit["truth_log_corroboration_passed"]
    assert audit["comparisons"]["shared_d1"]["entropy_auc_gain"] == pytest.approx(
        0.6
    )


def test_auditor_rejects_trace_metric_mismatch() -> None:
    result = _result()
    result["traces"]["11-11"]["width"][3]["steps"][0]["entropy_after"] += 0.2

    with pytest.raises(AssertionError):
        analyze(result)


def test_auditor_rejects_resume() -> None:
    result = copy.deepcopy(_result())
    result["resume"] = {
        "accepted_cells_reused": 1,
        "prior_error": "unexpected",
        "failure_artifact": "results/L1_FAILURE.json",
    }

    with pytest.raises(AssertionError):
        analyze(result)
