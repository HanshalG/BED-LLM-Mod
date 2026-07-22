import copy

import numpy as np
import pytest

from scripts.analyze_nonmyopic_rock_branch_result import ARMS, BASELINES
from scripts.analyze_nonmyopic_rocksample_15_15_llm import (
    EXPECTED_CONFIG,
    EXPECTED_MODEL,
    EXPECTED_RUN_ID,
    EXPECTED_RUNS,
    analyze,
    analyze_run,
    _assert_bootstrap_ci,
    render_summary,
)
from scripts.nonmyopic_rock_strategy_prior import _bootstrap_mean_ci, _stable_seed


def _trace(arm: str, trial_index: int) -> dict:
    entropy = 4.0 if arm in {"strategy_eig", "exhaustive_d2"} else 4.5
    return {
        "map_name": "15-15",
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
            for _ in range(15)
        ],
    }


def _result(*, gain: float = 0.5, run_key: str = "gemma") -> dict:
    expected = EXPECTED_RUNS[run_key]
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
            "entropy_auc_gain_ci95": [gain, gain],
            "entropy_auc_paired_values": [gain] * 30,
            "truth_log_probability_auc_gain_mean": gain,
            "truth_log_probability_auc_gain_ci95": [gain, gain],
            "truth_log_probability_auc_paired_values": [gain] * 30,
            "entropy_auc_wins_ties_losses": [30, 0, 0],
        }
        for baseline in BASELINES
    }
    paired["strategy_eig_minus_exhaustive_d2"] = {
        "entropy_auc_gain_mean": 0.0,
        "entropy_auc_gain_ci95": [0.0, 0.0],
        "entropy_auc_paired_values": [0.0] * 30,
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
            **EXPECTED_CONFIG,
            "seed": expected["seed"],
            "trial_concurrency": expected["trial_concurrency"],
        },
        "mechanics": mechanics,
        "candidate_requests": [{}],
        "invalid_responses": [{}],
        "resume": (
            {
                "accepted_cells_reused": expected["accepted_cells_reused"],
                "prior_error": "failed cell",
                "failure_artifact": "results/L1_FAILURE.json",
            }
            if expected["resumed"]
            else None
        ),
        "usage": {
            "backend": expected["backend"],
            "model": expected["model"],
            "requests": 2,
            "reasoning_tokens": 0,
            "forced_exits": 0,
            "run_cost_usd": 0.1,
            "model_usage": {expected["model"]: {}},
        },
        "traces": {"15-15": traces},
        "maps": {
            "15-15": {
                "gate_passed": True,
                "paired": paired,
                "summary": {
                    arm: {"round_entropy_mean": [4.0] * 15} for arm in ARMS
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
    result["traces"]["15-15"]["width"][3]["steps"][0]["entropy_after"] += 0.2

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


def test_auditor_rejects_unregistered_run_key() -> None:
    with pytest.raises(KeyError):
        analyze_run(_result(), "unregistered")


def test_auditor_accepts_registered_vllm_replication() -> None:
    audit = analyze_run(_result(run_key="vllm"), "vllm")

    assert audit["model"] == "google/gemma-4-26B-A4B-it"
    assert audit["usage"]["backend"] == "vllm"
    assert audit["primary_gate_passed"]


@pytest.mark.parametrize(
    "run_key", ["vllm_seed_24102", "vllm_seed_24103", "e4b_vllm"]
)
def test_auditor_accepts_registered_vllm_runs(run_key: str) -> None:
    audit = analyze_run(_result(run_key=run_key), run_key)

    assert audit["usage"]["backend"] == "vllm"
    assert audit["primary_gate_passed"]


def test_summary_reports_failed_gate_without_positive_claim() -> None:
    audit = analyze(_result())
    audit["primary_gate_passed"] = False

    summary = render_summary(audit)

    assert "fails its preregistered primary entropy-AUC gate" in summary
    assert "The registered fifteen-rock run" in summary
    assert EXPECTED_RUNS["gemma"]["label"] in summary
    assert "Positive paired gains favor StrategyEIG" not in summary


def test_summary_labels_positive_exhaustive_d2_comparison_as_advantage() -> None:
    audit = analyze(_result())
    audit["exact_d2_entropy_auc_gap"] = 0.1
    audit["exact_d2_entropy_auc_gap_ci95"] = [0.05, 0.15]

    summary = render_summary(audit)

    assert "advantage over terminal-objective exhaustive d2" in summary
    assert "remaining entropy-AUC gap" not in summary


def test_auditor_reconstructs_metric_as_separate_seed_component() -> None:
    values = [float(index) for index in range(30)]
    stored = _bootstrap_mean_ci(
        np.asarray(values),
        seed=_stable_seed(24101, "l1-bootstrap", "15-15-shared_d1", "entropy-auc"),
        replicates=10_000,
    )

    _assert_bootstrap_ci(
        values,
        list(stored),
        seed=24101,
        label="15-15-shared_d1",
        metric="entropy-auc",
    )
