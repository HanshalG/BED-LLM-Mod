import math

import numpy as np
import pytest

from core import BeliefState
from environments.location_finding.types import LocationObservation, LocationStrategyEntry
from helpers import Config
from scripts.location_fixed_root_depth_sweep import (
    _DepthBranch,
    _group_strategy_branch_indices,
    _metric_summary_by_policy,
    _observe_with_noise_z,
    _paired_delta_summary_by_policy,
    _paired_trial_delta_rows,
    _plot_paired_trial_differences,
    _policy_specs,
    _trial_window,
    _truth_augmented_state,
    _truth_log_probability,
    _write_depth_sweep_report,
)


def test_observe_with_noise_z_uses_configured_local_bump_signal():
    config = Config(
        task="location_finding",
        location_num_sources=1,
        location_signal_model="local_bump",
        location_signal_lengthscale=0.5,
        location_signal_amplitude=8.0,
        location_noise_sd=0.15,
    )
    hidden_state = np.asarray([[1.0, 0.0]], dtype=float)

    observation = _observe_with_noise_z((1.0, 0.0), hidden_state, config, noise_z=0.0)

    assert observation.query == (1.0, 0.0)
    assert observation.value == pytest.approx(8.1)


def test_truth_augmented_state_exposes_truth_log_probability():
    config = Config(
        task="location_finding",
        location_num_sources=1,
        location_noise_sd=0.2,
    )
    hidden_state = np.asarray([[0.0, 0.0]], dtype=float)
    belief = BeliefState(hypotheses=[((2.0, 2.0),)], probabilities=[1.0])
    observation = _observe_with_noise_z((0.0, 0.0), hidden_state, config, noise_z=0.0)

    state = _truth_augmented_state(
        belief,
        [(observation.query, observation)],
        hidden_state,
        config,
    )
    truth_log_probability = _truth_log_probability(state, hidden_state)

    assert len(state.hypotheses) == 2
    assert math.isfinite(truth_log_probability)
    assert math.exp(truth_log_probability) > 0.5


def test_policy_metric_summary_and_paired_delta_use_policy_labels():
    per_trial = [
        {
            "trial_index": 0,
            "policy_label": "EIG",
            "round_metrics": [
                {"source_rmse": 0.8, "posterior_entropy": 1.5, "truth_log_probability": -2.0},
                {"source_rmse": 0.5, "posterior_entropy": 1.0, "truth_log_probability": -1.0},
            ],
        },
        {
            "trial_index": 0,
            "policy_label": "StrategyEIG-d2",
            "round_metrics": [
                {"source_rmse": 0.7, "posterior_entropy": 1.2, "truth_log_probability": -1.5},
                {"source_rmse": 0.2, "posterior_entropy": 0.6, "truth_log_probability": -0.4},
            ],
        },
    ]

    summary = _metric_summary_by_policy(per_trial)
    deltas = _paired_delta_summary_by_policy(
        per_trial,
        baseline_label="EIG",
        rng=np.random.default_rng(0),
    )

    assert summary["EIG"]["source_rmse"]["final_mean"] == pytest.approx(0.5)
    assert summary["StrategyEIG-d2"]["posterior_entropy"]["mean_trace"] == pytest.approx([1.2, 0.6])
    assert deltas["StrategyEIG-d2"]["source_rmse"]["final_delta_mean"] == pytest.approx(-0.3)
    assert deltas["StrategyEIG-d2"]["truth_log_probability"]["final_delta_mean"] == pytest.approx(0.6)
    assert deltas["StrategyEIG-d2"]["source_rmse"]["final_delta_ci95"] == pytest.approx([-0.3, -0.3])
    assert deltas["StrategyEIG-d2"]["source_rmse"]["wilcoxon_signed_rank_p"] is not None


def test_policy_specs_add_myopic_controls_only_when_requested():
    without_controls = _policy_specs(3, include_myopic_controls=False)
    with_controls = _policy_specs(3, include_myopic_controls=True)

    assert ("StrategyEIG-myopic-d2", "StrategyEIG-myopic-control", 2, 1) not in without_controls
    assert ("StrategyEIG-d3", "StrategyEIG", 3, 3) in with_controls
    assert ("StrategyEIG-myopic-d2", "StrategyEIG-myopic-control", 2, 1) in with_controls
    assert ("StrategyEIG-myopic-d3", "StrategyEIG-myopic-control", 3, 1) in with_controls
    assert ("StrategyEIG-myopic-d1", "StrategyEIG-myopic-control", 1, 1) not in with_controls


def test_policy_specs_can_select_mpp_depth_subset():
    specs = _policy_specs(
        5,
        include_myopic_controls=True,
        strategy_depths=[1, 3, 5],
        myopic_control_depths=[3, 5],
    )

    assert ("StrategyEIG-d1", "StrategyEIG", 1, 1) in specs
    assert ("StrategyEIG-d3", "StrategyEIG", 3, 3) in specs
    assert ("StrategyEIG-d5", "StrategyEIG", 5, 5) in specs
    assert ("StrategyEIG-d2", "StrategyEIG", 2, 2) not in specs
    assert ("StrategyEIG-d4", "StrategyEIG", 4, 4) not in specs
    assert ("StrategyEIG-myopic-d3", "StrategyEIG-myopic-control", 3, 1) in specs
    assert ("StrategyEIG-myopic-d5", "StrategyEIG-myopic-control", 5, 1) in specs
    assert ("StrategyEIG-myopic-d2", "StrategyEIG-myopic-control", 2, 1) not in specs


def test_trial_window_supports_split_replay_offsets():
    assert _trial_window(50, num_trials=10, trial_offset=20) == (20, 10, 30, 30)
    assert _trial_window(50, num_trials=10, trial_offset=10, total_trials=30) == (10, 10, 30, 20)
    assert _trial_window(50, num_trials=None, trial_offset=0) == (0, 50, 50, 50)

    with pytest.raises(ValueError, match="trial_offset"):
        _trial_window(50, num_trials=10, trial_offset=-1)
    with pytest.raises(ValueError, match="num_trials"):
        _trial_window(50, num_trials=0, trial_offset=0)
    with pytest.raises(ValueError, match="total_trials"):
        _trial_window(50, num_trials=10, trial_offset=20, total_trials=29)


def test_strategy_state_grouping_shares_only_identical_branch_states():
    belief = BeliefState(hypotheses=[((0.0, 0.0),), ((1.0, 0.0),)], probabilities=[0.6, 0.4])
    shared_a = _DepthBranch(
        trial_index=0,
        policy_label="StrategyEIG-d1",
        policy_kind="StrategyEIG",
        policy_depth=1,
        selection_depth=1,
        hidden_state=np.asarray([[0.0, 0.0]]),
        belief_state=belief,
    )
    shared_b = _DepthBranch(
        trial_index=0,
        policy_label="StrategyEIG-d3",
        policy_kind="StrategyEIG",
        policy_depth=3,
        selection_depth=3,
        hidden_state=np.asarray([[0.0, 0.0]]),
        belief_state=belief,
    )
    changed_history = _DepthBranch(
        trial_index=0,
        policy_label="StrategyEIG-d5",
        policy_kind="StrategyEIG",
        policy_depth=5,
        selection_depth=5,
        hidden_state=np.asarray([[0.0, 0.0]]),
        belief_state=belief,
        history=[((0.0, 0.0), LocationObservation(query=(0.0, 0.0), value=1.0))],
    )
    changed_library = _DepthBranch(
        trial_index=0,
        policy_label="StrategyEIG-myopic-d2",
        policy_kind="StrategyEIG-myopic-control",
        policy_depth=2,
        selection_depth=1,
        hidden_state=np.asarray([[0.0, 0.0]]),
        belief_state=belief,
    )
    changed_library.library.replace_entries(
        [
            LocationStrategyEntry(
                strategy="probe the left branch",
                mean_score=0.5,
                score_variance=0.1,
                root_query_fingerprint="0,0",
                round_index=0,
                root_query=(0.0, 0.0),
            )
        ]
    )

    groups = _group_strategy_branch_indices(
        [shared_a, shared_b, changed_history, changed_library],
        [0, 1, 2, 3],
    )

    assert groups == [[0, 1], [2], [3]]


def test_paired_trial_delta_rows_keep_trial_labels_and_plot(tmp_path):
    per_trial = [
        {
            "trial_index": 0,
            "policy_label": "EIG",
            "round_metrics": [{"source_rmse": 0.5}],
        },
        {
            "trial_index": 0,
            "policy_label": "StrategyEIG-d2",
            "round_metrics": [{"source_rmse": 0.2}],
        },
        {
            "trial_index": 1,
            "policy_label": "EIG",
            "round_metrics": [{"source_rmse": 0.8}],
        },
        {
            "trial_index": 1,
            "policy_label": "StrategyEIG-d2",
            "round_metrics": [{"source_rmse": 0.9}],
        },
    ]

    rows = _paired_trial_delta_rows(per_trial)
    plot_path = tmp_path / "paired_trial_deltas.png"
    _plot_paired_trial_differences({"per_trial": per_trial}, plot_path)

    assert rows == [
        {
            "trial_index": 0,
            "policy_label": "StrategyEIG-d2",
            "metric_name": "source_rmse",
            "policy_value": 0.2,
            "baseline_label": "EIG",
            "baseline_value": 0.5,
            "delta": pytest.approx(-0.3),
        },
        {
            "trial_index": 1,
            "policy_label": "StrategyEIG-d2",
            "metric_name": "source_rmse",
            "policy_value": 0.9,
            "baseline_label": "EIG",
            "baseline_value": 0.8,
            "delta": pytest.approx(0.1),
        },
    ]
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_depth_sweep_report_contains_paired_delta_table(tmp_path):
    summary = {
        "config_path": "configs/example.yaml",
        "num_trials": 2,
        "num_rounds": 3,
        "location_seed": 1304,
        "location_source_prior": "branch_decoy",
        "location_signal_model": "local_bump",
        "location_max_step_radius": 0.5,
        "include_myopic_controls": True,
        "paired_trial_delta_plot_path": "plots/example_paired_trial_rmse_deltas.png",
        "run_metadata": {
            "questioner_model": "google/gemma-4-E4B-it",
            "hostname": "test-host",
            "slurm_job_id": "123",
            "slurm_partition": "llm",
            "cuda_visible_devices": "0",
        },
        "token_usage": {
            "total": {
                "calls": 2,
                "prompt_tokens": 11,
                "completion_tokens": 22,
                "total_tokens": 33,
                "unknown_completion_token_records": 0,
            },
            "by_call_type": {
                "batched_chat": {
                    "calls": 2,
                    "prompt_tokens": 11,
                    "completion_tokens": 22,
                    "total_tokens": 33,
                    "unknown_completion_token_records": 0,
                }
            },
        },
        "aggregate": {
            "EIG": {"source_rmse": {"final_mean": 0.5, "final_std": 0.1}},
            "StrategyEIG-d2": {"source_rmse": {"final_mean": 0.2, "final_std": 0.05}},
            "StrategyEIG-myopic-d2": {"source_rmse": {"final_mean": 0.4, "final_std": 0.07}},
        },
        "paired_delta_vs_eig": {
            "StrategyEIG-d2": {
                "source_rmse": {
                    "n": 2,
                    "final_delta_mean": -0.3,
                    "final_delta_ci95": [-0.4, -0.2],
                    "wilcoxon_signed_rank_p": 0.25,
                },
                "posterior_entropy": {
                    "n": 2,
                    "final_delta_mean": -0.2,
                    "final_delta_ci95": [-0.3, -0.1],
                    "wilcoxon_signed_rank_p": 0.25,
                },
                "truth_log_probability": {
                    "n": 2,
                    "final_delta_mean": 0.6,
                    "final_delta_ci95": [0.4, 0.8],
                    "wilcoxon_signed_rank_p": 0.25,
                },
            },
            "StrategyEIG-myopic-d2": {
                "source_rmse": {
                    "n": 2,
                    "final_delta_mean": -0.1,
                    "final_delta_ci95": [-0.2, 0.0],
                    "wilcoxon_signed_rank_p": 0.5,
                },
                "posterior_entropy": {
                    "n": 2,
                    "final_delta_mean": -0.05,
                    "final_delta_ci95": [-0.1, 0.0],
                    "wilcoxon_signed_rank_p": 0.5,
                },
                "truth_log_probability": {
                    "n": 2,
                    "final_delta_mean": 0.2,
                    "final_delta_ci95": [0.0, 0.4],
                    "wilcoxon_signed_rank_p": 0.5,
                },
            },
        },
    }
    report_path = tmp_path / "REPORT.md"

    _write_depth_sweep_report(report_path, summary)
    text = report_path.read_text(encoding="utf-8")

    assert "Fixed-Root Location Depth Sweep Report" in text
    assert "Questioner model: `google/gemma-4-E4B-it`" in text
    assert "Matched-compute myopic controls: True" in text
    assert "SLURM job: `123` on partition `llm`" in text
    assert "## LLM Token Usage" in text
    assert "| `batched_chat` | 2 | 11 | 22 | 33 |" in text
    assert "Paired trial delta plot: `plots/example_paired_trial_rmse_deltas.png`" in text
    assert "| `StrategyEIG-d2` | `source_rmse` | 2 | -0.3000 | [-0.4000, -0.2000] | 0.25 |" in text
    assert "| `StrategyEIG-myopic-d2` | `source_rmse` | 2 | -0.1000 | [-0.2000, 0.0000] | 0.5 |" in text
