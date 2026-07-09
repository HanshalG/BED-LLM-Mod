import json

import numpy as np
import pytest

from core import BeliefState
from environments.location_finding.types import LocationStrategyEvaluation
from helpers import Config
from scripts.strategy_ranking_fidelity import (
    _aggregate_variant_metrics,
    _copy_config_for_score_variant,
    _diagnostic_num_rounds,
    _generate_probe_states,
    _gate_assessment,
    _posterior_expected_rmse,
    _posterior_state_record,
    _state_depth_metrics,
    _write_report,
)


def _evaluation(mean_score: float, rollout_scores: list[float]) -> LocationStrategyEvaluation:
    return LocationStrategyEvaluation(
        strategy="test",
        mean_score=mean_score,
        score_variance=float(np.var(rollout_scores)),
        root_query_fingerprint="[0, 0]",
        rollout_scores=rollout_scores,
        root_query=(0.0, 0.0),
    )


def test_state_depth_metrics_reports_rank_regret_and_snr():
    evaluations = [
        _evaluation(0.1, [0.05, 0.15]),
        _evaluation(0.3, [0.25, 0.35]),
        _evaluation(0.2, [0.15, 0.25]),
    ]
    realized_entropy = [1.0, 3.0, 2.0]
    realized_rmse = [0.1, 0.3, 0.2]
    realized_truth = [-3.0, -1.0, -2.0]

    metrics = _state_depth_metrics(
        evaluations,
        realized_entropy,
        realized_rmse,
        realized_truth,
        {
            "within_strategy_query_distance_mean": 0.5,
            "between_strategy_query_distance_mean": 2.0,
            "between_over_within_query_distance": 4.0,
        },
    )

    assert metrics["n"] == 3
    assert metrics["spearman_entropy"] == pytest.approx(1.0)
    assert metrics["spearman_rmse"] == pytest.approx(1.0)
    assert metrics["spearman_truth_log_prob"] == pytest.approx(1.0)
    assert metrics["top1_regret_entropy"] == pytest.approx(0.0)
    assert metrics["score_var_between_strategies"] > 0.0
    assert metrics["score_var_within_strategy"] > 0.0
    assert metrics["snr_between_over_within"] > 0.0
    assert metrics["between_over_within_query_distance"] == pytest.approx(4.0)


def test_variant_aggregation_and_report_include_before_after_rows(tmp_path):
    baseline_metrics = {
        "2": {
            "n": 3,
            "spearman_entropy": -0.5,
            "pearson_entropy": -0.4,
            "spearman_rmse": -0.3,
            "pearson_rmse": -0.2,
            "spearman_truth_log_prob": -0.1,
            "pearson_truth_log_prob": -0.1,
            "top1_regret_entropy": 1.5,
            "score_var_between_strategies": 0.01,
            "score_var_within_strategy": 0.10,
            "snr_between_over_within": 0.1,
            "within_strategy_query_distance_mean": 0.5,
            "between_strategy_query_distance_mean": 0.6,
            "between_over_within_query_distance": 1.2,
        }
    }
    configured_metrics = {
        "2": {
            "n": 3,
            "spearman_entropy": 1.0,
            "pearson_entropy": 0.9,
            "spearman_rmse": 0.8,
            "pearson_rmse": 0.7,
            "spearman_truth_log_prob": 0.6,
            "pearson_truth_log_prob": 0.5,
            "top1_regret_entropy": 0.0,
            "score_var_between_strategies": 0.20,
            "score_var_within_strategy": 0.05,
            "snr_between_over_within": 4.0,
            "within_strategy_query_distance_mean": 0.5,
            "between_strategy_query_distance_mean": 2.0,
            "between_over_within_query_distance": 4.0,
        }
    }
    records = [
        {
            "score_variant_metrics": {
                "baseline": baseline_metrics,
                "configured": configured_metrics,
            }
        }
    ]

    aggregate = _aggregate_variant_metrics(
        records,
        [2],
        ["baseline", "configured"],
        np.random.default_rng(0),
    )

    assert aggregate["baseline"]["2"]["spearman_entropy"]["mean"] == pytest.approx(-0.5)
    assert aggregate["configured"]["2"]["spearman_entropy"]["mean"] == pytest.approx(1.0)

    summary = {
        "config_path": "configs/example.yaml",
        "num_probe_states": 1,
        "num_trials": 1,
        "state_rounds": [0],
        "depths": [2],
        "deployments": 2,
        "target_num_candidates": 8,
        "score_variants": ["baseline", "configured"],
        "token_usage": {
            "total": {
                "calls": 1,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "unknown_completion_token_records": 0,
            },
            "by_call_type": {
                "chat": {
                    "calls": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "unknown_completion_token_records": 0,
                }
            },
        },
        "aggregate": aggregate,
    }
    gate = _gate_assessment(summary)
    report_path = tmp_path / "REPORT.md"
    _write_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")

    assert gate["status"] == "proceed"
    assert gate["best_depth"] == 2
    assert gate["best_spearman_entropy"] == pytest.approx(1.0)
    assert gate["best_spearman_truth_log_prob"] == pytest.approx(0.6)
    assert "## Gate" in report
    assert "Status: `proceed`" in report
    assert "Target candidates per probe: 8" in report
    assert "Spearman truth log-prob" in report
    assert "query distance ratio" in report
    assert "| baseline | 2 |" in report
    assert "| configured | 2 |" in report
    assert "Top-1 regret" in report
    assert "## LLM Token Usage" in report
    assert "| `chat` | 1 | 10 | 20 | 30 |" in report


def test_gate_assessment_stops_below_threshold():
    summary = {
        "depths": [2, 3],
        "score_variants": ["configured"],
        "aggregate": {
            "configured": {
                "2": {"spearman_entropy": {"mean": 0.1}, "spearman_truth_log_prob": {"mean": 0.2}},
                "3": {"spearman_entropy": {"mean": 0.39}, "spearman_truth_log_prob": {"mean": 0.2}},
            }
        },
    }

    gate = _gate_assessment(summary)

    assert gate["status"] == "stop"
    assert gate["best_depth"] == 3
    assert gate["best_spearman_entropy"] == pytest.approx(0.39)


def test_gate_assessment_warns_when_entropy_passes_but_truth_does_not():
    summary = {
        "depths": [2],
        "score_variants": ["configured"],
        "aggregate": {
            "configured": {
                "2": {
                    "spearman_entropy": {"mean": 0.6},
                    "spearman_truth_log_prob": {"mean": 0.0},
                },
            }
        },
    }

    gate = _gate_assessment(summary)

    assert gate["status"] == "calibration_warning"
    assert "truth-log-prob" in gate["message"]


def test_baseline_score_variant_restores_legacy_refresh_settings():
    config = Config(
        task="location_finding",
        location_strategy_rollout_scoring_support_mode="fixed_common",
        location_strategy_rollout_final_refresh_enabled=False,
        location_strategy_rollout_refresh_hypotheses_each_step=True,
    )

    baseline = _copy_config_for_score_variant(config, "baseline")
    configured = _copy_config_for_score_variant(config, "configured")

    assert baseline.location_strategy_rollout_scoring_support_mode == "union"
    assert baseline.location_strategy_rollout_final_refresh_enabled is True
    assert baseline.location_strategy_rollout_refresh_hypotheses_each_step is False
    assert configured.location_strategy_rollout_scoring_support_mode == "fixed_common"
    assert configured.location_strategy_rollout_final_refresh_enabled is False
    assert configured.location_strategy_rollout_refresh_hypotheses_each_step is True


def test_posterior_state_record_includes_expected_rmse_and_support():
    belief = BeliefState(
        hypotheses=[((0.0, 0.0),), ((2.0, 0.0),)],
        probabilities=[0.25, 0.75],
    )
    hidden_state = np.asarray([[1.0, 0.0]], dtype=float)

    expected = _posterior_expected_rmse(belief, hidden_state)
    record = _posterior_state_record(
        belief,
        hidden_state,
        candidate_index=3,
        replicate_index=4,
    )

    assert expected == pytest.approx(np.sqrt(0.5))
    assert record["expected_rmse"] == pytest.approx(np.sqrt(0.5))
    assert record["candidate_index"] == 3
    assert record["replicate_index"] == 4
    assert record["hypotheses"] == [[[0.0, 0.0]], [[2.0, 0.0]]]
    assert record["probabilities"] == [0.25, 0.75]


def test_diagnostic_num_rounds_leaves_horizon_after_latest_probe_state():
    assert _diagnostic_num_rounds([0, 3, 6], [2, 3, 5]) == 11
    assert _diagnostic_num_rounds([0], [2, 3]) == 3


def test_generate_probe_states_uses_trial_offset_for_trial_ids():
    class FakeEnv:
        def sample_hidden_state_for_trial(self, trial_index, rng):
            del trial_index
            return rng.normal(size=(1, 2))

        def initial_belief_state(self, questioner, config):
            del questioner, config
            return BeliefState(hypotheses=[((0.0, 0.0),)], probabilities=[1.0])

    probes = _generate_probe_states(
        FakeEnv(),
        questioner=None,
        config=Config(task="location_finding", location_num_sources=1, location_seed=1304),
        num_trials=3,
        trial_offset=7,
        state_rounds=[0],
        rng=np.random.default_rng(0),
    )

    assert [probe.trial_index for probe in probes] == [7, 8, 9]

    repeated = _generate_probe_states(
        FakeEnv(),
        questioner=None,
        config=Config(task="location_finding", location_num_sources=1, location_seed=1304),
        num_trials=3,
        trial_offset=7,
        state_rounds=[0],
        rng=np.random.default_rng(0),
    )
    different_offset = _generate_probe_states(
        FakeEnv(),
        questioner=None,
        config=Config(task="location_finding", location_num_sources=1, location_seed=1304),
        num_trials=3,
        trial_offset=10,
        state_rounds=[0],
        rng=np.random.default_rng(0),
    )

    assert [probe.hidden_state.tolist() for probe in probes] == [
        probe.hidden_state.tolist()
        for probe in repeated
    ]
    assert [probe.hidden_state.tolist() for probe in probes] != [
        probe.hidden_state.tolist()
        for probe in different_offset
    ]
