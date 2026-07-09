import pytest

from scripts.combine_location_fixed_root_depth_sweeps import combine_depth_sweep_summaries


def _summary(trial_index: int, eig_rmse: float, strategy_rmse: float) -> dict:
    return {
        "config_path": "configs/example.yaml",
        "max_depth": 3,
        "strategy_depths": [1],
        "eval_depths": [1],
        "myopic_control_depths": [],
        "num_trials": 1,
        "num_rounds": 1,
        "location_seed": 1304,
        "location_source_prior": "branch_decoy",
        "location_source_radius": 2.2,
        "location_signal_model": "local_bump",
        "location_signal_lengthscale": 0.5,
        "location_signal_amplitude": 8.0,
        "location_noise_sd": 0.15,
        "location_max_step_radius": 0.5,
        "location_strategy_num_rollouts": 8,
        "location_strategy_num_candidates": 10,
        "location_strategy_discount_factor": 1.0,
        "location_strategy_rollout_score_mode": "entropy",
        "location_strategy_rollout_scoring_support_mode": "truth_plus_sampled",
        "location_strategy_rollout_refresh_hypotheses_each_step": True,
        "include_myopic_controls": False,
        "token_usage": {"total": {"calls": 2, "total_tokens": 10}},
        "run_metadata": {"questioner_model": "google/gemma-4-26B-A4B-it"},
        "ranking_fidelity": {"1": {"1": {"n": 1}}},
        "per_trial": [
            {
                "trial_index": trial_index,
                "policy_label": "EIG",
                "policy_kind": "EIG",
                "policy_depth": None,
                "round_metrics": [
                    {
                        "source_rmse": eig_rmse,
                        "posterior_entropy": 1.0,
                        "truth_log_probability": -1.0,
                    }
                ],
            },
            {
                "trial_index": trial_index,
                "policy_label": "StrategyEIG-d1",
                "policy_kind": "StrategyEIG",
                "policy_depth": 1,
                "round_metrics": [
                    {
                        "source_rmse": strategy_rmse,
                        "posterior_entropy": 0.8,
                        "truth_log_probability": -0.8,
                    }
                ],
            },
        ],
        "aggregate": {},
        "paired_delta_vs_eig": {},
    }


def test_combine_depth_sweep_summaries_recomputes_metrics_across_trial_blocks():
    combined = combine_depth_sweep_summaries(
        [
            _summary(0, eig_rmse=1.0, strategy_rmse=0.8),
            _summary(1, eig_rmse=0.6, strategy_rmse=0.4),
        ]
    )

    assert combined["num_trials"] == 2
    assert combined["trial_indices"] == [0, 1]
    assert combined["aggregate"]["EIG"]["source_rmse"]["final_mean"] == pytest.approx(0.8)
    assert combined["aggregate"]["StrategyEIG-d1"]["source_rmse"]["final_mean"] == pytest.approx(0.6)
    assert combined["paired_delta_vs_eig"]["StrategyEIG-d1"]["source_rmse"]["final_delta_mean"] == pytest.approx(-0.2)
    assert combined["token_usage"]["total"]["calls"] == 4
    assert combined["token_usage"]["total"]["total_tokens"] == 20


def test_combine_depth_sweep_summaries_rejects_overlapping_trial_policy_records():
    with pytest.raises(ValueError, match="Duplicate trial/policy"):
        combine_depth_sweep_summaries([_summary(0, 1.0, 0.8), _summary(0, 0.9, 0.7)])
