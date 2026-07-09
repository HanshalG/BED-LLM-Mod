import json

import pytest

from scripts.compare_location_depth_sweeps import (
    compare_depth_sweeps,
    headline_policy_labels,
    plot_comparison,
    plot_headline_rmse,
    write_comparison_report,
)


def _summary(max_step_radius):
    return {
        "config_path": "configs/example.yaml",
        "num_trials": 2,
        "num_rounds": 3,
        "location_source_prior": "branch_decoy",
        "location_signal_model": "local_bump",
        "location_max_step_radius": max_step_radius,
        "run_metadata": {
            "questioner_model": "google/gemma-4-E4B-it",
            "hostname": "test-host",
            "slurm_job_id": "123",
        },
        "token_usage": {
            "total": {
                "calls": 1,
                "prompt_tokens": 9,
                "completion_tokens": 10,
                "total_tokens": 19,
                "unknown_completion_token_records": 0,
            },
            "by_call_type": {
                "chat": {
                    "calls": 1,
                    "prompt_tokens": 9,
                    "completion_tokens": 10,
                    "total_tokens": 19,
                    "unknown_completion_token_records": 0,
                }
            },
        },
        "aggregate": {
            "EIG": {
                "source_rmse": {"final_mean": 0.5, "final_std": 0.1, "mean_trace": [0.9, 0.5]},
                "expected_posterior_rmse": {"final_mean": 0.55, "final_std": 0.1, "mean_trace": [0.95, 0.55]},
                "posterior_entropy": {"final_mean": 1.1, "final_std": 0.2, "mean_trace": [1.4, 1.1]},
                "truth_log_probability": {"final_mean": -1.0, "final_std": 0.2, "mean_trace": [-1.5, -1.0]},
            },
            "StrategyEIG-d2": {
                "source_rmse": {"final_mean": 0.2, "final_std": 0.05, "mean_trace": [0.8, 0.2]},
                "expected_posterior_rmse": {"final_mean": 0.25, "final_std": 0.05, "mean_trace": [0.85, 0.25]},
                "posterior_entropy": {"final_mean": 0.7, "final_std": 0.1, "mean_trace": [1.3, 0.7]},
                "truth_log_probability": {"final_mean": -0.4, "final_std": 0.2, "mean_trace": [-1.2, -0.4]},
            },
            "StrategyEIG-d1": {
                "source_rmse": {"final_mean": 0.45, "final_std": 0.07, "mean_trace": [0.88, 0.45]},
                "expected_posterior_rmse": {"final_mean": 0.5, "final_std": 0.07, "mean_trace": [0.9, 0.5]},
                "posterior_entropy": {"final_mean": 1.0, "final_std": 0.12, "mean_trace": [1.35, 1.0]},
                "truth_log_probability": {"final_mean": -0.9, "final_std": 0.2, "mean_trace": [-1.4, -0.9]},
            },
            "StrategyEIG-d3": {
                "source_rmse": {"final_mean": 0.18, "final_std": 0.04, "mean_trace": [0.78, 0.18]},
                "expected_posterior_rmse": {"final_mean": 0.23, "final_std": 0.04, "mean_trace": [0.82, 0.23]},
                "posterior_entropy": {"final_mean": 0.65, "final_std": 0.08, "mean_trace": [1.25, 0.65]},
                "truth_log_probability": {"final_mean": -0.35, "final_std": 0.2, "mean_trace": [-1.1, -0.35]},
            },
            "StrategyEIG-d5": {
                "source_rmse": {"final_mean": 0.16, "final_std": 0.03, "mean_trace": [0.76, 0.16]},
                "expected_posterior_rmse": {"final_mean": 0.2, "final_std": 0.03, "mean_trace": [0.8, 0.2]},
                "posterior_entropy": {"final_mean": 0.6, "final_std": 0.07, "mean_trace": [1.2, 0.6]},
                "truth_log_probability": {"final_mean": -0.3, "final_std": 0.2, "mean_trace": [-1.0, -0.3]},
            },
            "StrategyEIG-myopic-d2": {
                "source_rmse": {"final_mean": 0.4, "final_std": 0.06, "mean_trace": [0.85, 0.4]},
                "expected_posterior_rmse": {"final_mean": 0.45, "final_std": 0.06, "mean_trace": [0.9, 0.45]},
                "posterior_entropy": {"final_mean": 0.9, "final_std": 0.12, "mean_trace": [1.35, 0.9]},
                "truth_log_probability": {"final_mean": -0.8, "final_std": 0.2, "mean_trace": [-1.3, -0.8]},
            },
        },
        "paired_delta_vs_eig": {
            "StrategyEIG-d2": {
                "source_rmse": {
                    "final_delta_mean": -0.3,
                    "final_delta_ci95": [-0.4, -0.2],
                    "wilcoxon_signed_rank_p": 0.25,
                },
                "expected_posterior_rmse": {
                    "final_delta_mean": -0.3,
                    "final_delta_ci95": [-0.4, -0.2],
                    "wilcoxon_signed_rank_p": 0.25,
                },
                "posterior_entropy": {
                    "final_delta_mean": -0.4,
                    "final_delta_ci95": [-0.6, -0.1],
                    "wilcoxon_signed_rank_p": 0.125,
                },
                "truth_log_probability": {
                    "final_delta_mean": 0.6,
                    "final_delta_ci95": [0.4, 0.8],
                    "wilcoxon_signed_rank_p": 0.25,
                },
            },
            "StrategyEIG-d1": {
                "source_rmse": {
                    "final_delta_mean": -0.05,
                    "final_delta_ci95": [-0.12, 0.02],
                    "wilcoxon_signed_rank_p": 0.75,
                },
                "expected_posterior_rmse": {
                    "final_delta_mean": -0.05,
                    "final_delta_ci95": [-0.12, 0.02],
                    "wilcoxon_signed_rank_p": 0.75,
                },
                "posterior_entropy": {
                    "final_delta_mean": -0.1,
                    "final_delta_ci95": [-0.2, 0.0],
                    "wilcoxon_signed_rank_p": 0.5,
                },
                "truth_log_probability": {
                    "final_delta_mean": 0.1,
                    "final_delta_ci95": [-0.02, 0.22],
                    "wilcoxon_signed_rank_p": 0.75,
                },
            },
            "StrategyEIG-d3": {
                "source_rmse": {
                    "final_delta_mean": -0.32,
                    "final_delta_ci95": [-0.44, -0.2],
                    "wilcoxon_signed_rank_p": 0.125,
                },
                "expected_posterior_rmse": {
                    "final_delta_mean": -0.32,
                    "final_delta_ci95": [-0.44, -0.2],
                    "wilcoxon_signed_rank_p": 0.125,
                },
                "posterior_entropy": {
                    "final_delta_mean": -0.45,
                    "final_delta_ci95": [-0.65, -0.2],
                    "wilcoxon_signed_rank_p": 0.0625,
                },
                "truth_log_probability": {
                    "final_delta_mean": 0.65,
                    "final_delta_ci95": [0.45, 0.8],
                    "wilcoxon_signed_rank_p": 0.125,
                },
            },
            "StrategyEIG-d5": {
                "source_rmse": {
                    "final_delta_mean": -0.34,
                    "final_delta_ci95": [-0.46, -0.22],
                    "wilcoxon_signed_rank_p": 0.0625,
                },
                "expected_posterior_rmse": {
                    "final_delta_mean": -0.35,
                    "final_delta_ci95": [-0.47, -0.23],
                    "wilcoxon_signed_rank_p": 0.0625,
                },
                "posterior_entropy": {
                    "final_delta_mean": -0.5,
                    "final_delta_ci95": [-0.7, -0.25],
                    "wilcoxon_signed_rank_p": 0.0625,
                },
                "truth_log_probability": {
                    "final_delta_mean": 0.7,
                    "final_delta_ci95": [0.5, 0.9],
                    "wilcoxon_signed_rank_p": 0.0625,
                },
            },
            "StrategyEIG-myopic-d2": {
                "source_rmse": {
                    "final_delta_mean": -0.1,
                    "final_delta_ci95": [-0.2, 0.0],
                    "wilcoxon_signed_rank_p": 0.5,
                },
                "expected_posterior_rmse": {
                    "final_delta_mean": -0.1,
                    "final_delta_ci95": [-0.2, 0.0],
                    "wilcoxon_signed_rank_p": 0.5,
                },
                "posterior_entropy": {
                    "final_delta_mean": -0.2,
                    "final_delta_ci95": [-0.3, -0.05],
                    "wilcoxon_signed_rank_p": 0.25,
                },
                "truth_log_probability": {
                    "final_delta_mean": 0.2,
                    "final_delta_ci95": [0.0, 0.4],
                    "wilcoxon_signed_rank_p": 0.5,
                },
            }
        },
    }


def test_compare_depth_sweeps_extracts_constrained_and_unconstrained_fields():
    comparison = compare_depth_sweeps(_summary(0.5), _summary(None))

    assert comparison["constrained"]["max_step_radius"] == pytest.approx(0.5)
    assert comparison["unconstrained"]["max_step_radius"] is None
    assert comparison["constrained"]["run_metadata"]["questioner_model"] == "google/gemma-4-E4B-it"
    assert comparison["constrained"]["policies"]["EIG"]["final_rmse_mean"] == pytest.approx(0.5)
    assert comparison["constrained"]["policies"]["StrategyEIG-d2"]["paired_delta_vs_eig_mean"] == pytest.approx(-0.3)
    assert comparison["constrained"]["policies"]["StrategyEIG-d2"]["paired_delta_vs_eig_ci95"] == [-0.4, -0.2]
    assert list(comparison["constrained"]["policies"]) == [
        "EIG",
        "StrategyEIG-d1",
        "StrategyEIG-d2",
        "StrategyEIG-d3",
        "StrategyEIG-d5",
        "StrategyEIG-myopic-d2",
    ]
    assert headline_policy_labels(comparison) == ["EIG", "StrategyEIG-d1", "StrategyEIG-d3", "StrategyEIG-d5"]
    assert (
        comparison["constrained"]["policies"]["StrategyEIG-d2"]["paired_delta_vs_eig"]["posterior_entropy"]["mean"]
        == pytest.approx(-0.4)
    )
    assert (
        comparison["constrained"]["policies"]["StrategyEIG-d2"]["paired_delta_vs_eig"]["posterior_entropy"]["ci95"]
        == [-0.6, -0.1]
    )
    assert comparison["constrained"]["token_usage"]["total"]["total_tokens"] == 19


def test_write_comparison_report_mentions_both_sides(tmp_path):
    comparison = compare_depth_sweeps(_summary(0.5), _summary(None))
    report_path = tmp_path / "REPORT.md"

    write_comparison_report(report_path, comparison)
    text = report_path.read_text(encoding="utf-8")

    assert "Location Depth Sweep Contrast Report" in text
    assert "## Constrained" in text
    assert "## Unconstrained" in text
    assert "Questioner model: `google/gemma-4-E4B-it`" in text
    assert "SLURM job: `123`" in text
    assert "## LLM Token Usage" in text
    assert "| `chat` | 1 | 9 | 10 | 19 |" in text
    assert "## Headline Constrained Depths" in text
    assert "| `StrategyEIG-d3` | 0.1800 | 0.0400 | -0.3200 | [-0.4400, -0.2000] | 0.1250 |" in text
    assert "| `StrategyEIG-d2` | `source_rmse` | 0.2000 | -0.3000 | [-0.4000, -0.2000] | 0.2500 |" in text
    assert "| `StrategyEIG-d2` | `expected_posterior_rmse` | 0.2500 | -0.3000 | [-0.4000, -0.2000] | 0.2500 |" in text
    assert "| `StrategyEIG-d2` | `posterior_entropy` | 0.7000 | -0.4000 | [-0.6000, -0.1000] | 0.1250 |" in text
    assert "| `StrategyEIG-d2` | `truth_log_probability` | -0.4000 | 0.6000 | [0.4000, 0.8000] | 0.2500 |" in text
    assert "| `StrategyEIG-myopic-d2` | `source_rmse` | 0.4000 | -0.1000 | [-0.2000, 0.0000] | 0.5000 |" in text


def test_plot_comparison_includes_rmse_and_entropy_panels(tmp_path):
    comparison = compare_depth_sweeps(_summary(0.5), _summary(None))
    plot_path = tmp_path / "contrast.png"
    headline_plot_path = tmp_path / "headline.png"

    plot_comparison(plot_path, comparison)
    plot_headline_rmse(headline_plot_path, comparison)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert headline_plot_path.exists()
    assert headline_plot_path.stat().st_size > 0


def test_comparison_summary_is_json_serializable():
    comparison = compare_depth_sweeps(_summary(0.5), _summary(None))

    assert json.loads(json.dumps(comparison))["constrained"]["source_prior"] == "branch_decoy"
