import json
from pathlib import Path

from scripts.build_path_a_package import build_path_a_package


def _summary(max_step_radius):
    aggregate = {
        "EIG": {
            "source_rmse": {"final_mean": 0.5, "final_std": 0.1, "mean_trace": [0.9, 0.5]},
            "posterior_entropy": {"final_mean": 1.0, "final_std": 0.1, "mean_trace": [1.2, 1.0]},
        }
    }
    paired = {}
    for depth, final in [(1, 0.45), (3, 0.2), (5, 0.18)]:
        label = f"StrategyEIG-d{depth}"
        aggregate[label] = {
            "source_rmse": {"final_mean": final, "final_std": 0.05, "mean_trace": [0.8, final]},
            "posterior_entropy": {"final_mean": 0.8, "final_std": 0.1, "mean_trace": [1.1, 0.8]},
        }
        paired[label] = {
            "source_rmse": {
                "final_delta_mean": final - 0.5,
                "final_delta_ci95": [final - 0.6, final - 0.4],
                "wilcoxon_signed_rank_p": 0.25,
            },
            "posterior_entropy": {
                "final_delta_mean": -0.2,
                "final_delta_ci95": [-0.3, -0.1],
                "wilcoxon_signed_rank_p": 0.25,
            },
        }
    aggregate["StrategyEIG-myopic-d3"] = {
        "source_rmse": {"final_mean": 0.4, "final_std": 0.06, "mean_trace": [0.85, 0.4]},
        "posterior_entropy": {"final_mean": 0.9, "final_std": 0.12, "mean_trace": [1.15, 0.9]},
    }
    paired["StrategyEIG-myopic-d3"] = {
        "source_rmse": {
            "final_delta_mean": -0.1,
            "final_delta_ci95": [-0.2, 0.0],
            "wilcoxon_signed_rank_p": 0.5,
        },
        "posterior_entropy": {
            "final_delta_mean": -0.1,
            "final_delta_ci95": [-0.2, 0.0],
            "wilcoxon_signed_rank_p": 0.5,
        },
    }
    return {
        "config_path": "configs/example.yaml",
        "max_depth": 5,
        "num_trials": 2,
        "num_rounds": 2,
        "location_source_prior": "branch_decoy",
        "location_signal_model": "local_bump",
        "location_max_step_radius": max_step_radius,
        "location_strategy_num_candidates": 4,
        "location_strategy_num_rollouts": 8,
        "run_metadata": {"questioner_model": "google/gemma-4-26B-A4B-it"},
        "token_usage": {
            "total": {
                "calls": 1,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "unknown_completion_token_records": 0,
            },
            "by_call_type": {},
        },
        "aggregate": aggregate,
        "aggregate_by_strategy_depth": {},
        "paired_delta_vs_eig": paired,
        "per_trial": [
            {
                "trial_index": 0,
                "policy_label": "EIG",
                "policy_kind": "EIG",
                "hidden_state": [[1.0, 0.0]],
                "history": [
                    {"action": [0.0, 0.0], "observation": {"query": [0.0, 0.0], "value": 1.0}},
                    {"action": [0.5, 0.0], "observation": {"query": [0.5, 0.0], "value": 2.0}},
                ],
                "round_metrics": [{"source_rmse": 0.8}, {"source_rmse": 0.6, "truth_log_probability": -2.0}],
            },
            {
                "trial_index": 0,
                "policy_label": "StrategyEIG-d3",
                "policy_kind": "StrategyEIG",
                "hidden_state": [[1.0, 0.0]],
                "history": [
                    {"action": [0.0, 0.0], "observation": {"query": [0.0, 0.0], "value": 1.0}},
                    {"action": [1.0, 0.0], "observation": {"query": [1.0, 0.0], "value": 8.0}},
                ],
                "round_metrics": [{"source_rmse": 0.7}, {"source_rmse": 0.1, "truth_log_probability": -0.2}],
            },
        ],
    }


def _write_run(run_dir, summary):
    run_dir.mkdir(parents=True)
    (run_dir / "fixed_root_depth_sweep_metrics.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    decisions = [
        {
            "trial_index": 0,
            "policy_label": "StrategyEIG-d3",
            "round_index": 0,
            "selected_eig": 0.5,
            "selected_root_query": [0.0, 0.0],
            "selected_strategy": "Move toward the right branch, then exploit the strong signal.",
        }
    ]
    (run_dir / "fixed_root_depth_sweep_decisions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in decisions),
        encoding="utf-8",
    )


def _write_existing_gate_artifacts(root):
    (root / "results/ranking_fidelity").mkdir(parents=True)
    (root / "results/ranking_fidelity/REPORT.md").write_text(
        "Spearman top-1 SNR",
        encoding="utf-8",
    )
    (root / "results/constrained_oracle").mkdir(parents=True)
    (root / "results/constrained_oracle/REPORT.md").write_text(
        "Planner greedy RMSE",
        encoding="utf-8",
    )


def test_build_path_a_package_creates_reports_plots_costs_and_validates(tmp_path):
    constrained = tmp_path / "runs/constrained"
    unconstrained = tmp_path / "runs/unconstrained"
    _write_run(constrained, _summary(0.5))
    _write_run(unconstrained, _summary(None))
    _write_existing_gate_artifacts(tmp_path)

    payload = build_path_a_package(
        constrained=constrained,
        unconstrained=unconstrained,
        output_dir=tmp_path / "results/location_depth_sweeps",
        plot_dir=tmp_path / "plots/location_depth_sweeps",
        cost_dir=tmp_path / "results/cost_vs_depth",
        qualitative_dir=tmp_path / "results/location_qualitative",
        run_name="demo",
        validate_root=tmp_path,
    )

    assert payload["validation"]["ok"] is True
    for key in (
        "comparison_summary",
        "comparison_report",
        "contrast_plot",
        "headline_plot",
        "cost_json",
        "cost_report",
        "qualitative_constrained_json",
        "qualitative_constrained_report",
        "qualitative_unconstrained_json",
        "qualitative_unconstrained_report",
    ):
        assert Path(payload[key]).exists()
    assert (tmp_path / "results/location_depth_sweeps/demo_REPORT.md").exists()
    assert (tmp_path / "plots/location_depth_sweeps/demo_headline_rmse.png").stat().st_size > 0
    assert "StrategyEIG vs brute-force n-step EIG proxy" in (
        tmp_path / "results/cost_vs_depth/demo_cost_vs_depth.md"
    ).read_text(encoding="utf-8")
