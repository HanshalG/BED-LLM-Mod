import json

import pytest

from scripts.compare_location_rollout_ablation import (
    compare_rollout_ablation,
    plot_rollout_ablation,
    write_rollout_ablation_report,
)


def _summary(path, *, rollouts, rmse_delta, entropy_delta):
    payload = {
        "config_path": f"configs/rollouts{rollouts}.yaml",
        "num_trials": 50,
        "num_rounds": 6,
        "location_strategy_num_rollouts": rollouts,
        "location_max_step_radius": 0.5,
        "run_metadata": {"questioner_model": "google/gemma-4-E4B-it"},
        "paired_delta_vs_eig": {
            "StrategyEIG-d3": {
                "source_rmse": {
                    "final_delta_mean": rmse_delta,
                    "final_delta_ci95": [rmse_delta - 0.1, rmse_delta + 0.1],
                    "wilcoxon_signed_rank_p": 0.25,
                },
                "posterior_entropy": {
                    "final_delta_mean": entropy_delta,
                    "final_delta_ci95": [entropy_delta - 0.2, entropy_delta + 0.2],
                    "wilcoxon_signed_rank_p": 0.125,
                },
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compare_rollout_ablation_sorts_by_rollout_count(tmp_path):
    high = _summary(tmp_path / "high.json", rollouts=32, rmse_delta=-0.3, entropy_delta=-0.5)
    low = _summary(tmp_path / "low.json", rollouts=8, rmse_delta=-0.1, entropy_delta=-0.2)

    comparison = compare_rollout_ablation([high, low])

    assert [run["rollouts"] for run in comparison["runs"]] == [8, 32]
    assert (
        comparison["runs"][0]["policies"]["StrategyEIG-d3"]["source_rmse"]["final_delta_mean"]
        == pytest.approx(-0.1)
    )
    assert (
        comparison["runs"][1]["policies"]["StrategyEIG-d3"]["posterior_entropy"]["final_delta_mean"]
        == pytest.approx(-0.5)
    )


def test_rollout_ablation_report_and_plot(tmp_path):
    first = _summary(tmp_path / "r8.json", rollouts=8, rmse_delta=-0.1, entropy_delta=-0.2)
    second = _summary(tmp_path / "r32.json", rollouts=32, rmse_delta=-0.3, entropy_delta=-0.5)
    comparison = compare_rollout_ablation([first, second])
    report_path = tmp_path / "REPORT.md"
    plot_path = tmp_path / "plot.png"

    write_rollout_ablation_report(report_path, comparison)
    plot_rollout_ablation(plot_path, comparison)
    report = report_path.read_text(encoding="utf-8")

    assert "| 8 | `StrategyEIG-d3` | `source_rmse` | -0.1000 | [-0.2000, 0.0000] | 0.2500 |" in report
    assert "| 32 | `StrategyEIG-d3` | `posterior_entropy` | -0.5000 | [-0.7000, -0.3000] | 0.1250 |" in report
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
