import pytest

from scripts.constrained_oracle_check import _write_oracle_report


def test_write_oracle_report_includes_planner_gap_and_figure(tmp_path):
    summary = {
        "config": {
            "source_prior": "branch_decoy",
            "signal_model": "local_bump",
            "source_radius": 2.2,
            "signal_lengthscale": 0.5,
            "signal_amplitude": 8.0,
            "max_step_radius": 0.5,
            "num_trials": 120,
            "num_rounds": 6,
            "planner_depth": 2,
            "planning_support_size": 6,
            "seed": 1304,
        },
        "greedy": {
            "rmse": {"final_mean": 0.5606},
            "entropy": {"final_mean": 1.2},
        },
        "planner": {
            "rmse": {"final_mean": 0.1529},
            "entropy": {"final_mean": 0.8},
        },
        "lawnmower": {
            "rmse": {"final_mean": 0.44},
            "entropy": {"final_mean": 1.0},
        },
        "random": {
            "rmse": {"final_mean": 0.9},
            "entropy": {"final_mean": 1.5},
        },
        "paired_rmse_planner_minus_greedy": {
            "final_delta_mean": -0.4077,
            "auc_delta_mean": -0.9306,
            "final_planner_win_rate": 0.525,
        },
        "paired_rmse_planner_minus_lawnmower": {
            "final_delta_mean": -0.2871,
            "auc_delta_mean": -0.5,
            "final_planner_win_rate": 0.6,
        },
        "paired_entropy_planner_minus_greedy": {
            "final_delta_mean": -0.4,
            "auc_delta_mean": -1.1,
            "final_planner_win_rate": 0.7,
        },
        "paired_entropy_planner_minus_lawnmower": {
            "final_delta_mean": -0.2,
            "auc_delta_mean": -0.6,
            "final_planner_win_rate": 0.65,
        },
        "power_for_strategy_closing_half_oracle_gap": {
            "required_trials_80_power": 42.0,
            "target_effect": 0.20385,
            "paired_sd": 0.5,
        },
        "public_plot_path": "plots/constrained_oracle/example_rmse.png",
    }
    report_path = tmp_path / "oracle_REPORT.md"

    _write_oracle_report(report_path, summary)
    text = report_path.read_text(encoding="utf-8")

    assert "Constrained Oracle Check" in text
    assert "Source prior: `branch_decoy`" in text
    assert "| RMSE | 0.5606 | 0.4400 | 0.9000 | 0.1529 | -0.4077 | -0.2871 | 0.525 |" in text
    assert "Power Check" in text
    assert "`42.0`" in text
    assert "![Oracle RMSE trace](plots/constrained_oracle/example_rmse.png)" in text


def test_write_oracle_report_requires_summary_metrics(tmp_path):
    with pytest.raises(KeyError):
        _write_oracle_report(tmp_path / "bad.md", {"config": {}})
