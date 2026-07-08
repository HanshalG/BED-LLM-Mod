from scripts.constrained_oracle_check import OracleConfig
from scripts.constrained_oracle_robustness_sweep import run_robustness_sweep


def test_run_robustness_sweep_writes_summary_report_and_heatmap(tmp_path, monkeypatch):
    import scripts.constrained_oracle_robustness_sweep as sweep

    def fake_run_oracle_check(config, output_dir, run_name, plot_dir=None):
        del plot_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "greedy": {"rmse": {"final_mean": 0.6}},
            "planner": {"rmse": {"final_mean": 0.2}},
            "lawnmower": {"rmse": {"final_mean": 0.5}},
            "paired_rmse_planner_minus_greedy": {
                "final_delta_mean": -0.4,
                "final_delta_std": 0.1,
                "final_planner_win_rate": 0.75,
            },
            "paired_rmse_planner_minus_lawnmower": {
                "final_delta_mean": -0.3,
                "final_planner_win_rate": 0.7,
            },
            "power_for_strategy_closing_half_oracle_gap": {
                "required_trials_80_power": 12,
            },
        }
        (output_dir / f"{run_name}_summary.json").write_text("{}", encoding="utf-8")
        return summary

    monkeypatch.setattr(sweep, "run_oracle_check", fake_run_oracle_check)
    payload = run_robustness_sweep(
        base_config=OracleConfig(
            num_trials=2,
            num_rounds=2,
            num_particles=4,
            grid_size=5,
            arena=1.0,
            max_step_radius=1.0,
            noise_sd=0.1,
            planner_depth=2,
            planning_support_size=2,
            source_prior="branch_decoy",
            source_radius=1.0,
            seed=1304,
            num_sources=1,
            fixed_first_query=True,
            signal_model="local_bump",
            signal_lengthscale=0.5,
            signal_amplitude=8.0,
        ),
        signal_lengthscales=[0.5],
        max_step_radii=[0.5, 0.7],
        noise_sds=[0.1],
        output_dir=tmp_path / "results",
        plot_dir=tmp_path / "plots",
        run_name="demo",
    )

    assert len(payload["cells"]) == 2
    assert payload["cells"][0]["metrics"]["planner_minus_greedy_final_rmse"] == -0.4
    assert (tmp_path / "results/demo_summary.json").exists()
    assert (tmp_path / "results/demo_REPORT.md").exists()
    assert (tmp_path / "plots/demo_heatmap.png").stat().st_size > 0
