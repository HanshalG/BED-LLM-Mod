import json

from scripts.validate_path_a_package import summary_payload, validate_path_a_package


def _write(path, text="ok"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_depth_summary(root):
    policies = {
        label: {"final_rmse_mean": 0.5}
        for label in (
            "EIG",
            "StrategyEIG-d1",
            "StrategyEIG-d3",
            "StrategyEIG-d5",
            "StrategyEIG-myopic-d3",
            "StrategyEIG-myopic-d5",
        )
    }
    _write(
        root / "results/location_depth_sweeps/demo_summary.json",
        json.dumps(
            {
                "constrained": {"policies": policies},
                "unconstrained": {"policies": {"EIG": {"final_rmse_mean": 0.6}}},
            }
        ),
    )


def test_validate_path_a_package_passes_complete_mpp(tmp_path):
    _write(
        tmp_path / "results/ranking_fidelity/REPORT.md",
        "Spearman correlations, top-1 regret, and SNR all reported.",
    )
    ranking_plot = tmp_path / "results/ranking_fidelity/demo_plot.png"
    ranking_plot.parent.mkdir(parents=True, exist_ok=True)
    ranking_plot.write_bytes(b"png")
    _write(
        tmp_path / "results/constrained_oracle/REPORT.md",
        "Planner beats greedy on RMSE in the constrained oracle check.",
    )
    robustness_heatmap = tmp_path / "plots/constrained_oracle_robustness/demo_heatmap.png"
    robustness_heatmap.parent.mkdir(parents=True, exist_ok=True)
    robustness_heatmap.write_bytes(b"png")
    _write(
        tmp_path / "results/location_depth_sweeps/demo_REPORT.md",
        "\n".join(
            [
                "## Headline Constrained Depths",
                "`StrategyEIG-d1`",
                "`StrategyEIG-d3`",
                "`StrategyEIG-d5`",
                "`StrategyEIG-myopic-d3`",
                "`StrategyEIG-myopic-d5`",
                "`truth_log_probability`",
                "`expected_posterior_rmse`",
                "paired final delta vs EIG",
            ]
        ),
    )
    _write_depth_summary(tmp_path)
    plot_path = tmp_path / "plots/location_depth_sweeps/demo_headline_rmse.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_bytes(b"png")
    depth_contrast_plot = tmp_path / "plots/location_depth_sweeps/demo_depth_contrast.png"
    depth_contrast_plot.write_bytes(b"png")
    paired_plot_path = tmp_path / "plots/location_depth_sweeps/demo_paired_trial_rmse_deltas.png"
    paired_plot_path.write_bytes(b"png")
    truth_log_paired_plot_path = (
        tmp_path / "plots/location_depth_sweeps/demo_paired_trial_truth_log_probability_deltas.png"
    )
    truth_log_paired_plot_path.write_bytes(b"png")
    _write(
        tmp_path / "results/location_qualitative/demo_constrained_qualitative_examples.md",
        "Qualitative Location Strategy Examples\nRMSE delta vs EIG\nRoot query",
    )
    qualitative_plot = tmp_path / "results/location_qualitative/demo_constrained_qualitative_example_1.png"
    qualitative_plot.parent.mkdir(parents=True, exist_ok=True)
    qualitative_plot.write_bytes(b"png")
    _write(
        tmp_path / "results/cost_vs_depth/demo_cost_vs_depth.md",
        "LLM Cost vs Depth\nStrategyEIG vs brute-force n-step EIG proxy\nBF/Strategy root-set ratio",
    )
    cost_plot = tmp_path / "results/cost_vs_depth/demo_cost_vs_depth.png"
    cost_plot.write_bytes(b"png")

    results = validate_path_a_package(tmp_path)
    payload = summary_payload(results)

    assert payload["ok"] is True
    assert {item["name"] for item in payload["checks"]} == {
        "ranking_fidelity_gate",
        "ranking_fidelity_diagnostics_plot",
        "constrained_oracle",
        "constrained_oracle_robustness_heatmap",
        "depth_sweep_headline_and_control",
        "depth_sweep_summary",
        "depth_contrast_plot",
        "headline_rmse_plot",
        "paired_trial_delta_plot",
        "truth_log_paired_trial_delta_plot",
        "qualitative_strategy_examples",
        "cost_vs_depth",
        "cost_vs_depth_plot",
    }


def test_validate_path_a_package_reports_missing_artifacts(tmp_path):
    results = validate_path_a_package(tmp_path)
    payload = summary_payload(results)

    assert payload["ok"] is False
    assert all(item["ok"] is False for item in payload["checks"])
    assert "missing" in payload["checks"][0]["detail"]


def test_validate_path_a_package_accepts_later_valid_depth_summary(tmp_path):
    stale = tmp_path / "results/location_depth_sweeps/a_summary.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("{bad json", encoding="utf-8")
    _write_depth_summary(tmp_path)

    checks = summary_payload(validate_path_a_package(tmp_path))["checks"]
    summary_check = next(item for item in checks if item["name"] == "depth_sweep_summary")

    assert summary_check["ok"] is True
    assert summary_check["detail"].endswith("demo_summary.json")


def test_validate_path_a_package_rejects_invalid_depth_summaries(tmp_path):
    _write(
        tmp_path / "results/location_depth_sweeps/demo_summary.json",
        json.dumps({"constrained": {"policies": {"EIG": {}}}, "unconstrained": {"policies": {}}}),
    )

    checks = summary_payload(validate_path_a_package(tmp_path))["checks"]
    summary_check = next(item for item in checks if item["name"] == "depth_sweep_summary")

    assert summary_check["ok"] is False
    assert "StrategyEIG-d1" in summary_check["detail"]


def test_validate_path_a_package_accepts_later_nonempty_qualitative_report(tmp_path):
    _write(
        tmp_path / "results/ranking_fidelity/REPORT.md",
        "Spearman correlations, top-1 regret, and SNR all reported.",
    )
    ranking_plot = tmp_path / "plots/ranking_fidelity/demo_diagnostics.png"
    ranking_plot.parent.mkdir(parents=True, exist_ok=True)
    ranking_plot.write_bytes(b"png")
    _write(
        tmp_path / "results/constrained_oracle/REPORT.md",
        "Planner beats greedy on RMSE in the constrained oracle check.",
    )
    robustness_heatmap = tmp_path / "plots/constrained_oracle_robustness/demo_heatmap.png"
    robustness_heatmap.parent.mkdir(parents=True, exist_ok=True)
    robustness_heatmap.write_bytes(b"png")
    _write(
        tmp_path / "results/location_depth_sweeps/demo_REPORT.md",
        "\n".join(
            [
                "## Headline Constrained Depths",
                "`StrategyEIG-d1`",
                "`StrategyEIG-d3`",
                "`StrategyEIG-d5`",
                "`StrategyEIG-myopic-d3`",
                "`StrategyEIG-myopic-d5`",
                "`truth_log_probability`",
                "`expected_posterior_rmse`",
                "paired final delta vs EIG",
            ]
        ),
    )
    _write_depth_summary(tmp_path)
    plot_path = tmp_path / "plots/location_depth_sweeps/demo_headline_rmse.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_bytes(b"png")
    depth_contrast_plot = tmp_path / "plots/location_depth_sweeps/demo_depth_contrast.png"
    depth_contrast_plot.write_bytes(b"png")
    paired_plot_path = tmp_path / "plots/location_depth_sweeps/demo_paired_trial_rmse_deltas.png"
    paired_plot_path.write_bytes(b"png")
    truth_log_paired_plot_path = (
        tmp_path / "plots/location_depth_sweeps/demo_paired_trial_truth_log_probability_deltas.png"
    )
    truth_log_paired_plot_path.write_bytes(b"png")
    _write(
        tmp_path / "results/location_qualitative/a_constrained_qualitative_examples.md",
        "Qualitative Location Strategy Examples\nNo StrategyEIG examples were available.",
    )
    _write(
        tmp_path / "results/location_qualitative/z_constrained_qualitative_examples.md",
        "Qualitative Location Strategy Examples\nRMSE delta vs EIG\nRoot query",
    )
    qualitative_plot = tmp_path / "results/location_qualitative/z_constrained_qualitative_example_1.png"
    qualitative_plot.write_bytes(b"png")
    _write(
        tmp_path / "results/cost_vs_depth/demo_cost_vs_depth.md",
        "LLM Cost vs Depth\nStrategyEIG vs brute-force n-step EIG proxy\nBF/Strategy root-set ratio",
    )
    cost_plot = tmp_path / "plots/cost_vs_depth/demo_cost_vs_depth.png"
    cost_plot.parent.mkdir(parents=True, exist_ok=True)
    cost_plot.write_bytes(b"png")

    payload = summary_payload(validate_path_a_package(tmp_path))

    assert payload["ok"] is True


def test_validate_path_a_package_rejects_unconstrained_only_qualitative_examples(tmp_path):
    _write(
        tmp_path / "results/ranking_fidelity/REPORT.md",
        "Spearman correlations, top-1 regret, and SNR all reported.",
    )
    ranking_plot = tmp_path / "plots/ranking_fidelity/demo_diagnostics.png"
    ranking_plot.parent.mkdir(parents=True, exist_ok=True)
    ranking_plot.write_bytes(b"png")
    _write(
        tmp_path / "results/constrained_oracle/REPORT.md",
        "Planner beats greedy on RMSE in the constrained oracle check.",
    )
    robustness_heatmap = tmp_path / "plots/constrained_oracle_robustness/demo_heatmap.png"
    robustness_heatmap.parent.mkdir(parents=True, exist_ok=True)
    robustness_heatmap.write_bytes(b"png")
    _write(
        tmp_path / "results/location_depth_sweeps/demo_REPORT.md",
        "\n".join(
            [
                "## Headline Constrained Depths",
                "`StrategyEIG-d1`",
                "`StrategyEIG-d3`",
                "`StrategyEIG-d5`",
                "`StrategyEIG-myopic-d3`",
                "`StrategyEIG-myopic-d5`",
                "`truth_log_probability`",
                "`expected_posterior_rmse`",
                "paired final delta vs EIG",
            ]
        ),
    )
    _write_depth_summary(tmp_path)
    plot_path = tmp_path / "plots/location_depth_sweeps/demo_headline_rmse.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_bytes(b"png")
    depth_contrast_plot = tmp_path / "plots/location_depth_sweeps/demo_depth_contrast.png"
    depth_contrast_plot.write_bytes(b"png")
    paired_plot_path = tmp_path / "plots/location_depth_sweeps/demo_paired_trial_rmse_deltas.png"
    paired_plot_path.write_bytes(b"png")
    truth_log_paired_plot_path = (
        tmp_path / "plots/location_depth_sweeps/demo_paired_trial_truth_log_probability_deltas.png"
    )
    truth_log_paired_plot_path.write_bytes(b"png")
    _write(
        tmp_path / "results/location_qualitative/demo_unconstrained_qualitative_examples.md",
        "Qualitative Location Strategy Examples\nRMSE delta vs EIG\nRoot query",
    )
    qualitative_plot = tmp_path / "results/location_qualitative/demo_unconstrained_qualitative_example_1.png"
    qualitative_plot.write_bytes(b"png")
    _write(
        tmp_path / "results/cost_vs_depth/demo_cost_vs_depth.md",
        "LLM Cost vs Depth\nStrategyEIG vs brute-force n-step EIG proxy\nBF/Strategy root-set ratio",
    )
    cost_plot = tmp_path / "plots/cost_vs_depth/demo_cost_vs_depth.png"
    cost_plot.parent.mkdir(parents=True, exist_ok=True)
    cost_plot.write_bytes(b"png")

    checks = summary_payload(validate_path_a_package(tmp_path))["checks"]
    qualitative_check = next(item for item in checks if item["name"] == "qualitative_strategy_examples")

    assert qualitative_check["ok"] is False
    assert "constrained" in qualitative_check["detail"]
