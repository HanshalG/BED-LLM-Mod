from scripts.validate_path_a_package import summary_payload, validate_path_a_package


def _write(path, text="ok"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    plot_path = tmp_path / "plots/location_depth_sweeps/demo_headline_rmse.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_bytes(b"png")
    _write(
        tmp_path / "results/location_qualitative/demo_qualitative_examples.md",
        "Qualitative Location Strategy Examples\nRMSE delta vs EIG\nRoot query",
    )
    qualitative_plot = tmp_path / "results/location_qualitative/demo_qualitative_example_1.png"
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
        "headline_rmse_plot",
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
    plot_path = tmp_path / "plots/location_depth_sweeps/demo_headline_rmse.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_bytes(b"png")
    _write(
        tmp_path / "results/location_qualitative/a_empty_qualitative_examples.md",
        "Qualitative Location Strategy Examples\nNo StrategyEIG examples were available.",
    )
    _write(
        tmp_path / "results/location_qualitative/z_valid_qualitative_examples.md",
        "Qualitative Location Strategy Examples\nRMSE delta vs EIG\nRoot query",
    )
    qualitative_plot = tmp_path / "results/location_qualitative/z_valid_qualitative_example_1.png"
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
