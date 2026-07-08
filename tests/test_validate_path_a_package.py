from scripts.validate_path_a_package import summary_payload, validate_path_a_package


def _write(path, text="ok"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_validate_path_a_package_passes_complete_mpp(tmp_path):
    _write(
        tmp_path / "results/ranking_fidelity/REPORT.md",
        "Spearman correlations, top-1 regret, and SNR all reported.",
    )
    _write(
        tmp_path / "results/constrained_oracle/REPORT.md",
        "Planner beats greedy on RMSE in the constrained oracle check.",
    )
    _write(
        tmp_path / "results/location_depth_sweeps/demo_REPORT.md",
        "\n".join(
            [
                "## Headline Constrained Depths",
                "`StrategyEIG-d1`",
                "`StrategyEIG-d3`",
                "`StrategyEIG-d5`",
                "`StrategyEIG-myopic-d3`",
                "paired final delta vs EIG",
            ]
        ),
    )
    plot_path = tmp_path / "plots/location_depth_sweeps/demo_headline_rmse.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_bytes(b"png")
    _write(
        tmp_path / "results/cost_vs_depth/demo_cost_vs_depth.md",
        "LLM Cost vs Depth\nStrategyEIG vs brute-force n-step EIG proxy\nBF/Strategy root-set ratio",
    )

    results = validate_path_a_package(tmp_path)
    payload = summary_payload(results)

    assert payload["ok"] is True
    assert {item["name"] for item in payload["checks"]} == {
        "ranking_fidelity_gate",
        "constrained_oracle",
        "depth_sweep_headline_and_control",
        "headline_rmse_plot",
        "cost_vs_depth",
    }


def test_validate_path_a_package_reports_missing_artifacts(tmp_path):
    results = validate_path_a_package(tmp_path)
    payload = summary_payload(results)

    assert payload["ok"] is False
    assert all(item["ok"] is False for item in payload["checks"])
    assert "missing" in payload["checks"][0]["detail"]
