from pathlib import Path

from scripts.plot_paprika_headline import plot_headline


def test_plot_headline_writes_png_and_pdf(tmp_path: Path) -> None:
    arms = {
        name: {"resolution_curve": curve}
        for name, curve in {
            "arbitration": [0.1, 0.2, 0.3, 0.4, 0.6],
            "candidate0": [0.1, 0.2, 0.3, 0.3, 0.4],
            "naive_thinking": [0.1, 0.1, 0.2, 0.3, 0.4],
            "naive_nonthinking": [0.0, 0.1, 0.1, 0.2, 0.3],
            "best_n_eig": [0.1, 0.2, 0.3, 0.4, 0.5],
        }.items()
    }
    comparison = {"mean_censored_turn_delta": -0.4, "bootstrap_ci95": [-0.8, -0.1]}
    result = {
        "round_budget": 5,
        "arms": arms,
        "primary_arbitration_vs_naive_thinking": comparison,
        "coprimary_arbitration_vs_candidate0": comparison,
        "context_best_n_vs_naive_thinking": comparison,
        "context_best_n_vs_arbitration": {
            "mean_censored_turn_delta": 0.1,
            "bootstrap_ci95": [-0.2, 0.4],
        },
    }
    png_path, pdf_path = plot_headline(result, tmp_path / "headline")
    assert png_path.stat().st_size > 1_000
    assert pdf_path.stat().st_size > 1_000
