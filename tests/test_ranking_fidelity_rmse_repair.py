import json

import pytest

from scripts.ranking_fidelity_rmse_repair import analyze_rmse_repair, append_report_section, write_report


def _record(depth="2"):
    return {
        "trial_index": 0,
        "round_index": 0,
        "realized_by_depth": {
            depth: {
                "entropy_drop_mean": [0.1, 0.2, 0.3],
                "rmse_drop_mean": [1.0, 2.0, 3.0],
                "rmse_drop_std": [0.5, 0.5, 0.5],
                "truth_log_prob_mean": [-3.0, -2.0, -1.0],
                "strategy_execution_fidelity": {
                    "between_over_within_query_distance": 2.0,
                },
            }
        },
    }


def test_analyze_rmse_repair_computes_realized_realized_link():
    analysis = analyze_rmse_repair([_record()])

    depth = analysis["by_depth"]["2"]
    assert depth["spearman_realized_entropy_vs_rmse_drop"]["mean"] == pytest.approx(1.0)
    assert depth["spearman_truth_log_prob_vs_rmse_drop"]["mean"] == pytest.approx(1.0)
    assert depth["rmse_snr_between_over_within"]["mean"] == pytest.approx(4.0)
    assert depth["strategy_query_distance_ratio"]["mean"] == pytest.approx(2.0)
    assert analysis["expected_posterior_rmse"]["status"] == "unavailable_from_current_records"


def test_analyze_rmse_repair_uses_future_expected_posterior_rmse_drop_records():
    record = _record()
    record["realized_by_depth"]["2"]["expected_posterior_rmse_drop_mean"] = [0.1, 0.2, 0.3]

    analysis = analyze_rmse_repair([record])

    depth = analysis["by_depth"]["2"]
    assert analysis["expected_posterior_rmse"]["status"] == "available"
    assert depth["spearman_expected_posterior_rmse_drop_vs_rmse_drop"]["mean"] == pytest.approx(1.0)


def test_rmse_repair_report_and_append_section_are_written(tmp_path):
    analysis = analyze_rmse_repair([_record()])
    report_path = tmp_path / "RMSE_REPAIR.md"
    gate_path = tmp_path / "PHASE1.md"
    gate_path.write_text("# Gate\n\nold text\n", encoding="utf-8")

    write_report(report_path, analysis)
    append_report_section(gate_path, report_path, analysis)

    report = report_path.read_text(encoding="utf-8")
    gate = gate_path.read_text(encoding="utf-8")
    assert "Realized-Realized Link" in report
    assert "Expected Posterior RMSE" in report
    assert "## RMSE Repair Analysis" in gate
    assert "1.000 +/- 0.000" in gate
