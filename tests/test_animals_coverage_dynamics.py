import pytest

from scripts.animals_coverage_dynamics import summarize_probe


def test_summarize_probe_reports_spread_and_immediate_eig_coverage_regret():
    summary = summarize_probe(
        [
            {
                "candidate_dynamics": [
                    {
                        "immediate_eig": 0.7,
                        "expected_truth_coverage": 0.2,
                        "dynamic_brier_gain": -0.3,
                    },
                    {
                        "immediate_eig": 0.1,
                        "expected_truth_coverage": 0.8,
                        "dynamic_brier_gain": 0.4,
                    },
                ]
            },
            {
                "candidate_dynamics": [
                    {
                        "immediate_eig": 0.4,
                        "expected_truth_coverage": 0.5,
                        "dynamic_brier_gain": 0.0,
                    },
                    {
                        "immediate_eig": 0.2,
                        "expected_truth_coverage": 0.5,
                        "dynamic_brier_gain": 0.1,
                    },
                ]
            },
        ]
    )

    assert summary["num_states"] == 2
    assert summary["num_candidate_rows"] == 4
    assert summary["mean_within_state_coverage_spread"] == pytest.approx(0.3)
    assert summary["mean_immediate_eig_coverage_regret"] == pytest.approx(0.3)
    assert summary["states_with_immediate_eig_coverage_regret_at_least_0_20"] == 1
    assert summary["mean_immediate_eig_selected_expected_truth_coverage"] == pytest.approx(0.35)
    assert summary["mean_dynamic_brier_selected_expected_truth_coverage"] == pytest.approx(0.65)
    assert summary["mean_dynamic_brier_paired_coverage_gain"] == pytest.approx(0.3)
    assert summary["dynamic_brier_immediate_wins_ties_losses"] == [1, 1, 0]
    assert summary["spearman_dynamic_brier_gain_vs_expected_truth_coverage"] > 0.0
