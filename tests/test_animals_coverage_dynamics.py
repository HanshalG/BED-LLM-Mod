import pytest

from scripts.animals_coverage_dynamics import summarize_probe


def test_summarize_probe_reports_spread_and_immediate_eig_coverage_regret():
    summary = summarize_probe(
        [
            {
                "candidate_dynamics": [
                    {"immediate_eig": 0.7, "expected_truth_coverage": 0.2},
                    {"immediate_eig": 0.1, "expected_truth_coverage": 0.8},
                ]
            },
            {
                "candidate_dynamics": [
                    {"immediate_eig": 0.4, "expected_truth_coverage": 0.5},
                    {"immediate_eig": 0.2, "expected_truth_coverage": 0.5},
                ]
            },
        ]
    )

    assert summary["num_states"] == 2
    assert summary["num_candidate_rows"] == 4
    assert summary["mean_within_state_coverage_spread"] == pytest.approx(0.3)
    assert summary["mean_immediate_eig_coverage_regret"] == pytest.approx(0.3)
    assert summary["states_with_immediate_eig_coverage_regret_at_least_0_20"] == 1
