from scripts.analyze_animals_transition_selectors import (
    select_index,
    summarize_expected_coverage_artifact,
    within_state_pairwise_accuracy,
)


def test_select_index_uses_stable_first_tie_break():
    candidates = [{"score": 1.0}, {"score": 1.0}, {"score": 0.5}]
    assert select_index(candidates, "score") == 0


def test_pairwise_accuracy_compares_only_within_state():
    records = [
        {
            "candidate_dynamics": [
                {"score": 2.0, "endpoint": 1.0},
                {"score": 1.0, "endpoint": 0.0},
            ]
        },
        {
            "candidate_dynamics": [
                {"score": 100.0, "endpoint": 0.0},
                {"score": 99.0, "endpoint": 1.0},
            ]
        },
    ]
    result = within_state_pairwise_accuracy(
        records,
        "score",
        "endpoint",
    )
    assert result == {
        "accuracy": 0.5,
        "pairs": 2,
        "informative_states": 2,
    }


def test_expected_coverage_summary_compares_shared_candidates():
    records = [
        {
            "candidate_dynamics": [
                {
                    "expected_current_support_retention": 0.9,
                    "immediate_eig": 0.1,
                    "expected_truth_coverage": 0.8,
                },
                {
                    "expected_current_support_retention": 0.1,
                    "immediate_eig": 0.9,
                    "expected_truth_coverage": 0.2,
                },
            ]
        }
    ]
    summary = summarize_expected_coverage_artifact(records)
    assert summary == {
        "states": 1,
        "retention_mean_expected_truth_coverage": 0.8,
        "eig_mean_expected_truth_coverage": 0.2,
        "retention_minus_eig": 0.6000000000000001,
        "wins_ties_losses": [1, 0, 0],
        "selector_changes": 1,
    }
