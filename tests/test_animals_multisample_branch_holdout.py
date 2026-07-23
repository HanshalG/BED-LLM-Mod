from scripts.animals_multisample_branch_holdout import (
    evaluate_gates,
    union_covered_states,
)


def test_union_covered_states_counts_any_generated_branch_hit():
    records = [
        {
            "candidate_dynamics": [
                {"truth_covered_if_yes": False, "truth_covered_if_no": True}
            ]
        },
        {
            "candidate_dynamics": [
                {"truth_covered_if_yes": False, "truth_covered_if_no": False}
            ]
        },
    ]
    assert union_covered_states(records) == 1


def test_holdout_gates_require_positive_ranking_and_paired_evidence():
    summary = {
        "num_states": 60,
        "num_active_states": 22,
        "ranker_immediate_wins_ties_losses": [12, 43, 5],
        "spearman_belief_recall_score_vs_expected_truth_coverage": 0.2,
        "spearman_immediate_eig_vs_expected_truth_coverage": -0.1,
        "mean_active_state_regret_belief_recall": 0.1,
        "mean_active_state_regret_immediate_eig": 0.2,
    }
    gates = evaluate_gates(
        summary,
        {"ci95": [0.01, 0.2]},
        union_covered=22,
    )
    assert gates["all_pass"]

    gates = evaluate_gates(
        summary,
        {"ci95": [0.0, 0.2]},
        union_covered=22,
    )
    assert not gates["all_pass"]
