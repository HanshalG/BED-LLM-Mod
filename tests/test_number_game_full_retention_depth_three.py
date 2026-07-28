import pytest

from scripts.number_game_full_retention_depth_three import (
    BRIER_TOLERANCE,
    EXPECTED_REQUESTS,
    RUN_BUDGET_USD,
    TARGET_SEEDS,
    TREE_SEEDS,
    powered_gates,
)


def _comparison(
    *,
    gain: float = 0.06,
    wins: int = 8,
) -> dict:
    return {
        "relative_brier_reduction": gain,
        "brier_tree_wins": wins,
        "mean_candidate_minus_baseline_brier": -0.01,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.02, -0.001],
        "mean_candidate_minus_baseline_hamming": -0.01,
        "mean_coverage_difference": 0.01,
    }


def test_powered_protocol_constants_are_frozen():
    assert TREE_SEEDS == tuple(range(28000, 28020))
    assert TARGET_SEEDS == tuple(range(28100, 28120))
    assert EXPECTED_REQUESTS == 1000
    assert RUN_BUDGET_USD == pytest.approx(3.60)
    assert BRIER_TOLERANCE == pytest.approx(0.005)


def test_powered_gate_is_conjunctive():
    scored = [
        {
            "mechanics": {
                "initial_valid": 20,
                "minimum_first_branch_valid": 10,
                "minimum_retained_second_branch_valid": 10,
                "mean_first_branch_valid": 22.0,
                "mean_generated_first_branch_valid": 15.0,
                "mean_retained_second_branch_valid": 20.0,
                "mean_generated_second_branch_valid": 14.0,
                "target_valid": 20,
                "novel_targets": 10,
            }
        }
    ]
    live = [
        {
            "mechanics": {
                "minimum_first_branch_valid": 10,
                "minimum_second_branch_valid": 10,
            }
        }
    ]
    aggregate = {
        "comparisons": {
            "predictive_bayes_risk_depth_two": _comparison(gain=0.02),
            "retained_parent_only_depth_three": _comparison(gain=0.02),
            "generated_only_depth_three": _comparison(),
            "myopic_eig": _comparison(gain=0.06),
            "fixed_support_depth_three": _comparison(gain=0.06),
            "uniform_random_candidate_root": _comparison(),
        },
        "root_differences": {
            "predictive_bayes_risk_depth_two_root": 7,
            "retained_parent_only_depth_three_root": 6,
            "generated_only_depth_three_root": 4,
        },
        "ranking": {
            "predictive_risk_spearman_brier": {"mean": 0.5},
            "predictive_pairwise_concordance": {"mean": 0.7},
        },
        "novel_target_mean_differences": {
            "candidate_minus_baseline_brier": -0.01,
            "candidate_minus_baseline_hamming": -0.01,
            "coverage_difference": 0.01,
        },
    }
    usage = {
        "adapter_requests": 1000,
        "http_attempts": 1000,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 3.2,
    }

    gates = powered_gates(
        scored_trees=scored,
        live_trees=live,
        usage=usage,
        aggregate=aggregate,
    )

    assert all(gates.values())
    aggregate["comparisons"]["predictive_bayes_risk_depth_two"][
        "tree_cluster_brier_difference_95pct_bootstrap"
    ][1] = 0.0
    assert not powered_gates(
        scored_trees=scored,
        live_trees=live,
        usage=usage,
        aggregate=aggregate,
    )["brier_cluster_ci_vs_depth_two_below_zero"]
