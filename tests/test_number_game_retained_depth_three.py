import pytest

from scripts.number_game_generator_aware_bed import RuleHypothesis
from scripts.number_game_retained_depth_three import (
    BRIER_TOLERANCE,
    TARGET_SEEDS,
    TREE_SEEDS,
    confirmation_gates,
    retained_second_branches,
)


def _rule(name: str, positives: set[int]) -> RuleHypothesis:
    return RuleHypothesis(
        name=name,
        expression="n == 0",
        extension=tuple(number in positives for number in range(101)),
    )


def test_fresh_confirmation_seeds_and_tolerance_are_frozen():
    assert TREE_SEEDS == tuple(range(27800, 27806))
    assert TARGET_SEEDS == tuple(range(27900, 27906))
    assert BRIER_TOLERANCE == pytest.approx(0.005)


def test_retained_second_branches_preserve_parent_and_generation():
    generated = _rule("generated", {1, 2, 3})
    retained = _rule("retained", {1, 3})
    inconsistent = _rule("inconsistent", {2})
    first = {(1, True): [retained, inconsistent]}
    second = {(1, True, 3, True): [generated]}

    merged, parent_only, diagnostics = retained_second_branches(
        first_branches=first,
        generated_second_branches=second,
    )

    assert merged[(1, True, 3, True)] == [generated, retained]
    assert parent_only[(1, True, 3, True)] == [retained]
    assert diagnostics[(1, True, 3, True)][
        "retained_parent_novel_count"
    ] == 1


def test_confirmation_gate_is_conjunctive():
    scored = [
        {
            "mechanics": {
                "initial_valid": 20,
                "minimum_first_branch_valid": 9,
                "minimum_retained_second_branch_valid": 5,
                "target_valid": 20,
                "novel_targets": 10,
            }
        }
    ]
    live = [{"mechanics": {"minimum_second_branch_valid": 5}}]
    comparison = {
        "relative_brier_reduction": 0.02,
        "brier_tree_wins": 3,
        "mean_candidate_minus_baseline_brier": -0.01,
        "mean_candidate_minus_baseline_hamming": -0.01,
        "mean_coverage_difference": 0.01,
    }
    aggregate = {
        "comparisons": {
            "predictive_bayes_risk_depth_two": comparison,
            "retained_parent_only_depth_three": comparison,
            "generated_only_depth_three": comparison,
            "myopic_eig": comparison,
        },
        "root_differences": {
            "predictive_bayes_risk_depth_two_root": 3,
            "retained_parent_only_depth_three_root": 2,
            "generated_only_depth_three_root": 2,
        },
        "ranking": {
            "predictive_risk_spearman_brier": {"mean": 0.5},
            "predictive_pairwise_concordance": {"mean": 0.7},
        },
    }
    usage = {
        "adapter_requests": 300,
        "http_attempts": 300,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 1.0,
    }

    gates = confirmation_gates(
        scored_trees=scored,
        live_trees=live,
        usage=usage,
        aggregate=aggregate,
    )

    assert all(gates.values())
