from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import number_game_dynamic_support_mechanism96 as mechanism


def _row(
    *,
    realized: float,
    dynamic_rho: float = 0.6,
    fixed_rho: float = 0.4,
    dynamic_concordance: float = 0.7,
    fixed_concordance: float = 0.5,
) -> dict:
    novelty = {
        f"{stage}_mean_{measure}": value
        for stage, value in (
            ("first", 0.4),
            ("second", 0.5),
            ("combined", 0.45),
        )
        for measure in ("novel_count", "novel_fraction")
    }
    return {
        "tree_index": 0,
        "tree_seed": 80000,
        "roots_differ": True,
        "dynamic_root": 1,
        "fixed_root": 2,
        "oracle_root": 1,
        "dynamic_rank": {
            "spearman": dynamic_rho,
            "pairwise_concordance": dynamic_concordance,
        },
        "fixed_rank": {
            "spearman": fixed_rho,
            "pairwise_concordance": fixed_concordance,
        },
        "dynamic_predicted_advantage": 0.03,
        "fixed_counter_advantage": 0.02,
        "score_reversal_margin": 0.05,
        "realized_advantage": realized,
        "dynamic_oracle_regret": 0.0,
        "fixed_oracle_regret": realized,
        "dynamic_minus_fixed_oracle_regret": -realized,
        "dynamic_selected_novelty": novelty,
        "fixed_selected_novelty": {
            key: value - 0.1 for key, value in novelty.items()
        },
        "novelty_differences": {key: 0.1 for key in novelty},
        "all_branch_novelty": [
            {
                "stage": "first",
                "root": 1,
                "generated_count": 4,
                "parent_consistent_count": 2,
                "novel_count": 2,
                "novel_fraction": 0.5,
            },
            {
                "stage": "second",
                "root": 1,
                "generated_count": 5,
                "parent_consistent_count": 3,
                "novel_count": 2,
                "novel_fraction": 0.4,
            },
        ],
        "fixed_risk": {"1": 0.2, "2": 0.1},
    }


def test_changed_root_summary_uses_frozen_advantage_sign() -> None:
    rows = [_row(realized=0.03), _row(realized=-0.01)]

    summary = mechanism._changed_root_summary(rows)

    assert summary["mean_realized_advantage"] == pytest.approx(0.01)
    assert summary["wins"] == 1
    assert summary["ties"] == 0
    assert summary["losses"] == 1


def test_bootstrap_analysis_is_deterministic() -> None:
    rows = [
        _row(realized=0.01 + index / 10000)
        for index in range(mechanism.TREE_COUNT)
    ]

    first = mechanism.bootstrap_analysis(rows, seed=7, samples=100)
    second = mechanism.bootstrap_analysis(rows, seed=7, samples=100)

    assert first == second
    assert (
        first["changed_root_mean_realized_advantage_95pct"][0] > 0.0
    )
    assert (
        first["mean_dynamic_minus_fixed_oracle_regret_95pct"][1] < 0.0
    )


def test_summarize_rows_requires_complete_source_cohort() -> None:
    with pytest.raises(ValueError, match="expected 96 rows"):
        mechanism.summarize_rows([_row(realized=0.01)])


def test_novelty_summary_keeps_counts_descriptive() -> None:
    rows = [
        deepcopy(_row(realized=0.01))
        for _ in range(mechanism.TREE_COUNT)
    ]

    summary = mechanism._novelty_summary(rows)

    assert summary["branch_count"] == 2 * mechanism.TREE_COUNT
    assert summary["first_branch_count"] == mechanism.TREE_COUNT
    assert summary["second_branch_count"] == mechanism.TREE_COUNT
    assert summary["all_branches_mean_novel_count"] == pytest.approx(2.0)
    assert (
        summary["selected_roots"][
            "dynamic_minus_fixed_mean_combined_mean_novel_fraction"
        ]
        == pytest.approx(0.1)
    )
