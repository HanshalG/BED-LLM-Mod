from __future__ import annotations

import pytest

from scripts import number_game_pooled_second_refresh_ablation as analysis


def test_comparison_summary_is_paired_and_counts_root_changes() -> None:
    rows = [
        {
            "selected_roots": {
                "merged_retained_generated": 1,
                "parent_only": 2,
            },
            "endpoint_brier": {
                "merged_retained_generated": 0.1,
                "parent_only": 0.2,
            },
        },
        {
            "selected_roots": {
                "merged_retained_generated": 3,
                "parent_only": 3,
            },
            "endpoint_brier": {
                "merged_retained_generated": 0.3,
                "parent_only": 0.3,
            },
        },
    ]
    summary = analysis.comparison_summary(
        rows,
        baseline="parent_only",
        bootstrap_indices=[[0, 1], [0, 0], [1, 1]],
    )

    assert summary["candidate_mean_brier"] == pytest.approx(0.2)
    assert summary["baseline_mean_brier"] == pytest.approx(0.25)
    assert summary["relative_brier_reduction"] == pytest.approx(0.2)
    assert summary["root_differences"] == 1
    assert (summary["wins"], summary["ties"], summary["losses"]) == (1, 1, 0)


def test_source_artifacts_are_hash_bound() -> None:
    assert analysis.sha256_file(
        analysis.SOURCE_RESULT
    ) == analysis.SOURCE_RESULT_SHA256
    assert analysis.sha256_file(
        analysis.SOURCE_TREES
    ) == analysis.SOURCE_TREES_SHA256
