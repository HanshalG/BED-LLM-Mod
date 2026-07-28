from pathlib import Path

import pytest

from scripts.number_game_first_retention_two_query_audit import (
    run_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_depth_three_development_v2"
    / "number-game-depth-three-development-v2-20260728T193000Z"
    / "TREES.json"
)
CONFIRMATION = (
    REPO_ROOT
    / "results/nonmyopic/number_game_retained_depth_three_confirmation"
    / "number-game-retained-depth-three-confirmation-20260728T070902Z"
    / "TREES.json"
)


def test_two_query_audit_separates_support_and_selection_effects(
    tmp_path,
):
    output = tmp_path / "RESULT.json"

    result = run_audit(
        inputs=(
            ("development_8", DEVELOPMENT),
            ("fresh_confirmation_6", CONFIRMATION),
        ),
        output_path=output,
    )
    aggregate = result["aggregate"]
    complete = aggregate["complete_policy"]
    selection = aggregate["selection_only"]

    assert result["status"] == "completed_posthoc_audit"
    assert result["protocol"]["model_calls"] == 0
    assert aggregate["tree_count"] == 14
    assert aggregate["roots_changed"] == 7
    assert complete["relative_brier_reduction"] == pytest.approx(
        0.06935669505608225
    )
    assert complete["brier_tree_wins"] == 14
    assert complete["brier_tree_losses"] == 0
    assert complete["tree_bootstrap_brier_difference_95pct"][1] < 0.0
    assert (
        complete["tree_bootstrap_hamming_difference_95pct"][1]
        < 0.0
    )
    assert (
        complete["tree_bootstrap_coverage_difference_95pct"][0]
        > 0.0
    )
    assert selection["relative_brier_reduction"] == pytest.approx(
        0.006380766956072831
    )
    assert selection["brier_tree_wins"] == 3
    assert selection["brier_tree_losses"] == 4
    assert selection["brier_tree_ties"] == 7
    assert (
        selection["tree_bootstrap_brier_difference_95pct"][0]
        < 0.0
        < selection["tree_bootstrap_brier_difference_95pct"][1]
    )
    assert (
        aggregate["ranking"][
            "retained_source_risk_spearman_target_brier_mean"
        ]
        <= aggregate["ranking"][
            "generated_source_risk_spearman_target_brier_mean"
        ]
    )
    assert output.is_file()
