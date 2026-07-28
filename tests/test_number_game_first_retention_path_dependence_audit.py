from pathlib import Path

import pytest

from scripts.number_game_first_retention_path_dependence_audit import (
    audit_source,
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


def test_fresh_confirmation_has_first_refresh_path_dependence():
    source = audit_source(
        label="fresh_confirmation_6",
        path=CONFIRMATION,
    )
    aggregate = source["aggregate"]

    assert aggregate["tree_count"] == 6
    assert aggregate["branch_count"] == 96
    assert aggregate["all_recorded_second_queries_reproduced"]
    assert aggregate["trees_with_changed_second_query"] == 6
    assert aggregate["second_queries_changed"] == 75
    assert aggregate["second_query_change_rate"] == pytest.approx(
        0.78125
    )
    assert aggregate["target_paths"] == 1064
    assert aggregate["target_paths_recovered"] == 340
    assert aggregate["target_paths_lost"] == 0
    assert aggregate["generated_target_coverage_rate"] == pytest.approx(
        0.2161654135338346
    )
    assert aggregate["retained_target_coverage_rate"] == pytest.approx(
        0.5357142857142857
    )
    assert (
        aggregate[
            "tree_bootstrap_target_coverage_difference_95pct"
        ][0]
        > 0.0
    )


def test_combined_audit_is_zero_call_and_writes_result(tmp_path):
    output = tmp_path / "RESULT.json"

    result = run_audit(
        inputs=(
            ("development_8", DEVELOPMENT),
            ("fresh_confirmation_6", CONFIRMATION),
        ),
        output_path=output,
    )

    assert result["status"] == "passed"
    assert result["protocol"]["model_calls"] == 0
    assert result["aggregate"]["tree_count"] == 14
    assert result["aggregate"]["branch_count"] == 224
    assert result["aggregate"]["second_queries_changed"] == 159
    assert result["aggregate"]["target_paths_recovered"] == 738
    assert all(result["gates"].values())
    assert output.is_file()
