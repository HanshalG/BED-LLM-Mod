from __future__ import annotations

import json

from scripts import number_game_pooled_dynamic_vs_fixed64 as analysis
from scripts import number_game_pooled_replication_synthesis64 as pooled


def test_hash_bound_rows_reproduce_both_source_comparisons() -> None:
    first, _, second = pooled.load_sources()
    cohorts = analysis.comparison_rows(first, second)

    assert [len(cohort) for cohort in cohorts] == [32, 32]
    assert not (
        {row["tree_seed"] for row in cohorts[0]}
        & {row["tree_seed"] for row in cohorts[1]}
    )
    analysis.validate_source_reproduction(cohorts, (first, second))


def test_source_mean_difference() -> None:
    rows = [
        {
            "comparison": {
                "candidate_brier": 0.1,
                "baseline_brier": 0.2,
            }
        },
        {
            "comparison": {
                "candidate_brier": 0.3,
                "baseline_brier": 0.35,
            }
        },
    ]

    assert abs(
        analysis.source_mean_difference(rows, metric="brier") + 0.075
    ) < 1e-12


def test_zero_call_dynamic_support_synthesis_passes_frozen_gates(
    tmp_path,
) -> None:
    result = analysis.run_analysis(tmp_path)

    assert result["status"] == "retrospective_dynamic_support_positive"
    assert result["protocol"]["model_calls"] == 0
    assert result["protocol"]["cost_usd"] == 0.0
    assert all(result["gates"].values())
    assert result["pooled"]["brier"]["wins"] == 30
    assert (
        result["pooled"]["brier"]["stratified_bootstrap_difference_95pct"][1]
        < 0.0
    )
    assert json.loads((tmp_path / "RESULT.json").read_text())["status"] == (
        "retrospective_dynamic_support_positive"
    )
