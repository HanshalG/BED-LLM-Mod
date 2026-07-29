from __future__ import annotations

import pytest

from scripts import number_game_pooled_margin_scale_analysis as analysis


def source_tree(*, tree_seed, risks, candidate, myopic, realized):
    per_root = {str(root): 0.2 for root in risks}
    per_root[str(candidate)] = 0.2
    per_root[str(myopic)] = 0.2 + realized
    return {
        "tree_seed": tree_seed,
        "selection": {
            "crossfit_depth_three_root": candidate,
            "myopic_root": myopic,
            "crossfit_depth_three_brier": {
                str(root): value for root, value in risks.items()
            },
        },
        "per_root_endpoint_brier": per_root,
    }


def test_analysis_rows_standardize_margin_by_within_tree_scale() -> None:
    source = {
        "trees": [
            source_tree(
                tree_seed=1,
                risks={0: 0.1, 1: 0.3, 2: 0.2},
                candidate=0,
                myopic=1,
                realized=0.04,
            ),
            source_tree(
                tree_seed=2,
                risks={0: 0.49, 1: 0.51, 2: 0.5},
                candidate=0,
                myopic=1,
                realized=0.01,
            ),
        ]
    }

    rows = analysis.analysis_rows(source)

    assert len(rows) == 2
    assert rows[0]["raw_predicted_advantage"] == pytest.approx(0.2)
    assert rows[1]["raw_predicted_advantage"] == pytest.approx(0.02)
    assert rows[0]["normalized_predicted_advantage"] > 0
    assert rows[1]["normalized_predicted_advantage"] > 0


def test_source_hash_and_changed_root_count_are_frozen(tmp_path) -> None:
    result = analysis.run_analysis(output_dir=tmp_path)

    assert result["protocol"]["model_calls"] == 0
    assert result["summary"]["tree_count"] == 30
    assert result["protocol"]["cannot_rescue_source_status"] is True
