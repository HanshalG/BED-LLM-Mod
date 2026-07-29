from __future__ import annotations

from scripts import number_game_qwen_first_link_mechanism64 as mechanism
from scripts.number_game_qwen_external_canonical_pooled64 import (
    STUDIES,
    validate_source,
)


def test_advantage_uses_same_tree_roots_and_endpoint() -> None:
    sources = [validate_source(study) for study in STUDIES]
    row = mechanism.rows_for_baseline(
        sources,
        root_key="myopic_root",
    )[0]
    tree = sources[0]["trees"][0]
    selection = tree["selection"]
    candidate = str(selection["crossfit_depth_three_root"])
    baseline = str(selection["myopic_root"])

    assert row["predicted_advantage"] == (
        selection["crossfit_depth_three_brier"][baseline]
        - selection["crossfit_depth_three_brier"][candidate]
    )
    assert row["realized_advantage"] == (
        tree["per_root_endpoint_brier"][baseline]
        - tree["per_root_endpoint_brier"][candidate]
    )


def test_analysis_is_zero_call_and_preserves_boundary(tmp_path) -> None:
    result = mechanism.run_analysis(tmp_path)
    myopic = result["analyses"]["myopic_eig"]
    fixed = result["analyses"]["fixed_support_depth_three"]

    assert result["status"] == "retrospective_first_link_summary"
    assert result["protocol"]["model_calls"] == 0
    assert result["protocol"]["cost_usd"] == 0.0
    assert myopic["root_differences"] == 62
    assert myopic["divergent_trees"]["wins"] == 50
    assert (
        myopic["divergent_trees"][
            "score_to_realized_advantage_spearman"
        ]
        > 0.3
    )
    assert (
        fixed["divergent_trees"][
            "score_to_realized_advantage_spearman"
        ]
        < 0.2
    )
    assert result["interpretation"]["formal_status_unchanged"] is True
