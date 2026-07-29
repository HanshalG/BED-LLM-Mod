from __future__ import annotations

from scripts import number_game_crossplanner_canonical_pooled128 as pooled


def test_four_source_blocks_are_disjoint() -> None:
    blocks = pooled.load_blocks()
    seed_sets = [
        {int(tree["tree_seed"]) for tree in block["trees"]}
        for block in blocks
    ]

    assert len(blocks) == 4
    assert all(len(block["trees"]) == 32 for block in blocks)
    assert {block["family"] for block in blocks} == {
        "openai/gpt-5.4-mini",
        "qwen/qwen3.7-plus",
    }
    assert all(
        not seed_sets[left].intersection(seed_sets[right])
        for left in range(4)
        for right in range(left + 1, 4)
    )


def test_synthesis_is_zero_call_and_crossplanner_positive(tmp_path) -> None:
    result = pooled.run_synthesis(tmp_path)

    assert result["status"] == (
        "retrospective_crossplanner_robustness_positive"
    )
    assert result["protocol"]["analysis_is_retrospective"] is True
    assert result["protocol"]["model_calls"] == 0
    assert result["protocol"]["cost_usd"] == 0.0
    assert result["protocol"]["tree_count"] == 128
    assert all(result["robustness_checks"].values())
    assert result["pooled"]["comparisons"]["myopic_eig"]["wins"] >= 80
    assert "heterogeneous" in result["depth_interpretation"]
