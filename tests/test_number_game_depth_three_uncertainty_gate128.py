from __future__ import annotations

import pytest

from scripts import number_game_depth_three_uncertainty_gate128 as gate


def test_uncertainty_gate_requires_eight_draws() -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        gate.uncertainty_gate([0.1] * 7)


def test_uncertainty_gate_requires_consensus_and_positive_lower_bound() -> None:
    accepted = gate.uncertainty_gate(
        [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, -0.001, -0.001]
    )
    weak_mean = gate.uncertainty_gate(
        [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, -0.2, -0.2]
    )
    weak_consensus = gate.uncertainty_gate(
        [0.03, 0.03, 0.03, 0.03, 0.03, -0.001, -0.001, -0.001]
    )

    assert accepted["accepted"] is True
    assert accepted["positive_draws"] == 6
    assert accepted["one_se_lower_bound"] > 0.0
    assert weak_mean["accepted"] is False
    assert weak_consensus["accepted"] is False


def test_source_blocks_are_hash_bound_and_aligned() -> None:
    blocks = gate.load_source_blocks()

    assert len(blocks) == 4
    assert all(len(block["raw_trees"]) == 32 for block in blocks)
    assert all(len(block["scored_trees"]) == 32 for block in blocks)
    assert {block["family"] for block in blocks} == {
        "openai/gpt-5.4-mini",
        "qwen/qwen3.7-plus",
    }
    assert gate.MIN_POSITIVE_DRAWS == 6
    assert gate.LOWER_STANDARD_ERROR_MULTIPLIER == 1.0
