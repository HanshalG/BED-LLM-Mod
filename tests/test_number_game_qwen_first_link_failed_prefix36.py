from __future__ import annotations

import pytest

from scripts import number_game_qwen_first_link_failed_prefix36 as prefix


def test_replay_adapter_consumes_exact_batches() -> None:
    adapter = prefix.ReplayAdapter((("a",), ("b", "c")))

    assert adapter.chat_complete_messages_batched_structured([[]]) == ["a"]
    assert adapter.chat_complete_messages_batched_structured([[], []]) == [
        "b",
        "c",
    ]
    adapter.assert_exhausted()


def test_replay_adapter_rejects_shape_or_extra_calls() -> None:
    adapter = prefix.ReplayAdapter((("a",),))
    with pytest.raises(ValueError, match="batch length"):
        adapter.chat_complete_messages_batched_structured([[], []])

    adapter = prefix.ReplayAdapter((("a",),))
    adapter.chat_complete_messages_batched_structured([[]])
    with pytest.raises(ValueError, match="unexpected batch"):
        adapter.chat_complete_messages_batched_structured([[]])


def test_failed_prefix_constants_preserve_scope() -> None:
    assert len(prefix.TREE_SEEDS) == 36
    assert len(prefix.TARGET_SEEDS) == 36
    assert prefix.TREE_SEEDS == tuple(range(60_100, 60_136))
    assert prefix.BOOTSTRAP_SEED == 61_800
    assert prefix.BOOTSTRAP_SAMPLES == 20_000
