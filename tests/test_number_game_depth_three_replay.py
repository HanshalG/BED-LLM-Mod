from scripts.number_game_depth_three_replay import ReplayAdapter


def test_replay_adapter_returns_frozen_batches():
    adapter = ReplayAdapter(
        [["one"], ["two", "three"]],
        {"adapter_requests": 3},
    )

    assert adapter.chat_complete_messages_batched_structured([{}]) == ["one"]
    assert adapter.chat_complete_messages_batched_structured(
        [{}, {}]
    ) == ["two", "three"]
    assert adapter.usage_snapshot() == {"adapter_requests": 3}
