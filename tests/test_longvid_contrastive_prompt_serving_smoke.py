from __future__ import annotations

from scripts.longvid_contrastive_prompt_serving_smoke import (
    EXPECTED_REQUESTS,
    OBSERVATIONS,
    QUESTION,
    DeterministicFixtureModel,
    prompt_only_support_messages,
    support_template,
)
from scripts.longvid_contrastive_path_belief_mechanics import initial_messages


def test_support_template_has_every_flat_field() -> None:
    template = support_template()
    assert len(template) == 24
    assert set(template) == {
        f"{field}_{index}"
        for index in range(1, 7)
        for field in ("hypothesis", "weight", "anchor", "query")
    }


def test_prompt_only_message_exposes_exact_template() -> None:
    messages = prompt_only_support_messages(initial_messages(QUESTION))
    assert '"hypothesis_1"' in messages[-1]["content"]
    assert '"query_6"' in messages[-1]["content"]


def test_fixture_shape_matches_exact_ten_call_protocol() -> None:
    assert len(OBSERVATIONS) + 2 == EXPECTED_REQUESTS
    model = DeterministicFixtureModel()
    responses = model.chat_complete_messages_batched(
        [prompt_only_support_messages(initial_messages(QUESTION))],
        temperature=0.0,
        block_size=1,
    )
    assert len(responses) == 1
    assert model.usage_snapshot()["adapter_requests"] == 1
