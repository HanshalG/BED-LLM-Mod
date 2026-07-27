from __future__ import annotations

from scripts.longvid_contrastive_path_belief_mechanics import (
    layout_hash_for,
)
from scripts.longvid_contrastive_prompt_mechanics import (
    EXCLUDED_PRIOR_ROWS,
    TASK_LAYOUT,
    TASK_LAYOUT_HASH,
    PromptFixtureModel,
    PromptOnlyBridge,
)


def test_new_task_layout_hash_is_stable() -> None:
    assert layout_hash_for(TASK_LAYOUT) == TASK_LAYOUT_HASH


def test_new_layout_excludes_every_prior_model_task() -> None:
    assert not set(EXCLUDED_PRIOR_ROWS) & {
        row_index for row_index, _ in TASK_LAYOUT
    }


def test_prompt_bridge_executes_support_codec() -> None:
    bridge = PromptOnlyBridge(PromptFixtureModel())
    responses = bridge.chat_complete_messages_batched_structured(
        [
            [
                {"role": "system", "content": "fixture"},
                {
                    "role": "user",
                    "content": "QUESTION:\nfixture?\n\nGenerate support",
                },
            ]
        ],
        temperature=0.0,
        block_size=1,
        response_format={
            "json_schema": {"name": "longvid_semantic_belief_support"}
        },
    )
    assert len(responses) == 1
    assert '"hypothesis_1"' in responses[0]
