#!/usr/bin/env python3
"""Pool independently seeded Number Game support generations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from typing import Any, Sequence

from scripts.number_game_item_isolated_codec import (
    parse_proposals_item_isolated,
)


POOL_SIZE = 2
POOLED_FIELD = "pooled_responses"


def encode_pooled_responses(responses: Sequence[str]) -> str:
    if len(responses) != POOL_SIZE:
        raise ValueError(f"pooled support requires exactly {POOL_SIZE} draws")
    return json.dumps({POOLED_FIELD: list(responses)})


def parse_pooled_proposals(
    response: str,
    *,
    observations: Sequence[tuple[int, bool]] = (),
):
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        value = None
    if not (
        isinstance(value, dict)
        and set(value) == {POOLED_FIELD}
        and isinstance(value[POOLED_FIELD], list)
    ):
        support, diagnostic = parse_proposals_item_isolated(
            response,
            observations=observations,
        )
        diagnostic.update(
            {
                "pool_size": 1,
                "draw_valid_counts": [len(support)],
                "draw_novel_contributions": [len(support)],
            }
        )
        return support, diagnostic

    raw_responses = value[POOLED_FIELD]
    if (
        len(raw_responses) != POOL_SIZE
        or not all(isinstance(item, str) for item in raw_responses)
    ):
        raise ValueError(
            f"pooled response must contain exactly {POOL_SIZE} strings"
        )
    merged = []
    seen_extensions = set()
    draw_diagnostics = []
    draw_valid_counts = []
    draw_novel_contributions = []
    for raw_response in raw_responses:
        support, diagnostic = parse_proposals_item_isolated(
            raw_response,
            observations=observations,
        )
        draw_diagnostics.append(diagnostic)
        draw_valid_counts.append(len(support))
        contributed = 0
        for hypothesis in support:
            if hypothesis.extension in seen_extensions:
                continue
            seen_extensions.add(hypothesis.extension)
            merged.append(hypothesis)
            contributed += 1
        draw_novel_contributions.append(contributed)
    return merged, {
        "codec_mode": "pooled_independent_draws",
        "pool_size": POOL_SIZE,
        "raw_count": sum(
            diagnostic["raw_count"] for diagnostic in draw_diagnostics
        ),
        "valid_unique_count": len(merged),
        "draw_valid_counts": draw_valid_counts,
        "draw_novel_contributions": draw_novel_contributions,
        "draw_diagnostics": draw_diagnostics,
    }


class PooledStructuredAdapter:
    """Expose two independently seeded adapters as one pooled adapter."""

    def __init__(self, adapters: Sequence[Any]) -> None:
        if len(adapters) != POOL_SIZE:
            raise ValueError(
                f"pooled adapter requires exactly {POOL_SIZE} adapters"
            )
        self.adapters = tuple(adapters)

    def chat_complete_messages_batched_structured(
        self,
        batch_messages,
        *,
        temperature,
        block_size,
        response_format,
        max_new_tokens=None,
    ):
        def request(adapter):
            return adapter.chat_complete_messages_batched_structured(
                batch_messages,
                temperature=temperature,
                block_size=block_size,
                response_format=response_format,
                max_new_tokens=max_new_tokens,
            )

        with ThreadPoolExecutor(max_workers=POOL_SIZE) as executor:
            draws = list(executor.map(request, self.adapters))
        return [
            encode_pooled_responses(responses)
            for responses in zip(*draws, strict=True)
        ]

    def usage_snapshot(self) -> dict[str, Any]:
        snapshots = [adapter.usage_snapshot() for adapter in self.adapters]
        numeric_fields = {
            key
            for snapshot in snapshots
            for key, value in snapshot.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        merged = {
            key: sum(float(snapshot.get(key, 0)) for snapshot in snapshots)
            for key in numeric_fields
        }
        for key in numeric_fields:
            if all(
                isinstance(snapshot.get(key, 0), int)
                for snapshot in snapshots
            ):
                merged[key] = int(merged[key])
        return merged
