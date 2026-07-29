#!/usr/bin/env python3
"""Parse complete Number Game hypotheses from a damaged JSON envelope."""

from __future__ import annotations

import json
from typing import Any, Sequence

from scripts.discoverphysics_oscillator_belief_smoke import strict_json_object
from scripts.number_game_generator_aware_bed import (
    NUM_PROPOSALS,
    RuleHypothesis,
    parse_proposals,
)


def complete_hypothesis_items(response: str) -> list[dict[str, Any]]:
    """Return independently decodable hypothesis objects in source order."""
    decoder = json.JSONDecoder()
    items: list[dict[str, Any]] = []
    for index, character in enumerate(response):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(response, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == {"name", "expression"}:
            items.append(value)
    if len(items) > NUM_PROPOSALS:
        raise ValueError(
            "damaged response contains more complete hypothesis items "
            f"than expected: {len(items)} > {NUM_PROPOSALS}"
        )
    return items


def parse_proposals_item_isolated(
    response: str,
    *,
    observations: Sequence[tuple[int, bool]] = (),
) -> tuple[list[RuleHypothesis], dict[str, Any]]:
    """Use strict JSON when possible, otherwise validate complete items only."""
    try:
        strict_json_object(response, label="number-game hypotheses")
    except ValueError as strict_error:
        items = complete_hypothesis_items(response)
        if not items:
            raise ValueError(
                "number-game response has invalid JSON and no complete "
                "hypothesis items"
            ) from strict_error
        missing = NUM_PROPOSALS - len(items)
        padded = {
            "hypotheses": [*items, *([None] * missing)],
        }
        support, diagnostic = parse_proposals(
            json.dumps(padded),
            observations=observations,
        )
        diagnostic["raw_count"] = len(items)
        diagnostic["rejected"]["wrong_fields"] -= missing
        diagnostic["rejected"]["missing_or_incomplete"] = missing
        diagnostic.update(
            {
                "codec_mode": "complete_item_salvage",
                "complete_item_count": len(items),
                "expected_item_count": NUM_PROPOSALS,
            }
        )
        return support, diagnostic

    support, diagnostic = parse_proposals(
        response,
        observations=observations,
    )
    diagnostic.update(
        {
            "codec_mode": "strict_json",
            "complete_item_count": NUM_PROPOSALS,
            "expected_item_count": NUM_PROPOSALS,
            "rejected": {
                **diagnostic["rejected"],
                "missing_or_incomplete": 0,
            },
        }
    )
    return support, diagnostic
