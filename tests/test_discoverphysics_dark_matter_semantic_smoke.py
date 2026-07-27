from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from scripts.discoverphysics_dark_matter_semantic_smoke import (
    NUM_HYPOTHESES,
    NonReasoningOpenRouterAdapter,
    action_table,
    parse_refresh,
    parse_scorer,
    parse_tree,
    quadrant,
)


def _hypotheses(prefix: str = "h") -> list[dict]:
    return [
        {
            "description": f"{prefix}{index} semantic map",
            "probability": 1.0 / NUM_HYPOTHESES,
        }
        for index in range(NUM_HYPOTHESES)
    ]


def _tree_payload() -> dict:
    branches = [
        {
            "probability": 0.5,
            "observation": "strong local pull",
            "posterior_probabilities": [
                0.20,
                0.20,
                0.15,
                0.15,
                0.10,
                0.08,
                0.07,
                0.05,
            ],
        },
        {
            "probability": 0.5,
            "observation": "weak local pull",
            "posterior_probabilities": [
                0.05,
                0.07,
                0.08,
                0.10,
                0.15,
                0.15,
                0.20,
                0.20,
            ],
        },
    ]
    return {
        "hypotheses": _hypotheses(),
        "roots": [
            {"id": root_id, "branches": branches}
            for root_id in ("D", "B", "A", "C")
        ],
    }


def test_tree_parser_sorts_roots_and_computes_finite_eig():
    tree = parse_tree(json.dumps(_tree_payload()))

    assert [root["id"] for root in tree["roots"]] == ["A", "B", "C", "D"]
    assert all(
        np.isfinite(root["immediate_eig_nats"]) for root in tree["roots"]
    )


def test_refresh_parser_requires_fresh_action_and_exact_support():
    response = json.dumps(
        {
            "hypotheses": _hypotheses("fresh"),
            "continuation_action": "r2.5_a1",
            "expected_map_readiness": 73,
            "expected_learning": "Resolve northeast elongation.",
        }
    )

    parsed = parse_refresh(
        response,
        root_action_id="center",
        label="refresh",
    )

    assert parsed["continuation_action"] == "r2.5_a1"
    assert parsed["expected_map_readiness"] == 73

    with pytest.raises(ValueError, match="repeats the root action"):
        parse_refresh(
            response.replace('"r2.5_a1"', '"center"'),
            root_action_id="center",
            label="refresh",
        )


def test_scorer_parser_maps_blind_labels_to_root_ids():
    scores = parse_scorer(
        json.dumps({"scores": {"K": 60, "M": 90, "Q": 50, "T": 75}})
    )

    assert scores == {"A": 60, "B": 90, "C": 50, "D": 75}


def test_action_table_and_quadrants_are_frozen():
    actions = action_table()

    assert len(actions) == 25
    assert actions["center"] == [0.0, 0.0]
    assert quadrant("r4.5_a1") == (1, 1)
    assert quadrant("r4.5_a5") == (-1, -1)
    assert quadrant("center") is None


def test_nonreasoning_adapter_accepts_current_response_format_keyword():
    parameters = inspect.signature(
        NonReasoningOpenRouterAdapter._payload
    ).parameters

    assert "response_format" in parameters
