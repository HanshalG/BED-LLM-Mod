from __future__ import annotations

import json

import pytest

from scripts.hiddenbench_dynamic_belief_v3_codec import parse_refresh, parse_root, parse_routing


def root_value():
    return {
        "prior": [{"option_id": option, "probability": 1 / 3} for option in ("O1", "O2", "O3")],
        "queries": [
            {
                "query_id": query,
                "request": f"Request evidence dimension {index}",
                "target_dimension": f"dimension {index}",
                "channels": [{"channel_id": channel, "description": f"{query} pattern {c}"} for c, channel in enumerate(("C1", "C2", "C3"), 1)],
                "likelihoods": [{"option_id": option, "channels": [{"channel_id": channel, "probability": probability} for channel, probability in zip(("C1", "C2", "C3"), row, strict=True)]} for option, row in zip(("O1", "O2", "O3"), ([0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]), strict=True)],
            }
            for index, query in enumerate(("Q1", "Q2", "Q3", "Q4"), 1)
        ],
    }


def test_root_refresh_and_routing_parse_exact_coverage() -> None:
    root = parse_root(json.dumps(root_value()), ("O1", "O2", "O3"))
    assert set(root["queries"]) == {"Q1", "Q2", "Q3", "Q4"}
    refresh = {"branches": [{"query_id": query, "channel_id": channel, "belief": [{"option_id": option, "probability": probability} for option, probability in zip(("O1", "O2", "O3"), ([0.7, 0.2, 0.1] if channel == "C1" else [0.2, 0.6, 0.2] if channel == "C2" else [0.1, 0.2, 0.7]), strict=True)]} for query in ("Q1", "Q2", "Q3", "Q4") for channel in ("C1", "C2", "C3")]}
    assert len(parse_refresh(json.dumps(refresh), ("O1", "O2", "O3"))) == 4
    routing = {"tasks": [{"slot": slot, "mappings": [{"query_id": query, "fact_id": f"F{index}", "channel_id": f"C{index if index < 4 else 3}"} for index, query in enumerate(("Q1", "Q2", "Q3", "Q4"), 1)]} for slot in ("T1", "T2", "T3", "T4")]}
    parsed = parse_routing(json.dumps(routing), {slot: ("F1", "F2", "F3", "F4") for slot in ("T1", "T2", "T3", "T4")})
    assert len(parsed) == 4


def test_codec_rejects_direct_answer_and_duplicate_branch() -> None:
    value = root_value()
    value["queries"][0]["request"] = "Which option is correct?"
    with pytest.raises(ValueError, match="asks for the answer"):
        parse_root(json.dumps(value), ("O1", "O2", "O3"))
    refresh = {"branches": [{"query_id": "Q1", "channel_id": "C1", "belief": [{"option_id": option, "probability": 1 / 3} for option in ("O1", "O2", "O3")]}] * 12}
    with pytest.raises(ValueError, match="duplicated"):
        parse_refresh(json.dumps(refresh), ("O1", "O2", "O3"))
