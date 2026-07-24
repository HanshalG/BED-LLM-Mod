import json
import pytest

from scripts.movielens_semantic_ranking_gate_v6 import parse_ranking


def test_ranking_parser() -> None:
    payload = {
        "candidates": [
            {"id": f"q{i + 1}", "score": i / 4, "rationale": "reason"}
            for i in range(4)
        ]
    }
    assert parse_ranking(json.dumps(payload)) == [0.0, 0.25, 0.5, 0.75]
    payload["candidates"][0]["score"] = 2
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        parse_ranking(json.dumps(payload))
