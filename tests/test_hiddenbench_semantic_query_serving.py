from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from scripts import hiddenbench_semantic_query_serving as serving
from scripts.hiddenbench_semantic_query_custodian import project_rows


LIKELIHOODS = {
    "Q1": {
        "O1": [0.0493126163130284, 0.4932105096437303, 0.4574768740432413],
        "O2": [0.3303599698477637, 0.45629498901261956, 0.2133450411396168],
        "O3": [0.32534781803298624, 0.3430616360996095, 0.3315905458674043],
    },
    "Q2": {
        "O1": [0.4490899244329971, 0.271524037949386, 0.27938603761761693],
        "O2": [0.1918811349570642, 0.1993602651729771, 0.6087585998699587],
        "O3": [0.4055046059506606, 0.37853751095681976, 0.2159578830925196],
    },
    "Q3": {
        "O1": [0.42472871497686576, 0.5201272292581214, 0.05514405576501288],
        "O2": [0.28078177792432624, 0.38670056202014746, 0.33251766005552635],
        "O3": [0.533264496581508, 0.33982687849287707, 0.12690862492561492],
    },
    "Q4": {
        "O1": [0.4104786478552682, 0.17296297343103076, 0.416558378713701],
        "O2": [0.5493056961088721, 0.22377883904785295, 0.226915464843275],
        "O3": [0.5141167190803416, 0.3935702261681046, 0.09231305475155381],
    },
}


def source_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": 100 + index,
            "name": f"case {index}",
            "description": f"Decide the outcome for case {index}",
            "shared_information": ["shared alpha", "shared beta", "shared gamma"],
            "hidden_information": ["private north", "private south", "private east"],
            "possible_answers": ["answer red", "answer green", "answer blue"],
            "correct_answer": "answer green",
            "rationale": "this must stay sealed",
        }
        for index in range(4)
    ]


def projected_views() -> dict[str, Any]:
    return project_rows(source_rows())


def root_response() -> str:
    return json.dumps(
        {
            "prior": [
                {"id": option_id, "probability": 1 / 3}
                for option_id in ("O1", "O2", "O3")
            ],
            "queries": [
                {
                    "id": query_id,
                    "request": f"Provide evidence about dimension {index}",
                    "target_dimension": f"dimension {index}",
                }
                for index, query_id in enumerate(serving.QUERY_IDS, start=1)
            ],
        }
    )


def world_response() -> str:
    return json.dumps(
        {
            "query_models": [
                {
                    "id": query_id,
                    "channels": [
                        {"id": channel_id, "description": f"{query_id} response pattern {index}"}
                        for index, channel_id in enumerate(serving.CHANNEL_IDS, start=1)
                    ],
                    "likelihoods": [
                        {
                            "option_id": option_id,
                            "channels": [
                                {"id": channel_id, "probability": probability}
                                for channel_id, probability in zip(
                                    serving.CHANNEL_IDS,
                                    LIKELIHOODS[query_id][option_id],
                                    strict=True,
                                )
                            ],
                        }
                        for option_id in ("O1", "O2", "O3")
                    ],
                    "followups": [
                        {
                            "channel_id": channel_id,
                            "query_id": serving.QUERY_IDS[
                                (serving.QUERY_IDS.index(query_id) + offset) % 4
                            ],
                        }
                        for offset, channel_id in zip(
                            (1, 2, 3), serving.CHANNEL_IDS, strict=True
                        )
                    ],
                }
                for query_id in serving.QUERY_IDS
            ]
        }
    )


class FakeAdapter:
    def __init__(self) -> None:
        self.responses = [root_response()] * 4 + [world_response()] * 4 + [
            json.dumps({"addressed": True, "fact_id": "F1"}),
            json.dumps({"addressed": True, "fact_id": "F2"}),
        ]
        self.messages: list[list[dict[str, str]]] = []
        self.seeds: list[int] = []

    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        assert len(batch_messages) == len(seeds) == 1
        assert temperature == 0.0
        assert response_format["json_schema"]["strict"] is True
        assert max_new_tokens == serving.MAX_TOKENS
        self.messages.append(batch_messages[0])
        self.seeds.append(seeds[0])
        return [self.responses[len(self.messages) - 1]]

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": len(self.messages),
            "http_attempts": len(self.messages),
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "forced_final_requests": 0,
            "adapter_cost_usd": 0.001,
        }


def test_custodian_views_strip_label_designation_and_disjoin_options_facts() -> None:
    views = projected_views()
    encoded = json.dumps(views)
    assert "correct_answer" not in encoded
    assert "rationale" not in encoded
    assert "this must stay sealed" not in encoded
    assert all("options" in task and "private_facts" not in task for task in views["planner"])
    assert all("private_facts" in task and "options" not in task for task in views["router"])


def test_root_codec_rejects_direct_answer_request() -> None:
    value = json.loads(root_response())
    value["queries"][0]["request"] = "Which option is correct?"
    with pytest.raises(ValueError, match="directly"):
        serving.parse_root(json.dumps(value), ("O1", "O2", "O3"))


def test_world_codec_rejects_unnormalized_likelihood() -> None:
    value = json.loads(world_response())
    value["query_models"][0]["likelihoods"][0]["channels"][0]["probability"] = 0.9
    with pytest.raises(ValueError, match="normalized"):
        serving.parse_world(json.dumps(value), ("O1", "O2", "O3"))


def test_exact_depth_two_score_changes_first_query() -> None:
    world = {
        query_id: {
            "likelihoods": LIKELIHOODS[query_id],
        }
        for query_id in serving.QUERY_IDS
    }
    score = serving.score_model([1 / 3] * 3, world, ("O1", "O2", "O3"))
    assert score["greedy_query_id"] == "Q2"
    assert score["depth_two_query_id"] == "Q1"
    assert score["greedy_margin"] > serving.MIN_WIN_MARGIN
    assert score["depth_two_margin"] > serving.MIN_WIN_MARGIN


def test_ten_call_serving_passes_without_label_leakage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(serving, "load_projected_views", lambda source_path: projected_views())
    adapter = FakeAdapter()
    result = serving.run_serving(
        source_path=tmp_path / "unused.json",
        output_dir=tmp_path / "run",
        adapter=adapter,
    )
    assert result["status"] == "serving_pass"
    assert result["aggregate"]["changed_first_request_tasks"] == 4
    assert result["gates"]["exact_transport"] is True
    assert adapter.seeds == list(serving.MODEL_SEEDS)
    prompts = json.dumps(adapter.messages)
    assert "correct_answer" not in prompts
    assert "this must stay sealed" not in prompts
    assert "private north" not in json.dumps(adapter.messages[:8])
    assert "answer red" not in json.dumps(adapter.messages[8:])
    raw = json.loads((tmp_path / "run/private/RAW_RESPONSES.json").read_text())
    assert {key: len(value) for key, value in raw.items()} == {
        "root": 4,
        "world": 4,
        "router": 2,
    }
