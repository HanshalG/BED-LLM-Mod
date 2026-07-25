from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import bamboogle_cached_search_mechanics as mechanics
from scripts import bamboogle_cached_search_structured_serving as serving


def _write_fixture(path: Path) -> None:
    rows = [
        {
            "id": task_id,
            "question": f"Task-{index}-question",
            "golden_answers": [f"gold-{index}"],
        }
        for index, task_id in enumerate(mechanics.MECHANICS_IDS)
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _response(task_index: int) -> str:
    payload = {}
    weights = (40, 20, 10, 8, 7, 6, 5, 4)
    for index, weight in enumerate(weights, start=1):
        payload[f"hypothesis_{index}"] = f"answer-{task_index}-{index}"
        payload[f"weight_{index}"] = weight
    for index in range(1, mechanics.ROOT_COUNT + 1):
        payload[f"root_{index}_query"] = (
            f"task {task_index} root query {index}"
        )
        payload[f"fixed_{index}_query"] = (
            f"task {task_index} fixed query {index}"
        )
    return json.dumps(payload, separators=(",", ":"))


class FakeStructuredModel:
    def __init__(self) -> None:
        self.requests = 0
        self.response_format = None

    def chat_complete_messages_batched_structured(
        self,
        batch_messages,
        temperature,
        block_size,
        response_format,
        max_new_tokens=None,
    ):
        self.response_format = response_format
        responses = [_response(index) for index in range(len(batch_messages))]
        self.requests += len(responses)
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
        }


def test_structured_serving_gate_has_no_scientific_endpoint(
    tmp_path,
    monkeypatch,
):
    data_path = tmp_path / "bamboogle.jsonl"
    raw_path = tmp_path / "private" / "RAW_RESPONSES.json"
    _write_fixture(data_path)
    source_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    monkeypatch.setattr(mechanics, "SOURCE_SHA256", source_hash)
    model = FakeStructuredModel()

    result = serving.run_serving_gate(
        object(),
        data_path=data_path,
        raw_path=raw_path,
        model_adapter=model,
    )

    assert result["status"] == "passed"
    assert result["protocol"]["scientific_endpoints_evaluated"] is False
    assert result["summary"]["gates"]["exact_5_physical_requests"]
    assert model.response_format["json_schema"]["strict"]
    assert raw_path.exists()
