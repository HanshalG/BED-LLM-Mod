from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import bamboogle_cached_search_flat as flat
from scripts import bamboogle_cached_search_flat_serving as serving
from scripts import bamboogle_cached_search_mechanics as mechanics


WEIGHTS = (40, 20, 10, 8, 7, 6, 5, 4)


def _belief_lines(prefix: str) -> list[str]:
    return [
        f"H{index:02d}|{weight}|{prefix}-{index}"
        for index, weight in enumerate(WEIGHTS, start=1)
    ]


def _initial_response(task_index: int) -> str:
    return "\n".join(
        _belief_lines(f"answer-{task_index}")
        + [
            f"R{index:02d}|task {task_index} root query {index}"
            for index in range(1, 5)
        ]
        + [
            f"F{index:02d}|task {task_index} fixed query {index}"
            for index in range(1, 5)
        ]
    )


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


class FakeFlatModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages,
        temperature,
        block_size,
        max_new_tokens=None,
    ):
        responses = []
        for messages in batch_messages:
            payload = json.loads(messages[-1]["content"])
            question = payload["question"]
            task_index = int(question.split("-")[1])
            system = messages[0]["content"]
            if "exactly 16 ordered lines" in system:
                response = _initial_response(task_index)
            elif "exactly nine ordered lines" in system:
                root_query = payload["root_search"]["query"]
                root_index = int(root_query.rsplit(" ", 1)[1])
                response = "\n".join(
                    _belief_lines(f"root-{task_index}-{root_index}")
                    + [
                        "A01|task "
                        f"{task_index} adaptive query {root_index}"
                    ]
                )
            else:
                second_query = payload["evidence"][1]["query"]
                root_index = int(second_query.rsplit(" ", 1)[1])
                mode = "adaptive" if "adaptive" in second_query else "fixed"
                response = "\n".join(
                    _belief_lines(f"{mode}-{task_index}-{root_index}")
                )
            responses.append(response)
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


class FakeRetriever:
    def __init__(self) -> None:
        self.logical_actions = 0
        self.physical_requests = 0
        self.transport_retries = 0
        self.cache_hits = 0

    def retrieve(self, query):
        self.logical_actions += 1
        self.physical_requests += 1
        return [{"title": f"Page {query}", "extract": "Evidence"}]

    def cache_sha256(self):
        return "f" * 64


def test_flat_parsers_require_exact_order_and_no_extra_text():
    belief, roots, fixed = flat.parse_initial(_initial_response(0))
    assert belief.weights == WEIGHTS
    assert len(roots) == len(fixed) == 4

    for invalid in (
        _initial_response(0) + "\nextra",
        _initial_response(0).replace("H01|", "H02|", 1),
        _initial_response(0).replace("R01|", "R01|extra|", 1),
    ):
        try:
            flat.parse_initial(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid flat response was accepted")


def test_flat_prompts_do_not_contain_endpoint_answers():
    belief, _, _ = flat.parse_initial(_initial_response(0))
    documents = [{"title": "Visible", "extract": "Visible evidence"}]
    prompts = [
        flat.initial_messages("Task question"),
        flat.refresh_messages("Task question", belief, "root", documents),
        flat.terminal_messages(
            "Task question",
            belief,
            "root",
            documents,
            "second",
            documents,
        ),
    ]
    serialized = json.dumps(prompts).lower()
    assert "golden_answers" not in serialized
    assert "hidden-gold-answer" not in serialized


def test_flat_serving_gate_has_no_scientific_endpoint(
    tmp_path,
    monkeypatch,
):
    data_path = tmp_path / "bamboogle.jsonl"
    raw_path = tmp_path / "private" / "RAW_RESPONSES.json"
    _write_fixture(data_path)
    monkeypatch.setattr(
        mechanics,
        "SOURCE_SHA256",
        hashlib.sha256(data_path.read_bytes()).hexdigest(),
    )
    model = FakeFlatModel()

    result = serving.run_serving_gate(
        object(),
        data_path=data_path,
        raw_path=raw_path,
        model_adapter=model,
    )

    assert result["status"] == "passed"
    assert result["protocol"]["scientific_endpoints_evaluated"] is False
    assert result["summary"]["gates"]["all_pass"]
    assert model.requests == 5


def test_flat_codec_preserves_exact_mechanics_counts(tmp_path, monkeypatch):
    data_path = tmp_path / "bamboogle.jsonl"
    raw_path = tmp_path / "private" / "RAW_RESPONSES.json"
    _write_fixture(data_path)
    monkeypatch.setattr(
        mechanics,
        "SOURCE_SHA256",
        hashlib.sha256(data_path.read_bytes()).hexdigest(),
    )
    model = FakeFlatModel()
    retriever = FakeRetriever()

    result = mechanics.run_mechanics(
        object(),
        data_path=data_path,
        raw_path=raw_path,
        model_adapter=model,
        retriever=retriever,
        interface_version=flat.INTERFACE_VERSION,
        codec=flat.FLAT_CODEC,
    )

    assert model.requests == mechanics.EXPECTED_MODEL_REQUESTS == 65
    assert (
        retriever.logical_actions
        == mechanics.EXPECTED_RETRIEVAL_ACTIONS
        == 60
    )
    assert result["protocol"]["response_format"] == "strict_flat_lines"
