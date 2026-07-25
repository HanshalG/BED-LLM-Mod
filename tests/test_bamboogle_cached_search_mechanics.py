from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "bamboogle_cached_search_mechanics.py"
)
SPEC = importlib.util.spec_from_file_location(
    "bamboogle_cached_search_mechanics",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _belief_payload(
    *,
    prefix: str = "answer",
    weights: tuple[int, ...] = (40, 20, 10, 8, 7, 6, 5, 4),
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for index, weight in enumerate(weights, start=1):
        payload[f"hypothesis_{index}"] = f"{prefix}-{index}"
        payload[f"weight_{index}"] = weight
    return payload


def _initial_payload(task_index: int) -> dict[str, object]:
    payload = _belief_payload(prefix=f"initial-{task_index}")
    for index in range(1, MODULE.ROOT_COUNT + 1):
        payload[f"root_{index}_query"] = (
            f"task {task_index} root query {index}"
        )
        payload[f"fixed_{index}_query"] = (
            f"task {task_index} fixed query {index}"
        )
    return payload


class FakeModel:
    def __init__(self) -> None:
        self.requests = 0
        self.structured_formats = []

    def chat_complete_messages_batched(
        self,
        batch_messages,
        temperature,
        block_size,
        max_new_tokens=None,
    ):
        assert temperature == MODULE.TEMPERATURE
        responses = []
        for messages in batch_messages:
            payload = json.loads(messages[-1]["content"])
            question = payload["question"]
            task_index = int(question.split("-")[1])
            required = payload["required_output"]
            if "root_1_query" in required:
                response = _initial_payload(task_index)
            elif "adaptive_query" in required:
                root_query = payload["root_search"]["query"]
                root_index = int(root_query.rsplit(" ", 1)[1])
                response = _belief_payload(
                    prefix=f"root-{task_index}-{root_index}"
                )
                response["adaptive_query"] = (
                    f"task {task_index} adaptive query {root_index}"
                )
            else:
                second_query = payload["evidence"][1]["query"]
                root_index = int(second_query.rsplit(" ", 1)[1])
                mode = "adaptive" if "adaptive" in second_query else "fixed"
                weights = list((40, 20, 10, 8, 7, 6, 5, 4))
                weights[0] -= root_index
                weights[1] += root_index
                response = _belief_payload(
                    prefix=f"{mode}-{task_index}-{root_index}",
                    weights=tuple(weights),
                )
            responses.append(json.dumps(response, separators=(",", ":")))
        self.requests += len(responses)
        return responses

    def chat_complete_messages_batched_structured(
        self,
        batch_messages,
        temperature,
        block_size,
        response_format,
        max_new_tokens=None,
    ):
        self.structured_formats.append(response_format)
        return self.chat_complete_messages_batched(
            batch_messages,
            temperature,
            block_size,
            max_new_tokens=max_new_tokens,
        )

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
        return [
            {
                "title": f"Page for {query}",
                "extract": f"Evidence returned for {query}.",
            }
        ]

    def cache_sha256(self):
        return "f" * 64


def _write_fixture(path: Path) -> None:
    rows = [
        {
            "id": task_id,
            "question": f"Task-{task_index}-question",
            "golden_answers": [f"gold-{task_index}"],
        }
        for task_index, task_id in enumerate(MODULE.MECHANICS_IDS)
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_parse_belief_requires_unique_support_and_exact_weight_sum():
    belief = MODULE.parse_belief_payload(_belief_payload())

    assert belief.weights == (40, 20, 10, 8, 7, 6, 5, 4)
    assert belief.entropy_nats > 0.0

    bad_sum = _belief_payload()
    bad_sum["weight_8"] = 3
    try:
        MODULE.parse_belief_payload(bad_sum)
    except ValueError as exc:
        assert "sum to exactly 100" in str(exc)
    else:
        raise AssertionError("invalid weight sum was accepted")

    duplicate = _belief_payload()
    duplicate["hypothesis_8"] = duplicate["hypothesis_1"]
    try:
        MODULE.parse_belief_payload(duplicate)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate hypothesis was accepted")


def test_initial_parser_requires_all_queries_to_be_distinct():
    payload = _initial_payload(0)
    belief, roots, fixed = MODULE.parse_initial(json.dumps(payload))

    assert len(belief.hypotheses) == MODULE.HYPOTHESIS_COUNT
    assert len(roots) == MODULE.ROOT_COUNT
    assert len(fixed) == MODULE.ROOT_COUNT

    payload["fixed_4_query"] = payload["root_1_query"]
    try:
        MODULE.parse_initial(json.dumps(payload))
    except ValueError as exc:
        assert "must be distinct" in str(exc)
    else:
        raise AssertionError("duplicate initial query was accepted")


def test_prompts_do_not_contain_endpoint_answers():
    initial = MODULE.initial_messages("Task question")
    belief = MODULE.parse_belief_payload(_belief_payload())
    documents = [{"title": "Visible", "extract": "Visible evidence"}]
    refresh = MODULE.refresh_messages(
        "Task question",
        belief,
        "root query",
        documents,
    )
    terminal = MODULE.terminal_messages(
        "Task question",
        belief,
        "root query",
        documents,
        "second query",
        documents,
    )
    serialized = json.dumps([initial, refresh, terminal]).lower()

    assert "golden_answers" not in serialized
    assert "hidden-gold-answer" not in serialized


def test_wikipedia_parser_orders_pages_and_truncates_extracts():
    payload = {
        "query": {
            "pages": [
                {"index": 2, "title": "Second", "extract": "B"},
                {
                    "index": 1,
                    "title": " First ",
                    "extract": "A " * 2_000,
                },
            ]
        }
    }

    documents = MODULE.parse_wikipedia_response(payload)

    assert [document["title"] for document in documents] == [
        "First",
        "Second",
    ]
    assert len(documents[0]["extract"]) == MODULE.MAX_EXTRACT_CHARS


def test_response_formats_are_strict_closed_schemas():
    formats = [
        MODULE.initial_response_format(),
        MODULE.refresh_response_format(),
        MODULE.terminal_response_format(),
    ]

    assert all(value["type"] == "json_schema" for value in formats)
    assert all(value["json_schema"]["strict"] for value in formats)
    for value in formats:
        schema = value["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    initial = formats[0]["json_schema"]["schema"]["properties"]
    assert len(initial) == 24
    assert initial["weight_1"]["minimum"] == 1
    assert initial["weight_1"]["maximum"] == 100


def test_run_mechanics_uses_exact_frozen_call_counts(tmp_path, monkeypatch):
    data_path = tmp_path / "bamboogle.jsonl"
    raw_path = tmp_path / "private" / "RAW_RESPONSES.json"
    _write_fixture(data_path)
    source_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    monkeypatch.setattr(MODULE, "SOURCE_SHA256", source_hash)
    model = FakeModel()
    retriever = FakeRetriever()

    result = MODULE.run_mechanics(
        object(),
        data_path=data_path,
        raw_path=raw_path,
        model_adapter=model,
        retriever=retriever,
    )

    assert model.requests == MODULE.EXPECTED_MODEL_REQUESTS == 65
    assert retriever.logical_actions == MODULE.EXPECTED_RETRIEVAL_ACTIONS == 60
    assert result["usage"]["reasoning_tokens"] == 0
    assert result["protocol"]["scientific_retries_or_repairs"] == 0
    assert len(result["records"]) == len(MODULE.MECHANICS_IDS)
    assert raw_path.exists()

    serialized_public = json.dumps(result).lower()
    for forbidden in (
        "task-0-question",
        "root query",
        "adaptive query",
        "evidence returned",
        "initial-0-1",
        "gold-0",
    ):
        assert forbidden not in serialized_public


def test_structured_v2_uses_schema_for_all_four_model_batches(
    tmp_path,
    monkeypatch,
):
    data_path = tmp_path / "bamboogle.jsonl"
    raw_path = tmp_path / "private" / "RAW_RESPONSES.json"
    _write_fixture(data_path)
    source_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    monkeypatch.setattr(MODULE, "SOURCE_SHA256", source_hash)
    model = FakeModel()

    result = MODULE.run_mechanics(
        object(),
        data_path=data_path,
        raw_path=raw_path,
        model_adapter=model,
        retriever=FakeRetriever(),
        interface_version="fixture-v2",
        structured_outputs=True,
    )

    assert result["protocol"]["interface_version"] == "fixture-v2"
    assert result["protocol"]["response_format"] == "chat_strict_json_schema"
    assert [
        value["json_schema"]["name"] for value in model.structured_formats
    ] == [
        "bamboogle_initial_belief",
        "bamboogle_root_refresh",
        "bamboogle_terminal_belief",
        "bamboogle_terminal_belief",
    ]


def test_gold_mass_uses_exact_normalized_answer_support():
    payload = _belief_payload()
    payload["hypothesis_1"] = "The Richmond."
    belief = MODULE.parse_belief_payload(payload)

    assert MODULE.gold_mass(belief, ["Richmond"]) == 0.40
    assert MODULE.top_answer_correct(belief, ["Richmond"])
