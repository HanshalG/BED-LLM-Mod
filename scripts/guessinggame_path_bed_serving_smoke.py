#!/usr/bin/env python3
"""Qualify structured semantic hypothesis retrieval for GuessingGame."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import guessinggame_path_bed_source_audit as source
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "guessinggame-path-bed-serving-smoke-1"
SOURCE_MANIFEST_PATH = (
    REPO_ROOT
    / "results/nonmyopic/guessinggame_path_bed_source_audit/"
    "guessinggame-path-bed-source-audit-20260729T020019Z/MANIFEST.json"
)
SOURCE_MANIFEST_SHA256 = (
    "8d1269e197c9cf05865852353b16eeadb36cc8db6231af1d429b7650b263e7b8"
)
MODEL_ID = "openai/gpt-5.4-mini"
REQUEST_SEED = 39_500
TEMPERATURE = 0.3
NUM_CASES = 5
ACTIONS = ("material", "function")
SUPPORT_SIZE = 32
EXPECTED_REQUESTS = NUM_CASES * len(ACTIONS)
CONCURRENCY = EXPECTED_REQUESTS
MAX_TOKENS = 2_400
PROJECTED_COST_USD = 0.05
RUN_BUDGET_USD = 0.20
MIN_DISTINCT_WEIGHTS = 4
MAX_SUPPORT_JACCARD = 0.75
MIN_MATERIAL_RECALL = 2
MIN_FUNCTION_RECALL = 4
MIN_MATERIAL_TOP_HALF_RECALL = 1
MIN_FUNCTION_TOP_HALF_RECALL = 3


class StructuredModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class ServingCase:
    case_id: str
    source_row_sha256: str
    target_object_id: int
    material_question: str
    material_answer: str
    function_question: str
    function_answer: str

    def history(self, action: str) -> tuple[str, str]:
        if action == "material":
            return self.material_question, self.material_answer
        if action == "function":
            return self.function_question, self.function_answer
        raise ValueError(f"unknown action {action!r}")


@dataclass(frozen=True)
class Hypothesis:
    object_id: int
    weight: int


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def strict_json_object(response: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not exact JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys are {sorted(value)}, expected {sorted(expected)}"
        )


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "guessinggame_semantic_hypotheses",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": SUPPORT_SIZE,
                        "maxItems": SUPPORT_SIZE,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["object_id", "weight"],
                            "properties": {
                                "object_id": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": source.EXPECTED_OBJECTS - 1,
                                },
                                "weight": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def parse_response(
    response: str,
    *,
    vocabulary_size: int,
) -> list[Hypothesis]:
    value = strict_json_object(response, label="retrieval response")
    _exact_keys(value, {"hypotheses"}, "retrieval response")
    items = value["hypotheses"]
    if not isinstance(items, list) or len(items) != SUPPORT_SIZE:
        raise ValueError(
            f"retrieval response must contain exactly {SUPPORT_SIZE} rows"
        )
    hypotheses = []
    seen = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"hypothesis {index} is not an object")
        _exact_keys(item, {"object_id", "weight"}, f"hypothesis {index}")
        object_id = item["object_id"]
        weight = item["weight"]
        if (
            isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or not 0 <= object_id < vocabulary_size
        ):
            raise ValueError(f"hypothesis {index} has invalid object_id")
        if object_id in seen:
            raise ValueError("retrieval response contains duplicate object ids")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, int)
            or not 1 <= weight <= 100
        ):
            raise ValueError(f"hypothesis {index} has invalid weight")
        seen.add(object_id)
        hypotheses.append(Hypothesis(object_id=object_id, weight=weight))
    return sorted(
        hypotheses,
        key=lambda hypothesis: (-hypothesis.weight, hypothesis.object_id),
    )


def load_cases() -> tuple[list[str], list[ServingCase]]:
    if sha256_file(SOURCE_MANIFEST_PATH) != SOURCE_MANIFEST_SHA256:
        raise ValueError("GuessingGame source manifest changed")
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    specs = manifest["splits"]["serving_smoke"]
    if len(specs) != NUM_CASES:
        raise ValueError("source serving split no longer has five cases")
    vocabulary = source.load_objects()
    vocabulary_index = {
        object_name: index for index, object_name in enumerate(vocabulary)
    }
    rows_by_sha = {
        source.row_sha256(row): row for row in source.load_games()
    }
    cases = []
    for spec in specs:
        row = rows_by_sha[spec["source_row_sha256"]]
        if source.case_id(row) != spec["case_id"]:
            raise ValueError("opaque source case id changed")
        if source.eligibility_errors(row):
            raise ValueError(f"source case {spec['case_id']} is ineligible")
        cases.append(
            ServingCase(
                case_id=spec["case_id"],
                source_row_sha256=spec["source_row_sha256"],
                target_object_id=vocabulary_index[row.target],
                material_question=row.material_question,
                material_answer=row.material_answer,
                function_question=row.function_question,
                function_answer=row.function_answer,
            )
        )
    return vocabulary, cases


def retrieval_messages(
    *,
    vocabulary: Sequence[str],
    case: ServingCase,
    action: str,
) -> list[dict[str, Any]]:
    question, answer = case.history(action)
    request = {
        "task": (
            "Retrieve and weight the 32 most plausible hidden physical "
            "objects after the semantic observation."
        ),
        "prior": (
            "The hidden object is uniformly drawn from the complete supplied "
            "vocabulary. The object id is its zero-based list position."
        ),
        "observation_history": [
            {"question": question, "answer": answer}
        ],
        "object_vocabulary": [
            {"object_id": index, "name": name}
            for index, name in enumerate(vocabulary)
        ],
        "requirements": [
            "Return exactly 32 distinct object ids from the vocabulary.",
            "Use larger integer weights for more plausible objects.",
            "Use the observation semantics, not vocabulary order.",
            "Do not explain the choices.",
        ],
    }
    return [{"role": "user", "content": canonical_json(request)}]


def _entropy(hypotheses: Sequence[Hypothesis]) -> float:
    total = sum(item.weight for item in hypotheses)
    return -sum(
        (item.weight / total) * math.log(item.weight / total)
        for item in hypotheses
    )


def _rank(
    hypotheses: Sequence[Hypothesis],
    *,
    target_object_id: int,
) -> int | None:
    for index, hypothesis in enumerate(hypotheses, start=1):
        if hypothesis.object_id == target_object_id:
            return index
    return None


def _support_metrics(
    hypotheses: Sequence[Hypothesis],
    *,
    target_object_id: int,
) -> dict[str, Any]:
    rank = _rank(hypotheses, target_object_id=target_object_id)
    weights = [item.weight for item in hypotheses]
    return {
        "target_in_support": rank is not None,
        "target_rank": rank,
        "target_reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        "target_in_top_half": rank is not None and rank <= SUPPORT_SIZE // 2,
        "distinct_weight_count": len(set(weights)),
        "weight_ratio": max(weights) / min(weights),
        "entropy_nats": _entropy(hypotheses),
    }


def _jaccard(
    left: Sequence[Hypothesis],
    right: Sequence[Hypothesis],
) -> float:
    left_ids = {item.object_id for item in left}
    right_ids = {item.object_id for item in right}
    return len(left_ids & right_ids) / len(left_ids | right_ids)


def usage_snapshot(model: StructuredModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "adapter_requests": int(snapshot.get("adapter_requests", 0) or 0),
        "http_attempts": int(snapshot.get("http_attempts", 0) or 0),
        "retry_count": int(snapshot.get("retry_count", 0) or 0),
        "provider_error_retries": int(
            snapshot.get("provider_error_retries", 0) or 0
        ),
        "adapter_reasoning_tokens": int(
            snapshot.get("adapter_reasoning_tokens", 0) or 0
        ),
        "forced_exits": int(snapshot.get("forced_exits", 0) or 0),
        "adapter_prompt_tokens": int(
            snapshot.get("adapter_prompt_tokens", 0) or 0
        ),
        "adapter_completion_tokens": int(
            snapshot.get("adapter_completion_tokens", 0) or 0
        ),
        "run_cost_usd": float(
            snapshot.get("adapter_cost_usd", 0.0) or 0.0
        ),
        "model": snapshot,
    }


def _checkpoint_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def aggregate_gates(
    *,
    cases: Sequence[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, bool]:
    material = [case["material"] for case in cases]
    function = [case["function"] for case in cases]
    all_supports = material + function
    gates = {
        "exact_10_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "exact_10_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_ten_schemas_parse_with_32_unique_ids": (
            len(cases) == NUM_CASES
        ),
        "all_supports_have_weight_dynamic_range": all(
            item["distinct_weight_count"] >= MIN_DISTINCT_WEIGHTS
            and item["weight_ratio"] >= 2.0
            for item in all_supports
        ),
        "material_recovers_at_least_two_of_five_targets": (
            sum(item["target_in_support"] for item in material)
            >= MIN_MATERIAL_RECALL
        ),
        "function_recovers_at_least_four_of_five_targets": (
            sum(item["target_in_support"] for item in function)
            >= MIN_FUNCTION_RECALL
        ),
        "every_target_is_recovered_by_at_least_one_observation": all(
            case["material"]["target_in_support"]
            or case["function"]["target_in_support"]
            for case in cases
        ),
        "material_has_at_least_one_top_half_target": (
            sum(item["target_in_top_half"] for item in material)
            >= MIN_MATERIAL_TOP_HALF_RECALL
        ),
        "function_has_at_least_three_top_half_targets": (
            sum(item["target_in_top_half"] for item in function)
            >= MIN_FUNCTION_TOP_HALF_RECALL
        ),
        "material_and_function_supports_are_distinct": all(
            case["support_jaccard"] <= MAX_SUPPORT_JACCARD
            for case in cases
        ),
        "cost_at_most_0_20": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_smoke(
    *,
    vocabulary: Sequence[str],
    cases: Sequence[ServingCase],
    model: StructuredModel,
    raw_path: Path,
) -> dict[str, Any]:
    if len(vocabulary) != source.EXPECTED_OBJECTS:
        raise ValueError("vocabulary size changed")
    if len(cases) != NUM_CASES:
        raise ValueError("serving smoke requires exactly five cases")
    requests = [
        (case, action)
        for case in cases
        for action in ACTIONS
    ]
    raw: dict[str, Any] = {
        "requests": [
            {
                "case_id": case.case_id,
                "action": action,
                "target_object_id": case.target_object_id,
            }
            for case, action in requests
        ],
        "responses": [],
        "mechanics_accessed": False,
        "development_accessed": False,
        "confirmation_accessed": False,
    }
    try:
        responses = model.chat_complete_messages_batched_structured(
            [
                retrieval_messages(
                    vocabulary=vocabulary,
                    case=case,
                    action=action,
                )
                for case, action in requests
            ],
            temperature=TEMPERATURE,
            block_size=CONCURRENCY,
            response_format=response_format(),
            max_new_tokens=MAX_TOKENS,
        )
        raw["responses"] = list(responses)
        _checkpoint_private(raw_path, raw)
        if len(responses) != EXPECTED_REQUESTS:
            raise ValueError("retrieval response count changed")
        parsed = [
            parse_response(response, vocabulary_size=len(vocabulary))
            for response in responses
        ]
        parsed_by_case = [
            {
                action: parsed[
                    case_index * len(ACTIONS) + action_index
                ]
                for action_index, action in enumerate(ACTIONS)
            }
            for case_index in range(NUM_CASES)
        ]
        public_cases = []
        for case, supports in zip(cases, parsed_by_case, strict=True):
            material_metrics = _support_metrics(
                supports["material"],
                target_object_id=case.target_object_id,
            )
            function_metrics = _support_metrics(
                supports["function"],
                target_object_id=case.target_object_id,
            )
            public_cases.append(
                {
                    "case_id": case.case_id,
                    "material": material_metrics,
                    "function": function_metrics,
                    "support_jaccard": _jaccard(
                        supports["material"], supports["function"]
                    ),
                }
            )
        usage = usage_snapshot(model)
    except Exception as exc:
        _checkpoint_private(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            usage_snapshot(model),
        ) from exc

    gates = aggregate_gates(cases=public_cases, usage=usage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "model": MODEL_ID,
            "seed": REQUEST_SEED,
            "temperature": TEMPERATURE,
            "support_size": SUPPORT_SIZE,
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "mechanics_accessed": False,
            "development_accessed": False,
            "confirmation_accessed": False,
        },
        "metrics": {
            "case_count": len(public_cases),
            "material_target_recall": sum(
                case["material"]["target_in_support"]
                for case in public_cases
            ),
            "function_target_recall": sum(
                case["function"]["target_in_support"]
                for case in public_cases
            ),
            "material_top_half_recall": sum(
                case["material"]["target_in_top_half"]
                for case in public_cases
            ),
            "function_top_half_recall": sum(
                case["function"]["target_in_top_half"]
                for case in public_cases
            ),
            "union_target_recall": sum(
                case["material"]["target_in_support"]
                or case["function"]["target_in_support"]
                for case in public_cases
            ),
            "mean_support_jaccard": sum(
                case["support_jaccard"] for case in public_cases
            )
            / len(public_cases),
        },
        "cases": public_cases,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self, target_object_ids: Sequence[int]) -> None:
        self.target_object_ids = list(target_object_ids)
        self.requests = 0

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, response_format, max_new_tokens
        responses = []
        for offset, _messages in enumerate(batch_messages):
            request_index = self.requests + offset
            target_id = self.target_object_ids[request_index]
            action_index = request_index % len(ACTIONS)
            start = 50 + action_index * 400 + request_index * SUPPORT_SIZE
            ids = [target_id]
            cursor = start
            while len(ids) < SUPPORT_SIZE:
                candidate = cursor % source.EXPECTED_OBJECTS
                cursor += 1
                if candidate not in ids:
                    ids.append(candidate)
            responses.append(
                json.dumps(
                    {
                        "hypotheses": [
                            {
                                "object_id": object_id,
                                "weight": 100 - 2 * index,
                            }
                            for index, object_id in enumerate(ids)
                        ]
                    }
                )
            )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def fixture_model(cases: Sequence[ServingCase]) -> DeterministicFixtureModel:
    return DeterministicFixtureModel(
        [
            case.target_object_id
            for case in cases
            for _action in ACTIONS
        ]
    )


def _adapter(
    *,
    run_id: str,
    output_dir: Path,
) -> SeededStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=500.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return SeededStructuredAdapter(
        ModelSpec(
            model=MODEL_ID,
            backend="openrouter",
            max_model_len=262_144,
        ),
        config,
        request_seed=REQUEST_SEED,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    vocabulary, cases = load_cases()
    model: StructuredModel
    if args.dry_run:
        model = fixture_model(cases)
    else:
        model = _adapter(run_id=args.run_id, output_dir=args.output_dir)
    try:
        payload = run_smoke(
            vocabulary=vocabulary,
            cases=cases,
            model=model,
            raw_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
        checkpoint(args.output_dir / "SERVING.json", payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "metrics": payload["metrics"],
                    "gates": payload["gates"],
                    "usage": payload["usage"],
                },
                indent=2,
            )
        )
    except ServingExecutionError as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "expected_requests": EXPECTED_REQUESTS,
                "reasoning_requested": False,
                "repairs_or_reissues": 0,
            },
            "error": str(exc),
            "usage": exc.usage,
            "private_raw_sha256": sha256_file(raw_path),
        }
        checkpoint(args.output_dir / "FAILURE.json", failure)
        raise


if __name__ == "__main__":
    main()
