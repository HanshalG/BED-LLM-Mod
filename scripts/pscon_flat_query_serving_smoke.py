#!/usr/bin/env python3
"""Validate a flat, target-free PSCon semantic-query interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.pscon_semantic_tree_smoke import (
    CONVERSATION_ID,
    GENERATOR_MODEL_ID,
    Product,
    _conversation_row,
    _load_products,
    _visible_task,
    entropy,
    verify_source,
)


INTERFACE_VERSION = "pscon-flat-query-serving-smoke-1"
SEED = 24_386
QUERY_COUNT = 10
OPTION_COUNT = 3
MAX_COST_USD = 0.15


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def parse_flat_query(text: str, product_count: int) -> dict[str, Any]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 3:
        raise ValueError("flat query must have exactly three nonempty lines")
    prefixes = ("QUESTION: ", "OPTIONS: ", "ASSIGNMENTS: ")
    if any(not line.startswith(prefix) for line, prefix in zip(lines, prefixes)):
        raise ValueError("flat query line prefixes are invalid")
    question = lines[0][len(prefixes[0]) :].strip()
    options = tuple(
        value.strip() for value in lines[1][len(prefixes[1]) :].split(" || ")
    )
    assignments = lines[2][len(prefixes[2]) :].strip().upper()
    if (
        "\n" in question
        or not question.endswith(("?", "？"))
        or not 8 <= len(question) <= 240
    ):
        raise ValueError("question must be one bounded line ending in ? or fullwidth ?")
    if (
        len(options) != OPTION_COUNT
        or len(set(options)) != OPTION_COUNT
        or not all(1 <= len(option) <= 100 for option in options)
    ):
        raise ValueError("options must be three distinct bounded strings")
    if len(assignments) != product_count or set(assignments) - set("ABC"):
        raise ValueError("assignments must be one A/B/C character per product")
    if set(assignments) != set("ABC"):
        raise ValueError("every assignment label must be used")
    return {
        "question": question,
        "options": options,
        "assignments": assignments,
        "eig": entropy(list(assignments)),
    }


def _messages(
    *,
    user_history: list[str],
    products: list[Product],
    query_index: int,
) -> list[dict[str, str]]:
    request = {
        "user_history": user_history,
        "query_index": query_index,
        "candidate_products": [
            {
                "candidate_index": index,
                "title": product.title,
            }
            for index, product in enumerate(products, start=1)
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Create one useful multiple-choice clarification question for a "
                "shopping assistant using only the supplied conversation and product "
                "titles. The three options must be mutually exclusive and exhaustive "
                "for all candidates. Do not mention product IDs, candidate indices, "
                "or ask which listed product the user wants. Assign each candidate "
                "to exactly one option in input order using one character: A for the "
                "first option, B for the second, or C for the third. Output exactly "
                "three lines and nothing else:\n"
                "QUESTION: <one question ending in ?>\n"
                "OPTIONS: <option A> || <option B> || <option C>\n"
                "ASSIGNMENTS: <exactly one A/B/C character per candidate, no spaces>"
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _usage(model: ChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "adapter_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "generator": snapshot,
    }


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_serving_smoke(
    config: Config,
    *,
    source_root: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    conversation_path, graph_path = verify_source(source_root)
    row = _conversation_row(conversation_path)
    user_history, support_ids = _visible_task(row)
    products = _load_products(graph_path, support_ids)
    raw: dict[str, Any] = {
        "conversation_id": CONVERSATION_ID,
        "support_ids": [product.product_id for product in products],
        "target_loaded": False,
    }
    try:
        messages = [
            _messages(
                user_history=user_history,
                products=products,
                query_index=index,
            )
            for index in range(QUERY_COUNT)
        ]
        responses = model.chat_complete_messages_batched(
            messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=500,
        )
        raw["responses"] = responses
        _checkpoint(raw_path, raw)
        if len(responses) != QUERY_COUNT:
            raise ValueError("serving response count changed")
        queries = [
            parse_flat_query(response, len(products)) for response in responses
        ]
        raw["all_queries_parsed"] = True
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    questions = [query["question"] for query in queries]
    signatures = [query["assignments"] for query in queries]
    eigs = [query["eig"] for query in queries]
    gates = {
        "exact_request_count": usage["physical_requests"] == QUERY_COUNT,
        "exact_http_attempt_count": usage["http_attempts"] == QUERY_COUNT,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_ten_queries_parse": len(queries) == QUERY_COUNT,
        "at_least_eight_unique_questions": len(set(questions)) >= 8,
        "at_least_five_unique_partitions": len(set(signatures)) >= 5,
        "all_partitions_informative": all(eig >= 0.30 for eig in eigs),
        "eig_range_at_least_0_10": max(eigs) - min(eigs) >= 0.10,
        "cost_at_most_0_15": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "conversation_id": CONVERSATION_ID,
            "seed": SEED,
            "model": GENERATOR_MODEL_ID,
            "query_count": QUERY_COUNT,
            "support_size": len(products),
            "target_loaded": False,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
        },
        "metrics": {
            "unique_question_count": len(set(questions)),
            "unique_partition_count": len(set(signatures)),
            "mean_eig": statistics.fmean(eigs),
            "min_eig": min(eigs),
            "max_eig": max(eigs),
            "eig_range": max(eigs) - min(eigs),
        },
        "queries": [
            {
                "query_index": index,
                **query,
            }
            for index, query in enumerate(queries)
        ],
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            index = int(request["query_index"])
            count = len(request["candidate_products"])
            a_count = 1 + (index % 6)
            b_count = 1 + ((2 * index) % 7)
            labels = (
                ("A" * a_count)
                + ("B" * b_count)
                + ("C" * (count - a_count - b_count))
            )
            responses.append(
                f"QUESTION: Which fixture preference applies for query {index}?\n"
                f"OPTIONS: Fixture A {index} || Fixture B {index} || Fixture C {index}\n"
                f"ASSIGNMENTS: {labels}"
            )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
        }


def _nonthinking_spec(spec: Any) -> Any:
    return replace(
        spec,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )


def _build_model(config: Config) -> ChatModel:
    spec = _nonthinking_spec(config.model_pairs[0].questioner)
    if spec.model != GENERATOR_MODEL_ID:
        raise ValueError("flat PSCon config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.03
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 10
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = DeterministicFixtureModel() if args.dry_run else _build_model(config)
    try:
        payload = run_serving_smoke(
            config,
            source_root=args.source_root,
            raw_path=raw_path,
            model=model,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "SERVING_FAILURE.json", failure)
        raise
    output = args.output_dir / "SERVING.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
