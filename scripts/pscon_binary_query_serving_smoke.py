#!/usr/bin/env python3
"""Validate role-separated PSCon question and likelihood generation."""

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


INTERFACE_VERSION = "pscon-binary-query-serving-smoke-1"
SEED = 24_387
QUESTION_COUNT = 5
EXPECTED_REQUESTS = 2 * QUESTION_COUNT
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


def parse_question(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("question must have exactly one nonempty line")
    question = lines[0]
    if not question.endswith(("?", "？")) or not 8 <= len(question) <= 240:
        raise ValueError("question must be bounded and end in ? or fullwidth ?")
    return question


def parse_labels(text: str, product_count: int) -> str:
    value = text.strip().upper()
    if len(value) != product_count or set(value) - set("YNU"):
        raise ValueError("labels must be one Y/N/U character per product")
    return value


def _question_messages(
    user_history: list[str],
    products: list[Product],
    question_index: int,
) -> list[dict[str, str]]:
    request = {
        "user_history": user_history,
        "question_index": question_index,
        "candidate_titles": [product.title for product in products],
    }
    return [
        {
            "role": "system",
            "content": (
                "Write one useful neutral yes/no clarification question that helps "
                "distinguish the candidate products using only the supplied user "
                "history and titles. The question must concern one product property "
                "that a customer can answer. Do not mention product IDs, candidate "
                "numbers, specific model names, or ask whether they want a listed "
                "product. Different question_index values should explore different "
                "properties. Output exactly the question on one line and nothing else."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _classification_messages(
    question: str,
    products: list[Product],
) -> list[dict[str, str]]:
    request = {
        "question": question,
        "candidate_titles": [product.title for product in products],
    }
    return [
        {
            "role": "system",
            "content": (
                "Classify how each candidate title answers the yes/no question. Use "
                "Y only when the title clearly supports yes, N only when it clearly "
                "supports no, and U when the title is insufficient. Output exactly "
                "one Y/N/U character per candidate in input order, with no spaces, "
                "punctuation, labels, or prose."
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
        question_raw = model.chat_complete_messages_batched(
            [
                _question_messages(user_history, products, index)
                for index in range(QUESTION_COUNT)
            ],
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=120,
        )
        raw["questions"] = question_raw
        _checkpoint(raw_path, raw)
        if len(question_raw) != QUESTION_COUNT:
            raise ValueError("question response count changed")
        questions = [parse_question(response) for response in question_raw]

        label_raw = model.chat_complete_messages_batched(
            [
                _classification_messages(question, products)
                for question in questions
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=40,
        )
        raw["labels"] = label_raw
        _checkpoint(raw_path, raw)
        if len(label_raw) != QUESTION_COUNT:
            raise ValueError("classification response count changed")
        labels = [parse_labels(response, len(products)) for response in label_raw]
        raw["all_parsed"] = True
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    eigs = [entropy(list(value)) for value in labels]
    informative = [len(set(value)) >= 2 and eig >= 0.30 for value, eig in zip(labels, eigs)]
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_questions_and_labels_parse": (
            len(questions) == QUESTION_COUNT and len(labels) == QUESTION_COUNT
        ),
        "at_least_four_unique_questions": len(set(questions)) >= 4,
        "at_least_four_informative_partitions": sum(informative) >= 4,
        "at_least_four_unique_partitions": len(set(labels)) >= 4,
        "informative_eig_range_at_least_0_10": (
            max(eig for eig, keep in zip(eigs, informative) if keep)
            - min(eig for eig, keep in zip(eigs, informative) if keep)
            >= 0.10
        )
        if any(informative)
        else False,
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
            "question_count": QUESTION_COUNT,
            "expected_requests": EXPECTED_REQUESTS,
            "support_size": len(products),
            "target_loaded": False,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
        },
        "metrics": {
            "unique_question_count": len(set(questions)),
            "unique_partition_count": len(set(labels)),
            "informative_partition_count": sum(informative),
            "mean_eig": statistics.fmean(eigs),
            "eig_range": max(eigs) - min(eigs),
        },
        "queries": [
            {
                "query_index": index,
                "question": question,
                "labels": label,
                "eig": eig,
                "informative": keep,
            }
            for index, (question, label, eig, keep) in enumerate(
                zip(questions, labels, eigs, informative)
            )
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
            if "question_index" in request:
                responses.append(
                    f"Does the product have fixture property {request['question_index']}?"
                )
            else:
                count = len(request["candidate_titles"])
                offset = sum(ord(character) for character in request["question"]) % 7
                a_count = 1 + offset
                b_count = 1 + ((2 * offset) % 6)
                responses.append(
                    ("Y" * a_count)
                    + ("N" * b_count)
                    + ("U" * (count - a_count - b_count))
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
        raise ValueError("binary PSCon config selects the wrong model")
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
    config.openrouter_concurrency = 5
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
