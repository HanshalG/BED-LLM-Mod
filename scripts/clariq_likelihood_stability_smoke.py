#!/usr/bin/env python3
"""Test repeated GPT-5.4 facet-likelihood maps on fixed ClariQ questions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.pscon_binary_query_serving_smoke import parse_labels


INTERFACE_VERSION = "clariq-likelihood-stability-smoke-1"
SOURCE_REPOSITORY = "https://github.com/aliannejadi/ClariQ"
SOURCE_COMMIT = "46885a544581a0af8aff0681d29e4971807e2912"
DEV_SHA256 = "68d2a5f87eab73721979b5f45f64099a9b2f080db1d0ce4b979d9daa4249906e"
MODEL_ID = "openai/gpt-5.4"
TOPIC_ID = "133"
QUESTION_IDS = ("Q00796", "Q01384", "Q03514", "Q03741")
EXPECTED_FACET_IDS = ("F0134", "F0135", "F0136", "F0137", "F0138")
SEED = 24_390
EXPECTED_REQUESTS = 10
MAX_COST_USD = 0.50


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class StabilityExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def verify_source(source_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"ClariQ commit is {commit}, expected {SOURCE_COMMIT}")
    path = source_root / "data" / "dev.tsv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DEV_SHA256:
        raise ValueError(f"ClariQ dev hash is {digest}, expected {DEV_SHA256}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["topic_id"] == TOPIC_ID
        ]
    facets = {
        row["facet_id"]: row["facet_desc"]
        for row in rows
    }
    questions = {
        row["question_id"]: row["question"]
        for row in rows
    }
    if tuple(sorted(facets)) != EXPECTED_FACET_IDS:
        raise ValueError("frozen ClariQ facet support changed")
    if any(question_id not in questions for question_id in QUESTION_IDS):
        raise ValueError("frozen ClariQ question bank changed")
    return {
        "initial_request": rows[0]["initial_request"],
        "facets": [
            {"facet_id": facet_id, "description": facets[facet_id]}
            for facet_id in EXPECTED_FACET_IDS
        ],
        "questions": [
            {"question_id": question_id, "question": questions[question_id]}
            for question_id in QUESTION_IDS
        ],
    }


def _messages(task: dict[str, Any], question: dict[str, str]) -> list[dict[str, str]]:
    request = {
        "initial_request": task["initial_request"],
        "clarification_question": question["question"],
        "candidate_facets": task["facets"],
    }
    return [
        {
            "role": "system",
            "content": (
                "Classify how a user with each candidate information need would "
                "answer the clarification question. Use Y when the facet clearly "
                "implies yes, N when it clearly implies no, and U when the facet does "
                "not determine an answer. Output exactly one Y/N/U character per "
                "candidate facet in input order, with no spaces, punctuation, or prose."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _entropy(labels: str) -> float:
    return -sum(
        (labels.count(value) / len(labels))
        * math.log(labels.count(value) / len(labels))
        for value in set(labels)
    )


def _usage(model: ChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "adapter_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "model": snapshot,
    }


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_smoke(
    config: Config,
    *,
    source_root: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    task = verify_source(source_root)
    questions = task["questions"]
    request_indices = [0, 1, 2, 3, 0, 1, 2, 3, 0, 0]
    raw: dict[str, Any] = {
        "topic_id": TOPIC_ID,
        "question_ids": list(QUESTION_IDS),
        "request_indices": request_indices,
        "hidden_facet_loaded": False,
    }
    try:
        responses = model.chat_complete_messages_batched(
            [_messages(task, questions[index]) for index in request_indices],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=16,
        )
        raw["responses"] = responses
        _checkpoint(raw_path, raw)
        if len(responses) != EXPECTED_REQUESTS:
            raise ValueError("likelihood response count changed")
        labels = [
            parse_labels(response, len(EXPECTED_FACET_IDS))
            for response in responses
        ]
        raw["all_parsed"] = True
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise StabilityExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    by_question = {
        index: [
            label
            for request_index, label in zip(request_indices, labels)
            if request_index == index
        ]
        for index in range(len(questions))
    }
    base_labels = labels[:4]
    eigs = [_entropy(label) for label in base_labels]
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_ten_maps_parse": len(labels) == EXPECTED_REQUESTS,
        "every_question_repeat_agrees": all(
            len(set(values)) == 1 for values in by_question.values()
        ),
        "four_way_repeat_agrees": len(set(by_question[0])) == 1,
        "all_base_maps_informative": all(
            len(set(label)) >= 2 and eig >= 0.30
            for label, eig in zip(base_labels, eigs)
        ),
        "at_least_three_unique_base_maps": len(set(base_labels)) >= 3,
        "base_eig_range_at_least_0_10": max(eigs) - min(eigs) >= 0.10,
        "cost_at_most_0_50": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "dev_sha256": DEV_SHA256,
            "topic_id": TOPIC_ID,
            "facet_ids": list(EXPECTED_FACET_IDS),
            "question_ids": list(QUESTION_IDS),
            "seed": SEED,
            "model": MODEL_ID,
            "expected_requests": EXPECTED_REQUESTS,
            "hidden_facet_loaded": False,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
        },
        "metrics": {
            "unique_base_map_count": len(set(base_labels)),
            "informative_base_map_count": sum(
                len(set(label)) >= 2 and eig >= 0.30
                for label, eig in zip(base_labels, eigs)
            ),
            "base_eig_range": max(eigs) - min(eigs),
            "repeat_unique_counts": {
                QUESTION_IDS[index]: len(set(values))
                for index, values in by_question.items()
            },
        },
        "questions": questions,
        "base_labels": base_labels,
        "base_eigs": eigs,
        "repeat_labels": {
            QUESTION_IDS[index]: values
            for index, values in by_question.items()
        },
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
        patterns = {
            "are you looking for declaration of independence at the national archives": "NNYNN",
            "are you trying to look up a speech": "YYNNN",
            "would you like to know its interpretation by the us supreme court": "NNNYN",
            "would you like to learn about the author of that quote": "YNYNN",
        }
        responses = []
        for messages in batch_messages:
            question = json.loads(messages[-1]["content"])["clarification_question"]
            responses.append(patterns[question])
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
    if spec.model != MODEL_ID:
        raise ValueError("ClariQ config selects the wrong model")
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
    config.openrouter_projected_cost_usd = 0.15
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 10
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = DeterministicFixtureModel() if args.dry_run else _build_model(config)
    try:
        payload = run_smoke(
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
        if isinstance(exc, StabilityExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "STABILITY_FAILURE.json", failure)
        raise
    output = args.output_dir / "STABILITY.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
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
