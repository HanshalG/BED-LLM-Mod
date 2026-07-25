#!/usr/bin/env python3
"""Test GPT-5.4 semantic-likelihood stability on AmbigDocs."""

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
from scripts.ambigdocs_binary_serving_smoke import (
    EXPECTED_QID,
    ROW_INDEX,
    _label_messages,
    _question_messages,
    _usage,
    verify_source,
)
from scripts.pscon_binary_query_serving_smoke import parse_labels, parse_question
from scripts.pscon_semantic_tree_smoke import entropy


INTERFACE_VERSION = "ambigdocs-gpt54-stability-smoke-1"
MODEL_ID = "openai/gpt-5.4"
SEED = 24_389
QUESTION_COUNT = 4
REPEAT_COUNT = 3
EXPECTED_REQUESTS = QUESTION_COUNT + QUESTION_COUNT + REPEAT_COUNT - 1
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


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_smoke(
    config: Config,
    *,
    source_path: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    row = verify_source(source_path)
    count = len(row["documents"])
    raw: dict[str, Any] = {
        "row_index": ROW_INDEX,
        "qid": EXPECTED_QID,
        "target_sampled": False,
    }
    try:
        question_raw = model.chat_complete_messages_batched(
            [_question_messages(row, index) for index in range(QUESTION_COUNT)],
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=120,
        )
        raw["questions"] = question_raw
        _checkpoint(raw_path, raw)
        questions = [parse_question(value) for value in question_raw]
        if len(questions) != QUESTION_COUNT:
            raise ValueError("question response count changed")

        classification_questions = list(questions) + [questions[0], questions[0]]
        label_raw = model.chat_complete_messages_batched(
            [_label_messages(row, question) for question in classification_questions],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=20,
        )
        raw["labels"] = label_raw
        _checkpoint(raw_path, raw)
        labels = [parse_labels(value, count) for value in label_raw]
        if len(labels) != len(classification_questions):
            raise ValueError("label response count changed")
        raw["all_parsed"] = True
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise StabilityExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    base_labels = labels[:QUESTION_COUNT]
    repeats = [labels[0], labels[-2], labels[-1]]
    eigs = [entropy(list(value)) for value in base_labels]
    informative = [len(set(value)) >= 2 and eig >= 0.30 for value, eig in zip(base_labels, eigs)]
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_questions_and_labels_parse": (
            len(questions) == QUESTION_COUNT and len(labels) == 6
        ),
        "all_four_questions_unique": len(set(questions)) == QUESTION_COUNT,
        "at_least_three_unique_base_partitions": len(set(base_labels)) >= 3,
        "all_four_base_partitions_informative": all(informative),
        "base_eig_range_at_least_0_10": max(eigs) - min(eigs) >= 0.10,
        "three_exact_repeat_maps_agree": len(set(repeats)) == 1,
        "cost_at_most_0_50": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "seed": SEED,
            "row_index": ROW_INDEX,
            "qid": EXPECTED_QID,
            "support_size": count,
            "question_count": QUESTION_COUNT,
            "repeat_count": REPEAT_COUNT,
            "expected_requests": EXPECTED_REQUESTS,
            "target_sampled": False,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
        },
        "metrics": {
            "unique_question_count": len(set(questions)),
            "unique_base_partition_count": len(set(base_labels)),
            "informative_base_partition_count": sum(informative),
            "mean_base_eig": statistics.fmean(eigs),
            "base_eig_range": max(eigs) - min(eigs),
            "repeat_unique_map_count": len(set(repeats)),
        },
        "questions": questions,
        "base_labels": base_labels,
        "repeat_labels": repeats,
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
        patterns = {
            0: "YNNNNN",
            1: "YYNNNN",
            2: "YYYNNN",
            3: "YNYNNN",
        }
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            if "question_index" in request:
                responses.append(
                    f"Does the intended entity have fixture trait {request['question_index']}?"
                )
            else:
                index = int(request["clarification_question"].split()[-1][:-1])
                responses.append(patterns[index])
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
        raise ValueError("AmbigDocs stability config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.20
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 6
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = DeterministicFixtureModel() if args.dry_run else _build_model(config)
    try:
        payload = run_smoke(config, source_path=args.source_path, raw_path=raw_path, model=model)
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
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
            failure["private_raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        _checkpoint(args.output_dir / "STABILITY_FAILURE.json", failure)
        raise
    output = args.output_dir / "STABILITY.json"
    _checkpoint(output, payload)
    print(json.dumps({"status": payload["status"], "metrics": payload["metrics"], "gates": payload["gates"], "usage": payload["usage"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
