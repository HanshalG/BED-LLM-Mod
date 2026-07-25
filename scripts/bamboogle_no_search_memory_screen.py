#!/usr/bin/env python3
"""Screen Bamboogle mechanics tasks for frontier-model memory saturation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import string
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.bamboogle_semantic_bed_manifest import (
    SOURCE_SHA256,
    load_rows,
    sha256_file,
)


MODEL_ID = "openai/gpt-5.4"
INTERFACE_VERSION = "bamboogle-no-search-memory-screen-1"
MECHANICS_IDS = (
    "test_87",
    "test_110",
    "test_72",
    "test_69",
    "test_61",
)
SAMPLES_PER_TASK = 5
EXPECTED_REQUESTS = len(MECHANICS_IDS) * SAMPLES_PER_TASK
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 96
MAX_COST_USD = 0.15


def _strict_object(text: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("response must be one JSON object")
    return payload


def parse_answer(text: str) -> str:
    payload = _strict_object(text)
    if set(payload) != {"answer"}:
        raise ValueError("answer response has unexpected keys")
    answer = payload["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a nonempty string")
    return " ".join(answer.split())


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def answer_matches(prediction: str, golden_answers: Sequence[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    return normalized_prediction in {
        normalize_answer(answer) for answer in golden_answers
    }


def categorical_entropy(values: Sequence[str]) -> float:
    if not values:
        raise ValueError("cannot compute entropy of an empty sample")
    counts = Counter(values)
    total = len(values)
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
    )


def answer_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer from your existing knowledge without tools, search, "
                "citations, explanation, or reasoning. Return exactly one JSON "
                'object {"answer":"short answer"} and no markdown.'
            ),
        },
        {
            "role": "user",
            "content": json.dumps({"question": question}, separators=(",", ":")),
        },
    ]


def load_mechanics(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("Bamboogle source hash mismatch")
    rows = load_rows(path)
    row_by_id = {str(row["id"]): row for row in rows}
    if any(task_id not in row_by_id for task_id in MECHANICS_IDS):
        raise ValueError("Bamboogle mechanics IDs do not reproduce")
    return [row_by_id[task_id] for task_id in MECHANICS_IDS]


def _build_model(config: Config) -> Any:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("Bamboogle screen config selects the wrong model")
    return build_model_adapter(spec, config)


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "generator": snapshot,
    }


def summarize_records(
    tasks: Sequence[dict[str, Any]],
    answers_by_task: Sequence[Sequence[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(tasks) != len(answers_by_task):
        raise ValueError("task and answer groups have different lengths")
    records: list[dict[str, Any]] = []
    for task, predictions in zip(tasks, answers_by_task):
        if len(predictions) != SAMPLES_PER_TASK:
            raise ValueError("wrong number of answer samples")
        normalized = [normalize_answer(value) for value in predictions]
        matches = [
            answer_matches(value, task["golden_answers"])
            for value in predictions
        ]
        counts = Counter(normalized)
        modal_value = max(
            counts,
            key=lambda value: (counts[value], -normalized.index(value)),
        )
        records.append(
            {
                "task_id": str(task["id"]),
                "sample_match_flags": matches,
                "sample_accuracy": sum(matches) / len(matches),
                "unique_normalized_answers": len(counts),
                "answer_entropy_nats": categorical_entropy(normalized),
                "modal_answer_correct": modal_value
                in {
                    normalize_answer(answer)
                    for answer in task["golden_answers"]
                },
            }
        )

    modal_correct_count = sum(
        bool(record["modal_answer_correct"]) for record in records
    )
    sample_accuracy = sum(
        sum(bool(value) for value in record["sample_match_flags"])
        for record in records
    ) / EXPECTED_REQUESTS
    tasks_below_point_eight_gold_support = sum(
        float(record["sample_accuracy"]) < 0.8 for record in records
    )
    mean_entropy = sum(
        float(record["answer_entropy_nats"]) for record in records
    ) / len(records)
    gates = {
        "modal_correct_at_most_3_of_5": modal_correct_count <= 3,
        "sample_accuracy_at_most_point_70": sample_accuracy <= 0.70,
        "at_least_2_tasks_below_point_8_gold_support": (
            tasks_below_point_eight_gold_support >= 2
        ),
    }
    summary = {
        "modal_correct_count": modal_correct_count,
        "sample_accuracy": sample_accuracy,
        "tasks_below_point_eight_gold_support": (
            tasks_below_point_eight_gold_support
        ),
        "mean_answer_entropy_nats": mean_entropy,
        "gates": gates,
        "passed": all(gates.values()),
    }
    return records, summary


def run_screen(
    config: Config,
    *,
    data_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    tasks = load_mechanics(data_path)
    model = model_adapter if model_adapter is not None else _build_model(config)
    messages = [
        answer_messages(str(task["question"]))
        for task in tasks
        for _ in range(SAMPLES_PER_TASK)
    ]
    responses = model.chat_complete_messages_batched(
        messages,
        temperature=TEMPERATURE,
        block_size=EXPECTED_REQUESTS,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(
            {
                "interface_version": INTERFACE_VERSION,
                "task_ids": list(MECHANICS_IDS),
                "responses": responses,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if len(responses) != EXPECTED_REQUESTS:
        raise ValueError("Bamboogle memory screen returned the wrong call count")
    parsed = [parse_answer(response) for response in responses]
    answers_by_task = [
        parsed[index : index + SAMPLES_PER_TASK]
        for index in range(0, len(parsed), SAMPLES_PER_TASK)
    ]
    usage = _usage_snapshot(model)
    if usage["physical_requests"] != EXPECTED_REQUESTS:
        raise ValueError("Bamboogle memory screen physical call count changed")
    if usage["reasoning_tokens"] != 0:
        raise ValueError("Bamboogle memory screen used reasoning tokens")
    if usage["adapter_cost_usd"] > MAX_COST_USD:
        raise ValueError("Bamboogle memory screen exceeded its cost cap")
    records, summary = summarize_records(tasks, answers_by_task)
    return {
        "interface_version": INTERFACE_VERSION,
        "status": "passed" if summary["passed"] else "gate_failed",
        "protocol": {
            "model": MODEL_ID,
            "reasoning_disabled": True,
            "temperature": TEMPERATURE,
            "samples_per_task": SAMPLES_PER_TASK,
            "expected_requests": EXPECTED_REQUESTS,
            "max_cost_usd": MAX_COST_USD,
            "search_calls": 0,
            "gold_answers_hidden_from_model": True,
        },
        "records": records,
        "summary": summary,
        "usage": usage,
        "private_raw_sha256": hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Bamboogle no-search memory-saturation screen."
    )
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_screen(
        config,
        data_path=args.data_path,
        raw_path=args.raw_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
