#!/usr/bin/env python3
"""OpenRouter-native CA-BED shared-tree ranking gate for Animals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.animals_cabed_shared_tree_v10 import run_stage
from scripts.movielens_profile_dynamics_gate import _parse_json_object


SCHEMA_VERSION = 11
SELECTION_SEED = 24310
SMOKE_RUN_CAP_USD = 0.75
FORMAL_RUN_CAP_USD = 5.00

_LIKELIHOOD_PATTERN = re.compile(
    r"Hypothesized target animal:\s*\n(?P<animal>.*?)"
    r"\n\s*\nQuestion:\s*\n(?P<question>.*?)"
    r"\n\s*\nAllowed answer labels:",
    flags=re.DOTALL,
)


def parse_likelihood_conversation(
    messages: Sequence[dict[str, str]],
) -> tuple[str, str]:
    if not messages:
        raise ValueError("semantic likelihood conversation is empty")
    content = str(messages[-1].get("content", ""))
    match = _LIKELIHOOD_PATTERN.search(content)
    if match is None:
        raise ValueError("could not parse animal likelihood conversation")
    animal = " ".join(match.group("animal").split())
    question = " ".join(match.group("question").split())
    if not animal or not question:
        raise ValueError("animal likelihood conversation has an empty field")
    return animal, question


def batch_classification_messages(
    question: str,
    animals: Sequence[str],
) -> list[dict[str, str]]:
    schema = {
        "answers": [
            {"animal": animal, "answer": "Yes|No"}
            for animal in animals
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "Classify animal properties for Bayesian experimental design. "
                "Return strict JSON only, with no reasoning or explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "For every animal below, decide the answer an accurate 20 "
                "Questions answerer should give. Use exactly Yes or No even when "
                "the property is uncommon or context-dependent. Preserve the "
                "animal order and spelling. Do not omit or add animals.\n"
                f"Animals: {json.dumps(list(animals), separators=(',', ':'))}\n"
                "Return exactly this schema:\n"
                + json.dumps(schema, separators=(",", ":"))
            ),
        },
    ]


def parse_batch_classification(
    text: str,
    animals: Sequence[str],
) -> tuple[str, ...]:
    rows = _parse_json_object(text).get("answers")
    if not isinstance(rows, list) or len(rows) != len(animals):
        raise ValueError("batch classification lost an animal")
    answers: list[str] = []
    for expected, row in zip(animals, rows, strict=True):
        if not isinstance(row, dict) or row.get("animal") != expected:
            raise ValueError("batch classification changed animal order or name")
        answer = str(row.get("answer", "")).strip()
        if answer not in {"Yes", "No"}:
            raise ValueError("batch classification answer must be Yes or No")
        answers.append(answer)
    return tuple(answers)


class BatchedSemanticOpenRouterModel:
    """Delegate generation while batching one semantic table per question."""

    def __init__(self, delegate: Any, config: Config) -> None:
        self.delegate = delegate
        self.config = config
        self.classification_records: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        num_responses: int = 1,
    ) -> list[str]:
        return self.delegate.chat_complete(
            messages,
            temperature,
            num_responses=num_responses,
        )

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        return self.delegate.chat_complete_messages_batched(
            batch_messages,
            temperature,
            block_size,
            max_new_tokens=max_new_tokens,
        )

    def chat_probabilities_messages_batched(
        self,
        messages: list[list[dict[str, str]]],
        responses: list[str],
        temperature: float,
        block_size: int,
    ) -> list[dict[str, float]]:
        del temperature
        if responses != ["Yes", "No"]:
            raise ValueError("V11 supports exactly the Yes/No likelihood labels")

        grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for index, conversation in enumerate(messages):
            animal, question = parse_likelihood_conversation(conversation)
            grouped[question].append((index, animal))

        questions = list(grouped)
        animals_by_question = [
            [animal for _index, animal in grouped[question]]
            for question in questions
        ]
        prompts = [
            batch_classification_messages(question, animals)
            for question, animals in zip(
                questions,
                animals_by_question,
                strict=True,
            )
        ]
        raw = self.delegate.chat_complete_messages_batched(
            prompts,
            temperature=0.0,
            block_size=block_size,
            max_new_tokens=self.config.openrouter_max_output_tokens,
        )
        if len(raw) != len(questions):
            raise ValueError("semantic batch request count changed")

        output: list[dict[str, float] | None] = [None] * len(messages)
        for question, animals, text in zip(
            questions,
            animals_by_question,
            raw,
            strict=True,
        ):
            answers = parse_batch_classification(text, animals)
            self.classification_records.append(
                {
                    "question": question,
                    "answers": [
                        {"animal": animal, "answer": answer}
                        for animal, answer in zip(
                            animals,
                            answers,
                            strict=True,
                        )
                    ],
                }
            )
            for (index, _animal), answer in zip(
                grouped[question],
                answers,
                strict=True,
            ):
                output[index] = (
                    {"Yes": 1.0, "No": 0.0}
                    if answer == "Yes"
                    else {"Yes": 0.0, "No": 1.0}
                )
        if any(row is None for row in output):
            raise ValueError("semantic batch output lost an input row")
        return [row for row in output if row is not None]

    def usage_snapshot(self) -> dict[str, Any]:
        return self.delegate.usage_snapshot()


def run_openrouter_stage(
    config: Config,
    *,
    stage: str,
    model: BatchedSemanticOpenRouterModel,
) -> dict[str, Any]:
    payload = run_stage(
        config,
        stage=stage,
        questioner=model,
        answerer=model,
    )
    usage = model.usage_snapshot()
    payload["schema_version"] = SCHEMA_VERSION
    payload["protocol"].update(
        {
            "interface": "openrouter_batched_semantic_classification",
            "semantic_labels": ["Yes", "No"],
            "semantic_raw_probabilities": [1.0, 0.0],
            "semantic_probability_after_confidence_smoothing": [0.85, 0.15],
            "semantic_batches_are_target_blind": True,
            "semantic_batch_count": len(model.classification_records),
            "openrouter_model": config.model_pairs[0].questioner.model,
        }
    )
    payload["semantic_classifications"] = model.classification_records
    payload["usage"] = usage
    payload["summary"]["gates"]["zero_reasoning_tokens"] = (
        int(usage["adapter_reasoning_tokens"]) == 0
    )
    payload["summary"]["gates"]["all_pass"] = all(
        payload["summary"]["gates"].values()
    )
    payload["status"] = (
        "passed"
        if payload["summary"]["gates"]["all_pass"]
        else "gate_failed"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "formal"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.10
        config.openrouter_run_budget_usd = SMOKE_RUN_CAP_USD
    else:
        config.openrouter_projected_cost_usd = 1.00
        config.openrouter_run_budget_usd = FORMAL_RUN_CAP_USD
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "GATE.json"
    )
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )

    delegate = build_model_adapter(config.model_pairs[0].questioner, config)
    model = BatchedSemanticOpenRouterModel(delegate, config)
    try:
        payload = run_openrouter_stage(
            config,
            stage=args.stage,
            model=model,
        )
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "usage": model.usage_snapshot(),
            "completed_semantic_batches": model.classification_records,
        }
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": payload["status"], **payload["summary"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
