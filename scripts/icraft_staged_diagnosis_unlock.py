#!/usr/bin/env python3
"""Gate open-world diagnosis recovery from staged iCRAFT evidence."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence, TypeVar

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mediq.data import load_mediq_tasks_with_report
from environments.mediq.parsing import parse_json_object
from helpers import Config, load_config
from model_factory import build_model_adapter


PRIOR_MODEL_CALL_IDS = frozenset(
    {2, 23, 40, 60, 62, 64, 96, 99, 100, 117, 125, 132, 137}
)
DEVELOPMENT_IDS = (
    8,
    13,
    14,
    29,
    37,
    38,
    46,
    55,
    66,
    81,
    84,
    85,
    88,
    98,
    105,
    109,
    110,
    114,
    121,
    138,
)
HOLDOUT_IDS = (
    0,
    1,
    4,
    5,
    6,
    12,
    15,
    17,
    18,
    19,
    20,
    21,
    22,
    24,
    26,
    28,
    31,
    32,
    36,
    39,
    41,
    48,
    49,
    51,
    52,
    53,
    54,
    56,
    57,
    58,
    61,
    63,
    72,
    73,
    74,
    75,
    79,
    80,
    83,
    86,
    87,
    89,
    91,
    92,
    94,
    97,
    104,
    106,
    112,
    119,
    120,
    122,
    124,
    127,
    128,
    130,
    134,
    135,
    136,
    139,
)
DIAGNOSIS_COUNT = 8
COVERAGE_THRESHOLD = 0.80

T = TypeVar("T")


class StructuredBatchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        row: int,
        response: str,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.row = row
        self.response = response


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip()
        key = clean.casefold()
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
    return result


def parse_diagnoses(text: str, count: int = DIAGNOSIS_COUNT) -> list[str]:
    raw = parse_json_object(text).get("diagnoses")
    if not isinstance(raw, list):
        raise ValueError("diagnoses must be a list")
    values: list[str] = []
    for item in raw:
        if isinstance(item, str):
            value = item.strip()
        elif isinstance(item, dict):
            value = next(
                (
                    item[key].strip()
                    for key in ("diagnosis", "name", "label")
                    if isinstance(item.get(key), str) and item[key].strip()
                ),
                "",
            )
        else:
            value = ""
        if value:
            values.append(value)
    values = _dedupe(values)
    if len(values) < count:
        raise ValueError(
            f"parsed {len(values)} usable unique diagnoses; requires at least {count}; "
            f"raw_items={len(raw)}"
        )
    return values[:count]


def initial_differential_messages(
    visible_facts: Sequence[str],
    count: int = DIAGNOSIS_COUNT,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Generate a broad clinical differential diagnosis from only the "
                "evidence shown. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Visible patient evidence:\n- "
                + "\n- ".join(visible_facts)
                + f"\n\nReturn exactly {count} distinct diagnoses as "
                '{"diagnoses":["diagnosis name", ...]}. Each item must be one '
                "diagnosis-name string, not an object or explanation. Include "
                "plausible specific diagnoses, not symptoms. You are not given "
                "answer options or any hidden benchmark label."
            ),
        },
    ]


def workup_differential_messages(
    initial_facts: Sequence[str],
    workup_facts: Sequence[str],
    initial_diagnoses: Sequence[str],
    count: int = DIAGNOSIS_COUNT,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Regenerate a clinical differential after a diagnostic workup. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Initial patient evidence:\n- "
                + "\n- ".join(initial_facts)
                + "\n\nNew guaranteed workup evidence:\n- "
                + "\n- ".join(workup_facts)
                + "\n\nEarlier differential:\n- "
                + "\n- ".join(initial_diagnoses)
                + f"\n\nReturn exactly {count} updated distinct diagnoses as "
                '{"diagnoses":["diagnosis name", ...]}. Each item must be one '
                "diagnosis-name string, not an object or explanation. Use all "
                "shown evidence. You are not given answer options or any hidden "
                "benchmark label."
            ),
        },
    ]


def semantic_diagnosis_messages(
    true_diagnosis: str,
    supports: Sequence[tuple[str, Sequence[str]]],
) -> list[dict[str, str]]:
    payload = {
        "true_diagnosis": true_diagnosis,
        "supports": [
            {"id": support_id, "diagnoses": list(diagnoses)}
            for support_id, diagnoses in supports
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict diagnosis-equivalence evaluator. The true "
                "diagnosis was hidden from every differential generator. Return "
                "strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each support, score the closest generated diagnosis in [0,1]. "
                "Score at least 0.8 only for the same diagnosis or a standard "
                "clinically equivalent synonym. A symptom, broad parent category, "
                "related disease, alternative subtype, or common differential must "
                "score below 0.8. Return exactly "
                '{"supports":[{"id":"...","best_match_score":0.0,'
                '"reason":"brief"}]}; preserve IDs and order.\n'
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_semantic_diagnosis(
    text: str,
    support_ids: Sequence[str],
) -> list[dict[str, Any]]:
    raw = parse_json_object(text).get("supports")
    if not isinstance(raw, list) or len(raw) != len(support_ids):
        raise ValueError("semantic response must contain one row per support")
    parsed = []
    for support_id, row in zip(support_ids, raw, strict=True):
        if not isinstance(row, dict) or row.get("id") != support_id:
            raise ValueError("semantic response changed support IDs or order")
        score = row.get("best_match_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("best_match_score must be numeric")
        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("best_match_score must be in [0,1]")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("semantic response requires a reason")
        parsed.append(
            {
                "id": support_id,
                "best_match_score": score,
                "covered": score >= COVERAGE_THRESHOLD,
                "reason": reason.strip(),
            }
        )
    return parsed


def complete_parsed_many(
    model: Any,
    messages: Sequence[list[dict[str, str]]],
    *,
    temperature: float,
    max_new_tokens: int,
    parser: Callable[[str], T],
    stage: str,
    retries: int,
) -> list[T]:
    current_messages = [list(item) for item in messages]
    responses = model.chat_complete_messages_batched(
        current_messages,
        temperature=temperature,
        block_size=256,
        max_new_tokens=max_new_tokens,
    )
    results: list[T | None] = [None] * len(messages)
    active = list(range(len(messages)))
    for attempt in range(retries + 1):
        failed: list[tuple[int, ValueError]] = []
        for index in active:
            try:
                results[index] = parser(responses[index])
            except ValueError as exc:
                failed.append((index, exc))
        if not failed:
            return [result for result in results if result is not None]
        if attempt >= retries:
            index, error = failed[0]
            raise StructuredBatchError(
                str(error),
                stage=stage,
                row=index,
                response=responses[index],
            ) from error
        retry_messages = []
        retry_indices = []
        for index, error in failed:
            current_messages[index] = [
                *current_messages[index],
                {"role": "assistant", "content": responses[index]},
                {
                    "role": "user",
                    "content": (
                        f"That response could not be parsed: {error}. Return only "
                        "corrected strict JSON matching the original schema."
                    ),
                },
            ]
            retry_indices.append(index)
            retry_messages.append(current_messages[index])
        retry_responses = model.chat_complete_messages_batched(
            retry_messages,
            temperature=temperature,
            block_size=256,
            max_new_tokens=max_new_tokens,
        )
        for index, response in zip(retry_indices, retry_responses, strict=True):
            responses[index] = response
        active = retry_indices
    raise AssertionError("unreachable")


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    initial_covered = sum(record["initial_measurement"]["covered"] for record in records)
    workup_covered = sum(record["workup_measurement"]["covered"] for record in records)
    recovered = sum(
        not record["initial_measurement"]["covered"]
        and record["workup_measurement"]["covered"]
        for record in records
    )
    omissions = len(records) - initial_covered
    gains = [
        record["workup_measurement"]["best_match_score"]
        - record["initial_measurement"]["best_match_score"]
        for record in records
    ]
    summary = {
        "num_tasks": len(records),
        "initial_covered": initial_covered,
        "initial_omitted": omissions,
        "workup_generated_covered": workup_covered,
        "initially_omitted_recovered_by_workup": recovered,
        "recovery_fraction_among_initial_omissions": (
            recovered / omissions if omissions else 0.0
        ),
        "mean_workup_best_match_gain": float(np.mean(gains)),
    }
    gates = {
        "all_twenty_tasks_completed": len(records) == 20,
        "initial_support_not_saturated": initial_covered <= 10,
        "workup_support_covers_at_least_fourteen": workup_covered >= 14,
        "at_least_eight_initial_omissions_recovered": recovered >= 8,
        "recovery_fraction_at_least_0_60": (
            summary["recovery_fraction_among_initial_omissions"] >= 0.60
        ),
        "mean_match_gain_at_least_0_25": (
            summary["mean_workup_best_match_gain"] >= 0.25
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {**summary, "gates": gates}


def _build_models(config: Config, judge_model: str) -> tuple[Any, Any]:
    spec = config.model_pairs[0].questioner
    generator = build_model_adapter(spec, config)
    judge_spec = replace(
        spec,
        model=judge_model,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    judge = build_model_adapter(judge_spec, config)
    return generator, judge


def _load_tasks(config: Config) -> list[Any]:
    tasks, excluded, raw_count = load_mediq_tasks_with_report(
        config.mediq_data_path,
        dataset="icraft_md",
        verify_official_hash=bool(config.mediq_verify_official_hash),
        skip_unusable_tasks=True,
    )
    if excluded or raw_count != 140 or len(tasks) != 140:
        raise ValueError("staged iCRAFT gate requires all 140 pinned tasks")
    return tasks


def run_development(config: Config, judge_model: str) -> dict[str, Any]:
    generator, judge = _build_models(config, judge_model)
    tasks = _load_tasks(config)
    selected = [tasks[index] for index in DEVELOPMENT_IDS]
    initial_messages = [
        initial_differential_messages(task.facts[:2])
        for task in selected
    ]
    initial = complete_parsed_many(
        generator,
        initial_messages,
        temperature=float(config.generation_temperature_diverse),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=parse_diagnoses,
        stage="initial_differential",
        retries=int(config.mediq_structured_max_retries),
    )
    workup_messages = [
        workup_differential_messages(task.facts[:2], task.facts[2:], diagnoses)
        for task, diagnoses in zip(selected, initial, strict=True)
    ]
    workup = complete_parsed_many(
        generator,
        workup_messages,
        temperature=float(config.generation_temperature_diverse),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=parse_diagnoses,
        stage="workup_differential",
        retries=int(config.mediq_structured_max_retries),
    )
    support_ids = ("initial", "workup_generated", "merged")
    semantic_messages = [
        semantic_diagnosis_messages(
            task.option_text(task.answer_idx),
            (
                ("initial", initial_diagnoses),
                ("workup_generated", workup_diagnoses),
                ("merged", _dedupe([*initial_diagnoses, *workup_diagnoses])),
            ),
        )
        for task, initial_diagnoses, workup_diagnoses in zip(
            selected, initial, workup, strict=True
        )
    ]
    semantic = complete_parsed_many(
        judge,
        semantic_messages,
        temperature=0.0,
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=lambda text: parse_semantic_diagnosis(text, support_ids),
        stage="semantic_measurement",
        retries=int(config.mediq_structured_max_retries),
    )
    records = []
    for task_id, task, initial_diagnoses, workup_diagnoses, measurements in zip(
        DEVELOPMENT_IDS,
        selected,
        initial,
        workup,
        semantic,
        strict=True,
    ):
        records.append(
            {
                "source_id": task_id,
                "initial_visible_facts": list(task.facts[:2]),
                "workup_revealed_facts": list(task.facts[2:]),
                "true_diagnosis_measurement_only": task.option_text(task.answer_idx),
                "initial_diagnoses": initial_diagnoses,
                "workup_generated_diagnoses": workup_diagnoses,
                "initial_measurement": measurements[0],
                "workup_measurement": measurements[1],
                "merged_measurement": measurements[2],
            }
        )
    summary = summarize(records)
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "development_gate_failed",
        "protocol": {
            "dataset": "icraft_md",
            "selection_seed": 24288,
            "prior_model_call_ids": sorted(PRIOR_MODEL_CALL_IDS),
            "development_ids": list(DEVELOPMENT_IDS),
            "holdout_ids": list(HOLDOUT_IDS),
            "diagnosis_count": DIAGNOSIS_COUNT,
            "coverage_threshold": COVERAGE_THRESHOLD,
            "answer_options_hidden_from_generation": True,
            "truth_used_for_measurement_only": True,
            "judge_model": judge_model,
        },
        "summary": summary,
        "records": records,
        "usage": {
            "generator": generator.usage_snapshot(),
            "judge": judge.usage_snapshot(),
        },
    }


def run_serving_smoke(config: Config, judge_model: str) -> dict[str, Any]:
    """Exercise exactly ten physical requests over all three stages."""
    generator, judge = _build_models(config, judge_model)
    tasks = _load_tasks(config)
    selected = [tasks[index] for index in DEVELOPMENT_IDS[:4]]
    initial = complete_parsed_many(
        generator,
        [initial_differential_messages(task.facts[:2]) for task in selected],
        temperature=float(config.generation_temperature_diverse),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=parse_diagnoses,
        stage="smoke_initial",
        retries=0,
    )
    workup = complete_parsed_many(
        generator,
        [
            workup_differential_messages(task.facts[:2], task.facts[2:], diagnoses)
            for task, diagnoses in zip(selected, initial, strict=True)
        ],
        temperature=float(config.generation_temperature_diverse),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=parse_diagnoses,
        stage="smoke_workup",
        retries=0,
    )
    semantic = complete_parsed_many(
        judge,
        [
            semantic_diagnosis_messages(
                task.option_text(task.answer_idx),
                (("initial", initial[index]), ("workup_generated", workup[index])),
            )
            for index, task in enumerate(selected[:2])
        ],
        temperature=0.0,
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=lambda text: parse_semantic_diagnosis(
            text, ("initial", "workup_generated")
        ),
        stage="smoke_semantic",
        retries=0,
    )
    usage = {
        "generator": generator.usage_snapshot(),
        "judge": judge.usage_snapshot(),
    }
    requests = sum(int(value["adapter_requests"]) for value in usage.values())
    return {
        "schema_version": 1,
        "status": "passed" if requests == 10 else "failed",
        "physical_requests": requests,
        "expected_physical_requests": 10,
        "stage_counts": {"initial": 4, "workup": 4, "semantic": 2},
        "parsed_initial_sizes": [len(values) for values in initial],
        "parsed_workup_sizes": [len(values) for values in workup],
        "semantic_rows": semantic,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "development"),
        required=True,
    )
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            run_serving_smoke(config, args.judge_model)
            if args.stage == "serving_smoke"
            else run_development(config, args.judge_model)
        )
        filename = (
            "SERVING_SMOKE.json"
            if args.stage == "serving_smoke"
            else "DEVELOPMENT.json"
        )
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "status": "failed_closed",
            "error": f"{type(exc).__name__}: {exc}",
            "failing_stage": getattr(exc, "stage", None),
            "failing_row": getattr(exc, "row", None),
            "raw_failing_response": getattr(exc, "response", None),
        }
        filename = (
            "SERVING_SMOKE_FAILURE.json"
            if args.stage == "serving_smoke"
            else "DEVELOPMENT_FAILURE.json"
        )
        (args.output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / filename).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = (
        {"status": payload["status"], **payload["summary"]}
        if "summary" in payload
        else payload
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
