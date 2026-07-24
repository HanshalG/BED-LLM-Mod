#!/usr/bin/env python3
"""Measure action and order opportunity in ClinDiag belief regeneration."""

from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.clindiag_staged_generator_gate import (
    CLINDIAG_SOURCE_COMMIT,
    CLINDIAG_ZIP_SHA256,
    COVERAGE_THRESHOLD,
    DIAGNOSIS_COUNT,
    HOLDOUT_IDS,
    ClinDiagCase,
    _build_models,
    initial_differential_messages,
    load_selected_cases,
)
from scripts.icraft_staged_diagnosis_unlock import (
    complete_parsed_many,
    parse_diagnoses,
    parse_semantic_diagnosis,
    semantic_diagnosis_messages,
)


SELECTION_SEED = 24290
OPPORTUNITY_IDS = (
    "rare266",
    "rare179",
    "22112808",
    "rare128",
    "15509821",
    "rare215",
    "rare158",
    "26765577",
    "16148290",
    "14597471",
    "17113188",
    "rare85",
)
SMOKE_IDS = (
    "rare22",
    "11097981",
)
ACTIONS = (
    "history",
    "physical_exam",
    "laboratory_tests",
    "imaging",
    "other_tests",
)
ACTION_LABELS = {
    "history": "Medical history",
    "physical_exam": "Physical examination",
    "laboratory_tests": "Laboratory tests",
    "imaging": "Radiographic and imaging tests",
    "other_tests": "Other diagnostic tests",
}
SEQUENCES = tuple(
    (first, second)
    for first, second in permutations(ACTIONS, 2)
)
EXPECTED_FORMAL_REQUESTS = (
    len(OPPORTUNITY_IDS)
    * (1 + len(ACTIONS) + len(SEQUENCES) + 1 + 1)
)


def action_evidence(case: ClinDiagCase, action: str) -> Any:
    if action == "history":
        value = case.medical_history.get("medical_history")
    elif action == "physical_exam":
        value = case.physical_examination.get("physical_examinations")
    elif action == "laboratory_tests":
        value = case.diagnostic_test.get("laboratory_examinations")
    elif action == "imaging":
        value = case.diagnostic_test.get("radiographic_examinations")
    elif action == "other_tests":
        value = case.diagnostic_test.get("other_examinations")
    else:
        raise ValueError(f"unknown ClinDiag action: {action}")
    if not value:
        raise ValueError(f"{case.source_id}: action {action} has no evidence")
    return value


def refresh_differential_messages(
    case: ClinDiagCase,
    observed_actions: Sequence[str],
    previous_diagnoses: Sequence[str],
) -> list[dict[str, str]]:
    evidence = []
    for step, action in enumerate(observed_actions, start=1):
        evidence.append(
            f"Step {step} - {ACTION_LABELS[action]}:\n"
            + json.dumps(
                action_evidence(case, action),
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    return [
        {
            "role": "system",
            "content": (
                "Regenerate a precise open-world clinical differential after "
                "new evidence. Evidence order and the prior differential are part "
                "of the current belief state. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial presentation:\n{case.initial_information}\n\n"
                "Evidence acquired in order:\n"
                + "\n\n".join(evidence)
                + "\n\nDifferential before the latest evidence:\n- "
                + "\n- ".join(previous_diagnoses)
                + f"\n\nReturn exactly {DIAGNOSIS_COUNT} updated, distinct, "
                "specific, unifying diagnoses as "
                '{"diagnoses":["diagnosis name", ...]}. Each item must be one '
                "diagnosis-name string, not an explanation or object. Infer from "
                "all evidence shown, including rare conditions. No answer options, "
                "case title, final diagnosis, or hidden benchmark label is provided."
            ),
        },
    ]


def _parse_support(text: str) -> list[str]:
    return parse_diagnoses(text, DIAGNOSIS_COUNT)


def _support_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {value.strip().casefold() for value in left}
    right_set = {value.strip().casefold() for value in right}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _argmax_key(values: dict[str, float], order: Sequence[str]) -> str:
    return max(order, key=lambda key: (values[key], -order.index(key)))


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    one_step_scores = record["one_step_scores"]
    sequence_scores = record["sequence_scores"]
    one_step_spread = max(one_step_scores.values()) - min(one_step_scores.values())
    order_gaps = {
        f"{first}|{second}": abs(
            sequence_scores[f"{first}>{second}"]
            - sequence_scores[f"{second}>{first}"]
        )
        for index, first in enumerate(ACTIONS)
        for second in ACTIONS[index + 1 :]
    }
    sequence_order = [f"{first}>{second}" for first, second in SEQUENCES]
    greedy_action = _argmax_key(one_step_scores, ACTIONS)
    oracle_sequence = _argmax_key(sequence_scores, sequence_order)
    greedy_continuation = max(
        score
        for key, score in sequence_scores.items()
        if key.startswith(f"{greedy_action}>")
    )
    oracle_two_step = sequence_scores[oracle_sequence]
    best_one_step = max(one_step_scores.values())
    duplicate_sequence = record["duplicate_sequence"]
    duplicate_score_gap = abs(
        sequence_scores[duplicate_sequence] - record["duplicate_score"]
    )
    return {
        "one_step_spread": one_step_spread,
        "max_reverse_order_gap": max(order_gaps.values()),
        "reverse_order_gaps": order_gaps,
        "greedy_action": greedy_action,
        "oracle_sequence": oracle_sequence,
        "oracle_first_differs_from_greedy": (
            oracle_sequence.split(">", 1)[0] != greedy_action
        ),
        "greedy_continuation_score": greedy_continuation,
        "oracle_two_step_score": oracle_two_step,
        "nonmyopic_gap_over_greedy_continuation": (
            oracle_two_step - greedy_continuation
        ),
        "oracle_two_step_gain_over_best_one_step": (
            oracle_two_step - best_one_step
        ),
        "duplicate_score_gap": duplicate_score_gap,
        "duplicate_support_jaccard": _support_jaccard(
            record["sequence_supports"][duplicate_sequence],
            record["duplicate_support"],
        ),
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [analyze_record(record) for record in records]
    initial_covered = sum(record["initial_score"] >= COVERAGE_THRESHOLD for record in records)
    spread_cases = sum(item["one_step_spread"] >= 0.20 for item in diagnostics)
    order_cases = sum(item["max_reverse_order_gap"] >= 0.15 for item in diagnostics)
    gap_cases = sum(
        item["nonmyopic_gap_over_greedy_continuation"] >= 0.10
        for item in diagnostics
    )
    first_differs = sum(
        item["oracle_first_differs_from_greedy"] for item in diagnostics
    )
    mean_gap = float(
        np.mean(
            [
                item["nonmyopic_gap_over_greedy_continuation"]
                for item in diagnostics
            ]
        )
    )
    mean_two_step_gain = float(
        np.mean(
            [
                item["oracle_two_step_gain_over_best_one_step"]
                for item in diagnostics
            ]
        )
    )
    mean_duplicate_gap = float(
        np.mean([item["duplicate_score_gap"] for item in diagnostics])
    )
    max_duplicate_gap = max(item["duplicate_score_gap"] for item in diagnostics)
    mean_duplicate_jaccard = float(
        np.mean([item["duplicate_support_jaccard"] for item in diagnostics])
    )
    summary = {
        "num_cases": len(records),
        "initial_covered": initial_covered,
        "one_step_spread_at_least_0_20_cases": spread_cases,
        "reverse_order_gap_at_least_0_15_cases": order_cases,
        "nonmyopic_gap_at_least_0_10_cases": gap_cases,
        "oracle_first_differs_from_greedy_cases": first_differs,
        "mean_nonmyopic_gap_over_greedy_continuation": mean_gap,
        "mean_oracle_two_step_gain_over_best_one_step": mean_two_step_gain,
        "mean_duplicate_score_gap": mean_duplicate_gap,
        "max_duplicate_score_gap": max_duplicate_gap,
        "mean_duplicate_support_jaccard": mean_duplicate_jaccard,
    }
    gates = {
        "all_twelve_cases_completed": len(records) == 12,
        "initial_support_not_saturated": initial_covered <= 5,
        "one_step_spread_on_at_least_eight": spread_cases >= 8,
        "order_gap_on_at_least_five": order_cases >= 5,
        "nonmyopic_gap_on_at_least_four": gap_cases >= 4,
        "mean_nonmyopic_gap_at_least_0_05": mean_gap >= 0.05,
        "mean_two_step_gain_at_least_0_10": mean_two_step_gain >= 0.10,
        "mean_duplicate_gap_at_most_0_05": mean_duplicate_gap <= 0.05,
        "max_duplicate_gap_at_most_0_15": max_duplicate_gap <= 0.15,
        "mean_duplicate_jaccard_at_least_0_75": mean_duplicate_jaccard >= 0.75,
    }
    gates["all_pass"] = all(gates.values())
    return {
        **summary,
        "gates": gates,
        "case_diagnostics": [
            {"source_id": record["source_id"], **diagnostic}
            for record, diagnostic in zip(records, diagnostics, strict=True)
        ],
    }


def _generate_initial(
    model: Any,
    cases: Sequence[ClinDiagCase],
    config: Config,
    retries: int,
) -> list[list[str]]:
    return complete_parsed_many(
        model,
        [initial_differential_messages(case) for case in cases],
        temperature=float(config.generation_temperature_simple),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=_parse_support,
        stage="initial_support",
        retries=retries,
    )


def _generate_one_step(
    model: Any,
    cases: Sequence[ClinDiagCase],
    initial: Sequence[Sequence[str]],
    config: Config,
    retries: int,
) -> dict[str, list[list[str]]]:
    messages = [
        refresh_differential_messages(case, (action,), initial_support)
        for case, initial_support in zip(cases, initial, strict=True)
        for action in ACTIONS
    ]
    flat = complete_parsed_many(
        model,
        messages,
        temperature=float(config.generation_temperature_simple),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=_parse_support,
        stage="one_step_support",
        retries=retries,
    )
    result = {action: [] for action in ACTIONS}
    for index in range(len(cases)):
        for action_index, action in enumerate(ACTIONS):
            result[action].append(flat[index * len(ACTIONS) + action_index])
    return result


def _generate_sequences(
    model: Any,
    cases: Sequence[ClinDiagCase],
    one_step: dict[str, list[list[str]]],
    config: Config,
    retries: int,
) -> dict[str, list[list[str]]]:
    messages = [
        refresh_differential_messages(
            case,
            (first, second),
            one_step[first][case_index],
        )
        for case_index, case in enumerate(cases)
        for first, second in SEQUENCES
    ]
    flat = complete_parsed_many(
        model,
        messages,
        temperature=float(config.generation_temperature_simple),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=_parse_support,
        stage="two_step_support",
        retries=retries,
    )
    result = {f"{first}>{second}": [] for first, second in SEQUENCES}
    for case_index in range(len(cases)):
        for sequence_index, (first, second) in enumerate(SEQUENCES):
            result[f"{first}>{second}"].append(
                flat[case_index * len(SEQUENCES) + sequence_index]
            )
    return result


def _duplicate_schedule(num_cases: int) -> list[tuple[str, str]]:
    return [SEQUENCES[index % len(SEQUENCES)] for index in range(num_cases)]


def run_opportunity(
    config: Config,
    data_zip: Path,
    generator_model: str,
    judge_model: str,
) -> dict[str, Any]:
    generator, judge = _build_models(config, generator_model, judge_model)
    cases = load_selected_cases(data_zip, OPPORTUNITY_IDS)
    for case in cases:
        for action in ACTIONS:
            action_evidence(case, action)
    retries = int(config.mediq_structured_max_retries)
    initial = _generate_initial(generator, cases, config, retries)
    one_step = _generate_one_step(generator, cases, initial, config, retries)
    sequences = _generate_sequences(generator, cases, one_step, config, retries)
    duplicate_schedule = _duplicate_schedule(len(cases))
    duplicate_messages = [
        refresh_differential_messages(
            case,
            (first, second),
            one_step[first][index],
        )
        for index, (case, (first, second)) in enumerate(
            zip(cases, duplicate_schedule, strict=True)
        )
    ]
    duplicates = complete_parsed_many(
        generator,
        duplicate_messages,
        temperature=float(config.generation_temperature_simple),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=_parse_support,
        stage="identity_duplicate_support",
        retries=retries,
    )
    support_ids = (
        "initial",
        *(f"one:{action}" for action in ACTIONS),
        *(f"seq:{first}>{second}" for first, second in SEQUENCES),
        "identity_duplicate",
    )
    semantic = complete_parsed_many(
        judge,
        [
            semantic_diagnosis_messages(
                case.final_diagnosis,
                (
                    ("initial", initial[index]),
                    *(
                        (f"one:{action}", one_step[action][index])
                        for action in ACTIONS
                    ),
                    *(
                        (
                            f"seq:{first}>{second}",
                            sequences[f"{first}>{second}"][index],
                        )
                        for first, second in SEQUENCES
                    ),
                    ("identity_duplicate", duplicates[index]),
                ),
            )
            for index, case in enumerate(cases)
        ],
        temperature=0.0,
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=lambda text: parse_semantic_diagnosis(text, support_ids),
        stage="semantic_measurement",
        retries=retries,
    )
    records = []
    for index, (case, measurements) in enumerate(
        zip(cases, semantic, strict=True)
    ):
        score_by_id = {
            measurement["id"]: measurement["best_match_score"]
            for measurement in measurements
        }
        first, second = duplicate_schedule[index]
        duplicate_sequence = f"{first}>{second}"
        records.append(
            {
                "source_id": case.source_id,
                "subset": case.subset,
                "initial_information": case.initial_information,
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "initial_support": initial[index],
                "initial_score": score_by_id["initial"],
                "one_step_supports": {
                    action: one_step[action][index] for action in ACTIONS
                },
                "one_step_scores": {
                    action: score_by_id[f"one:{action}"] for action in ACTIONS
                },
                "sequence_supports": {
                    f"{first}>{second}": sequences[f"{first}>{second}"][index]
                    for first, second in SEQUENCES
                },
                "sequence_scores": {
                    f"{first}>{second}": score_by_id[f"seq:{first}>{second}"]
                    for first, second in SEQUENCES
                },
                "duplicate_sequence": duplicate_sequence,
                "duplicate_support": duplicates[index],
                "duplicate_score": score_by_id["identity_duplicate"],
            }
        )
    summary = summarize(records)
    return {
        "schema_version": 1,
        "status": (
            "passed" if summary["gates"]["all_pass"] else "opportunity_gate_failed"
        ),
        "protocol": {
            "dataset": "ClinDiag-Benchmark",
            "source_commit": CLINDIAG_SOURCE_COMMIT,
            "archive_sha256": CLINDIAG_ZIP_SHA256,
            "selection_seed": SELECTION_SEED,
            "opportunity_ids": list(OPPORTUNITY_IDS),
            "sealed_generator_holdout_ids": list(HOLDOUT_IDS),
            "actions": list(ACTIONS),
            "ordered_sequences": [
                f"{first}>{second}" for first, second in SEQUENCES
            ],
            "diagnosis_count": DIAGNOSIS_COUNT,
            "coverage_threshold": COVERAGE_THRESHOLD,
            "temperature": float(config.generation_temperature_simple),
            "generator_model": generator_model,
            "judge_model": judge_model,
            "truth_used_for_measurement_only": True,
            "reasoning_disabled": True,
            "expected_physical_requests": EXPECTED_FORMAL_REQUESTS,
        },
        "summary": summary,
        "records": records,
        "usage": {
            "generator": generator.usage_snapshot(),
            "judge": judge.usage_snapshot(),
        },
    }


def run_serving_smoke(
    config: Config,
    data_zip: Path,
    generator_model: str,
    judge_model: str,
) -> dict[str, Any]:
    generator, judge = _build_models(config, generator_model, judge_model)
    cases = load_selected_cases(data_zip, SMOKE_IDS)
    initial = _generate_initial(generator, cases, config, retries=0)
    first_actions = ("history", "laboratory_tests")
    second_actions = ("imaging", "other_tests")
    one_step = complete_parsed_many(
        generator,
        [
            refresh_differential_messages(case, (action,), initial[index])
            for index, (case, action) in enumerate(
                zip(cases, first_actions, strict=True)
            )
        ],
        temperature=float(config.generation_temperature_simple),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=_parse_support,
        stage="smoke_one_step",
        retries=0,
    )
    second_messages = [
        refresh_differential_messages(
            case,
            (first_actions[index], second_actions[index]),
            one_step[index],
        )
        for index, case in enumerate(cases)
    ]
    two_step = complete_parsed_many(
        generator,
        second_messages,
        temperature=float(config.generation_temperature_simple),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=_parse_support,
        stage="smoke_two_step",
        retries=0,
    )
    duplicates = complete_parsed_many(
        generator,
        second_messages,
        temperature=float(config.generation_temperature_simple),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=_parse_support,
        stage="smoke_identity_duplicate",
        retries=0,
    )
    support_ids = ("initial", "one_step", "two_step", "identity_duplicate")
    semantic = complete_parsed_many(
        judge,
        [
            semantic_diagnosis_messages(
                case.final_diagnosis,
                (
                    ("initial", initial[index]),
                    ("one_step", one_step[index]),
                    ("two_step", two_step[index]),
                    ("identity_duplicate", duplicates[index]),
                ),
            )
            for index, case in enumerate(cases)
        ],
        temperature=0.0,
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=lambda text: parse_semantic_diagnosis(text, support_ids),
        stage="smoke_semantic",
        retries=0,
    )
    usage = {
        "generator": generator.usage_snapshot(),
        "judge": judge.usage_snapshot(),
    }
    requests = sum(int(item["adapter_requests"]) for item in usage.values())
    reasoning = sum(int(item["adapter_reasoning_tokens"]) for item in usage.values())
    all_supports = [*initial, *one_step, *two_step, *duplicates]
    return {
        "schema_version": 1,
        "status": (
            "passed"
            if requests == 10
            and reasoning == 0
            and all(len(values) == DIAGNOSIS_COUNT for values in all_supports)
            else "failed"
        ),
        "physical_requests": requests,
        "expected_physical_requests": 10,
        "reasoning_tokens": reasoning,
        "support_sizes": [len(values) for values in all_supports],
        "records": [
            {
                "source_id": case.source_id,
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "sequence": f"{first_actions[index]}>{second_actions[index]}",
                "initial_support": initial[index],
                "one_step_support": one_step[index],
                "two_step_support": two_step[index],
                "identity_duplicate_support": duplicates[index],
                "measurements": semantic[index],
                "duplicate_support_jaccard": _support_jaccard(
                    two_step[index], duplicates[index]
                ),
            }
            for index, case in enumerate(cases)
        ],
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "opportunity"),
        required=True,
    )
    parser.add_argument("--generator-model", default="openai/gpt-5.4")
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            run_serving_smoke(
                config,
                args.data_zip,
                args.generator_model,
                args.judge_model,
            )
            if args.stage == "serving_smoke"
            else run_opportunity(
                config,
                args.data_zip,
                args.generator_model,
                args.judge_model,
            )
        )
        filename = (
            "SERVING_SMOKE.json"
            if args.stage == "serving_smoke"
            else "OPPORTUNITY.json"
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
            else "OPPORTUNITY_FAILURE.json"
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
        else {
            "status": payload["status"],
            "physical_requests": payload["physical_requests"],
            "reasoning_tokens": payload["reasoning_tokens"],
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
