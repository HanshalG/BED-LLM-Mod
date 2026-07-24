#!/usr/bin/env python3
"""Measure replicated two-step truth recovery on fixed-slot ClinDiag cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mediq.parsing import parse_json_object
from helpers import Config, load_config
from scripts.clindiag_fixed_slot_audit import ACTION_IDS, fixed_evidence_slots
from scripts.clindiag_fixed_slot_opportunity_screen import (
    PREVALENCE_IDS,
    source_evidence_contains_target,
)
from scripts.clindiag_fixed_slot_support_gate import (
    refresh_differential_messages,
)
from scripts.clindiag_mc_joint_model_gate import _build_role
from scripts.clindiag_staged_generator_gate import (
    DIAGNOSIS_COUNT,
    load_selected_cases,
)
from scripts.icraft_staged_diagnosis_unlock import (
    complete_parsed_many,
    parse_diagnoses,
    parse_semantic_diagnosis,
    semantic_diagnosis_messages,
)


SELECTION_SEED = 24296
PAIR_SCREEN_IDS = (
    "31597024",
    "24283228",
    "rare287",
    "rare79",
    "rare66",
    "rare207",
    "rare243",
)
PREVALENCE_SHA256 = (
    "a66bbef5f9c8619785e32cb12ad711be61ab97627ac42a52e20072fbd5375b74"
)
PAIR_MEASUREMENT_CHUNK_SIZE = 14
PAIR_IDS = tuple(
    f"pair__{first}__then__{second}"
    for first in ACTION_IDS
    for second in ACTION_IDS
    if first != second
)
PAIR_GENERATION_REQUESTS = len(PAIR_SCREEN_IDS) * len(PAIR_IDS)
PAIR_MEASUREMENT_REQUESTS = (
    len(PAIR_IDS) // PAIR_MEASUREMENT_CHUNK_SIZE
) * len(PAIR_SCREEN_IDS)
EXPECTED_REQUESTS = (
    PAIR_GENERATION_REQUESTS
    + PAIR_MEASUREMENT_REQUESTS
    + len(PAIR_SCREEN_IDS)
    + len(PAIR_SCREEN_IDS)
)
TRUTH_COVERAGE_THRESHOLD = 0.80
MIN_VALIDATED_GAIN = 0.30
MIN_DUPLICATE_OVERLAP = 0.80
MAX_DUPLICATE_SCORE_GAP = 0.05
MIN_VALIDATED_UNLOCKS = 3
MIN_MEAN_VALIDATED_GAIN = 0.20


def pair_support_id(first: str, second: str) -> str:
    return f"pair__{first}__then__{second}"


def parse_pair_support_id(value: str) -> tuple[str, str]:
    prefix = "pair__"
    marker = "__then__"
    if not value.startswith(prefix) or marker not in value:
        raise ValueError("invalid pair support ID")
    first, second = value[len(prefix) :].split(marker, 1)
    if first not in ACTION_IDS or second not in ACTION_IDS or first == second:
        raise ValueError("invalid ordered action pair")
    return first, second


def load_prevalence_records(path: Path) -> list[dict[str, Any]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PREVALENCE_SHA256:
        raise ValueError(
            f"prevalence artifact hash mismatch: expected {PREVALENCE_SHA256}, "
            f"got {digest}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError("prevalence artifact did not pass")
    if payload.get("protocol", {}).get("screen_ids") != list(PREVALENCE_IDS):
        raise ValueError("prevalence case split changed")
    if payload.get("summary", {}).get("two_step_room_case_ids") != list(
        PAIR_SCREEN_IDS
    ):
        raise ValueError("qualifying pair-screen IDs changed")
    by_id = {record["source_id"]: record for record in payload["records"]}
    records = [by_id[source_id] for source_id in PAIR_SCREEN_IDS]
    if any(not record.get("two_step_room") for record in records):
        raise ValueError("pair screen includes an ineligible case")
    return records


def oracle_validation_messages(
    true_diagnosis: str,
    original_support: Sequence[str],
    duplicate_support: Sequence[str],
) -> list[dict[str, str]]:
    payload = {
        "true_diagnosis": true_diagnosis,
        "supports": [
            {"id": "oracle_original", "diagnoses": list(original_support)},
            {"id": "oracle_duplicate", "diagnoses": list(duplicate_support)},
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict diagnosis-equivalence and set-overlap evaluator. "
                "The true diagnosis was hidden from both differential generators. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each support, score its closest diagnosis to the truth in [0,1]. "
                "Use at least 0.8 only for the same diagnosis or a standard clinical "
                "synonym. Then report the fraction of diagnoses in each direction "
                "having a same-disease or standard-synonym match in the other list; "
                "related diseases and broad parents do not count. Return exactly "
                '{"supports":['
                '{"id":"oracle_original","best_match_score":0.0,'
                '"reason":"brief"},'
                '{"id":"oracle_duplicate","best_match_score":0.0,'
                '"reason":"brief"}],'
                '"original_to_duplicate_overlap":0.0,'
                '"duplicate_to_original_overlap":0.0,'
                '"overlap_reason":"brief"}; preserve IDs and order.\n'
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_oracle_validation(text: str) -> dict[str, Any]:
    raw = parse_json_object(text)
    support_ids = ("oracle_original", "oracle_duplicate")
    rows = raw.get("supports")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("validation must contain two support rows")
    parsed = []
    for support_id, row in zip(support_ids, rows, strict=True):
        if not isinstance(row, dict) or row.get("id") != support_id:
            raise ValueError("validation changed support IDs or order")
        score = row.get("best_match_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("best_match_score must be numeric")
        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("best_match_score must be in [0,1]")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("each validation score requires a reason")
        parsed.append(
            {
                "id": support_id,
                "best_match_score": score,
                "reason": reason.strip(),
            }
        )
    overlaps = []
    for field in (
        "original_to_duplicate_overlap",
        "duplicate_to_original_overlap",
    ):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in [0,1]")
        overlaps.append(value)
    reason = raw.get("overlap_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("overlap_reason must be nonempty")
    return {
        "supports": parsed,
        "original_to_duplicate_overlap": overlaps[0],
        "duplicate_to_original_overlap": overlaps[1],
        "duplicate_semantic_overlap": min(overlaps),
        "overlap_reason": reason.strip(),
    }


def _usage(models: dict[str, Any]) -> dict[str, Any]:
    snapshots = {name: model.usage_snapshot() for name, model in models.items()}
    return {
        "snapshots": snapshots,
        "physical_requests": sum(
            int(item["adapter_requests"]) for item in snapshots.values()
        ),
        "reasoning_tokens": sum(
            int(item["adapter_reasoning_tokens"]) for item in snapshots.values()
        ),
    }


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    all_pair_supports_size_twelve: bool,
    duplicate_prompts_exact: bool,
    source_target_leaks: int,
) -> dict[str, Any]:
    unlock_ids = [
        record["source_id"] for record in records if record["validated_unlock"]
    ]
    gains = [float(record["validated_gain"]) for record in records]
    mean_gain = sum(gains) / len(gains) if gains else 0.0
    gates = {
        "exactly_434_requests": usage["physical_requests"] == EXPECTED_REQUESTS,
        "zero_reasoning": usage["reasoning_tokens"] == 0,
        "all_pair_and_duplicate_supports_size_twelve": (
            all_pair_supports_size_twelve
        ),
        "duplicate_prompts_exact": duplicate_prompts_exact,
        "no_full_target_in_source_evidence": source_target_leaks == 0,
        "at_least_three_validated_unlocks": (
            len(unlock_ids) >= MIN_VALIDATED_UNLOCKS
        ),
        "mean_validated_gain_at_least_0_20": (
            mean_gain >= MIN_MEAN_VALIDATED_GAIN
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "physical_requests": usage["physical_requests"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "validated_unlock_case_ids": unlock_ids,
        "num_validated_unlocks": len(unlock_ids),
        "validated_gains": gains,
        "mean_validated_gain": mean_gain,
        "source_target_leaks": source_target_leaks,
        "gates": gates,
    }


def _complete_supports(
    model: Any,
    messages: Sequence[list[dict[str, str]]],
    config: Config,
    stage: str,
) -> list[list[str]]:
    return complete_parsed_many(
        model,
        messages,
        temperature=float(config.generation_temperature_diverse),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT),
        stage=stage,
        retries=0,
    )


def _protocol(generator_model: str, judge_model: str) -> dict[str, Any]:
    return {
        "selection_seed": SELECTION_SEED,
        "pair_screen_ids": list(PAIR_SCREEN_IDS),
        "prevalence_artifact_sha256": PREVALENCE_SHA256,
        "action_ids": list(ACTION_IDS),
        "ordered_pairs_per_case": len(PAIR_IDS),
        "pair_measurement_chunk_size": PAIR_MEASUREMENT_CHUNK_SIZE,
        "generator_model": generator_model,
        "judge_model": judge_model,
        "expected_requests": EXPECTED_REQUESTS,
        "support_size": DIAGNOSIS_COUNT,
        "truth_coverage_threshold": TRUTH_COVERAGE_THRESHOLD,
        "minimum_validated_gain": MIN_VALIDATED_GAIN,
        "minimum_duplicate_overlap": MIN_DUPLICATE_OVERLAP,
        "maximum_duplicate_score_gap": MAX_DUPLICATE_SCORE_GAP,
        "minimum_validated_unlocks": MIN_VALIDATED_UNLOCKS,
        "minimum_mean_validated_gain": MIN_MEAN_VALIDATED_GAIN,
        "reasoning_disabled": True,
        "retries": 0,
        "truth_hidden_from_generators": True,
        "truth_enters_only_post_generation_measurement": True,
    }


def run_pair_screen(
    config: Config,
    data_zip: Path,
    prevalence_path: Path,
    *,
    generator_model: str,
    judge_model: str,
) -> dict[str, Any]:
    prevalence_records = load_prevalence_records(prevalence_path)
    models = {
        "pair": _build_role(config, generator_model),
        "pair_judge": _build_role(config, judge_model),
        "duplicate": _build_role(config, generator_model),
        "validation_judge": _build_role(config, judge_model),
    }
    cases = load_selected_cases(data_zip, PAIR_SCREEN_IDS)
    slots = [fixed_evidence_slots(case) for case in cases]

    pair_messages = []
    pair_keys = []
    for case_index, (case, case_slots, prevalence_record) in enumerate(
        zip(cases, slots, prevalence_records, strict=True)
    ):
        for pair_id in PAIR_IDS:
            first, second = parse_pair_support_id(pair_id)
            pair_keys.append((case_index, pair_id))
            pair_messages.append(
                refresh_differential_messages(
                    case,
                    [
                        (first, case_slots[first]),
                        (second, case_slots[second]),
                    ],
                    prevalence_record["supports"][f"one_step__{first}"],
                )
            )
    pair_values = _complete_supports(
        models["pair"],
        pair_messages,
        config,
        "ordered_pair_supports",
    )
    pair_by_case: list[dict[str, list[str]]] = [
        {} for _ in PAIR_SCREEN_IDS
    ]
    for (case_index, pair_id), support in zip(
        pair_keys, pair_values, strict=True
    ):
        pair_by_case[case_index][pair_id] = support

    measurement_messages = []
    measurement_keys = []
    for case_index, (case, case_pairs) in enumerate(
        zip(cases, pair_by_case, strict=True)
    ):
        for start in range(0, len(PAIR_IDS), PAIR_MEASUREMENT_CHUNK_SIZE):
            chunk_ids = PAIR_IDS[start : start + PAIR_MEASUREMENT_CHUNK_SIZE]
            measurement_keys.append((case_index, chunk_ids))
            measurement_messages.append(
                semantic_diagnosis_messages(
                    case.final_diagnosis,
                    [(pair_id, case_pairs[pair_id]) for pair_id in chunk_ids],
                )
            )
    raw_measurements = models["pair_judge"].chat_complete_messages_batched(
        measurement_messages,
        temperature=0.0,
        block_size=256,
        max_new_tokens=int(config.openrouter_max_output_tokens),
    )
    pair_scores: list[dict[str, dict[str, Any]]] = [
        {} for _ in PAIR_SCREEN_IDS
    ]
    measurement_errors = []
    for index, (key, response) in enumerate(
        zip(measurement_keys, raw_measurements, strict=True)
    ):
        case_index, chunk_ids = key
        try:
            rows = parse_semantic_diagnosis(response, chunk_ids)
        except ValueError as exc:
            measurement_errors.append(
                {
                    "row": index,
                    "case_index": case_index,
                    "error": str(exc),
                    "raw_response": response,
                }
            )
            continue
        pair_scores[case_index].update({row["id"]: row for row in rows})

    usage_before_duplicates = _usage(models)
    if measurement_errors:
        return {
            "schema_version": 1,
            "status": "runtime_failure",
            "error_type": "StructuredPairMeasurementError",
            "error": measurement_errors[0]["error"],
            "protocol": _protocol(generator_model, judge_model),
            "summary": {
                "physical_requests": usage_before_duplicates[
                    "physical_requests"
                ],
                "reasoning_tokens": usage_before_duplicates["reasoning_tokens"],
                "structured_measurement_errors": len(measurement_errors),
            },
            "raw_pair_measurement_responses": raw_measurements,
            "measurement_errors": measurement_errors,
            "usage": usage_before_duplicates,
        }

    oracle_pair_ids = []
    duplicate_messages = []
    original_messages_by_case = []
    for case, case_slots, prevalence_record, case_pairs, case_scores in zip(
        cases,
        slots,
        prevalence_records,
        pair_by_case,
        pair_scores,
        strict=True,
    ):
        best_score = max(
            case_scores[pair_id]["best_match_score"] for pair_id in PAIR_IDS
        )
        oracle_pair_id = next(
            pair_id
            for pair_id in PAIR_IDS
            if case_scores[pair_id]["best_match_score"] == best_score
        )
        oracle_pair_ids.append(oracle_pair_id)
        first, second = parse_pair_support_id(oracle_pair_id)
        message = refresh_differential_messages(
            case,
            [(first, case_slots[first]), (second, case_slots[second])],
            prevalence_record["supports"][f"one_step__{first}"],
        )
        duplicate_messages.append(message)
        original_index = pair_keys.index(
            (PAIR_SCREEN_IDS.index(case.source_id), oracle_pair_id)
        )
        original_messages_by_case.append(pair_messages[original_index])
    duplicate_values = _complete_supports(
        models["duplicate"],
        duplicate_messages,
        config,
        "oracle_pair_duplicates",
    )

    validation_messages = [
        oracle_validation_messages(
            case.final_diagnosis,
            case_pairs[oracle_pair_id],
            duplicate_support,
        )
        for case, case_pairs, oracle_pair_id, duplicate_support in zip(
            cases,
            pair_by_case,
            oracle_pair_ids,
            duplicate_values,
            strict=True,
        )
    ]
    raw_validations = models[
        "validation_judge"
    ].chat_complete_messages_batched(
        validation_messages,
        temperature=0.0,
        block_size=256,
        max_new_tokens=int(config.openrouter_max_output_tokens),
    )
    validations: list[dict[str, Any] | None] = []
    validation_errors = []
    for index, response in enumerate(raw_validations):
        try:
            validations.append(parse_oracle_validation(response))
        except ValueError as exc:
            validations.append(None)
            validation_errors.append(
                {
                    "row": index,
                    "error": str(exc),
                    "raw_response": response,
                }
            )
    usage = _usage(models)
    if validation_errors:
        return {
            "schema_version": 1,
            "status": "runtime_failure",
            "error_type": "StructuredValidationError",
            "error": validation_errors[0]["error"],
            "protocol": _protocol(generator_model, judge_model),
            "summary": {
                "physical_requests": usage["physical_requests"],
                "reasoning_tokens": usage["reasoning_tokens"],
                "structured_validation_errors": len(validation_errors),
            },
            "raw_pair_measurement_responses": raw_measurements,
            "raw_validation_responses": raw_validations,
            "validation_errors": validation_errors,
            "usage": usage,
        }

    records = []
    for (
        case,
        prevalence_record,
        case_pairs,
        case_scores,
        oracle_pair_id,
        duplicate_support,
        validation,
    ) in zip(
        cases,
        prevalence_records,
        pair_by_case,
        pair_scores,
        oracle_pair_ids,
        duplicate_values,
        validations,
        strict=True,
    ):
        assert validation is not None
        validation_scores = {
            row["id"]: row["best_match_score"]
            for row in validation["supports"]
        }
        validated_score = min(validation_scores.values())
        best_one_step = float(prevalence_record["best_one_step_truth_score"])
        validated_gain = validated_score - best_one_step
        score_gap = abs(
            validation_scores["oracle_original"]
            - validation_scores["oracle_duplicate"]
        )
        validated_unlock = (
            validated_score >= TRUTH_COVERAGE_THRESHOLD
            and validated_gain >= MIN_VALIDATED_GAIN
            and validation["duplicate_semantic_overlap"]
            >= MIN_DUPLICATE_OVERLAP
            and score_gap <= MAX_DUPLICATE_SCORE_GAP
        )
        records.append(
            {
                "source_id": case.source_id,
                "subset": case.subset,
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "best_one_step_truth_score": best_one_step,
                "pair_supports": case_pairs,
                "pair_semantic_measurements": case_scores,
                "oracle_pair_id": oracle_pair_id,
                "oracle_pair_batch_score": case_scores[oracle_pair_id][
                    "best_match_score"
                ],
                "oracle_duplicate_support": duplicate_support,
                "oracle_validation": validation,
                "oracle_validation_score_gap": score_gap,
                "validated_truth_score": validated_score,
                "validated_gain": validated_gain,
                "validated_unlock": validated_unlock,
            }
        )

    summary = summarize(
        records,
        usage,
        all_pair_supports_size_twelve=(
            all(
                len(support) == DIAGNOSIS_COUNT
                for case_pairs in pair_by_case
                for support in case_pairs.values()
            )
            and all(
                len(support) == DIAGNOSIS_COUNT
                for support in duplicate_values
            )
        ),
        duplicate_prompts_exact=(
            original_messages_by_case == duplicate_messages
        ),
        source_target_leaks=sum(
            source_evidence_contains_target(case, case_slots)
            for case, case_slots in zip(cases, slots, strict=True)
        ),
    )
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "failed",
        "protocol": _protocol(generator_model, judge_model),
        "summary": summary,
        "records": records,
        "raw_pair_measurement_responses": raw_measurements,
        "raw_validation_responses": raw_validations,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--prevalence-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generator-model", default="openai/gpt-5.4")
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "PAIR_SCREEN.json"
    try:
        payload = run_pair_screen(
            config,
            args.data_zip,
            args.prevalence_artifact,
            generator_model=args.generator_model,
            judge_model=args.judge_model,
        )
    except Exception as exc:
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "runtime_failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
