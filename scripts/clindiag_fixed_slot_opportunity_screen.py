#!/usr/bin/env python3
"""Screen fixed-slot ClinDiag cases for two-step truth-recovery headroom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.clindiag_fixed_slot_audit import ACTION_IDS, fixed_evidence_slots
from scripts.clindiag_fixed_slot_support_gate import (
    refresh_differential_messages,
)
from scripts.clindiag_mc_joint_model_gate import _build_role
from scripts.clindiag_staged_generator_gate import (
    DIAGNOSIS_COUNT,
    ClinDiagCase,
    _normalized_text,
    initial_differential_messages,
    load_selected_cases,
)
from scripts.icraft_staged_diagnosis_unlock import (
    complete_parsed_many,
    parse_diagnoses,
    parse_semantic_diagnosis,
    semantic_diagnosis_messages,
)


SELECTION_SEED = 24295
SCREEN_IDS = ("20220188", "11222813", "rare140", "rare122")
SUPPORT_IDS = ("initial", *(f"one_step__{action_id}" for action_id in ACTION_IDS))
EXPECTED_REQUESTS = len(SCREEN_IDS) * (len(ACTION_IDS) + 2)
TRUTH_COVERAGE_THRESHOLD = 0.80


def source_evidence_contains_target(
    case: ClinDiagCase,
    slots: dict[str, Any],
) -> bool:
    target = _normalized_text(case.final_diagnosis)
    visible_source = (
        _normalized_text(case.initial_information),
        *(_normalized_text(slots[action_id]) for action_id in ACTION_IDS),
    )
    return bool(target) and any(target in value for value in visible_source)


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
    all_supports_size_twelve: bool,
    source_target_leaks: int,
) -> dict[str, Any]:
    room_ids = [
        record["source_id"] for record in records if record["two_step_room"]
    ]
    gates = {
        "exactly_40_requests": usage["physical_requests"] == EXPECTED_REQUESTS,
        "zero_reasoning": usage["reasoning_tokens"] == 0,
        "all_supports_size_twelve": all_supports_size_twelve,
        "no_full_target_in_source_evidence": source_target_leaks == 0,
        "at_least_two_cases_have_two_step_room": len(room_ids) >= 2,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "physical_requests": usage["physical_requests"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "source_target_leaks": source_target_leaks,
        "two_step_room_case_ids": room_ids,
        "num_cases_with_two_step_room": len(room_ids),
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
        "screen_ids": list(SCREEN_IDS),
        "action_ids": list(ACTION_IDS),
        "generator_model": generator_model,
        "judge_model": judge_model,
        "expected_requests": EXPECTED_REQUESTS,
        "support_size": DIAGNOSIS_COUNT,
        "truth_coverage_threshold": TRUTH_COVERAGE_THRESHOLD,
        "reasoning_disabled": True,
        "retries": 0,
        "truth_hidden_from_generators": True,
        "truth_enters_only_semantic_measurement": True,
    }


def run_one_step_screen(
    config: Config,
    data_zip: Path,
    *,
    generator_model: str,
    judge_model: str,
) -> dict[str, Any]:
    models = {
        "initial": _build_role(config, generator_model),
        "one_step": _build_role(config, generator_model),
        "judge": _build_role(config, judge_model),
    }
    cases = load_selected_cases(data_zip, SCREEN_IDS)
    slots = [fixed_evidence_slots(case) for case in cases]
    initial = _complete_supports(
        models["initial"],
        [initial_differential_messages(case) for case in cases],
        config,
        "initial_support",
    )

    one_step_messages = []
    one_step_keys = []
    for case_index, (case, case_slots, initial_support) in enumerate(
        zip(cases, slots, initial, strict=True)
    ):
        for action_id in ACTION_IDS:
            one_step_keys.append((case_index, action_id))
            one_step_messages.append(
                refresh_differential_messages(
                    case,
                    [(action_id, case_slots[action_id])],
                    initial_support,
                )
            )
    one_step_values = _complete_supports(
        models["one_step"],
        one_step_messages,
        config,
        "one_step_supports",
    )
    one_step_by_case: list[dict[str, list[str]]] = [
        {} for _ in SCREEN_IDS
    ]
    for (case_index, action_id), support in zip(
        one_step_keys, one_step_values, strict=True
    ):
        one_step_by_case[case_index][action_id] = support

    support_sets = []
    for initial_support, case_one_step in zip(
        initial, one_step_by_case, strict=True
    ):
        support_sets.append(
            [
                ("initial", initial_support),
                *[
                    (f"one_step__{action_id}", case_one_step[action_id])
                    for action_id in ACTION_IDS
                ],
            ]
        )
    audit_messages = [
        semantic_diagnosis_messages(case.final_diagnosis, case_supports)
        for case, case_supports in zip(cases, support_sets, strict=True)
    ]
    raw_audits = models["judge"].chat_complete_messages_batched(
        audit_messages,
        temperature=0.0,
        block_size=256,
        max_new_tokens=int(config.openrouter_max_output_tokens),
    )
    audits: list[list[dict[str, Any]] | None] = []
    audit_errors = []
    for index, response in enumerate(raw_audits):
        try:
            audits.append(parse_semantic_diagnosis(response, SUPPORT_IDS))
        except ValueError as exc:
            audits.append(None)
            audit_errors.append(
                {
                    "row": index,
                    "error": str(exc),
                    "raw_response": response,
                }
            )

    usage = _usage(models)
    if audit_errors:
        return {
            "schema_version": 1,
            "status": "runtime_failure",
            "error_type": "StructuredAuditError",
            "error": audit_errors[0]["error"],
            "protocol": _protocol(generator_model, judge_model),
            "summary": {
                "physical_requests": usage["physical_requests"],
                "reasoning_tokens": usage["reasoning_tokens"],
                "structured_audit_errors": len(audit_errors),
            },
            "partial_records": [
                {
                    "source_id": case.source_id,
                    "subset": case.subset,
                    "supports": {
                        support_id: support
                        for support_id, support in case_supports
                    },
                }
                for case, case_supports in zip(cases, support_sets, strict=True)
            ],
            "raw_audit_responses": raw_audits,
            "audit_errors": audit_errors,
            "usage": usage,
        }

    records = []
    for case, case_slots, case_supports, audit in zip(
        cases, slots, support_sets, audits, strict=True
    ):
        assert audit is not None
        measurements = {row["id"]: row for row in audit}
        one_step_scores = {
            action_id: measurements[f"one_step__{action_id}"][
                "best_match_score"
            ]
            for action_id in ACTION_IDS
        }
        best_score = max(one_step_scores.values())
        records.append(
            {
                "source_id": case.source_id,
                "subset": case.subset,
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "stored_observations": {
                    action_id: case_slots[action_id]
                    for action_id in ACTION_IDS
                },
                "supports": {
                    support_id: support
                    for support_id, support in case_supports
                },
                "semantic_measurements": measurements,
                "initial_truth_score": measurements["initial"][
                    "best_match_score"
                ],
                "one_step_truth_scores": one_step_scores,
                "best_one_step_truth_score": best_score,
                "best_one_step_action_ids": [
                    action_id
                    for action_id in ACTION_IDS
                    if one_step_scores[action_id] == best_score
                ],
                "two_step_room": (
                    measurements["initial"]["best_match_score"]
                    < TRUTH_COVERAGE_THRESHOLD
                    and best_score < TRUTH_COVERAGE_THRESHOLD
                ),
            }
        )

    summary = summarize(
        records,
        usage,
        all_supports_size_twelve=all(
            len(support) == DIAGNOSIS_COUNT
            for case_supports in support_sets
            for _, support in case_supports
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
        "raw_audit_responses": raw_audits,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generator-model", default="openai/gpt-5.4")
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "ONE_STEP_SCREEN.json"
    try:
        payload = run_one_step_screen(
            config,
            args.data_zip,
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
