#!/usr/bin/env python3
"""Gate three-round BED-LLM filtered-retention recovery on fresh ClinDiag cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.clindiag_filtered_retention_support_gate import (
    FILTER_THRESHOLD,
    StructuredStageError,
    _complete_stage,
    _usage,
    compatibility_messages,
    generated_candidate_messages,
    merge_support,
    parse_compatibility,
    retain_by_threshold,
)
from scripts.clindiag_fixed_slot_audit import fixed_evidence_slots
from scripts.clindiag_mc_joint_model_gate import _build_role
from scripts.clindiag_staged_generator_gate import (
    DIAGNOSIS_COUNT,
    ClinDiagCase,
    _normalized_text,
    initial_differential_messages,
    load_selected_cases,
)
from scripts.icraft_staged_diagnosis_unlock import (
    parse_diagnoses,
    parse_semantic_diagnosis,
    semantic_diagnosis_messages,
)


SELECTION_SEED = 24300
SMOKE_IDS = ("27223150", "rare130")
ACTION_ID = "lab_2"
NUM_CANDIDATE_ROUNDS = 3
EXPECTED_REQUESTS = 30
SUPPORT_IDS = ("initial", "filtered_retention", "filtered_retention_duplicate")


def source_evidence_contains_target(
    case: ClinDiagCase,
    stored_observation: Any,
) -> bool:
    target = _normalized_text(case.final_diagnosis)
    visible = (
        _normalized_text(case.initial_information),
        _normalized_text(stored_observation),
    )
    return bool(target) and any(target in value for value in visible)


def _group_by_case(
    values: Sequence[Any],
    *,
    num_cases: int,
    rounds: int = NUM_CANDIDATE_ROUNDS,
) -> list[list[Any]]:
    if len(values) != num_cases * rounds:
        raise ValueError("candidate rows do not match cases times rounds")
    return [
        list(values[index * rounds : (index + 1) * rounds])
        for index in range(num_cases)
    ]


def _exact_overlap(left: Sequence[str], right: Sequence[str]) -> dict[str, float]:
    left_keys = {value.casefold().strip() for value in left}
    right_keys = {value.casefold().strip() for value in right}
    intersection = left_keys & right_keys
    union = left_keys | right_keys
    return {
        "intersection": len(intersection),
        "left_fraction": len(intersection) / len(left_keys) if left_keys else 0.0,
        "right_fraction": len(intersection) / len(right_keys) if right_keys else 0.0,
        "jaccard": len(intersection) / len(union) if union else 1.0,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    candidate_generation_prompts_exact: bool,
) -> dict[str, Any]:
    truth_gaps = [float(row["duplicate_truth_score_gap"]) for row in records]
    substantive = [
        row["source_id"]
        for row in records
        if row["num_old_pruned"] >= 4
        and row["num_new_introduced"] >= 4
        and row["num_duplicate_new_introduced"] >= 4
    ]
    gates = {
        "exactly_30_requests": usage["physical_requests"] == EXPECTED_REQUESTS,
        "zero_reasoning": usage["reasoning_tokens"] == 0,
        "candidate_generation_prompts_exact": candidate_generation_prompts_exact,
        "all_final_supports_size_twelve": all(
            len(row["supports"][support_id]) == DIAGNOSIS_COUNT
            for row in records
            for support_id in ("filtered_retention", "filtered_retention_duplicate")
        ),
        "at_least_one_substantive_transition": len(substantive) >= 1,
        "both_duplicate_truth_score_gaps_at_most_0_05": (
            len(truth_gaps) == 2 and all(value <= 0.05 for value in truth_gaps)
        ),
        "truth_coverage_not_lost": all(
            min(row["final_truth_score"], row["duplicate_truth_score"])
            >= row["initial_truth_score"] - 0.05
            for row in records
        ),
        "no_hidden_target_in_source_evidence": all(
            not row["source_target_leak"] for row in records
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "physical_requests": usage["physical_requests"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "substantive_transition_case_ids": substantive,
        "old_pruned": [row["num_old_pruned"] for row in records],
        "new_introduced": [row["num_new_introduced"] for row in records],
        "duplicate_new_introduced": [
            row["num_duplicate_new_introduced"] for row in records
        ],
        "truth_score_traces": [
            [
                row["initial_truth_score"],
                row["final_truth_score"],
                row["duplicate_truth_score"],
            ]
            for row in records
        ],
        "duplicate_truth_score_gaps": truth_gaps,
        "exact_overlap_jaccards_descriptive": [
            row["exact_overlap"]["jaccard"] for row in records
        ],
        "gates": gates,
    }


def run_gate(
    config: Config,
    data_zip: Path,
    *,
    generator_model: str,
    filter_model: str,
    judge_model: str,
) -> dict[str, Any]:
    models = {
        "initial": _build_role(config, generator_model),
        "old_filter": _build_role(config, filter_model),
        "candidate": _build_role(config, generator_model),
        "candidate_duplicate": _build_role(config, generator_model),
        "new_filter": _build_role(config, filter_model),
        "new_filter_duplicate": _build_role(config, filter_model),
        "judge": _build_role(config, judge_model),
    }
    cases = load_selected_cases(data_zip, SMOKE_IDS)
    slots = [fixed_evidence_slots(case) for case in cases]
    raw_by_stage: dict[str, list[str]] = {}

    initial, raw_by_stage["initial"] = _complete_stage(
        models["initial"],
        [initial_differential_messages(case) for case in cases],
        config=config,
        temperature=0.0,
        parser=lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT),
        stage="initial",
    )
    old_filter_messages = [
        compatibility_messages(
            case,
            support,
            [(ACTION_ID, case_slots[ACTION_ID])],
        )
        for case, case_slots, support in zip(cases, slots, initial, strict=True)
    ]
    old_compatibility, raw_by_stage["old_filter"] = _complete_stage(
        models["old_filter"],
        old_filter_messages,
        config=config,
        temperature=0.0,
        parser=lambda text: parse_compatibility(
            text,
            num_diagnoses=DIAGNOSIS_COUNT,
            evidence_ids=(ACTION_ID,),
        ),
        stage="old_filter",
    )
    retained_old = [
        retain_by_threshold(support, rows)
        for support, rows in zip(initial, old_compatibility, strict=True)
    ]

    candidate_messages = [
        generated_candidate_messages(case, [(ACTION_ID, case_slots[ACTION_ID])])
        for case, case_slots in zip(cases, slots, strict=True)
        for _ in range(NUM_CANDIDATE_ROUNDS)
    ]
    duplicate_candidate_messages = [
        generated_candidate_messages(case, [(ACTION_ID, case_slots[ACTION_ID])])
        for case, case_slots in zip(cases, slots, strict=True)
        for _ in range(NUM_CANDIDATE_ROUNDS)
    ]
    generation_temperature = float(config.generation_temperature_diverse)
    candidates, raw_by_stage["candidate"] = _complete_stage(
        models["candidate"],
        candidate_messages,
        config=config,
        temperature=generation_temperature,
        parser=lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT),
        stage="candidate",
    )
    duplicate_candidates, raw_by_stage["candidate_duplicate"] = _complete_stage(
        models["candidate_duplicate"],
        duplicate_candidate_messages,
        config=config,
        temperature=generation_temperature,
        parser=lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT),
        stage="candidate_duplicate",
    )

    candidate_groups = _group_by_case(candidates, num_cases=len(cases))
    duplicate_candidate_groups = _group_by_case(
        duplicate_candidates, num_cases=len(cases)
    )
    new_filter_messages = []
    duplicate_filter_messages = []
    for case, case_slots, case_candidates, case_duplicate_candidates in zip(
        cases,
        slots,
        candidate_groups,
        duplicate_candidate_groups,
        strict=True,
    ):
        evidence = (
            ("initial", case.initial_information),
            (ACTION_ID, case_slots[ACTION_ID]),
        )
        new_filter_messages.extend(
            compatibility_messages(case, support, evidence)
            for support in case_candidates
        )
        duplicate_filter_messages.extend(
            compatibility_messages(case, support, evidence)
            for support in case_duplicate_candidates
        )

    new_compatibility, raw_by_stage["new_filter"] = _complete_stage(
        models["new_filter"],
        new_filter_messages,
        config=config,
        temperature=0.0,
        parser=lambda text: parse_compatibility(
            text,
            num_diagnoses=DIAGNOSIS_COUNT,
            evidence_ids=("initial", ACTION_ID),
        ),
        stage="new_filter",
    )
    duplicate_compatibility, raw_by_stage["new_filter_duplicate"] = _complete_stage(
        models["new_filter_duplicate"],
        duplicate_filter_messages,
        config=config,
        temperature=0.0,
        parser=lambda text: parse_compatibility(
            text,
            num_diagnoses=DIAGNOSIS_COUNT,
            evidence_ids=("initial", ACTION_ID),
        ),
        stage="new_filter_duplicate",
    )
    compatibility_groups = _group_by_case(
        new_compatibility, num_cases=len(cases)
    )
    duplicate_compatibility_groups = _group_by_case(
        duplicate_compatibility, num_cases=len(cases)
    )

    filtered_groups = []
    duplicate_filtered_groups = []
    final_supports = []
    duplicate_final_supports = []
    for old, case_candidates, case_rows, case_duplicate_candidates, duplicate_rows in zip(
        retained_old,
        candidate_groups,
        compatibility_groups,
        duplicate_candidate_groups,
        duplicate_compatibility_groups,
        strict=True,
    ):
        filtered = [
            retain_by_threshold(support, rows)
            for support, rows in zip(case_candidates, case_rows, strict=True)
        ]
        duplicate_filtered = [
            retain_by_threshold(support, rows)
            for support, rows in zip(
                case_duplicate_candidates, duplicate_rows, strict=True
            )
        ]
        filtered_groups.append(filtered)
        duplicate_filtered_groups.append(duplicate_filtered)
        final_supports.append(
            merge_support(old, [value for batch in filtered for value in batch])
        )
        duplicate_final_supports.append(
            merge_support(
                old,
                [value for batch in duplicate_filtered for value in batch],
            )
        )

    support_sets = [
        list(
            zip(
                SUPPORT_IDS,
                (initial_support, final_support, duplicate_support),
                strict=True,
            )
        )
        for initial_support, final_support, duplicate_support in zip(
            initial, final_supports, duplicate_final_supports, strict=True
        )
    ]
    semantic_rows, raw_by_stage["semantic_audit"] = _complete_stage(
        models["judge"],
        [
            semantic_diagnosis_messages(case.final_diagnosis, supports)
            for case, supports in zip(cases, support_sets, strict=True)
        ],
        config=config,
        temperature=0.0,
        parser=lambda text: parse_semantic_diagnosis(text, SUPPORT_IDS),
        stage="semantic_audit",
    )

    records = []
    for (
        case,
        case_slots,
        initial_support,
        old_rows,
        old,
        case_candidates,
        case_duplicate_candidates,
        case_rows,
        duplicate_rows,
        filtered,
        duplicate_filtered,
        final_support,
        duplicate_support,
        semantic,
    ) in zip(
        cases,
        slots,
        initial,
        old_compatibility,
        retained_old,
        candidate_groups,
        duplicate_candidate_groups,
        compatibility_groups,
        duplicate_compatibility_groups,
        filtered_groups,
        duplicate_filtered_groups,
        final_supports,
        duplicate_final_supports,
        semantic_rows,
        strict=True,
    ):
        scores = {row["id"]: float(row["best_match_score"]) for row in semantic}
        old_keys = {value.casefold() for value in old}
        records.append(
            {
                "source_id": case.source_id,
                "subset": case.subset,
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "action_id": ACTION_ID,
                "stored_observation": case_slots[ACTION_ID],
                "source_target_leak": source_evidence_contains_target(
                    case, case_slots[ACTION_ID]
                ),
                "supports": {
                    "initial": initial_support,
                    "retained_old": old,
                    "candidate_batches": case_candidates,
                    "filtered_candidate_batches": filtered,
                    "duplicate_candidate_batches": case_duplicate_candidates,
                    "duplicate_filtered_candidate_batches": duplicate_filtered,
                    "filtered_retention": final_support,
                    "filtered_retention_duplicate": duplicate_support,
                },
                "old_compatibility": old_rows,
                "new_compatibility_batches": case_rows,
                "duplicate_new_compatibility_batches": duplicate_rows,
                "num_old_pruned": len(initial_support) - len(old),
                "num_new_introduced": sum(
                    value.casefold() not in old_keys for value in final_support
                ),
                "num_duplicate_new_introduced": sum(
                    value.casefold() not in old_keys for value in duplicate_support
                ),
                "semantic_audit": semantic,
                "initial_truth_score": scores["initial"],
                "final_truth_score": scores["filtered_retention"],
                "duplicate_truth_score": scores[
                    "filtered_retention_duplicate"
                ],
                "duplicate_truth_score_gap": abs(
                    scores["filtered_retention"]
                    - scores["filtered_retention_duplicate"]
                ),
                "exact_overlap": _exact_overlap(
                    final_support, duplicate_support
                ),
            }
        )

    prompts_exact = candidate_messages == duplicate_candidate_messages
    usage = _usage(models)
    summary = summarize(
        records,
        usage,
        candidate_generation_prompts_exact=prompts_exact,
    )
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "failed",
        "protocol": {
            "selection_seed": SELECTION_SEED,
            "smoke_ids": list(SMOKE_IDS),
            "action_id": ACTION_ID,
            "filter_threshold": FILTER_THRESHOLD,
            "support_size": DIAGNOSIS_COUNT,
            "candidate_rounds_per_path": NUM_CANDIDATE_ROUNDS,
            "generator_model": generator_model,
            "filter_model": filter_model,
            "judge_model": judge_model,
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_disabled": True,
            "generation_temperature": generation_temperature,
            "filter_temperature": 0.0,
            "audit_temperature": 0.0,
            "retries": 0,
            "truth_hidden_from_generation_and_filtering": True,
            "truth_enters_only_semantic_audit": True,
            "candidate_generation_prompts_exact": prompts_exact,
            "set_identity_metrics_descriptive_only": True,
        },
        "summary": summary,
        "records": records,
        "raw_responses": raw_by_stage,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generator-model", default="openai/gpt-5.4")
    parser.add_argument("--filter-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "RECOVERY_GATE.json"
    try:
        payload = run_gate(
            config,
            args.data_zip,
            generator_model=args.generator_model,
            filter_model=args.filter_model,
            judge_model=args.judge_model,
        )
    except StructuredStageError as exc:
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "runtime_failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "stage": exc.stage,
                    "row": exc.row,
                    "raw_responses": exc.raw_responses,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
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
