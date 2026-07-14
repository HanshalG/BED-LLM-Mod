#!/usr/bin/env python3
"""Validate a frozen naive MediQ interaction bank before scorer calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from environments.mediq import load_mediq_tasks_with_report
from environments.mediq.data import MEDIQ_COMMIT, MEDIQ_IMEDQA_DEV_SHA256
from environments.mediq.env import (
    PATIENT_CANNOT_ANSWER,
    UNAVAILABLE_OUTCOME,
    _query_contract_error,
    _queries_semantically_equivalent,
)
from run_management import write_json


def analyze_bank(
    item_dir: Path,
    *,
    data_path: Path,
    task_offset: int,
    expected_tasks: int,
    expected_rounds: int,
    minimum_available_turns: int,
    verify_official_hash: bool = True,
) -> dict[str, Any]:
    interactions_path = item_dir / "mediq_interactions.json"
    manifest_path = item_dir / "mediq_data_manifest.json"
    records = json.loads(interactions_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    all_tasks, excluded, raw_count = load_mediq_tasks_with_report(
        data_path,
        dataset="imedqa",
        verify_official_hash=verify_official_hash,
        skip_unusable_tasks=True,
    )
    expected = all_tasks[task_offset : task_offset + expected_tasks]
    expected_ids = [task.task_id for task in expected]
    expected_source_ids = [task.source_id for task in expected]
    task_by_id = {task.task_id: task for task in expected}

    grounding_errors: list[str] = []
    action_errors: list[str] = []
    duplicate_errors: list[str] = []
    validation_errors: list[str] = []
    available_turns = 0
    unavailable_turns = 0
    total_turns = 0

    for record in records:
        task = task_by_id.get(record.get("task_id"))
        if task is None:
            validation_errors.append(f"unexpected task {record.get('task_id')!r}")
            continue
        prior_queries: list[str] = []
        for round_index, turn in enumerate(record.get("turns", []), 1):
            total_turns += 1
            query = str(turn.get("query", ""))
            outcomes = tuple(turn.get("outcomes", []))
            if outcomes != ("Yes", "No", UNAVAILABLE_OUTCOME):
                action_errors.append(
                    f"{task.task_id} round {round_index}: noncanonical outcomes"
                )
            contract_error = _query_contract_error(task, query)
            if contract_error is not None:
                action_errors.append(
                    f"{task.task_id} round {round_index}: {contract_error}"
                )
            if any(
                _queries_semantically_equivalent(query, prior)
                for prior in prior_queries
            ):
                duplicate_errors.append(
                    f"{task.task_id} round {round_index}: repeats {query!r}"
                )
            prior_queries.append(query)

            mapped_outcome = turn.get("mapped_outcome")
            if mapped_outcome == UNAVAILABLE_OUTCOME:
                unavailable_turns += 1
            else:
                available_turns += 1
            if not turn.get("mapped_cleanly"):
                grounding_errors.append(
                    f"{task.task_id} round {round_index}: unclean mapping"
                )
            if not turn.get("grounded") or not turn.get("relevant"):
                grounding_errors.append(
                    f"{task.task_id} round {round_index}: grounding/relevance failure"
                )

            selected = tuple(turn.get("selected_fact_indices", []))
            reply = turn.get("reply")
            cannot_answer = bool(turn.get("cannot_answer"))
            if selected:
                expected_reply = "\n".join(task.facts[index] for index in selected)
                if (
                    reply != expected_reply
                    or cannot_answer
                    or mapped_outcome == UNAVAILABLE_OUTCOME
                ):
                    grounding_errors.append(
                        f"{task.task_id} round {round_index}: selected-fact contract"
                    )
            elif (
                reply != PATIENT_CANNOT_ANSWER
                or not cannot_answer
                or mapped_outcome != UNAVAILABLE_OUTCOME
            ):
                grounding_errors.append(
                    f"{task.task_id} round {round_index}: unavailable contract"
                )

            details = turn.get("candidate_details") or []
            selected_details = [
                detail for detail in details if detail.get("query") == query
            ]
            if len(selected_details) != 1:
                validation_errors.append(
                    f"{task.task_id} round {round_index}: selected candidate detail"
                )
            else:
                semantic = selected_details[0].get("semantic_validation") or {}
                if semantic.get("valid") is not True:
                    validation_errors.append(
                        f"{task.task_id} round {round_index}: candidate validation"
                    )
            set_validation = turn.get("candidate_set_semantic_validation") or {}
            if set_validation.get("valid") is not True:
                validation_errors.append(
                    f"{task.task_id} round {round_index}: set validation"
                )

            metrics = turn.get("metrics") or {}
            for metric in (
                "structured_parse_failures",
                "candidate_validation_failures",
                "patient_relevance_failures",
            ):
                if float(metrics.get(metric, 0.0)) != 0.0:
                    validation_errors.append(
                        f"{task.task_id} round {round_index}: {metric}"
                    )

    checks = {
        "expected_task_ids": [record.get("task_id") for record in records]
        == expected_ids,
        "expected_round_count": len(records) == expected_tasks
        and all(len(record.get("turns", [])) == expected_rounds for record in records),
        "pinned_data_manifest": manifest.get("commit") == MEDIQ_COMMIT
        and (
            not verify_official_hash
            or manifest.get("expected_sha256") == MEDIQ_IMEDQA_DEV_SHA256
        )
        and manifest.get("raw_row_count") == raw_count
        and manifest.get("excluded_source_ids") == list(excluded)
        and manifest.get("selected_source_ids") == expected_source_ids,
        "canonical_valid_actions": not action_errors,
        "no_history_duplicates": not duplicate_errors,
        "clean_grounded_relevant_observations": not grounding_errors,
        "logged_validation_contract": not validation_errors,
        "minimum_available_turns": available_turns >= minimum_available_turns,
    }
    automated_pass = all(checks.values())
    return {
        "status": (
            "automated_pass_manual_review_pending" if automated_pass else "fail"
        ),
        "automated_pass": automated_pass,
        "manual_review_required": True,
        "endpoint_accuracy_is_ignored": True,
        "checks": checks,
        "item_dir": str(item_dir.resolve()),
        "interactions_path": str(interactions_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "task_offset": task_offset,
        "num_tasks": len(records),
        "num_turns": total_turns,
        "available_turns": available_turns,
        "unavailable_turns": unavailable_turns,
        "minimum_available_turns": minimum_available_turns,
        "grounding_errors": grounding_errors,
        "action_errors": action_errors,
        "duplicate_errors": duplicate_errors,
        "validation_errors": validation_errors,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MediQ Calibration Bank Validation",
        "",
        f"Status: **{report['status']}**",
        "",
        "Endpoint accuracy is ignored. Manual review of every selected interaction is required.",
        "",
        f"- Tasks: {report['num_tasks']}",
        f"- Turns: {report['num_turns']}",
        f"- Available Yes/No: {report['available_turns']}",
        f"- Unavailable: {report['unavailable_turns']}",
        "",
        "## Checks",
        "",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    for title, key in (
        ("Grounding", "grounding_errors"),
        ("Actions", "action_errors"),
        ("Duplicates", "duplicate_errors"),
        ("Validation", "validation_errors"),
    ):
        if report[key]:
            lines.extend(["", f"## {title} Errors", ""])
            lines.extend(f"- {error}" for error in report[key])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-item", default="000_naive")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("external/MediQ/data/all_dev_good.jsonl"),
    )
    parser.add_argument("--task-offset", type=int, default=5)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--expected-rounds", type=int, default=3)
    parser.add_argument("--minimum-available-turns", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_bank(
        args.run_dir / "items" / args.source_item,
        data_path=args.data_path,
        task_offset=args.task_offset,
        expected_tasks=args.expected_tasks,
        expected_rounds=args.expected_rounds,
        minimum_available_turns=args.minimum_available_turns,
    )
    if args.output:
        write_json(args.output, report)
        args.output.with_suffix(".md").write_text(
            render_markdown(report), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
