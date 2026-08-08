#!/usr/bin/env python3
"""Run endpoint-blind Bongard development blocks and paired analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import bongard_openworld_image_integrity_audit as image_audit
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_partition_integrity_audit as partition_audit
from scripts import bongard_openworld_sample_size_expansion_audit as expansion
from scripts import bongard_openworld_source_protocol_audit as source_audit
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-vlm-development64-14"
MODEL_ID = serving.MODEL_ID
BLOCK_SIZES = {"a": 16, "b": 16, "c": 16, "d": 16}
BLOCK_OFFSETS = {"a": 0, "b": 16, "c": 32, "d": 48}
BLOCK_EARLIEST_DATES = {
    "a": "2026-08-11",
    "b": "2026-08-12",
    "c": "2026-08-13",
    "d": "2026-08-14",
}
BLOCK_MODEL_SEEDS = {
    "a": 2_026_081_101,
    "b": 2_026_081_201,
    "c": 2_026_081_301,
    "d": 2_026_081_401,
}
BLOCK_ORDER = tuple(BLOCK_SIZES)
TASKS = sum(BLOCK_SIZES.values())
CASES_PER_TASK = 33
MAX_FINALS_PER_TASK = 10
CONCURRENCY = 24
MAX_TOKENS = serving.MAX_TOKENS
TEMPERATURE = 0.0
RUN_BUDGET_USD = 4.75
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_081_501
MIN_CHANGED_FINAL_HISTORIES = 24
MIN_RELATIVE_BRIER_IMPROVEMENT = 0.03
MIN_BOOTSTRAP_IMPROVEMENT_PROBABILITY = 0.80
EXPANSION_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_sample_size_expansion_audit/"
    "bongard-openworld-sample-size-expansion-audit-20260808/MANIFEST_V2.json"
)
EXPANSION_MANIFEST_SHA256 = (
    "eb4d8284db118eb3326eb3ee5c854bdb4eb42fb7ea5a786070f8608cf2f0f335"
)
IMPLEMENTATION_PATHS = (
    "scripts/bongard_openworld_vlm_bed.py",
    "scripts/bongard_openworld_partition_integrity_audit.py",
    "scripts/bongard_openworld_sample_size_expansion_audit.py",
    "scripts/bongard_openworld_luna_vlm_serving_smoke.py",
    "scripts/bongard_openworld_luna_vlm_mechanics_tree.py",
    "scripts/bongard_openworld_luna_vlm_development.py",
    "scripts/bongard_openworld_luna_claim_report.py",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_VLM_MECHANICS_PREREGISTRATION.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_FULL_MECHANICS_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_SEMANTIC_VALIDITY_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_SHUFFLED_CONTROL_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_HISTORY_BLIND_CONTROL_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_TERMINAL_CRN_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_TERMINAL_BATCH_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_CONTRASTIVE_PROMPT_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_PATH_DEPENDENT_CLAIM_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_BRANCH_OBEDIENCE_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_MATCHED_FIXED_SCORE_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_ENDPOINT_PREDICTIVE_UTILITY_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_PARTITION_INTEGRITY_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_TERMINAL_OBEDIENCE_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_TRANSPORT_RETRY_AMENDMENT.md",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_DEVELOPMENT32_PREREGISTRATION.md",
    "results/nonmyopic/BONGARD_OPENWORLD_DEVELOPMENT64_POWER_AMENDMENT.md",
    "results/nonmyopic/bongard_openworld_sample_size_expansion_audit/"
    "bongard-openworld-sample-size-expansion-audit-20260808/MANIFEST_V2.json",
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_CLAIM_DECISION_PLAN.md",
)


class StructuredModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, Any]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def message_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(bed.canonical_json(messages).encode("utf-8")).hexdigest()


def _expected_first_stage_requests(block_id: str) -> int:
    return BLOCK_SIZES[block_id] * CASES_PER_TASK


def _max_requests(block_id: str) -> int:
    return _expected_first_stage_requests(block_id) + (
        BLOCK_SIZES[block_id] * MAX_FINALS_PER_TASK
    )


def _max_http_attempts(block_id: str) -> int:
    accepted = _max_requests(block_id)
    return accepted + serving.transport_retry_allowance(accepted)


def development_tasks_for_block(
    tasks: Sequence[bed.VisualTask], block_id: str
) -> list[bed.VisualTask]:
    if block_id not in BLOCK_SIZES:
        raise ValueError(f"unknown development block {block_id!r}")
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks) or len(by_id) != TASKS:
        raise ValueError(f"development requires exactly {TASKS} tasks")
    expected = development_task_ids_by_block()[block_id]
    if set(by_id) != {
        task_id
        for ids in development_task_ids_by_block().values()
        for task_id in ids
    }:
        raise ValueError("development task identities changed")
    return [by_id[task_id] for task_id in expected]


@lru_cache(maxsize=1)
def development_rows_by_block() -> dict[str, list[Mapping[str, Any]]]:
    _, rows, _, _ = expansion.expanded_validation_rows()
    original = sorted(
        rows[: expansion.ORIGINAL_DEVELOPMENT_TASKS],
        key=lambda row: source_audit._task_layout(row)["task_id"],
    )
    additions = sorted(
        rows[expansion.ORIGINAL_DEVELOPMENT_TASKS :],
        key=lambda row: source_audit._task_layout(row)["task_id"],
    )
    if len(original) != 32 or len(additions) != 32:
        raise ValueError("expanded development rows changed")
    return {
        block_id: [
            *original[index * 8 : (index + 1) * 8],
            *additions[index * 8 : (index + 1) * 8],
        ]
        for index, block_id in enumerate(BLOCK_ORDER)
    }


@lru_cache(maxsize=1)
def development_task_ids_by_block() -> dict[str, tuple[str, ...]]:
    return {
        block_id: tuple(
            source_audit._task_layout(row)["task_id"] for row in rows
        )
        for block_id, rows in development_rows_by_block().items()
    }


def seal_endpoint_labels(task: bed.VisualTask) -> bed.VisualTask:
    endpoint_ids = set(task.endpoint_ids)
    labels = {
        image_id: label
        for image_id, label in task.actual_labels.items()
        if image_id not in endpoint_ids
    }
    expected = set(task.image_ids) - endpoint_ids
    if set(labels) != expected:
        raise ValueError("task does not contain exactly the non-endpoint labels")
    return replace(task, actual_labels=labels)


def load_block_tasks(block_id: str) -> list[bed.VisualTask]:
    tasks = bed.load_validation_partition_tasks(
        "development", include_endpoint_labels=False
    )
    return [
        seal_endpoint_labels(task)
        for task in development_tasks_for_block(tasks, block_id)
    ]


def build_protocol_manifest(*, output_path: Path) -> dict[str, Any]:
    expansion_manifest = json.loads(EXPANSION_MANIFEST.read_text(encoding="utf-8"))
    _, development_rows, _, _ = expansion.expanded_validation_rows()
    rows_by_task_id = {}
    for source_row in development_rows:
        layout = source_audit._task_layout(source_row)
        task_id = layout["task_id"]
        rows_by_task_id[task_id] = {
            "task_id": task_id,
            "source_row_sha256": source_audit.row_sha256(source_row),
        }
    rows = [
        {**rows_by_task_id[task_id], "block_id": block_id}
        for block_id in BLOCK_ORDER
        for task_id in development_task_ids_by_block()[block_id]
    ]
    counts = {
        block_id: sum(row["block_id"] == block_id for row in rows)
        for block_id in BLOCK_ORDER
    }
    gates = {
        "exact_64_unique_opaque_tasks": (
            len(rows) == len({row["task_id"] for row in rows}) == TASKS
        ),
        "exact_frozen_block_sizes": counts == BLOCK_SIZES,
        "source_commit_and_tree_are_bound": bool(
            source_audit.SOURCE_COMMIT and source_audit.SOURCE_TREE
        ),
        "full_image_archive_sha256_is_bound": len(image_audit.ARCHIVE_SHA256) == 64,
        "partition_integrity_manifest_is_bound": (
            sha256_file(partition_audit.PARTITION_INTEGRITY_MANIFEST)
            == partition_audit.PARTITION_INTEGRITY_MANIFEST_SHA256
        ),
        "sample_size_expansion_manifest_is_bound": (
            sha256_file(EXPANSION_MANIFEST) == EXPANSION_MANIFEST_SHA256
            and expansion_manifest.get("status")
            == "development64_confirmation96_partition_integrity_pass"
            and expansion_manifest.get("gates", {}).get("all_pass") is True
        ),
        "manifest_contains_no_source_uid_concept_caption_or_path": all(
            set(row) == {"task_id", "source_row_sha256", "block_id"}
            for row in rows
        ),
        "model_calls_are_zero": True,
        "endpoint_labels_remain_unaccessed": True,
        "confirmation_remains_unaccessed": True,
        "sealed_test_remains_unaccessed": True,
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen" if gates["all_pass"] else "failed",
        "interface_version": INTERFACE_VERSION,
        "source_commit": source_audit.SOURCE_COMMIT,
        "source_tree": source_audit.SOURCE_TREE,
        "image_archive_sha256": image_audit.ARCHIVE_SHA256,
        "partition_integrity_manifest_sha256": (
            partition_audit.PARTITION_INTEGRITY_MANIFEST_SHA256
        ),
        "sample_size_expansion_manifest_sha256": EXPANSION_MANIFEST_SHA256,
        "model": MODEL_ID,
        "reasoning": False,
        "blocks": {
            block_id: {
                "size": BLOCK_SIZES[block_id],
                "offset": BLOCK_OFFSETS[block_id],
                "earliest_london_date": BLOCK_EARLIEST_DATES[block_id],
                "model_seed": BLOCK_MODEL_SEEDS[block_id],
                "maximum_requests": _max_requests(block_id),
                "maximum_http_attempts": _max_http_attempts(block_id),
                "maximum_precharged_exposure_usd": (
                    _max_http_attempts(block_id) * serving.MAX_REQUEST_COST_USD
                ),
            }
            for block_id in BLOCK_ORDER
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "implementation_sha256": {
            path: sha256_file(REPO_ROOT / path) for path in IMPLEMENTATION_PATHS
        },
        "tasks": rows,
        "gates": gates,
    }
    checkpoint(output_path, result)
    return result


def verify_protocol_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    current_hashes = {
        item: sha256_file(REPO_ROOT / item) for item in IMPLEMENTATION_PATHS
    }
    expected_blocks = {
        block_id: {
            "size": BLOCK_SIZES[block_id],
            "offset": BLOCK_OFFSETS[block_id],
            "earliest_london_date": BLOCK_EARLIEST_DATES[block_id],
            "model_seed": BLOCK_MODEL_SEEDS[block_id],
            "maximum_requests": _max_requests(block_id),
            "maximum_http_attempts": _max_http_attempts(block_id),
            "maximum_precharged_exposure_usd": (
                _max_http_attempts(block_id) * serving.MAX_REQUEST_COST_USD
            ),
        }
        for block_id in BLOCK_ORDER
    }
    _, development_rows, _, _ = expansion.expanded_validation_rows()
    rows_by_task_id = {
        source_audit._task_layout(row)["task_id"]: {
            "task_id": source_audit._task_layout(row)["task_id"],
            "source_row_sha256": source_audit.row_sha256(row),
        }
        for row in development_rows
    }
    expected_tasks = [
        {**rows_by_task_id[task_id], "block_id": block_id}
        for block_id in BLOCK_ORDER
        for task_id in development_task_ids_by_block()[block_id]
    ]
    if (
        manifest.get("status") != "frozen"
        or manifest.get("interface_version") != INTERFACE_VERSION
        or manifest.get("model") != MODEL_ID
        or manifest.get("blocks") != expected_blocks
        or manifest.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or manifest.get("bootstrap_seed") != BOOTSTRAP_SEED
        or manifest.get("implementation_sha256") != current_hashes
        or manifest.get("source_commit") != source_audit.SOURCE_COMMIT
        or manifest.get("source_tree") != source_audit.SOURCE_TREE
        or manifest.get("image_archive_sha256") != image_audit.ARCHIVE_SHA256
        or manifest.get("partition_integrity_manifest_sha256")
        != partition_audit.PARTITION_INTEGRITY_MANIFEST_SHA256
        or manifest.get("sample_size_expansion_manifest_sha256")
        != EXPANSION_MANIFEST_SHA256
        or sha256_file(EXPANSION_MANIFEST) != EXPANSION_MANIFEST_SHA256
        or json.loads(EXPANSION_MANIFEST.read_text(encoding="utf-8")).get("status")
        != "development64_confirmation96_partition_integrity_pass"
        or sha256_file(partition_audit.PARTITION_INTEGRITY_MANIFEST)
        != partition_audit.PARTITION_INTEGRITY_MANIFEST_SHA256
        or manifest.get("tasks") != expected_tasks
        or not manifest.get("gates")
        or not all(manifest["gates"].values())
    ):
        raise ValueError("development protocol manifest is stale or invalid")
    return {
        "verified": True,
        "manifest_sha256": sha256_file(path),
        "implementation_sha256": current_hashes,
        "task_ids_by_block": {
            block_id: [
                row["task_id"]
                for row in expected_tasks
                if row["block_id"] == block_id
            ]
            for block_id in BLOCK_ORDER
        },
    }


def first_stage_cases(tasks: Sequence[bed.VisualTask]) -> list[mechanics.BeliefCase]:
    ordered = sorted(tasks, key=lambda task: task.task_id)
    roots = [
        mechanics.BeliefCase(
            case_id=f"{task.task_id}-root",
            task=task,
            history=task.initial_history,
            kind="root",
        )
        for task in ordered
    ]
    paired_branches = []
    for task in ordered:
        for candidate_id in task.candidate_ids:
            for label in (False, True):
                suffix = "positive" if label else "negative"
                paired_branches.extend(
                    [
                        mechanics.BeliefCase(
                            case_id=f"{task.task_id}-{candidate_id}-{suffix}",
                            task=task,
                            history=tuple(
                                sorted(
                                    (*task.initial_history, (candidate_id, label))
                                )
                            ),
                            kind="branch",
                            candidate_id=candidate_id,
                            simulated_label=label,
                        ),
                        mechanics.BeliefCase(
                            case_id=(
                                f"{task.task_id}-history-blind-"
                                f"{candidate_id}-{suffix}"
                            ),
                            task=task,
                            history=task.initial_history,
                            kind="history_blind",
                            candidate_id=candidate_id,
                            simulated_label=label,
                        ),
                    ]
                )
    cases = roots + paired_branches
    if len(cases) != len(tasks) * CASES_PER_TASK:
        raise AssertionError("development first-stage request count changed")
    return cases


def verify_mechanics_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    if (
        result.get("status") != "mechanics_pass"
        or protocol.get("interface_version") != mechanics.INTERFACE_VERSION
        or protocol.get("model") != MODEL_ID
        or protocol.get("development_accessed") is not False
        or protocol.get("confirmation_accessed") is not False
        or protocol.get("sealed_test_accessed") is not False
        or not result.get("gates")
        or not all(result["gates"].values())
        or not raw_path.is_file()
        or sha256_file(raw_path) != result.get("raw_responses_sha256")
    ):
        raise ValueError("mechanics result is not the frozen clean pass")
    return {
        "verified": True,
        "result_sha256": sha256_file(path),
        "raw_responses_sha256": sha256_file(raw_path),
        "cost_usd": float(result["usage"]["run_cost_usd"]),
        "request_count": int(result["protocol"]["expected_total_requests"]),
    }


def _parse_stage(
    cases: Sequence[mechanics.BeliefCase], responses: Sequence[str]
) -> tuple[
    dict[str, bed.SemanticBelief],
    dict[str, dict[tuple[str, bool], bed.SemanticBelief]],
    dict[str, dict[tuple[str, bool], bed.SemanticBelief]],
]:
    if len(responses) != len(cases):
        raise ValueError("first-stage response count changed")
    beliefs = [
        bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        for case, response in zip(cases, responses, strict=True)
    ]
    roots = {
        case.task.task_id: belief
        for case, belief in zip(cases, beliefs, strict=True)
        if case.kind == "root"
    }
    branches: dict[str, dict[tuple[str, bool], bed.SemanticBelief]] = {
        case.task.task_id: {} for case in cases if case.kind == "root"
    }
    history_blind: dict[
        str, dict[tuple[str, bool], bed.SemanticBelief]
    ] = {
        case.task.task_id: {} for case in cases if case.kind == "root"
    }
    for case, belief in zip(cases, beliefs, strict=True):
        if case.kind == "branch":
            branches[case.task.task_id][
                (str(case.candidate_id), bool(case.simulated_label))
            ] = belief
        elif case.kind == "history_blind":
            history_blind[case.task.task_id][
                (str(case.candidate_id), bool(case.simulated_label))
            ] = belief
    return roots, branches, history_blind


def _all_first_action_paths(
    *,
    task: bed.VisualTask,
    branches: Mapping[tuple[str, bool], bed.SemanticBelief],
) -> dict[str, dict[str, Any]]:
    paths = {}
    for first in task.candidate_ids:
        first_label = bool(task.actual_labels[first])
        remaining = tuple(
            candidate for candidate in task.candidate_ids if candidate != first
        )
        second_scores = bed.candidate_endpoint_eigs(
            branches[(first, first_label)], remaining, task.endpoint_ids
        )
        second = bed.select_best(second_scores)
        second_label = bool(task.actual_labels[second])
        final_history = tuple(
            sorted(
                (
                    *task.initial_history,
                    (first, first_label),
                    (second, second_label),
                )
            )
        )
        paths[first] = {
            "first_image_id": first,
            "first_label": bed.LABELS[first_label],
            "second_image_id": second,
            "second_label": bed.LABELS[second_label],
            "second_scores": second_scores,
            "final_history": final_history,
            "final_history_key": mechanics.history_key(final_history),
        }
    return paths


def _development_final_cases(
    *,
    tasks: Sequence[bed.VisualTask],
    plans: Mapping[str, Mapping[str, Any]],
    action_paths: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[mechanics.BeliefCase]:
    cases = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        histories = [
            plans[task.task_id]["policies"][policy]["final_history"]
            for policy in (
                "dynamic_depth2",
                "history_blind_depth2",
                "myopic_width",
                "fixed_depth2",
                "shuffled_dynamic_depth2",
                "random",
            )
        ] + [
            action_paths[task.task_id][first]["final_history"]
            for first in sorted(action_paths[task.task_id])
        ]
        seen = set()
        for history in histories:
            key = mechanics.history_key(history)
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                mechanics.BeliefCase(
                    case_id=(
                        f"{task.task_id}-final-"
                        f"{hashlib.sha256(key.encode()).hexdigest()[:12]}"
                    ),
                    task=task,
                    history=history,
                    kind="final",
                )
            )
    if not len(tasks) * 4 <= len(cases) <= len(tasks) * MAX_FINALS_PER_TASK:
        raise ValueError("development all-action final count is outside bounds")
    return cases


def _build_artifacts(
    *,
    tasks: Sequence[bed.VisualTask],
    stage_responses: Sequence[str],
    final_responses: Sequence[str],
    base_seed: int,
) -> dict[str, Any]:
    stage_cases = first_stage_cases(tasks)
    stage_messages = [
        bed.build_belief_messages(case.task, case.history)
        for case in stage_cases
    ]
    stage_seeds = mechanics.request_seeds_for_cases(
        stage_cases, base_seed=base_seed
    )
    paired_requests = mechanics.paired_request_diagnostics(
        cases=stage_cases,
        messages=stage_messages,
        seeds=stage_seeds,
    )
    roots, branches, history_blind = _parse_stage(
        stage_cases, stage_responses
    )
    branch_obedience = serving.branch_label_obedience(
        [
            (candidate_id, label, belief)
            for task_branches in branches.values()
            for (candidate_id, label), belief in task_branches.items()
        ]
    )
    plans = {
        task.task_id: mechanics.plan_task_policies(
            task=task,
            root=roots[task.task_id],
            branches=branches[task.task_id],
            history_blind_branches=history_blind[task.task_id],
        )
        for task in tasks
    }
    action_paths = {
        task.task_id: _all_first_action_paths(
            task=task, branches=branches[task.task_id]
        )
        for task in tasks
    }
    final_cases = _development_final_cases(
        tasks=tasks, plans=plans, action_paths=action_paths
    )
    if len(final_responses) != len(final_cases):
        raise ValueError("final response count changed")
    final_seeds = mechanics.request_seeds_for_cases(
        final_cases, base_seed=base_seed, final_stage=True
    )
    final_pairing = mechanics.final_request_diagnostics(
        cases=final_cases,
        seeds=final_seeds,
        plans=plans,
    )
    final_beliefs = [
        bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        for case, response in zip(final_cases, final_responses, strict=True)
    ]
    terminal_obedience = serving.terminal_label_obedience(
        [
            (case.task.initial_history, belief)
            for case, belief in zip(final_cases, final_beliefs, strict=True)
        ]
    )
    final_by_history = {
        (case.task.task_id, mechanics.history_key(case.history)): belief
        for case, belief in zip(final_cases, final_beliefs, strict=True)
    }
    diagnostics = [
        row
        for task in tasks
        for row in mechanics._branch_diagnostics(
            task=task, branches=branches[task.task_id]
        )
    ]
    trees = []
    for task in tasks:
        plan = plans[task.task_id]
        policy_rows = {}
        for policy in mechanics.POLICIES:
            selected = plan["policies"][policy]
            final_belief = final_by_history[
                (task.task_id, selected["final_history_key"])
            ]
            policy_rows[policy] = {
                **{
                    key: value
                    for key, value in selected.items()
                    if key != "final_history"
                },
                "final_belief": bed.public_belief_summary(final_belief),
            }
        trees.append(
            {
                "task_id": task.task_id,
                "score_objective": plan["score_objective"],
                "root_hypothesis_eig_diagnostics": plan[
                    "root_hypothesis_eig_diagnostics"
                ],
                "root_scores": plan["root_scores"],
                "shuffled_branch_mapping": plan["shuffled_branch_mapping"],
                "continuation_values": plan["continuation_values"],
                "root_belief": bed.public_belief_summary(roots[task.task_id]),
                "branch_diagnostics": mechanics._branch_diagnostics(
                    task=task, branches=branches[task.task_id]
                ),
                "policies": policy_rows,
                "all_first_action_paths": {
                    first: {
                        **{
                            key: value
                            for key, value in path.items()
                            if key != "final_history"
                        },
                        "final_belief": bed.public_belief_summary(
                            final_by_history[
                                (task.task_id, path["final_history_key"])
                            ]
                        ),
                    }
                    for first, path in action_paths[task.task_id].items()
                },
            }
        )
    return {
        "stage_cases": stage_cases,
        "stage_messages": stage_messages,
        "stage_seeds": stage_seeds,
        "paired_request_diagnostics": paired_requests,
        "roots": roots,
        "plans": plans,
        "action_paths": action_paths,
        "final_cases": final_cases,
        "final_request_pairing": final_pairing,
        "final_by_history": final_by_history,
        "branch_diagnostics": diagnostics,
        "branch_label_obedience": branch_obedience,
        "terminal_label_obedience": terminal_obedience,
        "trees": trees,
    }


def _block_gates(
    *,
    block_id: str,
    task_count: int,
    artifacts: Mapping[str, Any],
    usage: Mapping[str, Any],
    prompt_errors: Sequence[Sequence[str]],
    mechanics_verification: Mapping[str, Any],
) -> dict[str, bool]:
    final_count = len(artifacts["final_cases"])
    expected = _expected_first_stage_requests(block_id) + final_count
    finite = all(
        math.isfinite(value)
        for tree in artifacts["trees"]
        for scores in tree["root_scores"].values()
        for value in scores.values()
    ) and all(
        row["second_scores"] is None
        or all(math.isfinite(value) for value in row["second_scores"].values())
        for tree in artifacts["trees"]
        for row in tree["policies"].values()
    ) and all(
        math.isfinite(value)
        for tree in artifacts["trees"]
        for value in tree["root_hypothesis_eig_diagnostics"].values()
    )
    shuffled_control_exact = all(
        all(
            math.isclose(
                tree["continuation_values"]["shuffled_expected_continuation_utility"][
                    target
                ],
                tree["continuation_values"]["dynamic_expected_continuation_utility"][
                    source
                ],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for target, source in tree["shuffled_branch_mapping"].items()
        )
        for tree in artifacts["trees"]
    )
    gates = {
        "mechanics_result_is_hash_bound_clean_pass": (
            mechanics_verification.get("verified") is True
        ),
        "exact_frozen_task_count": task_count == BLOCK_SIZES[block_id],
        **serving.transport_retry_gates(usage, expected_requests=expected),
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "all_responses_parse_and_scores_are_finite": finite,
        "all_policies_use_endpoint_predictive_information_gain": all(
            tree.get("score_objective") == mechanics.SCORE_OBJECTIVE
            for tree in artifacts["trees"]
        ),
        "simulated_branch_labels_beat_constant_half_brier_in_both_classes": (
            serving.branch_label_obedience_passes(
                artifacts["branch_label_obedience"]
            )
        ),
        "terminal_beliefs_retain_both_queried_labels_better_than_constant_half": (
            serving.terminal_label_obedience_passes(
                artifacts["terminal_label_obedience"]
            )
        ),
        "exact_paired_seed_and_prompt_difference_accounting": (
            artifacts["paired_request_diagnostics"]["pair_count"]
            == task_count * ((CASES_PER_TASK - 1) // 2)
            and artifacts["paired_request_diagnostics"]["gates"]["all_pass"]
        ),
        "terminal_histories_use_task_level_common_random_numbers": (
            artifacts["final_request_pairing"]["task_count"] == task_count
            and artifacts["final_request_pairing"]["gates"]["all_pass"]
        ),
        "shuffled_control_exactly_permutes_complete_continuation_values": (
            shuffled_control_exact
        ),
        "fixed_score_dynamic_update_exactly_matches_fixed_first_and_dynamic_second": all(
            mechanics.fixed_score_dynamic_update_is_exact(tree)
            for tree in artifacts["trees"]
        ),
        "all_final_histories_generated_once_and_mapped": (
            task_count * 4 <= final_count <= task_count * MAX_FINALS_PER_TASK
        ),
        "all_prompts_hide_bound_source_truth": not any(prompt_errors),
        "endpoint_labels_remain_sealed": all(
            not (set(task.endpoint_ids) & set(task.actual_labels))
            for task in (case.task for case in artifacts["stage_cases"] if case.kind == "root")
        ),
        "cost_at_most_4_75": (
            float(usage.get("run_cost_usd", math.inf)) <= RUN_BUDGET_USD
        ),
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _adapter(
    *, output_dir: Path, run_id: str, block_id: str
) -> serving.LunaVisionAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=500.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=RUN_BUDGET_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=serving.MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return serving.LunaVisionAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=1_000_000),
        config,
        request_seed=BLOCK_MODEL_SEEDS[block_id],
    )


def _generate_checkpointed(
    *,
    model: StructuredModel,
    cases: Sequence[mechanics.BeliefCase],
    messages: Sequence[list[dict[str, Any]]],
    seeds: Sequence[int],
    progress_path: Path,
    dispatch_batches: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    if progress_path.exists():
        raise FileExistsError(f"response progress already exists: {progress_path}")
    if not (len(cases) == len(messages) == len(seeds)):
        raise ValueError("checkpoint request manifest lengths differ")
    batches = list(dispatch_batches or [])
    if not batches:
        batches = [
            {
                "batch_index": batch_index,
                "start": start,
                "stop": min(start + CONCURRENCY, len(cases)),
            }
            for batch_index, start in enumerate(
                range(0, len(cases), CONCURRENCY)
            )
        ]
    responses: list[str] = []
    request_hashes = [message_sha256(message) for message in messages]
    expected_start = 0
    for batch_index, batch in enumerate(batches):
        start = int(batch.get("start", -1))
        stop = int(batch.get("stop", -1))
        if (
            batch.get("batch_index") != batch_index
            or start != expected_start
            or start < 0
            or start >= stop
            or stop > len(cases)
            or stop - start > CONCURRENCY
        ):
            raise ValueError("checkpoint dispatch manifest is invalid")
        case_chunk = cases[start:stop]
        message_chunk = list(messages[start:stop])
        seed_chunk = list(seeds[start:stop])
        response_chunk = model.chat_complete_seeded_messages_batched_structured(
            message_chunk,
            seed_chunk,
            temperature=TEMPERATURE,
            response_format=bed.belief_response_format(),
            max_new_tokens=MAX_TOKENS,
        )
        if len(response_chunk) != len(case_chunk):
            raise ValueError("model returned the wrong checkpoint chunk size")
        for case, response in zip(case_chunk, response_chunk, strict=True):
            bed.parse_belief_response(
                response,
                image_ids=case.task.image_ids,
                history=case.history,
            )
        responses.extend(response_chunk)
        expected_start = stop
        checkpoint(
            progress_path,
            {
                "accepted_case_ids": [
                    case.case_id for case in cases[: len(responses)]
                ],
                "accepted_request_sha256": request_hashes[: len(responses)],
                "accepted_request_seeds": list(seeds[: len(responses)]),
                "accepted_responses": responses,
                "accepted_count": len(responses),
                "expected_count": len(cases),
                "complete": len(responses) == len(cases),
                "completed_dispatch_batches": batch_index + 1,
                "expected_dispatch_batches": len(batches),
            },
        )
    if expected_start != len(cases):
        raise ValueError("checkpoint dispatch manifest does not cover all cases")
    return responses


def run_block(
    *,
    output_dir: Path,
    run_id: str,
    block_id: str,
    mechanics_result: Path,
    protocol_manifest: Path,
    all_development_tasks: Sequence[bed.VisualTask] | None = None,
    adapter: StructuredModel | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_verification = verify_protocol_manifest(protocol_manifest)
    mechanics_verification = verify_mechanics_result(mechanics_result)
    source_tasks = (
        list(all_development_tasks)
        if all_development_tasks is not None
        else bed.load_validation_partition_tasks(
            "development", include_endpoint_labels=False
        )
    )
    tasks = [
        seal_endpoint_labels(task)
        for task in development_tasks_for_block(source_tasks, block_id)
    ]
    if [task.task_id for task in tasks] != manifest_verification[
        "task_ids_by_block"
    ][block_id]:
        raise ValueError("development block task IDs do not match the manifest")
    model = adapter or _adapter(
        output_dir=output_dir, run_id=run_id, block_id=block_id
    )
    stage_cases = first_stage_cases(tasks)
    stage_messages = [
        bed.build_belief_messages(case.task, case.history) for case in stage_cases
    ]
    stage_seeds = mechanics.request_seeds_for_cases(
        stage_cases, base_seed=BLOCK_MODEL_SEEDS[block_id]
    )
    paired_requests = mechanics.paired_request_diagnostics(
        cases=stage_cases,
        messages=stage_messages,
        seeds=stage_seeds,
    )
    if not paired_requests["gates"]["all_pass"]:
        raise ValueError("development paired history-blind request audit failed")
    stage_prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, messages)
        for case, messages in zip(stage_cases, stage_messages, strict=True)
    ]
    if any(stage_prompt_errors):
        raise ValueError("development first-stage hidden-state audit failed")
    stage_responses = _generate_checkpointed(
        model=model,
        cases=stage_cases,
        messages=stage_messages,
        seeds=stage_seeds,
        progress_path=output_dir / "private/FIRST_STAGE_PROGRESS.json",
    )
    roots, branches, history_blind = _parse_stage(
        stage_cases, stage_responses
    )
    plans = {
        task.task_id: mechanics.plan_task_policies(
            task=task,
            root=roots[task.task_id],
            branches=branches[task.task_id],
            history_blind_branches=history_blind[task.task_id],
        )
        for task in tasks
    }
    action_paths = {
        task.task_id: _all_first_action_paths(
            task=task, branches=branches[task.task_id]
        )
        for task in tasks
    }
    selected_final_cases = _development_final_cases(
        tasks=tasks, plans=plans, action_paths=action_paths
    )
    final_messages = [
        bed.build_belief_messages(case.task, case.history)
        for case in selected_final_cases
    ]
    final_seeds = mechanics.request_seeds_for_cases(
        selected_final_cases,
        base_seed=BLOCK_MODEL_SEEDS[block_id],
        final_stage=True,
    )
    final_dispatch_batches = mechanics.task_preserving_dispatch_batches(
        selected_final_cases
    )
    final_pairing = mechanics.final_request_diagnostics(
        cases=selected_final_cases,
        seeds=final_seeds,
        plans=plans,
        dispatch_batches=final_dispatch_batches,
    )
    if not final_pairing["gates"]["all_pass"]:
        raise ValueError("development terminal common-random-number audit failed")
    final_prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, messages)
        for case, messages in zip(selected_final_cases, final_messages, strict=True)
    ]
    if any(final_prompt_errors):
        raise ValueError("development final hidden-state audit failed")
    final_responses = _generate_checkpointed(
        model=model,
        cases=selected_final_cases,
        messages=final_messages,
        seeds=final_seeds,
        progress_path=output_dir / "private/FINAL_STAGE_PROGRESS.json",
        dispatch_batches=final_dispatch_batches,
    )
    artifacts = _build_artifacts(
        tasks=tasks,
        stage_responses=stage_responses,
        final_responses=final_responses,
        base_seed=BLOCK_MODEL_SEEDS[block_id],
    )
    usage = summarize_usage(model.usage_snapshot())
    gates = _block_gates(
        block_id=block_id,
        task_count=len(tasks),
        artifacts=artifacts,
        usage=usage,
        prompt_errors=[*stage_prompt_errors, *final_prompt_errors],
        mechanics_verification=mechanics_verification,
    )
    raw_path = output_dir / "private/RAW_RESPONSES.json"
    checkpoint(
        raw_path,
        {
            "block_id": block_id,
            "first_stage_case_ids": [case.case_id for case in stage_cases],
            "first_stage_request_sha256": [
                message_sha256(messages) for messages in stage_messages
            ],
            "first_stage_request_seeds": stage_seeds,
            "first_stage_responses": list(stage_responses),
            "final_case_ids": [
                case.case_id for case in artifacts["final_cases"]
            ],
            "final_responses": list(final_responses),
            "final_request_sha256": [
                message_sha256(messages) for messages in final_messages
            ],
            "final_request_seeds": final_seeds,
            "final_request_pairing": final_pairing,
            "candidate_labels_accessed_after_root_selection": True,
            "endpoint_labels_accessed": False,
            "combined_science_accessed": False,
            "confirmation_accessed": False,
            "sealed_test_accessed": False,
        },
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "block_mechanics_pass" if gates["all_pass"] else "failed_closed",
        "scientific_outcome_status": "sealed_until_all_blocks_complete",
        "authorizes_next_block": (
            gates["all_pass"] and block_id != BLOCK_ORDER[-1]
        ),
        "authorizes_combined_analysis": (
            gates["all_pass"] and block_id == BLOCK_ORDER[-1]
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_DEVELOPMENT32_PREREGISTRATION.md"
            ),
            "sample_size_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_DEVELOPMENT64_POWER_AMENDMENT.md"
            ),
            "endpoint_predictive_utility_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_ENDPOINT_PREDICTIVE_UTILITY_AMENDMENT.md"
            ),
            "score_objective": mechanics.SCORE_OBJECTIVE,
            "semantic_validity_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_SEMANTIC_VALIDITY_AMENDMENT.md"
            ),
            "shuffled_control_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_SHUFFLED_CONTROL_AMENDMENT.md"
            ),
            "history_blind_control_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_HISTORY_BLIND_CONTROL_AMENDMENT.md"
            ),
            "terminal_crn_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_TERMINAL_CRN_AMENDMENT.md"
            ),
            "terminal_obedience_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_TERMINAL_OBEDIENCE_AMENDMENT.md"
            ),
            "block_id": block_id,
            "block_size": BLOCK_SIZES[block_id],
            "block_offset": BLOCK_OFFSETS[block_id],
            "model": MODEL_ID,
            "model_seed": BLOCK_MODEL_SEEDS[block_id],
            "reasoning": False,
            "first_stage_requests": _expected_first_stage_requests(block_id),
            "root_requests": BLOCK_SIZES[block_id],
            "conditioned_branch_requests": BLOCK_SIZES[block_id] * 16,
            "history_blind_branch_requests": BLOCK_SIZES[block_id] * 16,
            "distinct_final_history_requests": len(artifacts["final_cases"]),
            "expected_total_requests": (
                _expected_first_stage_requests(block_id)
                + len(artifacts["final_cases"])
            ),
            "run_budget_usd": RUN_BUDGET_USD,
            "protocol_manifest_sha256": manifest_verification["manifest_sha256"],
            "endpoint_labels_accessed": False,
            "confirmation_accessed": False,
            "sealed_test_accessed": False,
        },
        "mechanics_verification": mechanics_verification,
        "protocol_manifest_verification": manifest_verification,
        "usage": usage,
        "branch_material_count": sum(
            row["material"] for row in artifacts["branch_diagnostics"]
        ),
        "branch_label_obedience": artifacts["branch_label_obedience"],
        "terminal_label_obedience": artifacts["terminal_label_obedience"],
        "paired_request_diagnostics": paired_requests,
        "final_request_pairing": final_pairing,
        "gates": gates,
        "trees": artifacts["trees"],
        "raw_responses_sha256": sha256_file(raw_path),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def replay_block(
    *,
    result_path: Path,
    all_development_tasks: Sequence[bed.VisualTask],
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    block_id = protocol.get("block_id")
    if (
        block_id not in BLOCK_SIZES
        or result.get("status") != "block_mechanics_pass"
        or protocol.get("interface_version") != INTERFACE_VERSION
        or protocol.get("model") != MODEL_ID
        or protocol.get("endpoint_labels_accessed") is not False
        or protocol.get("confirmation_accessed") is not False
        or protocol.get("sealed_test_accessed") is not False
        or (result.get("protocol_manifest_verification") or {}).get("verified")
        is not True
        or protocol.get("protocol_manifest_sha256")
        != (result.get("protocol_manifest_verification") or {}).get(
            "manifest_sha256"
        )
        or not result.get("gates")
        or not all(result["gates"].values())
    ):
        raise ValueError("development block is not a clean endpoint-blind pass")
    raw_path = result_path.parent / "private/RAW_RESPONSES.json"
    if sha256_file(raw_path) != result.get("raw_responses_sha256"):
        raise ValueError("development raw response hash changed")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    tasks = [
        seal_endpoint_labels(task)
        for task in development_tasks_for_block(all_development_tasks, block_id)
    ]
    stage_cases = first_stage_cases(tasks)
    if raw.get("first_stage_case_ids") != [case.case_id for case in stage_cases]:
        raise ValueError("development first-stage case order changed")
    stage_messages = [
        bed.build_belief_messages(case.task, case.history) for case in stage_cases
    ]
    if raw.get("first_stage_request_sha256") != [
        message_sha256(messages) for messages in stage_messages
    ]:
        raise ValueError("development first-stage request payload changed")
    stage_seeds = mechanics.request_seeds_for_cases(
        stage_cases, base_seed=BLOCK_MODEL_SEEDS[block_id]
    )
    if raw.get("first_stage_request_seeds") != stage_seeds:
        raise ValueError("development first-stage request seeds changed")
    artifacts = _build_artifacts(
        tasks=tasks,
        stage_responses=raw.get("first_stage_responses") or [],
        final_responses=raw.get("final_responses") or [],
        base_seed=BLOCK_MODEL_SEEDS[block_id],
    )
    if raw.get("final_case_ids") != [
        case.case_id for case in artifacts["final_cases"]
    ]:
        raise ValueError("development final case order changed")
    final_messages = [
        bed.build_belief_messages(case.task, case.history)
        for case in artifacts["final_cases"]
    ]
    if raw.get("final_request_sha256") != [
        message_sha256(messages) for messages in final_messages
    ]:
        raise ValueError("development final request payload changed")
    final_seeds = mechanics.request_seeds_for_cases(
        artifacts["final_cases"],
        base_seed=BLOCK_MODEL_SEEDS[block_id],
        final_stage=True,
    )
    if raw.get("final_request_seeds") != final_seeds:
        raise ValueError("development final request seeds changed")
    if (
        raw.get("final_request_pairing")
        != artifacts["final_request_pairing"]
        or result.get("final_request_pairing")
        != artifacts["final_request_pairing"]
    ):
        raise ValueError("development terminal pairing does not replay")
    if bed.canonical_json(artifacts["trees"]) != bed.canonical_json(result["trees"]):
        raise ValueError("development block tree does not replay")
    return {
        "block_id": block_id,
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "result_sha256": sha256_file(result_path),
        "raw_responses_sha256": sha256_file(raw_path),
        "tasks": tasks,
        "artifacts": artifacts,
        "verified": True,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_summary(
    values: Sequence[float], *, seed: int
) -> dict[str, Any]:
    values = tuple(float(value) for value in values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("paired values must be finite and nonempty")
    rng = random.Random(seed)
    samples = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return {
        "n": len(values),
        "mean_difference": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
        "bootstrap_probability_improvement": sum(value < 0 for value in samples)
        / len(samples),
        "wins": sum(value < 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value > 0 for value in values),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman vectors must have equal length at least two")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_scale == 0 or right_scale == 0:
        return 0.0
    return numerator / (left_scale * right_scale)


def analyze_combined(
    *,
    block_results: Sequence[Path],
    output_path: Path,
    all_development_tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    planning_tasks = sorted(
        [seal_endpoint_labels(task) for task in all_development_tasks]
        if all_development_tasks is not None
        else bed.load_validation_partition_tasks(
            "development", include_endpoint_labels=False
        ),
        key=lambda task: task.task_id,
    )
    if len(planning_tasks) != TASKS:
        raise ValueError(f"combined development requires exactly {TASKS} tasks")
    replays = [
        replay_block(result_path=path, all_development_tasks=planning_tasks)
        for path in block_results
    ]
    by_block = {replay["block_id"]: replay for replay in replays}
    if set(by_block) != set(BLOCK_ORDER) or len(replays) != len(BLOCK_ORDER):
        raise ValueError("combined analysis requires every frozen development block")
    if len(
        {replay["protocol_manifest_sha256"] for replay in replays}
    ) != 1:
        raise ValueError("development blocks do not share one protocol manifest")
    tasks = sorted(
        list(all_development_tasks)
        if all_development_tasks is not None
        else bed.load_validation_partition_tasks(
            "development", include_endpoint_labels=True
        ),
        key=lambda task: task.task_id,
    )
    if any(
        not set(task.endpoint_ids).issubset(task.actual_labels) for task in tasks
    ):
        raise ValueError("combined endpoint scorer lacks endpoint labels")
    full_by_id = {task.task_id: task for task in tasks}
    trees = []
    for block_id in BLOCK_ORDER:
        replay = by_block[block_id]
        for tree in replay["artifacts"]["trees"]:
            task = full_by_id[tree["task_id"]]
            endpoint_labels = {
                image_id: bool(task.actual_labels[image_id])
                for image_id in task.endpoint_ids
            }
            policy_rows = {}
            for policy in mechanics.POLICIES:
                row = tree["policies"][policy]
                final_belief = replay["artifacts"]["final_by_history"][
                    (task.task_id, row["final_history_key"])
                ]
                policy_rows[policy] = {
                    **row,
                    "endpoint": bed.endpoint_metrics(final_belief, endpoint_labels),
                }
            action_rows = {}
            for first, row in tree["all_first_action_paths"].items():
                final_belief = replay["artifacts"]["final_by_history"][
                    (task.task_id, row["final_history_key"])
                ]
                action_rows[first] = {
                    **row,
                    "endpoint": bed.endpoint_metrics(final_belief, endpoint_labels),
                }
            first_ids = tuple(sorted(action_rows))
            endpoint_utility = [
                -action_rows[first]["endpoint"]["mean_brier"]
                for first in first_ids
            ]
            ranking = {
                score_name: spearman_correlation(
                    [tree["root_scores"][score_name][first] for first in first_ids],
                    endpoint_utility,
                )
                for score_name in mechanics.SCORE_POLICIES
            }
            root = replay["artifacts"]["roots"][task.task_id]
            candidate_brier = statistics.fmean(
                (
                    bed.predictive_probability(root, image_id)
                    - float(task.actual_labels[image_id])
                )
                ** 2
                for image_id in task.candidate_ids
            )
            trees.append(
                {
                    **tree,
                    "policies": policy_rows,
                    "all_first_action_paths": action_rows,
                    "ranking_fidelity": ranking,
                    "root_candidate_brier": candidate_brier,
                }
            )
    trees.sort(key=lambda tree: tree["task_id"])
    if len(trees) != TASKS or len({tree["task_id"] for tree in trees}) != TASKS:
        raise ValueError("combined task coverage is not exact and unique")
    pooled = mechanics._pooled_policy_metrics(trees)
    ranking_fidelity = {
        score_name: {
            "mean_spearman": statistics.fmean(
                tree["ranking_fidelity"][score_name] for tree in trees
            ),
            "sample_sd": statistics.stdev(
                tree["ranking_fidelity"][score_name] for tree in trees
            ),
        }
        for score_name in mechanics.SCORE_POLICIES
    }
    mean_root_candidate_brier = statistics.fmean(
        tree["root_candidate_brier"] for tree in trees
    )
    comparisons: dict[str, Any] = {}
    for policy_index, policy in enumerate(mechanics.POLICIES):
        if policy == "myopic_width":
            continue
        comparisons[policy] = {}
        for metric_index, metric in enumerate(("mean_brier", "mean_log_loss")):
            values = [
                tree["policies"][policy]["endpoint"][metric]
                - tree["policies"]["myopic_width"]["endpoint"][metric]
                for tree in trees
            ]
            comparisons[policy][metric] = paired_summary(
                values,
                seed=BOOTSTRAP_SEED + policy_index * 10 + metric_index,
            )
    dynamic_vs_history_blind = {}
    for metric_index, metric in enumerate(("mean_brier", "mean_log_loss")):
        values = [
            tree["policies"]["dynamic_depth2"]["endpoint"][metric]
            - tree["policies"]["history_blind_depth2"]["endpoint"][metric]
            for tree in trees
        ]
        dynamic_vs_history_blind[metric] = paired_summary(
            values,
            seed=BOOTSTRAP_SEED + 1_000 + metric_index,
        )
    dynamic_vs_fixed_depth2 = {}
    for metric_index, metric in enumerate(("mean_brier", "mean_log_loss")):
        values = [
            tree["policies"]["dynamic_depth2"]["endpoint"][metric]
            - tree["policies"]["fixed_depth2"]["endpoint"][metric]
            for tree in trees
        ]
        dynamic_vs_fixed_depth2[metric] = paired_summary(
            values,
            seed=BOOTSTRAP_SEED + 2_000 + metric_index,
        )
    dynamic_vs_matched_fixed = {}
    for metric_index, metric in enumerate(("mean_brier", "mean_log_loss")):
        values = [
            tree["policies"]["dynamic_depth2"]["endpoint"][metric]
            - tree["policies"]["fixed_score_dynamic_update"]["endpoint"][metric]
            for tree in trees
        ]
        dynamic_vs_matched_fixed[metric] = paired_summary(
            values,
            seed=BOOTSTRAP_SEED + 3_000 + metric_index,
        )
    changed = sum(
        tree["policies"]["dynamic_depth2"]["final_history_key"]
        != tree["policies"]["myopic_width"]["final_history_key"]
        for tree in trees
    )
    robust_changed = sum(
        (
            dynamic_first := tree["policies"]["dynamic_depth2"][
                "first_image_id"
            ]
        )
        != (
            myopic_first := tree["policies"]["myopic_width"][
                "first_image_id"
            ]
        )
        and tree["root_scores"]["dynamic_depth2"][dynamic_first]
        - tree["root_scores"]["dynamic_depth2"][myopic_first]
        >= mechanics.MIN_ACTION_MARGIN_NATS
        for tree in trees
    )
    dynamic_blind_changed = sum(
        tree["policies"]["dynamic_depth2"]["final_history_key"]
        != tree["policies"]["history_blind_depth2"]["final_history_key"]
        for tree in trees
    )
    robust_dynamic_blind_changed = sum(
        tree["policies"]["dynamic_depth2"]["first_image_id"]
        != tree["policies"]["history_blind_depth2"]["first_image_id"]
        and tree["policies"]["dynamic_depth2"]["first_score_margin"]
        >= mechanics.MIN_ACTION_MARGIN_NATS
        and tree["policies"]["history_blind_depth2"]["first_score_margin"]
        >= mechanics.MIN_ACTION_MARGIN_NATS
        for tree in trees
    )
    dynamic_fixed_changed = sum(
        tree["policies"]["dynamic_depth2"]["final_history_key"]
        != tree["policies"]["fixed_depth2"]["final_history_key"]
        for tree in trees
    )
    robust_dynamic_fixed_changed = sum(
        (
            dynamic_first := tree["policies"]["dynamic_depth2"][
                "first_image_id"
            ]
        )
        != (
            fixed_first := tree["policies"]["fixed_depth2"][
                "first_image_id"
            ]
        )
        and tree["root_scores"]["dynamic_depth2"][dynamic_first]
        - tree["root_scores"]["dynamic_depth2"][fixed_first]
        >= mechanics.MIN_ACTION_MARGIN_NATS
        for tree in trees
    )
    dynamic_matched_fixed_changed = sum(
        tree["policies"]["dynamic_depth2"]["final_history_key"]
        != tree["policies"]["fixed_score_dynamic_update"]["final_history_key"]
        for tree in trees
    )
    robust_dynamic_matched_fixed_changed = sum(
        tree["policies"]["dynamic_depth2"]["first_image_id"]
        != tree["policies"]["fixed_score_dynamic_update"]["first_image_id"]
        and tree["policies"]["dynamic_depth2"]["first_score_margin"]
        >= mechanics.MIN_ACTION_MARGIN_NATS
        and tree["policies"]["fixed_score_dynamic_update"][
            "first_score_margin"
        ]
        >= mechanics.MIN_ACTION_MARGIN_NATS
        for tree in trees
    )
    task_ids_by_block = {
        block_id: {
            task.task_id for task in by_block[block_id]["tasks"]
        }
        for block_id in BLOCK_ORDER
    }
    blockwise = {}
    for block_id in BLOCK_ORDER:
        block_trees = [
            tree
            for tree in trees
            if tree["task_id"] in task_ids_by_block[block_id]
        ]
        blockwise[block_id] = {
            "tasks": len(block_trees),
            "dynamic_minus_myopic_mean_brier": statistics.fmean(
                tree["policies"]["dynamic_depth2"]["endpoint"]["mean_brier"]
                - tree["policies"]["myopic_width"]["endpoint"]["mean_brier"]
                for tree in block_trees
            ),
            "dynamic_minus_myopic_mean_log_loss": statistics.fmean(
                tree["policies"]["dynamic_depth2"]["endpoint"]["mean_log_loss"]
                - tree["policies"]["myopic_width"]["endpoint"]["mean_log_loss"]
                for tree in block_trees
            ),
            "dynamic_myopic_changed_final_histories": sum(
                tree["policies"]["dynamic_depth2"]["final_history_key"]
                != tree["policies"]["myopic_width"]["final_history_key"]
                for tree in block_trees
            ),
            "dynamic_history_blind_changed_final_histories": sum(
                tree["policies"]["dynamic_depth2"]["final_history_key"]
                != tree["policies"]["history_blind_depth2"]["final_history_key"]
                for tree in block_trees
            ),
            "dynamic_fixed_changed_final_histories": sum(
                tree["policies"]["dynamic_depth2"]["final_history_key"]
                != tree["policies"]["fixed_depth2"]["final_history_key"]
                for tree in block_trees
            ),
            "dynamic_matched_fixed_changed_final_histories": sum(
                tree["policies"]["dynamic_depth2"]["final_history_key"]
                != tree["policies"]["fixed_score_dynamic_update"][
                    "final_history_key"
                ]
                for tree in block_trees
            ),
            "dynamic_minus_history_blind_mean_brier": statistics.fmean(
                tree["policies"]["dynamic_depth2"]["endpoint"]["mean_brier"]
                - tree["policies"]["history_blind_depth2"]["endpoint"]["mean_brier"]
                for tree in block_trees
            ),
            "dynamic_minus_history_blind_mean_log_loss": statistics.fmean(
                tree["policies"]["dynamic_depth2"]["endpoint"]["mean_log_loss"]
                - tree["policies"]["history_blind_depth2"]["endpoint"]["mean_log_loss"]
                for tree in block_trees
            ),
            "dynamic_mean_spearman": statistics.fmean(
                tree["ranking_fidelity"]["dynamic_depth2"]
                for tree in block_trees
            ),
        }
    myopic_brier = pooled["myopic_width"]["mean_brier"]
    relative_brier_gain = (
        (myopic_brier - pooled["dynamic_depth2"]["mean_brier"]) / myopic_brier
        if myopic_brier > 0
        else -math.inf
    )
    history_blind_brier = pooled["history_blind_depth2"]["mean_brier"]
    dynamic_vs_history_blind_relative_brier_gain = (
        (
            history_blind_brier
            - pooled["dynamic_depth2"]["mean_brier"]
        )
        / history_blind_brier
        if history_blind_brier > 0
        else -math.inf
    )
    fixed_brier = pooled["fixed_depth2"]["mean_brier"]
    dynamic_vs_fixed_relative_brier_gain = (
        (fixed_brier - pooled["dynamic_depth2"]["mean_brier"])
        / fixed_brier
        if fixed_brier > 0
        else -math.inf
    )
    matched_fixed_brier = pooled["fixed_score_dynamic_update"]["mean_brier"]
    dynamic_vs_matched_fixed_relative_brier_gain = (
        (matched_fixed_brier - pooled["dynamic_depth2"]["mean_brier"])
        / matched_fixed_brier
        if matched_fixed_brier > 0
        else -math.inf
    )
    dynamic_brier = comparisons["dynamic_depth2"]["mean_brier"]
    dynamic_log = comparisons["dynamic_depth2"]["mean_log_loss"]
    dynamic_vs_fixed_brier = dynamic_vs_fixed_depth2["mean_brier"]
    dynamic_vs_fixed_log = dynamic_vs_fixed_depth2["mean_log_loss"]
    dynamic_vs_matched_fixed_brier = dynamic_vs_matched_fixed["mean_brier"]
    dynamic_vs_matched_fixed_log = dynamic_vs_matched_fixed["mean_log_loss"]
    science_gates = {
        "all_four_endpoint_blind_blocks_independently_replay": all(
            replay["verified"] for replay in replays
        ),
        "exact_64_disjoint_development_tasks": len(trees) == TASKS,
        "at_least_24_dynamic_final_histories_differ_from_myopic": (
            changed >= MIN_CHANGED_FINAL_HISTORIES
        ),
        "at_least_24_dynamic_action_changes_clear_numerical_tie_margin": (
            robust_changed >= MIN_CHANGED_FINAL_HISTORIES
        ),
        "dynamic_and_myopic_differ_in_every_execution_block": all(
            row["dynamic_myopic_changed_final_histories"] >= 1
            for row in blockwise.values()
        ),
        "at_least_24_dynamic_final_histories_differ_from_history_blind": (
            dynamic_blind_changed >= MIN_CHANGED_FINAL_HISTORIES
        ),
        "dynamic_and_history_blind_differ_in_every_execution_block": all(
            row["dynamic_history_blind_changed_final_histories"] >= 1
            for row in blockwise.values()
        ),
        "root_candidate_brier_beats_constant_half": (
            mean_root_candidate_brier < 0.25
        ),
        "dynamic_score_has_positive_mean_endpoint_ranking_fidelity": (
            ranking_fidelity["dynamic_depth2"]["mean_spearman"] > 0
        ),
        "dynamic_score_ranking_fidelity_is_not_worse_than_myopic": (
            ranking_fidelity["dynamic_depth2"]["mean_spearman"]
            >= ranking_fidelity["myopic_width"]["mean_spearman"]
        ),
        "dynamic_brier_relative_improvement_at_least_3_percent": (
            relative_brier_gain >= MIN_RELATIVE_BRIER_IMPROVEMENT
        ),
        "dynamic_brier_bootstrap_improvement_probability_at_least_0_80": (
            dynamic_brier["bootstrap_probability_improvement"]
            >= MIN_BOOTSTRAP_IMPROVEMENT_PROBABILITY
        ),
        "dynamic_log_loss_is_not_worse_than_myopic": (
            dynamic_log["mean_difference"] <= 0
        ),
        "dynamic_brier_relative_improvement_vs_history_blind_at_least_3_percent": (
            dynamic_vs_history_blind_relative_brier_gain
            >= MIN_RELATIVE_BRIER_IMPROVEMENT
        ),
        "dynamic_brier_vs_history_blind_bootstrap_probability_at_least_0_80": (
            dynamic_vs_history_blind["mean_brier"][
                "bootstrap_probability_improvement"
            ]
            >= MIN_BOOTSTRAP_IMPROVEMENT_PROBABILITY
        ),
        "dynamic_log_loss_is_not_worse_than_history_blind": (
            dynamic_vs_history_blind["mean_log_loss"]["mean_difference"] <= 0
        ),
        "dynamic_ranking_fidelity_is_not_worse_than_history_blind": (
            ranking_fidelity["dynamic_depth2"]["mean_spearman"]
            >= ranking_fidelity["history_blind_depth2"]["mean_spearman"]
        ),
        "at_least_24_dynamic_final_histories_differ_from_fixed_depth2": (
            dynamic_fixed_changed >= MIN_CHANGED_FINAL_HISTORIES
        ),
        "at_least_24_dynamic_action_changes_from_fixed_clear_numerical_tie_margin": (
            robust_dynamic_fixed_changed >= MIN_CHANGED_FINAL_HISTORIES
        ),
        "dynamic_and_fixed_depth2_differ_in_every_execution_block": all(
            row["dynamic_fixed_changed_final_histories"] >= 1
            for row in blockwise.values()
        ),
        "dynamic_brier_relative_improvement_vs_fixed_depth2_at_least_3_percent": (
            dynamic_vs_fixed_relative_brier_gain
            >= MIN_RELATIVE_BRIER_IMPROVEMENT
        ),
        "dynamic_brier_vs_fixed_depth2_bootstrap_probability_at_least_0_80": (
            dynamic_vs_fixed_brier["bootstrap_probability_improvement"]
            >= MIN_BOOTSTRAP_IMPROVEMENT_PROBABILITY
        ),
        "dynamic_log_loss_is_not_worse_than_fixed_depth2": (
            dynamic_vs_fixed_log["mean_difference"] <= 0
        ),
        "dynamic_ranking_fidelity_is_not_worse_than_fixed_depth2": (
            ranking_fidelity["dynamic_depth2"]["mean_spearman"]
            >= ranking_fidelity["fixed_depth2"]["mean_spearman"]
        ),
        "at_least_24_dynamic_final_histories_differ_from_fixed_score_dynamic_update": (
            dynamic_matched_fixed_changed >= MIN_CHANGED_FINAL_HISTORIES
        ),
        "at_least_24_dynamic_action_changes_from_fixed_score_dynamic_update_clear_numerical_tie_margin": (
            robust_dynamic_matched_fixed_changed >= MIN_CHANGED_FINAL_HISTORIES
        ),
        "dynamic_and_fixed_score_dynamic_update_differ_in_every_execution_block": all(
            row["dynamic_matched_fixed_changed_final_histories"] >= 1
            for row in blockwise.values()
        ),
        "dynamic_brier_relative_improvement_vs_fixed_score_dynamic_update_at_least_3_percent": (
            dynamic_vs_matched_fixed_relative_brier_gain
            >= MIN_RELATIVE_BRIER_IMPROVEMENT
        ),
        "dynamic_brier_vs_fixed_score_dynamic_update_bootstrap_probability_at_least_0_80": (
            dynamic_vs_matched_fixed_brier[
                "bootstrap_probability_improvement"
            ]
            >= MIN_BOOTSTRAP_IMPROVEMENT_PROBABILITY
        ),
        "dynamic_log_loss_is_not_worse_than_fixed_score_dynamic_update": (
            dynamic_vs_matched_fixed_log["mean_difference"] <= 0
        ),
        "dynamic_brier_is_not_worse_than_shuffled_control": (
            pooled["dynamic_depth2"]["mean_brier"]
            <= pooled["shuffled_dynamic_depth2"]["mean_brier"]
        ),
        "all_endpoint_metrics_are_finite": all(
            math.isfinite(value)
            for policy_metrics in pooled.values()
            for value in policy_metrics.values()
        ),
        "confirmation_and_sealed_test_remain_unopened": True,
    }
    science_gates["all_pass"] = all(science_gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "development_signal" if science_gates["all_pass"] else "development_null"
        ),
        "authorizes_confirmation_preregistration": science_gates["all_pass"],
        "authorizes_confirmation_execution": False,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_DEVELOPMENT32_PREREGISTRATION.md"
            ),
            "sample_size_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_DEVELOPMENT64_POWER_AMENDMENT.md"
            ),
            "endpoint_predictive_utility_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_ENDPOINT_PREDICTIVE_UTILITY_AMENDMENT.md"
            ),
            "score_objective": mechanics.SCORE_OBJECTIVE,
            "semantic_validity_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_SEMANTIC_VALIDITY_AMENDMENT.md"
            ),
            "shuffled_control_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_SHUFFLED_CONTROL_AMENDMENT.md"
            ),
            "history_blind_control_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_HISTORY_BLIND_CONTROL_AMENDMENT.md"
            ),
            "model": MODEL_ID,
            "blocks": list(BLOCK_ORDER),
            "task_count": TASKS,
            "endpoint_images_per_task": 2,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "endpoint_labels_accessed_only_after_all_blocks_replayed": True,
            "confirmation_accessed": False,
            "sealed_test_accessed": False,
        },
        "block_verification": {
            block_id: {
                key: value
                for key, value in by_block[block_id].items()
                if key
                in {
                    "block_id",
                    "protocol_manifest_sha256",
                    "result_sha256",
                    "raw_responses_sha256",
                    "verified",
                }
            }
            for block_id in BLOCK_ORDER
        },
        "pooled_policy_metrics": pooled,
        "mean_root_candidate_brier": mean_root_candidate_brier,
        "ranking_fidelity": ranking_fidelity,
        "blockwise_dynamic_vs_myopic": blockwise,
        "blockwise_dynamic_vs_history_blind": {
            block_id: {
                key: value
                for key, value in row.items()
                if key
                in {
                    "tasks",
                    "dynamic_history_blind_changed_final_histories",
                    "dynamic_minus_history_blind_mean_brier",
                    "dynamic_minus_history_blind_mean_log_loss",
                }
            }
            for block_id, row in blockwise.items()
        },
        "comparisons_vs_myopic": comparisons,
        "dynamic_vs_history_blind": dynamic_vs_history_blind,
        "dynamic_vs_fixed_depth2": dynamic_vs_fixed_depth2,
        "dynamic_vs_fixed_score_dynamic_update": dynamic_vs_matched_fixed,
        "dynamic_vs_myopic_changed_final_histories": changed,
        "dynamic_vs_myopic_robust_action_changes": robust_changed,
        "dynamic_vs_myopic_relative_brier_improvement": relative_brier_gain,
        "dynamic_vs_history_blind_changed_final_histories": (
            dynamic_blind_changed
        ),
        "dynamic_vs_history_blind_robust_action_changes": (
            robust_dynamic_blind_changed
        ),
        "dynamic_vs_history_blind_relative_brier_improvement": (
            dynamic_vs_history_blind_relative_brier_gain
        ),
        "dynamic_vs_fixed_depth2_changed_final_histories": (
            dynamic_fixed_changed
        ),
        "dynamic_vs_fixed_depth2_robust_action_changes": (
            robust_dynamic_fixed_changed
        ),
        "dynamic_vs_fixed_depth2_relative_brier_improvement": (
            dynamic_vs_fixed_relative_brier_gain
        ),
        "dynamic_vs_fixed_score_dynamic_update_changed_final_histories": (
            dynamic_matched_fixed_changed
        ),
        "dynamic_vs_fixed_score_dynamic_update_robust_action_changes": (
            robust_dynamic_matched_fixed_changed
        ),
        "dynamic_vs_fixed_score_dynamic_update_relative_brier_improvement": (
            dynamic_vs_matched_fixed_relative_brier_gain
        ),
        "gates": science_gates,
        "trees": trees,
    }
    checkpoint(output_path, result)
    return result


def _validate_daily_ledger(
    ledger: Mapping[str, Any], *, block_id: str, now: datetime | None = None
) -> None:
    timezone = ZoneInfo(serving.TIMEZONE)
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    if local_now.date().isoformat() < BLOCK_EARLIEST_DATES[block_id]:
        raise RuntimeError(
            f"development block {block_id} is forbidden before "
            f"{BLOCK_EARLIEST_DATES[block_id]}"
        )
    if ledger.get("date") != local_now.date().isoformat():
        raise RuntimeError("development requires the current daily ledger")
    if ledger.get("timezone") != serving.TIMEZONE:
        raise RuntimeError("development ledger timezone changed")
    if float(ledger.get("daily_cap_usd", 0.0)) != 5.0:
        raise RuntimeError("development ledger no longer has the exact $5 cap")
    if f"bongard_luna_vlm_development_block_{block_id}" in ledger:
        raise RuntimeError("development block is already recorded")


def _initialize_daily_ledger(
    *,
    path: Path,
    live: Mapping[str, float],
    block_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    timezone = ZoneInfo(serving.TIMEZONE)
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        _validate_daily_ledger(ledger, block_id=block_id, now=local_now)
        return ledger
    if local_now.date().isoformat() < BLOCK_EARLIEST_DATES[block_id]:
        raise RuntimeError("development spend date gate is closed")
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "date": local_now.date().isoformat(),
        "timezone": serving.TIMEZONE,
        "daily_cap_usd": 5.0,
        "opening_total_credits_usd": float(live["total_credits_usd"]),
        "opening_total_usage_usd": float(live["total_usage_usd"]),
        "opening_balance_usd": float(live["balance_usd"]),
        "opening_frozen_at_london": local_now.isoformat(),
        "recorded_actual_spend_usd": 0.0,
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "first_authorized_block": {
            "interface_version": INTERFACE_VERSION,
            "block_id": block_id,
            "model": MODEL_ID,
            "maximum_cost_usd": RUN_BUDGET_USD,
            "status": "authorized_pending",
        },
        "additional_paid_blocks_authorized": False,
    }
    checkpoint(path, ledger)
    return ledger


def _reconcile_ledger(
    *,
    ledger: Mapping[str, Any],
    block_id: str,
    measured_cost_usd: float,
    live_after: Mapping[str, float],
    status: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    previous = float(updated.get("recorded_actual_spend_usd", 0.0))
    posted = max(0.0, float(live_after["total_usage_usd"]) - opening)
    local = previous + measured_cost_usd
    recorded = max(posted, local)
    key = f"bongard_luna_vlm_development_block_{block_id}"
    old_cost = float((updated.get(key) or {}).get("actual_cost_usd", 0.0))
    block_cost = max(old_cost, measured_cost_usd)
    updated["recorded_actual_spend_usd"] = recorded
    updated[key] = {
        "status": status,
        "actual_cost_usd": block_cost,
        "maximum_cost_usd": RUN_BUDGET_USD,
        "interface_version": INTERFACE_VERSION,
        "model": MODEL_ID,
    }
    authorization = updated.get("first_authorized_block") or {}
    if authorization.get("interface_version") == INTERFACE_VERSION:
        authorization["status"] = status
        authorization["actual_cost_usd"] = block_cost
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live_after["total_credits_usd"]),
        "live_total_usage_usd": float(live_after["total_usage_usd"]),
        "live_balance_usd": float(live_after["balance_usd"]),
        "posted_spend_since_opening_usd": posted,
        "locally_measured_spend_usd": local,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": max(0.0, 5.0 - recorded),
    }
    return updated


def _verify_previous_blocks(
    *,
    block_id: str,
    previous_results: Sequence[Path],
    all_development_tasks: Sequence[bed.VisualTask],
    protocol_manifest_sha256: str,
) -> None:
    required = BLOCK_ORDER[: BLOCK_ORDER.index(block_id)]
    if len(previous_results) != len(required):
        raise ValueError(f"block {block_id} requires previous blocks {required}")
    replays = [
        replay_block(
            result_path=path, all_development_tasks=all_development_tasks
        )
        for path in previous_results
    ]
    observed = {replay["block_id"] for replay in replays}
    if observed != set(required):
        raise ValueError("previous development blocks are incomplete or mismatched")
    if any(
        replay["protocol_manifest_sha256"] != protocol_manifest_sha256
        for replay in replays
    ):
        raise ValueError("previous blocks used a different protocol manifest")


def execute_block(
    *,
    output_dir: Path,
    run_id: str,
    block_id: str,
    mechanics_result: Path,
    protocol_manifest: Path,
    previous_results: Sequence[Path],
    ledger_path: Path,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    block_runner: Callable[..., dict[str, Any]] = run_block,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    all_tasks = bed.load_validation_partition_tasks(
        "development", include_endpoint_labels=False
    )
    manifest_verification = verify_protocol_manifest(protocol_manifest)
    _verify_previous_blocks(
        block_id=block_id,
        previous_results=previous_results,
        all_development_tasks=all_tasks,
        protocol_manifest_sha256=manifest_verification["manifest_sha256"],
    )
    mechanics_verification = verify_mechanics_result(mechanics_result)
    projection = (
        mechanics_verification["cost_usd"]
        / mechanics_verification["request_count"]
        * _max_requests(block_id)
        * 1.5
    )
    if projection > RUN_BUDGET_USD + 1e-12:
        raise RuntimeError(
            f"mechanics cost projects ${projection:.6f}, above block cap"
        )
    live = live_reader()
    ledger = _initialize_daily_ledger(
        path=ledger_path, live=live, block_id=block_id, now=now
    )
    _validate_daily_ledger(ledger, block_id=block_id, now=now)
    require_budget(
        ledger,
        projected_cost_usd=RUN_BUDGET_USD,
        total_usage_usd=live["total_usage_usd"],
        now=now,
    )
    if live["balance_usd"] + 1e-12 < RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below the development block cap")
    try:
        result = block_runner(
            output_dir=output_dir,
            run_id=run_id,
            block_id=block_id,
            mechanics_result=mechanics_result,
            protocol_manifest=protocol_manifest,
            all_development_tasks=all_tasks,
        )
    except Exception as exc:
        reconciliation_error = None
        try:
            reconciled = _reconcile_ledger(
                ledger=ledger,
                block_id=block_id,
                measured_cost_usd=0.0,
                live_after=live_reader(),
                status="failed_closed_posted_spend_reconciled",
            )
            checkpoint(ledger_path, reconciled)
        except Exception as reconciliation_exc:
            reconciliation_error = (
                f"{type(reconciliation_exc).__name__}: {reconciliation_exc}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint(
            output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "block_id": block_id,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ledger_reconciliation_error": reconciliation_error,
            },
        )
        raise
    measured = float(result["usage"]["run_cost_usd"])
    local = _reconcile_ledger(
        ledger=ledger,
        block_id=block_id,
        measured_cost_usd=measured,
        live_after=live,
        status=result["status"],
    )
    checkpoint(ledger_path, local)
    final = _reconcile_ledger(
        ledger=local,
        block_id=block_id,
        measured_cost_usd=0.0,
        live_after=live_reader(),
        status=result["status"],
    )
    checkpoint(ledger_path, final)
    checkpoint(
        output_dir / "EXECUTION.json",
        {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "block_id": block_id,
            "status": "complete_reconciled",
            "result_sha256": sha256_file(output_dir / "RESULT.json"),
            "ledger_sha256": sha256_file(ledger_path),
            "recorded_daily_spend_usd": final["recorded_actual_spend_usd"],
            "remaining_daily_allowance_usd": final["reconciliation"][
                "remaining_daily_allowance_usd"
            ],
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    block_parser = subparsers.add_parser("block")
    block_parser.add_argument("--block-id", choices=BLOCK_ORDER, required=True)
    block_parser.add_argument("--output-dir", type=Path, required=True)
    block_parser.add_argument("--run-id", required=True)
    block_parser.add_argument("--mechanics-result", type=Path, required=True)
    block_parser.add_argument("--protocol-manifest", type=Path, required=True)
    block_parser.add_argument("--previous-result", type=Path, action="append", default=[])
    block_parser.add_argument("--daily-ledger", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--block-result", type=Path, action="append", required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "block":
        result = execute_block(
            output_dir=args.output_dir,
            run_id=args.run_id,
            block_id=args.block_id,
            mechanics_result=args.mechanics_result,
            protocol_manifest=args.protocol_manifest,
            previous_results=args.previous_result,
            ledger_path=args.daily_ledger,
        )
        summary = {
            "status": result["status"],
            "block_id": args.block_id,
            "usage": result["usage"],
            "gates": result["gates"],
        }
    elif args.command == "analyze":
        result = analyze_combined(
            block_results=args.block_result,
            output_path=args.output,
        )
        summary = {
            "status": result["status"],
            "pooled_policy_metrics": result["pooled_policy_metrics"],
            "comparisons_vs_myopic": result["comparisons_vs_myopic"],
            "gates": result["gates"],
        }
    else:
        result = build_protocol_manifest(output_path=args.output)
        summary = {
            "status": result["status"],
            "blocks": result["blocks"],
            "gates": result["gates"],
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
