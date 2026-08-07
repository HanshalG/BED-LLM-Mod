#!/usr/bin/env python3
"""Run and replay the frozen Bongard-OpenWorld confirmation64 blocks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import bongard_openworld_luna_claim_report as claim_report
from scripts import bongard_openworld_luna_confirmation64_verify as freeze_verify
from scripts import bongard_openworld_luna_development32_daily_execute as daily
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-confirmation64-2"
MODEL_ID = development.MODEL_ID
BLOCK_ORDER = freeze_verify.BLOCK_ORDER
BLOCK_SIZES = {block_id: 16 for block_id in BLOCK_ORDER}
BLOCK_OFFSETS = {
    block_id: index * 16 for index, block_id in enumerate(BLOCK_ORDER)
}
BLOCK_DATES = dict(freeze_verify.BLOCK_DATES)
BLOCK_MODEL_SEEDS = dict(freeze_verify.BLOCK_SEEDS)
TASKS = 64
CASES_PER_TASK = development.CASES_PER_TASK
MAX_FINALS_PER_TASK = development.MAX_FINALS_PER_TASK
MAX_REQUESTS_PER_BLOCK = 688
CONCURRENCY = development.CONCURRENCY
RUN_BUDGET_USD = development.RUN_BUDGET_USD
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_081_901
MIN_CHANGED_FINAL_HISTORIES = 24
MIN_RELATIVE_BRIER_IMPROVEMENT = 0.03
PROTOCOL_MANIFEST = freeze_verify.MANIFEST
PROTOCOL_MANIFEST_SHA256 = freeze_verify.MANIFEST_SHA256
DEVELOPMENT_ROOT = freeze_verify.DEVELOPMENT_ROOT
DEVELOPMENT_COMBINED = DEVELOPMENT_ROOT / "COMBINED_RESULT.json"
DEVELOPMENT_CLAIM_REPORT = DEVELOPMENT_ROOT / "CLAIM_REPORT.json"
DEVELOPMENT_BLOCK_RESULTS = tuple(
    daily.BLOCK_DIRS[block_id] / "RESULT.json"
    for block_id in development.BLOCK_ORDER
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol_manifest(path: Path = PROTOCOL_MANIFEST) -> dict[str, Any]:
    verification = freeze_verify.verify_manifest(
        path,
        expected_sha256=PROTOCOL_MANIFEST_SHA256,
        require_unopened_predecessors=False,
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks") or []
    task_ids_by_block = {
        block_id: [
            row["task_id"] for row in tasks if row.get("block_id") == block_id
        ]
        for block_id in BLOCK_ORDER
    }
    if any(len(task_ids_by_block[block_id]) != 16 for block_id in BLOCK_ORDER):
        raise ValueError("confirmation manifest block task counts changed")
    return {**verification, "task_ids_by_block": task_ids_by_block}


def _development_claim_expected() -> dict[str, Any]:
    verification = daily.verify_combined_result(
        result_path=DEVELOPMENT_COMBINED,
        block_results=DEVELOPMENT_BLOCK_RESULTS,
    )
    result = json.loads(DEVELOPMENT_COMBINED.read_text(encoding="utf-8"))
    return claim_report.build_claim_report(
        result,
        result_sha256=development.sha256_file(DEVELOPMENT_COMBINED),
        independent_verification=verification,
    )


def verify_development_authorization() -> dict[str, Any]:
    if not DEVELOPMENT_CLAIM_REPORT.is_file():
        raise RuntimeError("development claim report is missing")
    expected = _development_claim_expected()
    observed = json.loads(DEVELOPMENT_CLAIM_REPORT.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("development claim report does not replay exactly")
    if (
        observed.get("interface_version") != claim_report.INTERFACE_VERSION
        or observed.get("claim_tier")
        != "full_path_dependent_llm_native_development_signal"
        or observed.get("authorizes_confirmation_preregistration") is not True
        or observed.get("confirmation_execution_remains_unauthorized") is not True
    ):
        raise RuntimeError("development claim tier does not authorize frozen confirmation")
    block_verifications = {
        block_id: daily.validate_block_result(
            path=daily.BLOCK_DIRS[block_id] / "RESULT.json",
            block_id=block_id,
            ledger_path=daily.LEDGERS[block_id],
        )
        for block_id in development.BLOCK_ORDER
    }
    return {
        "verified": True,
        "claim_tier": observed["claim_tier"],
        "claim_report_sha256": sha256_file(DEVELOPMENT_CLAIM_REPORT),
        "combined_result_sha256": sha256_file(DEVELOPMENT_COMBINED),
        "block_result_sha256s": {
            block_id: verification["result_sha256"]
            for block_id, verification in block_verifications.items()
        },
        "authorization_amendment_sha256": (
            freeze_verify.AUTHORIZATION_AMENDMENT_SHA256
        ),
    }


def confirmation_tasks_for_block(
    tasks: Sequence[bed.VisualTask], block_id: str
) -> list[bed.VisualTask]:
    if block_id not in BLOCK_ORDER:
        raise ValueError(f"unknown confirmation block {block_id!r}")
    ordered = sorted(tasks, key=lambda task: task.task_id)
    if len(ordered) != TASKS:
        raise ValueError("confirmation requires exactly 64 tasks")
    start = BLOCK_OFFSETS[block_id]
    return ordered[start : start + BLOCK_SIZES[block_id]]


def load_planning_tasks() -> list[bed.VisualTask]:
    return [
        development.seal_endpoint_labels(task)
        for task in bed.load_validation_partition_tasks(
            "confirmation", include_endpoint_labels=False
        )
    ]


def _block_gates(
    *,
    task_count: int,
    artifacts: Mapping[str, Any],
    usage: Mapping[str, Any],
    prompt_errors: Sequence[Sequence[str]],
    mechanics_verification: Mapping[str, Any],
) -> dict[str, bool]:
    final_count = len(artifacts["final_cases"])
    expected = task_count * CASES_PER_TASK + final_count
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
    )
    shuffled_exact = all(
        all(
            math.isclose(
                tree["continuation_values"]["shuffled_expected_future_eig"][target],
                tree["continuation_values"]["dynamic_expected_future_eig"][source],
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
        "exact_frozen_task_count": task_count == 16,
        "exact_expected_accepted_requests": usage.get("adapter_requests") == expected,
        "exact_expected_http_attempts": usage.get("http_attempts") == expected,
        "zero_retries": usage.get("retry_count") == 0,
        "zero_provider_error_retries": usage.get("provider_error_retries", 0) == 0,
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "all_responses_parse_and_scores_are_finite": finite,
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
            shuffled_exact
        ),
        "all_final_histories_generated_once_and_mapped": (
            task_count * 4 <= final_count <= task_count * MAX_FINALS_PER_TASK
        ),
        "all_prompts_hide_bound_source_truth": not any(prompt_errors),
        "endpoint_labels_remain_sealed": all(
            not (set(task.endpoint_ids) & set(task.actual_labels))
            for task in (
                case.task
                for case in artifacts["stage_cases"]
                if case.kind == "root"
            )
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
        openrouter_max_output_tokens=development.MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return serving.LunaVisionAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=1_000_000),
        config,
        request_seed=BLOCK_MODEL_SEEDS[block_id],
    )


def run_block(
    *,
    output_dir: Path,
    run_id: str,
    block_id: str,
    mechanics_result: Path,
    protocol_manifest: Path = PROTOCOL_MANIFEST,
    all_confirmation_tasks: Sequence[bed.VisualTask] | None = None,
    adapter: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = verify_protocol_manifest(protocol_manifest)
    development_authorization = verify_development_authorization()
    mechanics_verification = development.verify_mechanics_result(mechanics_result)
    source_tasks = list(all_confirmation_tasks or load_planning_tasks())
    tasks = [
        development.seal_endpoint_labels(task)
        for task in confirmation_tasks_for_block(source_tasks, block_id)
    ]
    if [task.task_id for task in tasks] != manifest["task_ids_by_block"][block_id]:
        raise ValueError("confirmation block task IDs do not match manifest")
    model = adapter or _adapter(
        output_dir=output_dir, run_id=run_id, block_id=block_id
    )
    stage_cases = development.first_stage_cases(tasks)
    stage_messages = [
        bed.build_belief_messages(case.task, case.history) for case in stage_cases
    ]
    stage_seeds = mechanics.request_seeds_for_cases(
        stage_cases, base_seed=BLOCK_MODEL_SEEDS[block_id]
    )
    paired = mechanics.paired_request_diagnostics(
        cases=stage_cases, messages=stage_messages, seeds=stage_seeds
    )
    if not paired["gates"]["all_pass"]:
        raise ValueError("confirmation paired history-blind request audit failed")
    stage_prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, messages)
        for case, messages in zip(stage_cases, stage_messages, strict=True)
    ]
    if any(stage_prompt_errors):
        raise ValueError("confirmation first-stage hidden-state audit failed")
    stage_responses = development._generate_checkpointed(
        model=model,
        cases=stage_cases,
        messages=stage_messages,
        seeds=stage_seeds,
        progress_path=output_dir / "private/FIRST_STAGE_PROGRESS.json",
    )
    roots, branches, history_blind = development._parse_stage(
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
        task.task_id: development._all_first_action_paths(
            task=task, branches=branches[task.task_id]
        )
        for task in tasks
    }
    final_cases = development._development_final_cases(
        tasks=tasks, plans=plans, action_paths=action_paths
    )
    final_messages = [
        bed.build_belief_messages(case.task, case.history) for case in final_cases
    ]
    final_seeds = mechanics.request_seeds_for_cases(
        final_cases, base_seed=BLOCK_MODEL_SEEDS[block_id], final_stage=True
    )
    final_batches = mechanics.task_preserving_dispatch_batches(final_cases)
    final_pairing = mechanics.final_request_diagnostics(
        cases=final_cases,
        seeds=final_seeds,
        plans=plans,
        dispatch_batches=final_batches,
    )
    if not final_pairing["gates"]["all_pass"]:
        raise ValueError("confirmation terminal CRN audit failed")
    final_prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, messages)
        for case, messages in zip(final_cases, final_messages, strict=True)
    ]
    if any(final_prompt_errors):
        raise ValueError("confirmation final hidden-state audit failed")
    final_responses = development._generate_checkpointed(
        model=model,
        cases=final_cases,
        messages=final_messages,
        seeds=final_seeds,
        progress_path=output_dir / "private/FINAL_STAGE_PROGRESS.json",
        dispatch_batches=final_batches,
    )
    artifacts = development._build_artifacts(
        tasks=tasks,
        stage_responses=stage_responses,
        final_responses=final_responses,
        base_seed=BLOCK_MODEL_SEEDS[block_id],
    )
    usage = summarize_usage(model.usage_snapshot())
    gates = _block_gates(
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
                development.message_sha256(messages) for messages in stage_messages
            ],
            "first_stage_request_seeds": stage_seeds,
            "first_stage_responses": list(stage_responses),
            "final_case_ids": [case.case_id for case in artifacts["final_cases"]],
            "final_responses": list(final_responses),
            "final_request_sha256": [
                development.message_sha256(messages) for messages in final_messages
            ],
            "final_request_seeds": final_seeds,
            "final_request_pairing": final_pairing,
            "candidate_labels_accessed_after_root_selection": True,
            "endpoint_labels_accessed": False,
            "intermediate_science_accessed": False,
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
            "protocol_manifest_sha256": manifest["manifest_sha256"],
            "development_claim_report_sha256": development_authorization[
                "claim_report_sha256"
            ],
            "authorization_amendment_sha256": (
                freeze_verify.AUTHORIZATION_AMENDMENT_SHA256
            ),
            "block_id": block_id,
            "block_size": 16,
            "block_offset": BLOCK_OFFSETS[block_id],
            "model": MODEL_ID,
            "model_seed": BLOCK_MODEL_SEEDS[block_id],
            "reasoning": False,
            "first_stage_requests": 16 * CASES_PER_TASK,
            "distinct_final_history_requests": len(artifacts["final_cases"]),
            "expected_total_requests": 16 * CASES_PER_TASK
            + len(artifacts["final_cases"]),
            "maximum_requests": MAX_REQUESTS_PER_BLOCK,
            "run_budget_usd": RUN_BUDGET_USD,
            "endpoint_labels_accessed": False,
            "intermediate_science_accessed": False,
            "sealed_test_accessed": False,
        },
        "development_authorization": development_authorization,
        "mechanics_verification": mechanics_verification,
        "protocol_manifest_verification": manifest,
        "usage": usage,
        "paired_request_diagnostics": paired,
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
    all_confirmation_tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    block_id = protocol.get("block_id")
    if (
        block_id not in BLOCK_ORDER
        or result.get("status") != "block_mechanics_pass"
        or protocol.get("interface_version") != INTERFACE_VERSION
        or protocol.get("protocol_manifest_sha256") != PROTOCOL_MANIFEST_SHA256
        or protocol.get("model") != MODEL_ID
        or protocol.get("endpoint_labels_accessed") is not False
        or protocol.get("intermediate_science_accessed") is not False
        or protocol.get("sealed_test_accessed") is not False
        or not result.get("gates")
        or not all(result["gates"].values())
    ):
        raise ValueError("confirmation block is not a clean endpoint-blind pass")
    manifest = verify_protocol_manifest()
    if result.get("protocol_manifest_verification") != manifest:
        raise ValueError("confirmation manifest verification changed")
    development_authorization = verify_development_authorization()
    if result.get("development_authorization") != development_authorization:
        raise ValueError("confirmation development authorization changed")
    expected_protocol = {
        "interface_version": INTERFACE_VERSION,
        "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        "development_claim_report_sha256": development_authorization[
            "claim_report_sha256"
        ],
        "authorization_amendment_sha256": (
            freeze_verify.AUTHORIZATION_AMENDMENT_SHA256
        ),
        "block_id": block_id,
        "block_size": 16,
        "block_offset": BLOCK_OFFSETS[block_id],
        "model": MODEL_ID,
        "model_seed": BLOCK_MODEL_SEEDS[block_id],
        "reasoning": False,
        "first_stage_requests": 16 * CASES_PER_TASK,
        "distinct_final_history_requests": protocol.get(
            "distinct_final_history_requests"
        ),
        "expected_total_requests": protocol.get("expected_total_requests"),
        "maximum_requests": MAX_REQUESTS_PER_BLOCK,
        "run_budget_usd": RUN_BUDGET_USD,
        "endpoint_labels_accessed": False,
        "intermediate_science_accessed": False,
        "sealed_test_accessed": False,
    }
    if protocol != expected_protocol:
        raise ValueError("confirmation protocol fields changed")
    raw_path = result_path.parent / "private/RAW_RESPONSES.json"
    if sha256_file(raw_path) != result.get("raw_responses_sha256"):
        raise ValueError("confirmation raw response hash changed")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    source_tasks = list(all_confirmation_tasks or load_planning_tasks())
    tasks = [
        development.seal_endpoint_labels(task)
        for task in confirmation_tasks_for_block(source_tasks, block_id)
    ]
    if [task.task_id for task in tasks] != manifest["task_ids_by_block"][block_id]:
        raise ValueError("confirmation replay task IDs changed")
    stage_cases = development.first_stage_cases(tasks)
    stage_messages = [
        bed.build_belief_messages(case.task, case.history) for case in stage_cases
    ]
    if raw.get("first_stage_case_ids") != [case.case_id for case in stage_cases]:
        raise ValueError("confirmation first-stage case order changed")
    if raw.get("first_stage_request_sha256") != [
        development.message_sha256(messages) for messages in stage_messages
    ]:
        raise ValueError("confirmation first-stage payload changed")
    stage_seeds = mechanics.request_seeds_for_cases(
        stage_cases, base_seed=BLOCK_MODEL_SEEDS[block_id]
    )
    if raw.get("first_stage_request_seeds") != stage_seeds:
        raise ValueError("confirmation first-stage seeds changed")
    artifacts = development._build_artifacts(
        tasks=tasks,
        stage_responses=raw.get("first_stage_responses") or [],
        final_responses=raw.get("final_responses") or [],
        base_seed=BLOCK_MODEL_SEEDS[block_id],
    )
    final_cases = artifacts["final_cases"]
    if raw.get("final_case_ids") != [case.case_id for case in final_cases]:
        raise ValueError("confirmation final case order changed")
    final_messages = [
        bed.build_belief_messages(case.task, case.history) for case in final_cases
    ]
    if raw.get("final_request_sha256") != [
        development.message_sha256(messages) for messages in final_messages
    ]:
        raise ValueError("confirmation final payload changed")
    final_seeds = mechanics.request_seeds_for_cases(
        final_cases, base_seed=BLOCK_MODEL_SEEDS[block_id], final_stage=True
    )
    if raw.get("final_request_seeds") != final_seeds:
        raise ValueError("confirmation final seeds changed")
    if (
        raw.get("final_request_pairing") != artifacts["final_request_pairing"]
        or result.get("final_request_pairing")
        != artifacts["final_request_pairing"]
    ):
        raise ValueError("confirmation terminal pairing changed")
    if bed.canonical_json(artifacts["trees"]) != bed.canonical_json(result["trees"]):
        raise ValueError("confirmation trees do not replay")
    if (
        raw.get("candidate_labels_accessed_after_root_selection") is not True
        or raw.get("endpoint_labels_accessed") is not False
        or raw.get("intermediate_science_accessed") is not False
        or raw.get("sealed_test_accessed") is not False
    ):
        raise ValueError("confirmation raw privacy boundary changed")
    prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, messages)
        for case, messages in zip(stage_cases, stage_messages, strict=True)
    ] + [
        bed.prompt_hidden_state_errors(case.task, case.history, messages)
        for case, messages in zip(final_cases, final_messages, strict=True)
    ]
    mechanics_verification = development.verify_mechanics_result(
        freeze_verify.MECHANICS_RESULT
    )
    if result.get("mechanics_verification") != mechanics_verification:
        raise ValueError("confirmation mechanics authorization changed")
    replayed_gates = _block_gates(
        task_count=len(tasks),
        artifacts=artifacts,
        usage=result.get("usage") or {},
        prompt_errors=prompt_errors,
        mechanics_verification=mechanics_verification,
    )
    if result.get("gates") != replayed_gates or not replayed_gates["all_pass"]:
        raise ValueError("confirmation block gates do not replay")
    final_count = len(final_cases)
    if (
        protocol["distinct_final_history_requests"] != final_count
        or protocol["expected_total_requests"] != 16 * CASES_PER_TASK + final_count
    ):
        raise ValueError("confirmation request accounting changed")
    return {
        "verified": True,
        "block_id": block_id,
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "result_sha256": sha256_file(result_path),
        "raw_responses_sha256": sha256_file(raw_path),
        "tasks": tasks,
        "artifacts": artifacts,
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


def paired_summary(values: Sequence[float], *, seed: int) -> dict[str, Any]:
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


def _build_scored_trees(
    *,
    replays: Sequence[Mapping[str, Any]],
    endpoint_tasks: Sequence[bed.VisualTask],
) -> list[dict[str, Any]]:
    full_by_id = {task.task_id: task for task in endpoint_tasks}
    trees = []
    for replay in replays:
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
                -action_rows[first]["endpoint"]["mean_brier"] for first in first_ids
            ]
            ranking = {
                score_name: development.spearman_correlation(
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
        raise ValueError("confirmation task coverage is not exact and unique")
    return trees


def analyze_combined(
    *,
    block_results: Sequence[Path],
    output_path: Path,
    all_confirmation_tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    planning_tasks = sorted(
        [development.seal_endpoint_labels(task) for task in all_confirmation_tasks]
        if all_confirmation_tasks is not None
        else load_planning_tasks(),
        key=lambda task: task.task_id,
    )
    if len(planning_tasks) != TASKS:
        raise ValueError("combined confirmation requires exactly 64 tasks")
    replays = [
        replay_block(result_path=path, all_confirmation_tasks=planning_tasks)
        for path in block_results
    ]
    by_block = {replay["block_id"]: replay for replay in replays}
    if set(by_block) != set(BLOCK_ORDER) or len(replays) != len(BLOCK_ORDER):
        raise ValueError("combined confirmation requires all four blocks")
    if {replay["protocol_manifest_sha256"] for replay in replays} != {
        PROTOCOL_MANIFEST_SHA256
    }:
        raise ValueError("confirmation blocks do not share the frozen manifest")
    endpoint_tasks = sorted(
        list(all_confirmation_tasks)
        if all_confirmation_tasks is not None
        else bed.load_validation_partition_tasks(
            "confirmation", include_endpoint_labels=True
        ),
        key=lambda task: task.task_id,
    )
    if len(endpoint_tasks) != TASKS or any(
        not set(task.endpoint_ids).issubset(task.actual_labels)
        for task in endpoint_tasks
    ):
        raise ValueError("confirmation endpoint labels are incomplete")
    trees = _build_scored_trees(replays=replays, endpoint_tasks=endpoint_tasks)
    pooled = mechanics._pooled_policy_metrics(trees)
    ranking = {
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
    comparisons = {}
    for policy_index, policy in enumerate(mechanics.POLICIES):
        if policy == "myopic_width":
            continue
        comparisons[policy] = {}
        for metric_index, metric in enumerate(("mean_brier", "mean_log_loss")):
            comparisons[policy][metric] = paired_summary(
                [
                    tree["policies"][policy]["endpoint"][metric]
                    - tree["policies"]["myopic_width"]["endpoint"][metric]
                    for tree in trees
                ],
                seed=BOOTSTRAP_SEED + policy_index * 10 + metric_index,
            )
    dynamic_vs_blind = {
        metric: paired_summary(
            [
                tree["policies"]["dynamic_depth2"]["endpoint"][metric]
                - tree["policies"]["history_blind_depth2"]["endpoint"][metric]
                for tree in trees
            ],
            seed=BOOTSTRAP_SEED + 1_000 + metric_index,
        )
        for metric_index, metric in enumerate(("mean_brier", "mean_log_loss"))
    }
    dynamic_vs_fixed = {
        metric: paired_summary(
            [
                tree["policies"]["dynamic_depth2"]["endpoint"][metric]
                - tree["policies"]["fixed_depth2"]["endpoint"][metric]
                for tree in trees
            ],
            seed=BOOTSTRAP_SEED + 2_000 + metric_index,
        )
        for metric_index, metric in enumerate(("mean_brier", "mean_log_loss"))
    }
    task_ids_by_block = {
        block_id: {task.task_id for task in by_block[block_id]["tasks"]}
        for block_id in BLOCK_ORDER
    }
    blockwise = {}
    for block_id in BLOCK_ORDER:
        block_trees = [
            tree for tree in trees if tree["task_id"] in task_ids_by_block[block_id]
        ]
        blockwise[block_id] = {
            "tasks": len(block_trees),
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
            "dynamic_minus_myopic_mean_brier": statistics.fmean(
                tree["policies"]["dynamic_depth2"]["endpoint"]["mean_brier"]
                - tree["policies"]["myopic_width"]["endpoint"]["mean_brier"]
                for tree in block_trees
            ),
            "dynamic_minus_history_blind_mean_brier": statistics.fmean(
                tree["policies"]["dynamic_depth2"]["endpoint"]["mean_brier"]
                - tree["policies"]["history_blind_depth2"]["endpoint"]["mean_brier"]
                for tree in block_trees
            ),
        }
    changed = sum(
        tree["policies"]["dynamic_depth2"]["final_history_key"]
        != tree["policies"]["myopic_width"]["final_history_key"]
        for tree in trees
    )
    robust_changed = sum(
        (
            dynamic_first := tree["policies"]["dynamic_depth2"]["first_image_id"]
        )
        != (myopic_first := tree["policies"]["myopic_width"]["first_image_id"])
        and tree["root_scores"]["dynamic_depth2"][dynamic_first]
        - tree["root_scores"]["dynamic_depth2"][myopic_first]
        >= mechanics.MIN_ACTION_MARGIN_NATS
        for tree in trees
    )
    blind_changed = sum(
        tree["policies"]["dynamic_depth2"]["final_history_key"]
        != tree["policies"]["history_blind_depth2"]["final_history_key"]
        for tree in trees
    )
    fixed_changed = sum(
        tree["policies"]["dynamic_depth2"]["final_history_key"]
        != tree["policies"]["fixed_depth2"]["final_history_key"]
        for tree in trees
    )
    robust_fixed_changed = sum(
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
    myopic_brier = pooled["myopic_width"]["mean_brier"]
    blind_brier = pooled["history_blind_depth2"]["mean_brier"]
    dynamic_brier = pooled["dynamic_depth2"]["mean_brier"]
    fixed_brier = pooled["fixed_depth2"]["mean_brier"]
    relative_gain = (
        (myopic_brier - dynamic_brier) / myopic_brier
        if myopic_brier > 0
        else -math.inf
    )
    blind_relative_gain = (
        (blind_brier - dynamic_brier) / blind_brier
        if blind_brier > 0
        else -math.inf
    )
    fixed_relative_gain = (
        (fixed_brier - dynamic_brier) / fixed_brier
        if fixed_brier > 0
        else -math.inf
    )
    mean_root_candidate_brier = statistics.fmean(
        tree["root_candidate_brier"] for tree in trees
    )
    dynamic_comparison = comparisons["dynamic_depth2"]
    gates = {
        "all_four_endpoint_blind_blocks_independently_replay": all(
            replay["verified"] for replay in replays
        ),
        "exact_64_disjoint_confirmation_tasks": len(trees) == TASKS,
        "root_candidate_brier_beats_constant_half": (
            mean_root_candidate_brier < 0.25
        ),
        "all_endpoint_metrics_are_finite": all(
            math.isfinite(value)
            for metrics in pooled.values()
            for value in metrics.values()
        ),
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
        "dynamic_score_has_positive_mean_endpoint_ranking_fidelity": (
            ranking["dynamic_depth2"]["mean_spearman"] > 0
        ),
        "dynamic_score_ranking_fidelity_is_not_worse_than_myopic": (
            ranking["dynamic_depth2"]["mean_spearman"]
            >= ranking["myopic_width"]["mean_spearman"]
        ),
        "dynamic_brier_relative_improvement_at_least_3_percent": (
            relative_gain >= MIN_RELATIVE_BRIER_IMPROVEMENT
        ),
        "dynamic_brier_paired_tree_bootstrap_95pct_upper_below_zero": (
            dynamic_comparison["mean_brier"]["ci95"][1] < 0
        ),
        "dynamic_log_loss_is_not_worse_than_myopic": (
            dynamic_comparison["mean_log_loss"]["mean_difference"] <= 0
        ),
        "at_least_24_dynamic_final_histories_differ_from_fixed_depth2": (
            fixed_changed >= MIN_CHANGED_FINAL_HISTORIES
        ),
        "at_least_24_dynamic_action_changes_from_fixed_clear_numerical_tie_margin": (
            robust_fixed_changed >= MIN_CHANGED_FINAL_HISTORIES
        ),
        "dynamic_and_fixed_depth2_differ_in_every_execution_block": all(
            row["dynamic_fixed_changed_final_histories"] >= 1
            for row in blockwise.values()
        ),
        "dynamic_brier_relative_improvement_vs_fixed_depth2_at_least_3_percent": (
            fixed_relative_gain >= MIN_RELATIVE_BRIER_IMPROVEMENT
        ),
        "dynamic_brier_vs_fixed_depth2_paired_tree_bootstrap_95pct_upper_below_zero": (
            dynamic_vs_fixed["mean_brier"]["ci95"][1] < 0
        ),
        "dynamic_log_loss_is_not_worse_than_fixed_depth2": (
            dynamic_vs_fixed["mean_log_loss"]["mean_difference"] <= 0
        ),
        "dynamic_ranking_fidelity_is_not_worse_than_fixed_depth2": (
            ranking["dynamic_depth2"]["mean_spearman"]
            >= ranking["fixed_depth2"]["mean_spearman"]
        ),
        "dynamic_brier_is_not_worse_than_shuffled_control": (
            dynamic_brier <= pooled["shuffled_dynamic_depth2"]["mean_brier"]
        ),
        "at_least_24_dynamic_final_histories_differ_from_history_blind": (
            blind_changed >= MIN_CHANGED_FINAL_HISTORIES
        ),
        "dynamic_and_history_blind_differ_in_every_execution_block": all(
            row["dynamic_history_blind_changed_final_histories"] >= 1
            for row in blockwise.values()
        ),
        "dynamic_brier_relative_improvement_vs_history_blind_at_least_3_percent": (
            blind_relative_gain >= MIN_RELATIVE_BRIER_IMPROVEMENT
        ),
        "dynamic_brier_vs_history_blind_paired_tree_bootstrap_95pct_upper_below_zero": (
            dynamic_vs_blind["mean_brier"]["ci95"][1] < 0
        ),
        "dynamic_log_loss_is_not_worse_than_history_blind": (
            dynamic_vs_blind["mean_log_loss"]["mean_difference"] <= 0
        ),
        "dynamic_ranking_fidelity_is_not_worse_than_history_blind": (
            ranking["dynamic_depth2"]["mean_spearman"]
            >= ranking["history_blind_depth2"]["mean_spearman"]
        ),
    }
    manifest_science_gates = json.loads(
        PROTOCOL_MANIFEST.read_text(encoding="utf-8")
    )["science_gates"]
    expected_gate_names = {
        name for family in manifest_science_gates.values() for name in family
    }
    if set(gates) != expected_gate_names:
        raise AssertionError("confirmation science gate names diverged from manifest")
    gates["all_pass"] = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "confirmation_pass" if gates["all_pass"] else "confirmation_null",
        "claim_tier": (
            "full_llm_native_confirmation" if gates["all_pass"] else "confirmation_null"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
            "blocks": list(BLOCK_ORDER),
            "task_count": TASKS,
            "model": MODEL_ID,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "endpoint_labels_accessed_only_after_all_blocks_replayed": True,
            "sealed_test_accessed": False,
            "reserve_accessed": False,
        },
        "block_verification": {
            replay["block_id"]: {
                key: replay[key]
                for key in (
                    "block_id",
                    "protocol_manifest_sha256",
                    "result_sha256",
                    "raw_responses_sha256",
                    "verified",
                )
            }
            for replay in replays
        },
        "pooled_policy_metrics": pooled,
        "mean_root_candidate_brier": mean_root_candidate_brier,
        "ranking_fidelity": ranking,
        "blockwise": blockwise,
        "comparisons_vs_myopic": comparisons,
        "dynamic_vs_history_blind": dynamic_vs_blind,
        "dynamic_vs_fixed_depth2": dynamic_vs_fixed,
        "dynamic_vs_myopic_changed_final_histories": changed,
        "dynamic_vs_myopic_robust_action_changes": robust_changed,
        "dynamic_vs_myopic_relative_brier_improvement": relative_gain,
        "dynamic_vs_history_blind_changed_final_histories": blind_changed,
        "dynamic_vs_history_blind_relative_brier_improvement": blind_relative_gain,
        "dynamic_vs_fixed_depth2_changed_final_histories": fixed_changed,
        "dynamic_vs_fixed_depth2_robust_action_changes": robust_fixed_changed,
        "dynamic_vs_fixed_depth2_relative_brier_improvement": fixed_relative_gain,
        "gates": gates,
        "sealed_test_authorized": False,
        "unregistered_model_swap_authorized": False,
        "trees": trees,
    }
    checkpoint(output_path, result)
    return result


def verify_combined_result(
    *,
    result_path: Path,
    block_results: Sequence[Path],
    all_confirmation_tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    observed = json.loads(result_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="bongard-confirmation-replay-") as tmp:
        replay = analyze_combined(
            block_results=block_results,
            output_path=Path(tmp) / "COMBINED_RESULT.json",
            all_confirmation_tasks=all_confirmation_tasks,
        )
    if bed.canonical_json(replay) != bed.canonical_json(observed):
        raise RuntimeError("combined confirmation does not independently replay")
    return {
        "verified": True,
        "status": observed["status"],
        "claim_tier": observed["claim_tier"],
        "result_sha256": sha256_file(result_path),
        "sealed_test_authorized": False,
    }
