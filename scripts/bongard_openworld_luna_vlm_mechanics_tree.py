#!/usr/bin/env python3
"""Run the shared four-task Luna semantic-belief mechanics tree."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-vlm-mechanics-tree-1"
MODEL_ID = serving.MODEL_ID
MODEL_SEED = 2_026_081_021
RANDOM_SEED = 2_026_081_022
ROOT_REQUESTS = 4
BRANCH_REQUESTS = 64
FIRST_STAGE_REQUESTS = ROOT_REQUESTS + BRANCH_REQUESTS
MAX_FINAL_REQUESTS = 20
MAX_REQUESTS = FIRST_STAGE_REQUESTS + MAX_FINAL_REQUESTS
CONCURRENCY = 24
MAX_TOKENS = serving.MAX_TOKENS
TEMPERATURE = 0.0
RUN_BUDGET_USD = 1.50
MIN_MATERIAL_BRANCH_PAIRS = 24
MIN_MYOPIC_BRIER = 0.03
MIN_MYOPIC_LOG_LOSS = 0.15
POLICIES = (
    "myopic_width",
    "fixed_depth2",
    "dynamic_depth2",
    "shuffled_dynamic_depth2",
    "random",
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

    def usage_snapshot(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BeliefCase:
    case_id: str
    task: bed.VisualTask
    history: tuple[tuple[str, bool], ...]
    kind: str
    candidate_id: str | None = None
    simulated_label: bool | None = None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_key(history: Sequence[tuple[str, bool]]) -> str:
    return ";".join(
        f"{image_id}:{int(label)}" for image_id, label in sorted(history)
    )


def first_stage_cases(tasks: Sequence[bed.VisualTask]) -> list[BeliefCase]:
    ordered = sorted(tasks, key=lambda task: task.task_id)
    if len(ordered) != ROOT_REQUESTS:
        raise ValueError("full mechanics tree requires exactly four tasks")
    roots = [
        BeliefCase(
            case_id=f"{task.task_id}-root",
            task=task,
            history=task.initial_history,
            kind="root",
        )
        for task in ordered
    ]
    branches = [
        BeliefCase(
            case_id=(
                f"{task.task_id}-{candidate_id}-"
                f"{'positive' if label else 'negative'}"
            ),
            task=task,
            history=tuple(
                sorted((*task.initial_history, (candidate_id, label)))
            ),
            kind="branch",
            candidate_id=candidate_id,
            simulated_label=label,
        )
        for task in ordered
        for candidate_id in task.candidate_ids
        for label in (False, True)
    ]
    cases = roots + branches
    if len(cases) != FIRST_STAGE_REQUESTS:
        raise AssertionError("first-stage request count changed")
    return cases


def rotate_branch_map(
    task: bed.VisualTask,
    branches: Mapping[tuple[str, bool], bed.SemanticBelief],
) -> tuple[
    dict[tuple[str, bool], bed.SemanticBelief],
    dict[str, str],
]:
    candidates = tuple(sorted(task.candidate_ids))
    mapping = {
        candidate: candidates[(index + 1) % len(candidates)]
        for index, candidate in enumerate(candidates)
    }
    if any(source == target for source, target in mapping.items()):
        raise AssertionError("shuffled branch mapping has a fixed point")
    return (
        {
            (candidate, label): branches[(mapping[candidate], label)]
            for candidate in candidates
            for label in (False, True)
        },
        mapping,
    )


def _random_choices(task: bed.VisualTask) -> tuple[str, str]:
    offset = int.from_bytes(
        hashlib.sha256(task.task_id.encode()).digest()[:8], "big"
    )
    rng = random.Random(RANDOM_SEED + offset)
    return tuple(rng.sample(list(task.candidate_ids), 2))  # type: ignore[return-value]


def plan_task_policies(
    *,
    task: bed.VisualTask,
    root: bed.SemanticBelief,
    branches: Mapping[tuple[str, bool], bed.SemanticBelief],
) -> dict[str, Any]:
    candidates = tuple(task.candidate_ids)
    myopic_scores = bed.candidate_eigs(root, candidates)
    fixed_scores = bed.fixed_support_depth_two_scores(root, candidates)
    dynamic_scores = bed.dynamic_support_depth_two_scores(
        root, candidates, branches
    )
    shuffled_branches, shuffled_mapping = rotate_branch_map(task, branches)
    shuffled_scores = bed.dynamic_support_depth_two_scores(
        root, candidates, shuffled_branches
    )
    first_by_policy = {
        "myopic_width": bed.select_best(myopic_scores),
        "fixed_depth2": bed.select_best(fixed_scores),
        "dynamic_depth2": bed.select_best(dynamic_scores),
        "shuffled_dynamic_depth2": bed.select_best(shuffled_scores),
    }
    random_first, random_second = _random_choices(task)
    first_by_policy["random"] = random_first

    policies = {}
    for policy in POLICIES:
        first = first_by_policy[policy]
        first_label = bool(task.actual_labels[first])
        remaining = tuple(
            candidate for candidate in candidates if candidate != first
        )
        if policy == "random":
            second = random_second
            second_scores = None
        elif policy == "fixed_depth2":
            fixed_weights = bed.updated_weights_for_label(
                root, first, first_label
            )
            second_scores = bed.candidate_eigs(
                root, remaining, weights=fixed_weights
            )
            second = bed.select_best(second_scores)
        else:
            realized_branch = branches[(first, first_label)]
            second_scores = bed.candidate_eigs(realized_branch, remaining)
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
        policies[policy] = {
            "first_image_id": first,
            "first_label": bed.LABELS[first_label],
            "second_image_id": second,
            "second_label": bed.LABELS[second_label],
            "final_history": final_history,
            "final_history_key": history_key(final_history),
            "first_score": (
                None
                if policy == "random"
                else {
                    "myopic_width": myopic_scores,
                    "fixed_depth2": fixed_scores,
                    "dynamic_depth2": dynamic_scores,
                    "shuffled_dynamic_depth2": shuffled_scores,
                }[policy][first]
            ),
            "second_scores": second_scores,
        }
    return {
        "root_scores": {
            "myopic_width": myopic_scores,
            "fixed_depth2": fixed_scores,
            "dynamic_depth2": dynamic_scores,
            "shuffled_dynamic_depth2": shuffled_scores,
        },
        "shuffled_branch_mapping": shuffled_mapping,
        "policies": policies,
    }


def final_cases(
    tasks: Sequence[bed.VisualTask],
    plans: Mapping[str, Mapping[str, Any]],
) -> list[BeliefCase]:
    cases = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        seen = set()
        for policy in POLICIES:
            policy_row = plans[task.task_id]["policies"][policy]
            key = policy_row["final_history_key"]
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                BeliefCase(
                    case_id=(
                        f"{task.task_id}-final-"
                        f"{hashlib.sha256(key.encode()).hexdigest()[:12]}"
                    ),
                    task=task,
                    history=policy_row["final_history"],
                    kind="final",
                )
            )
    if not len(tasks) <= len(cases) <= len(tasks) * len(POLICIES):
        raise ValueError("distinct final history count is outside task/policy bounds")
    return cases


def verify_serving_result(
    path: Path,
    *,
    tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    if (
        result.get("status") != "passed"
        or protocol.get("interface_version") != serving.INTERFACE_VERSION
        or protocol.get("model") != MODEL_ID
        or protocol.get("actual_candidate_labels_accessed") is not False
        or protocol.get("endpoint_labels_accessed") is not False
        or not result.get("gates")
        or not all(result["gates"].values())
    ):
        raise ValueError("serving result is not the frozen clean pass")
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    if sha256_file(raw_path) != result.get("raw_responses_sha256"):
        raise ValueError("serving raw response hash changed")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    cases = serving.build_smoke_cases(tasks or bed.load_mechanics_tasks())
    if raw.get("case_ids") != [case.case_id for case in cases]:
        raise ValueError("serving raw case order changed")
    responses = raw.get("responses") or []
    if len(responses) != serving.EXPECTED_REQUESTS:
        raise ValueError("serving raw response count changed")
    messages = [
        bed.build_belief_messages(case.task, case.history) for case in cases
    ]
    prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, message)
        for case, message in zip(cases, messages, strict=True)
    ]
    beliefs = [
        bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        for case, response in zip(cases, responses, strict=True)
    ]
    metrics = serving.serving_metrics(cases, beliefs)
    gates = serving.serving_gates(
        cases=cases,
        beliefs=beliefs,
        metrics=metrics,
        prompt_errors=prompt_errors,
        usage=result["usage"],
    )
    if bed.canonical_json(metrics) != bed.canonical_json(result.get("metrics")):
        raise ValueError("serving metrics do not replay")
    if gates != result.get("gates") or not gates["all_pass"]:
        raise ValueError("serving gates do not replay")
    return {
        "result_sha256": sha256_file(path),
        "raw_responses_sha256": sha256_file(raw_path),
        "cost_usd": float(result["usage"]["run_cost_usd"]),
        "verified": True,
    }


def _branch_diagnostics(
    *,
    task: bed.VisualTask,
    branches: Mapping[tuple[str, bool], bed.SemanticBelief],
) -> list[dict[str, Any]]:
    return [
        serving.branch_sensitivity(
            branches[(candidate, False)],
            branches[(candidate, True)],
            candidate_id=candidate,
        )
        for candidate in task.candidate_ids
    ]


def _pooled_policy_metrics(trees: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pooled = {}
    for policy in POLICIES:
        rows = [tree["policies"][policy]["endpoint"] for tree in trees]
        pooled[policy] = {
            "mean_brier": sum(row["mean_brier"] for row in rows) / len(rows),
            "mean_log_loss": sum(row["mean_log_loss"] for row in rows) / len(rows),
            "mean_accuracy": sum(row["accuracy"] for row in rows) / len(rows),
            "mean_truth_probability": sum(
                row["mean_truth_probability"] for row in rows
            )
            / len(rows),
        }
    return pooled


def mechanics_gates(
    *,
    trees: Sequence[dict[str, Any]],
    branch_diagnostics: Sequence[dict[str, Any]],
    final_case_count: int,
    usage: Mapping[str, Any],
    prompt_errors: Sequence[Sequence[str]],
    serving_verification: Mapping[str, Any],
) -> dict[str, bool]:
    expected_requests = FIRST_STAGE_REQUESTS + final_case_count
    pooled = _pooled_policy_metrics(trees)
    dynamic_changes = sum(
        tree["policies"]["dynamic_depth2"]["first_image_id"]
        != tree["policies"]["myopic_width"]["first_image_id"]
        for tree in trees
    )
    distinct_control_policies = sum(
        any(
            tree["policies"][policy]["final_history_key"]
            != tree["policies"]["myopic_width"]["final_history_key"]
            for tree in trees
        )
        for policy in POLICIES
        if policy != "myopic_width"
    )
    finite_scores = all(
        math.isfinite(value)
        for tree in trees
        for score_map in tree["root_scores"].values()
        for value in score_map.values()
    ) and all(
        row["second_scores"] is None
        or all(math.isfinite(value) for value in row["second_scores"].values())
        for tree in trees
        for row in tree["policies"].values()
    )
    finite_endpoints = all(
        math.isfinite(value)
        for values in pooled.values()
        for value in values.values()
    )
    gates = {
        "serving_result_independently_replays": serving_verification.get("verified") is True,
        "exact_expected_accepted_requests": usage.get("adapter_requests") == expected_requests,
        "exact_expected_http_attempts": usage.get("http_attempts") == expected_requests,
        "zero_retries": usage.get("retry_count") == 0,
        "zero_provider_error_retries": usage.get("provider_error_retries", 0) == 0,
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "all_root_branch_and_final_responses_parse": (
            len(trees) == ROOT_REQUESTS
            and len(branch_diagnostics) == BRANCH_REQUESTS // 2
            and 4 <= final_case_count <= MAX_FINAL_REQUESTS
        ),
        "all_scores_are_finite_and_executable": finite_scores,
        "at_least_24_of_32_branch_pairs_are_materially_label_sensitive": sum(
            row["material"] for row in branch_diagnostics
        )
        >= MIN_MATERIAL_BRANCH_PAIRS,
        "dynamic_depth2_changes_at_least_one_myopic_first_action": dynamic_changes >= 1,
        "at_least_two_controls_have_a_distinct_final_history": distinct_control_policies >= 2,
        "all_distinct_final_supports_generated_once_and_mapped": 4 <= final_case_count <= 20,
        "myopic_endpoint_is_not_saturated": (
            pooled["myopic_width"]["mean_brier"] >= MIN_MYOPIC_BRIER
            or pooled["myopic_width"]["mean_log_loss"] >= MIN_MYOPIC_LOG_LOSS
        ),
        "all_endpoint_metrics_are_finite": finite_endpoints,
        "all_prompts_hide_bound_source_truth": not any(prompt_errors),
        "cost_at_most_1_50": float(usage.get("run_cost_usd", math.inf)) <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _adapter(*, output_dir: Path, run_id: str) -> serving.LunaVisionAdapter:
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
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return serving.LunaVisionAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=1_000_000),
        config,
        request_seed=MODEL_SEED,
    )


def run_mechanics(
    *,
    output_dir: Path,
    run_id: str,
    serving_result: Path,
    tasks: Sequence[bed.VisualTask] | None = None,
    adapter: StructuredModel | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(tasks or bed.load_mechanics_tasks(), key=lambda task: task.task_id)
    serving_verification = verify_serving_result(serving_result, tasks=tasks)
    model = adapter or _adapter(output_dir=output_dir, run_id=run_id)

    stage_cases = first_stage_cases(tasks)
    stage_messages = [
        bed.build_belief_messages(case.task, case.history)
        for case in stage_cases
    ]
    stage_prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, messages)
        for case, messages in zip(stage_cases, stage_messages, strict=True)
    ]
    if any(stage_prompt_errors):
        raise ValueError("first-stage hidden-state prompt audit failed")
    stage_responses = model.chat_complete_messages_batched_structured(
        stage_messages,
        temperature=TEMPERATURE,
        block_size=len(stage_messages),
        response_format=bed.belief_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    stage_beliefs = [
        bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        for case, response in zip(stage_cases, stage_responses, strict=True)
    ]
    roots = {
        case.task.task_id: belief
        for case, belief in zip(stage_cases, stage_beliefs, strict=True)
        if case.kind == "root"
    }
    branches_by_task: dict[
        str, dict[tuple[str, bool], bed.SemanticBelief]
    ] = {task.task_id: {} for task in tasks}
    for case, belief in zip(stage_cases, stage_beliefs, strict=True):
        if case.kind == "branch":
            branches_by_task[case.task.task_id][
                (str(case.candidate_id), bool(case.simulated_label))
            ] = belief

    plans = {
        task.task_id: plan_task_policies(
            task=task,
            root=roots[task.task_id],
            branches=branches_by_task[task.task_id],
        )
        for task in tasks
    }
    endpoint_accessed_after_selection = True
    selected_final_cases = final_cases(tasks, plans)
    selected_final_messages = [
        bed.build_belief_messages(case.task, case.history)
        for case in selected_final_cases
    ]
    final_prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, messages)
        for case, messages in zip(
            selected_final_cases, selected_final_messages, strict=True
        )
    ]
    if any(final_prompt_errors):
        raise ValueError("final hidden-state prompt audit failed")
    final_responses = model.chat_complete_messages_batched_structured(
        selected_final_messages,
        temperature=TEMPERATURE,
        block_size=len(selected_final_messages),
        response_format=bed.belief_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    final_beliefs = [
        bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        for case, response in zip(
            selected_final_cases, final_responses, strict=True
        )
    ]
    final_by_task_history = {
        (case.task.task_id, history_key(case.history)): belief
        for case, belief in zip(
            selected_final_cases, final_beliefs, strict=True
        )
    }

    trees = []
    all_branch_diagnostics = []
    for task in tasks:
        task_plan = plans[task.task_id]
        diagnostics = _branch_diagnostics(
            task=task, branches=branches_by_task[task.task_id]
        )
        all_branch_diagnostics.extend(diagnostics)
        policy_rows = {}
        endpoint_labels = {
            image_id: bool(task.actual_labels[image_id])
            for image_id in task.endpoint_ids
        }
        for policy in POLICIES:
            plan = task_plan["policies"][policy]
            final_belief = final_by_task_history[
                (task.task_id, plan["final_history_key"])
            ]
            policy_rows[policy] = {
                **{
                    key: value
                    for key, value in plan.items()
                    if key != "final_history"
                },
                "endpoint": bed.endpoint_metrics(
                    final_belief, endpoint_labels
                ),
                "final_belief": bed.public_belief_summary(final_belief),
            }
        trees.append(
            {
                "task_id": task.task_id,
                "root_scores": task_plan["root_scores"],
                "shuffled_branch_mapping": task_plan[
                    "shuffled_branch_mapping"
                ],
                "root_belief": bed.public_belief_summary(
                    roots[task.task_id]
                ),
                "branch_diagnostics": diagnostics,
                "policies": policy_rows,
            }
        )

    usage = summarize_usage(model.usage_snapshot())
    gates = mechanics_gates(
        trees=trees,
        branch_diagnostics=all_branch_diagnostics,
        final_case_count=len(selected_final_cases),
        usage=usage,
        prompt_errors=[*stage_prompt_errors, *final_prompt_errors],
        serving_verification=serving_verification,
    )
    raw_path = output_dir / "private/RAW_RESPONSES.json"
    checkpoint(
        raw_path,
        {
            "first_stage_case_ids": [case.case_id for case in stage_cases],
            "first_stage_responses": stage_responses,
            "final_case_ids": [case.case_id for case in selected_final_cases],
            "final_responses": final_responses,
            "actual_candidate_labels_accessed_after_root_selection": True,
            "endpoint_labels_accessed_after_all_query_selection": (
                endpoint_accessed_after_selection
            ),
            "development_accessed": False,
            "confirmation_accessed": False,
            "sealed_test_accessed": False,
        },
    )
    pooled = _pooled_policy_metrics(trees)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "mechanics_pass" if gates["all_pass"] else "gated_null",
        "scientific_opportunity_status": "untested",
        "authorizes_development": False,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_VLM_MECHANICS_PREREGISTRATION.md"
            ),
            "amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_FULL_MECHANICS_AMENDMENT.md"
            ),
            "model": MODEL_ID,
            "model_seed": MODEL_SEED,
            "random_seed": RANDOM_SEED,
            "reasoning": False,
            "first_stage_requests": FIRST_STAGE_REQUESTS,
            "distinct_final_history_requests": len(selected_final_cases),
            "expected_total_requests": (
                FIRST_STAGE_REQUESTS + len(selected_final_cases)
            ),
            "run_budget_usd": RUN_BUDGET_USD,
            "development_accessed": False,
            "confirmation_accessed": False,
            "sealed_test_accessed": False,
        },
        "serving_verification": serving_verification,
        "usage": usage,
        "pooled_policy_metrics": pooled,
        "comparisons_vs_myopic": {
            policy: {
                "brier_difference": (
                    pooled[policy]["mean_brier"]
                    - pooled["myopic_width"]["mean_brier"]
                ),
                "log_loss_difference": (
                    pooled[policy]["mean_log_loss"]
                    - pooled["myopic_width"]["mean_log_loss"]
                ),
            }
            for policy in POLICIES
            if policy != "myopic_width"
        },
        "gates": gates,
        "trees": trees,
        "raw_responses_sha256": sha256_file(raw_path),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def validate_daily_ledger(
    ledger: Mapping[str, Any], *, now: datetime | None = None
) -> None:
    timezone = ZoneInfo(serving.TIMEZONE)
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    if ledger.get("date") != local_now.date().isoformat():
        raise RuntimeError("full mechanics requires the current daily ledger")
    if ledger.get("timezone") != serving.TIMEZONE:
        raise RuntimeError("daily ledger timezone changed")
    if float(ledger.get("daily_cap_usd", 0.0)) != 5.0:
        raise RuntimeError("daily ledger no longer has the exact $5 cap")
    smoke = ledger.get("bongard_luna_vlm_serving_smoke") or {}
    if (
        smoke.get("status") != "passed"
        or smoke.get("interface_version") != serving.INTERFACE_VERSION
        or smoke.get("model") != MODEL_ID
    ):
        raise RuntimeError("daily ledger lacks the passed Luna serving gate")
    if "bongard_luna_vlm_mechanics_tree" in ledger:
        raise RuntimeError("full mechanics tree is already recorded")


def reconcile_ledger(
    *,
    ledger: Mapping[str, Any],
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
    previous_block = float(
        (updated.get("bongard_luna_vlm_mechanics_tree") or {}).get(
            "actual_cost_usd", 0.0
        )
    )
    block_cost = max(previous_block, measured_cost_usd)
    updated["recorded_actual_spend_usd"] = recorded
    updated["bongard_luna_vlm_mechanics_tree"] = {
        "status": status,
        "actual_cost_usd": block_cost,
        "maximum_cost_usd": RUN_BUDGET_USD,
        "interface_version": INTERFACE_VERSION,
        "model": MODEL_ID,
    }
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


def execute_mechanics(
    *,
    output_dir: Path,
    run_id: str,
    serving_result: Path,
    ledger_path: Path,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    mechanics_runner: Callable[..., dict[str, Any]] = run_mechanics,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    validate_daily_ledger(ledger, now=now)
    verification = verify_serving_result(serving_result)
    observed_projection = (
        verification["cost_usd"]
        / serving.EXPECTED_REQUESTS
        * MAX_REQUESTS
        * 1.5
    )
    if observed_projection > RUN_BUDGET_USD + 1e-12:
        raise RuntimeError(
            f"serving cost projects ${observed_projection:.6f}, above full cap"
        )
    live = live_reader()
    require_budget(
        ledger,
        projected_cost_usd=RUN_BUDGET_USD,
        total_usage_usd=live["total_usage_usd"],
        now=now,
    )
    if live["balance_usd"] + 1e-12 < RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below the full mechanics cap")
    try:
        result = mechanics_runner(
            output_dir=output_dir,
            run_id=run_id,
            serving_result=serving_result,
        )
    except Exception as exc:
        reconciliation_error = None
        try:
            reconciled = reconcile_ledger(
                ledger=ledger,
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
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ledger_reconciliation_error": reconciliation_error,
            },
        )
        raise
    measured = float(result["usage"]["run_cost_usd"])
    local = reconcile_ledger(
        ledger=ledger,
        measured_cost_usd=measured,
        live_after=live,
        status=result["status"],
    )
    checkpoint(ledger_path, local)
    final = reconcile_ledger(
        ledger=local,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--serving-result", type=Path, required=True)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    args = parser.parse_args()
    result = execute_mechanics(
        output_dir=args.output_dir,
        run_id=args.run_id,
        serving_result=args.serving_result,
        ledger_path=args.daily_ledger,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "usage": result["usage"],
                "pooled_policy_metrics": result["pooled_policy_metrics"],
                "comparisons_vs_myopic": result["comparisons_vs_myopic"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
