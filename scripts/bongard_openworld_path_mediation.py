#!/usr/bin/env python3
"""Replay and report Bongard's belief-to-action-to-endpoint mediation chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_dinov2_outcome as stage_outcome
from scripts import bongard_openworld_luna_confirmation64 as confirmation
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-path-mediation-1"
BOOTSTRAP_SEED = 20262809
STAGES = stage_outcome.STAGES


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_scores(scores: Mapping[str, Any]) -> str:
    return bed.canonical_json({str(key): float(value) for key, value in scores.items()})


def _mean_absolute_difference(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("prediction vectors must be nonempty and equally sized")
    return statistics.fmean(abs(a - b) for a, b in zip(left, right, strict=True))


def _rule_jaccard(
    left: bed.SemanticBelief, right: bed.SemanticBelief
) -> float:
    left_rules = {bed.canonical_rule(row.rule) for row in left.hypotheses}
    right_rules = {bed.canonical_rule(row.rule) for row in right.hypotheses}
    union = left_rules | right_rules
    if not union:
        raise ValueError("belief supports contain no rules")
    return len(left_rules & right_rules) / len(union)


def task_mediation_row(
    *,
    task: bed.VisualTask,
    tree: Mapping[str, Any],
    dynamic_branches: Mapping[tuple[str, bool], bed.SemanticBelief],
    history_blind_branches: Mapping[tuple[str, bool], bed.SemanticBelief],
) -> dict[str, Any]:
    policies = tree.get("policies")
    if not isinstance(policies, Mapping):
        raise ValueError("scored tree lacks policies")
    dynamic = policies.get("dynamic_depth2")
    blind_policy = policies.get("history_blind_update_matched_first")
    if not isinstance(dynamic, Mapping) or not isinstance(blind_policy, Mapping):
        raise ValueError("scored tree lacks matched mediation policies")
    first = dynamic.get("first_image_id")
    label_by_name = {name: label for label, name in bed.LABELS.items()}
    first_label = label_by_name.get(dynamic.get("first_label"))
    if (
        not isinstance(first, str)
        or first not in task.candidate_ids
        or not isinstance(first_label, bool)
        or blind_policy.get("first_image_id") != first
        or blind_policy.get("first_label") != dynamic.get("first_label")
    ):
        raise ValueError("matched mediation policies do not share first query and label")
    key = (first, first_label)
    if key not in dynamic_branches or key not in history_blind_branches:
        raise ValueError("realized dynamic or history-blind branch is missing")
    dynamic_belief = dynamic_branches[key]
    blind_belief = history_blind_branches[key]
    blind_weights = bed.updated_weights_for_label(
        blind_belief, first, first_label
    )
    remaining = tuple(image_id for image_id in task.candidate_ids if image_id != first)
    dynamic_scores = bed.candidate_endpoint_eigs(
        dynamic_belief, remaining, task.endpoint_ids
    )
    blind_scores = bed.candidate_endpoint_eigs(
        blind_belief,
        remaining,
        task.endpoint_ids,
        weights=blind_weights,
    )
    if (
        not isinstance(dynamic.get("second_scores"), Mapping)
        or not isinstance(blind_policy.get("second_scores"), Mapping)
        or _canonical_scores(dynamic_scores)
        != _canonical_scores(dynamic["second_scores"])
        or _canonical_scores(blind_scores)
        != _canonical_scores(blind_policy["second_scores"])
    ):
        raise ValueError("replayed second-query score maps changed")
    dynamic_second = dynamic.get("second_image_id")
    blind_second = blind_policy.get("second_image_id")
    if (
        dynamic_second != bed.select_best(dynamic_scores)
        or blind_second != bed.select_best(blind_scores)
    ):
        raise ValueError("stored second action is not the replayed argmax")
    candidate_dynamic = [
        bed.predictive_probability(dynamic_belief, image_id)
        for image_id in remaining
    ]
    candidate_blind = [
        bed.predictive_probability(blind_belief, image_id, weights=blind_weights)
        for image_id in remaining
    ]
    endpoint_dynamic = [
        bed.predictive_probability(dynamic_belief, image_id)
        for image_id in task.endpoint_ids
    ]
    endpoint_blind = [
        bed.predictive_probability(blind_belief, image_id, weights=blind_weights)
        for image_id in task.endpoint_ids
    ]
    dynamic_endpoint = dynamic.get("endpoint")
    blind_endpoint = blind_policy.get("endpoint")
    if not isinstance(dynamic_endpoint, Mapping) or not isinstance(
        blind_endpoint, Mapping
    ):
        raise ValueError("scored policies lack endpoint metrics")
    brier_difference = float(dynamic_endpoint["mean_brier"]) - float(
        blind_endpoint["mean_brier"]
    )
    log_loss_difference = float(dynamic_endpoint["mean_log_loss"]) - float(
        blind_endpoint["mean_log_loss"]
    )
    action_changed = dynamic_second != blind_second
    dynamic_gap = (
        float(dynamic_scores[dynamic_second]) - float(dynamic_scores[blind_second])
        if action_changed
        else 0.0
    )
    blind_gap = (
        float(blind_scores[blind_second]) - float(blind_scores[dynamic_second])
        if action_changed
        else 0.0
    )
    robust_change = (
        action_changed
        and float(dynamic.get("second_score_margin", -math.inf))
        >= mechanics.MIN_ACTION_MARGIN_NATS
        and float(blind_policy.get("second_score_margin", -math.inf))
        >= mechanics.MIN_ACTION_MARGIN_NATS
    )
    values = [
        *dynamic_scores.values(),
        *blind_scores.values(),
        brier_difference,
        log_loss_difference,
        dynamic_gap,
        blind_gap,
    ]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("mediation row contains non-finite values")
    return {
        "task_id": task.task_id,
        "first_image_id": first,
        "first_label": dynamic["first_label"],
        "dynamic_second_image_id": dynamic_second,
        "history_blind_second_image_id": blind_second,
        "second_action_changed": action_changed,
        "robust_second_action_changed": robust_change,
        "dynamic_support_rule_jaccard_vs_history_blind": _rule_jaccard(
            dynamic_belief, blind_belief
        ),
        "candidate_predictive_probability_mae": _mean_absolute_difference(
            candidate_dynamic, candidate_blind
        ),
        "endpoint_predictive_probability_mae": _mean_absolute_difference(
            endpoint_dynamic, endpoint_blind
        ),
        "second_query_score_spearman": development.spearman_correlation(
            [dynamic_scores[image_id] for image_id in remaining],
            [blind_scores[image_id] for image_id in remaining],
        ),
        "dynamic_support_preference_for_dynamic_action_nats": dynamic_gap,
        "history_blind_support_preference_for_blind_action_nats": blind_gap,
        "both_supports_robustly_prefer_own_action": (
            action_changed
            and dynamic_gap >= mechanics.MIN_ACTION_MARGIN_NATS
            and blind_gap >= mechanics.MIN_ACTION_MARGIN_NATS
        ),
        "dynamic_minus_history_blind_endpoint_brier": brier_difference,
        "dynamic_minus_history_blind_endpoint_log_loss": log_loss_difference,
    }


def _optional_paired_summary(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "status": "not_estimable_no_selected_tasks"}
    return development.paired_summary(values, seed=seed)


def build_report(
    *,
    stage: str,
    stage_result: Mapping[str, Any],
    contexts: Mapping[
        str,
        tuple[
            bed.VisualTask,
            Mapping[tuple[str, bool], bed.SemanticBelief],
            Mapping[tuple[str, bool], bed.SemanticBelief],
        ],
    ],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    trees = stage_result.get("trees")
    expected = stage_outcome.STAGE_TASK_COUNTS[stage]
    if (
        authorization.get("verified") is not True
        or not isinstance(trees, list)
        or len(trees) != expected
        or len(contexts) != expected
        or {str(tree.get("task_id")) for tree in trees} != set(contexts)
    ):
        raise ValueError("mediation inputs are incomplete or unauthorized")
    tree_by_task = {str(tree["task_id"]): tree for tree in trees}
    rows = []
    for task_id in sorted(contexts):
        task, dynamic_branches, blind_branches = contexts[task_id]
        rows.append(
            task_mediation_row(
                task=task,
                tree=tree_by_task[task_id],
                dynamic_branches=dynamic_branches,
                history_blind_branches=blind_branches,
            )
        )
    changed = [row for row in rows if row["second_action_changed"]]
    robust = [row for row in rows if row["robust_second_action_changed"]]
    endpoint_shifts = [
        float(row["endpoint_predictive_probability_mae"]) for row in rows
    ]
    realized_benefits = [
        -float(row["dynamic_minus_history_blind_endpoint_brier"]) for row in rows
    ]
    dynamic_gaps = [
        float(row["dynamic_support_preference_for_dynamic_action_nats"])
        for row in rows
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "path_mediation_complete",
        "stage": stage,
        "stage_status": stage_result.get("status"),
        "stage_claim_tier": stage_result.get("claim_tier"),
        "stage_authorization": dict(authorization),
        "task_count": len(rows),
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "changes_claim_tier": False,
        "interpretation_scope": (
            "Descriptive mediation only: replayed answer-conditioned versus "
            "same-seed history-blind intermediate beliefs, their second-query "
            "choices, and already-opened endpoint metrics."
        ),
        "summary": {
            "second_action_changed": len(changed),
            "robust_second_action_changed": len(robust),
            "both_supports_robustly_prefer_own_action": sum(
                row["both_supports_robustly_prefer_own_action"] for row in rows
            ),
            "mean_rule_jaccard": statistics.fmean(
                row["dynamic_support_rule_jaccard_vs_history_blind"] for row in rows
            ),
            "mean_candidate_predictive_probability_mae": statistics.fmean(
                row["candidate_predictive_probability_mae"] for row in rows
            ),
            "mean_endpoint_predictive_probability_mae": statistics.fmean(
                endpoint_shifts
            ),
            "mean_second_query_score_spearman": statistics.fmean(
                row["second_query_score_spearman"] for row in rows
            ),
            "endpoint_shift_vs_realized_brier_benefit_spearman": (
                development.spearman_correlation(endpoint_shifts, realized_benefits)
            ),
            "dynamic_action_gap_vs_realized_brier_benefit_spearman": (
                development.spearman_correlation(dynamic_gaps, realized_benefits)
            ),
        },
        "endpoint_effects": {
            subset: {
                "dynamic_minus_history_blind_brier": _optional_paired_summary(
                    [
                        row["dynamic_minus_history_blind_endpoint_brier"]
                        for row in selected
                    ],
                    seed=BOOTSTRAP_SEED + subset_index * 10,
                ),
                "dynamic_minus_history_blind_log_loss": _optional_paired_summary(
                    [
                        row["dynamic_minus_history_blind_endpoint_log_loss"]
                        for row in selected
                    ],
                    seed=BOOTSTRAP_SEED + subset_index * 10 + 1,
                ),
            }
            for subset_index, (subset, selected) in enumerate(
                (("all_tasks", rows), ("changed_actions", changed), ("robust_changes", robust))
            )
        },
        "rows": rows,
    }


def _belief_contexts_from_cases(
    *,
    cases: Sequence[mechanics.BeliefCase],
    responses: Sequence[str],
) -> dict[
    str,
    tuple[
        bed.VisualTask,
        Mapping[tuple[str, bool], bed.SemanticBelief],
        Mapping[tuple[str, bool], bed.SemanticBelief],
    ],
]:
    if len(cases) != len(responses):
        raise ValueError("first-stage case and response counts differ")
    tasks: dict[str, bed.VisualTask] = {}
    dynamic: dict[str, dict[tuple[str, bool], bed.SemanticBelief]] = {}
    blind: dict[str, dict[tuple[str, bool], bed.SemanticBelief]] = {}
    for case, response in zip(cases, responses, strict=True):
        tasks[case.task.task_id] = case.task
        dynamic.setdefault(case.task.task_id, {})
        blind.setdefault(case.task.task_id, {})
        if case.kind not in {"branch", "history_blind"}:
            continue
        belief = bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        key = (str(case.candidate_id), bool(case.simulated_label))
        target = dynamic if case.kind == "branch" else blind
        target[case.task.task_id][key] = belief
    return {
        task_id: (tasks[task_id], dynamic[task_id], blind[task_id])
        for task_id in sorted(tasks)
    }


def replay_contexts(
    *, stage: str, result_path: Path, block_results: Sequence[Path]
) -> dict[
    str,
    tuple[
        bed.VisualTask,
        Mapping[tuple[str, bool], bed.SemanticBelief],
        Mapping[tuple[str, bool], bed.SemanticBelief],
    ],
]:
    if stage == "mechanics":
        result = _load(result_path)
        raw_path = result_path.parent / "private/RAW_RESPONSES.json"
        if mechanics.sha256_file(raw_path) != result.get("raw_responses_sha256"):
            raise ValueError("mechanics raw response hash changed")
        raw = _load(raw_path)
        tasks = sorted(bed.load_mechanics_tasks(), key=lambda task: task.task_id)
        cases = mechanics.first_stage_cases(tasks)
        if raw.get("first_stage_case_ids") != [case.case_id for case in cases]:
            raise ValueError("mechanics first-stage case order changed")
        return _belief_contexts_from_cases(
            cases=cases, responses=raw.get("first_stage_responses") or []
        )
    if stage == "development":
        source_tasks = bed.load_validation_partition_tasks(
            "development", include_endpoint_labels=False
        )
        replays = [
            development.replay_block(
                result_path=path, all_development_tasks=source_tasks
            )
            for path in block_results
        ]
    elif stage == "confirmation":
        source_tasks = confirmation.load_planning_tasks()
        replays = [
            confirmation.replay_block(
                result_path=path, all_confirmation_tasks=source_tasks
            )
            for path in block_results
        ]
    else:
        raise ValueError(f"unknown mediation stage {stage!r}")
    contexts = {}
    for result_path, replay in zip(block_results, replays, strict=True):
        if replay.get("verified") is not True:
            raise ValueError("mediation block replay did not verify")
        raw_path = result_path.parent / "private/RAW_RESPONSES.json"
        if mechanics.sha256_file(raw_path) != replay.get("raw_responses_sha256"):
            raise ValueError("mediation block raw response hash changed")
        raw = _load(raw_path)
        block_contexts = _belief_contexts_from_cases(
            cases=replay["artifacts"]["stage_cases"],
            responses=raw.get("first_stage_responses") or [],
        )
        for task_id, context in block_contexts.items():
            if task_id in contexts:
                raise ValueError("mediation task appears in multiple blocks")
            contexts[task_id] = context
    return contexts


def run_report(
    *,
    stage: str,
    result_path: Path,
    output_path: Path,
    block_results: Sequence[Path] = (),
    wrapper_result: Path | None = None,
    authorizer: Callable[..., dict[str, Any]] = stage_outcome.authorize_stage,
    context_loader: Callable[..., Mapping[str, Any]] = replay_contexts,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown mediation stage {stage!r}")
    if output_path.exists():
        raise FileExistsError(output_path)
    authorization = authorizer(
        stage=stage,
        result_path=result_path,
        block_results=block_results,
        wrapper_result=wrapper_result,
    )
    if authorization.get("verified") is not True:
        raise ValueError("mediation stage authorization failed")
    # Raw beliefs and endpoint-bearing combined results are read only after replay.
    contexts = context_loader(
        stage=stage, result_path=result_path, block_results=block_results
    )
    stage_result = _load(result_path)
    report = build_report(
        stage=stage,
        stage_result=stage_result,
        contexts=contexts,
        authorization=authorization,
    )
    report["stage_result_path"] = str(result_path)
    report["stage_result_sha256"] = mechanics.sha256_file(result_path)
    report["block_result_sha256"] = [
        mechanics.sha256_file(path) for path in block_results
    ]
    checkpoint(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--block-result", type=Path, action="append", default=[])
    parser.add_argument("--wrapper-result", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_report(
        stage=args.stage,
        result_path=args.result_path,
        output_path=args.output_path,
        block_results=args.block_result,
        wrapper_result=args.wrapper_result,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
