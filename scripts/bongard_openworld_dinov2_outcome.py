#!/usr/bin/env python3
"""Evaluate frozen DINO plans only after an existing Bongard stage is verified."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_dinov2_classical_baseline as baseline
from scripts import bongard_openworld_luna_confirmation64 as confirmation
from scripts import bongard_openworld_luna_development32_daily_execute as development_daily
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-dinov2-outcome-1"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_DINOV2_CLASSICAL_BASELINE_PROTOCOL_20260808.md"
)
PROTOCOL_SHA256 = "578e2ce911457a0c33f275e3f982f78dfa6d6216c30595277415e778264bbde8"
PLANS_DIR = (
    REPO_ROOT
    / "results/nonmyopic/bongard_openworld_dinov2_classical_baseline/"
    "plans-20260808"
)
PLANS_MANIFEST_SHA256 = "56e9d538ded501f4518dc662d8403f3f15ba254c8cedb132e4bd09d10949e504"
PLANS_SHA256 = "58154f052424f1632c302f9b40af030c172fc4d626908c9f977211c1e8299847"
BASELINE_IMPLEMENTATION_SHA256 = (
    "ab30cb7add964eecfdb85194bd591c5be3dc6f78cd6a54d07c46868740a8bbfc"
)
STAGES = ("mechanics", "development", "confirmation")
STAGE_TASK_COUNTS = {"mechanics": 4, "development": 64, "confirmation": 96}
COMPARISON_SEED = 20261808


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_frozen_inputs() -> dict[str, str]:
    observed = {
        "protocol": baseline.sha256_file(PROTOCOL),
        "plans_manifest": baseline.sha256_file(PLANS_DIR / "MANIFEST.json"),
        "plans": baseline.sha256_file(PLANS_DIR / "PLANS.json"),
        "baseline_implementation": baseline.sha256_file(
            REPO_ROOT / "scripts/bongard_openworld_dinov2_classical_baseline.py"
        ),
    }
    expected = {
        "protocol": PROTOCOL_SHA256,
        "plans_manifest": PLANS_MANIFEST_SHA256,
        "plans": PLANS_SHA256,
        "baseline_implementation": BASELINE_IMPLEMENTATION_SHA256,
    }
    if observed != expected:
        raise ValueError(
            f"frozen DINO outcome inputs changed: expected {expected}, observed {observed}"
        )
    manifest = _load(PLANS_DIR / "MANIFEST.json")
    if (
        manifest.get("all_gates_pass") is not True
        or manifest.get("endpoint_labels_accessed") is not False
        or manifest.get("candidate_labels_accessed") is not False
        or manifest.get("plans_sha256") != PLANS_SHA256
    ):
        raise ValueError("DINO plans manifest is not the endpoint-sealed pass")
    return observed


def load_plan_rows(stage: str) -> list[dict[str, Any]]:
    plans = _load(PLANS_DIR / "PLANS.json")
    rows = [row for row in plans.get("tasks", []) if row.get("partition") == stage]
    if len(rows) != STAGE_TASK_COUNTS[stage] or len(
        {row.get("task_id") for row in rows}
    ) != len(rows):
        raise ValueError(f"frozen DINO {stage} plans are incomplete")
    return rows


def authorize_stage(
    *,
    stage: str,
    result_path: Path,
    block_results: Sequence[Path],
    wrapper_result: Path | None,
) -> dict[str, Any]:
    """Replay the pre-existing stage before any labelled task is loaded here."""
    if stage == "mechanics":
        if wrapper_result is None or block_results:
            raise ValueError("mechanics requires one wrapper and no block results")
        verification = development_daily.validate_aug10_authorization(
            wrapper_result=wrapper_result,
            mechanics_result=result_path,
        )
        if verification.get("verified") is not True:
            raise ValueError("mechanics authorization did not independently verify")
        return verification
    expected_blocks = (
        len(development.BLOCK_ORDER)
        if stage == "development"
        else len(confirmation.BLOCK_ORDER)
    )
    if wrapper_result is not None or len(block_results) != expected_blocks:
        raise ValueError(
            f"{stage} requires exactly {expected_blocks} block results and no wrapper"
        )
    verification = (
        development_daily.verify_combined_result(
            result_path=result_path,
            block_results=block_results,
        )
        if stage == "development"
        else confirmation.verify_combined_result(
            result_path=result_path,
            block_results=block_results,
        )
    )
    if verification.get("verified") is not True:
        raise ValueError(f"{stage} combined result did not independently verify")
    return verification


def load_opened_tasks(stage: str) -> list[bed.VisualTask]:
    tasks = (
        bed.load_mechanics_tasks()
        if stage == "mechanics"
        else bed.load_validation_partition_tasks(
            stage, include_endpoint_labels=True
        )
    )
    if len(tasks) != STAGE_TASK_COUNTS[stage] or any(
        not set(task.candidate_ids).issubset(task.actual_labels)
        or not set(task.endpoint_ids).issubset(task.actual_labels)
        for task in tasks
    ):
        raise ValueError(f"opened {stage} labels are incomplete")
    return tasks


def _pooled_luna_metrics(trees: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    policies = ("myopic_width", "dynamic_depth2")
    metrics = ("mean_brier", "mean_log_loss", "accuracy", "mean_truth_probability")
    return {
        policy: {
            metric: sum(
                float(tree["policies"][policy]["endpoint"][metric])
                for tree in trees
            )
            / len(trees)
            for metric in metrics
        }
        for policy in policies
    }


def _paired_luna_minus_dino(
    *,
    trees: Sequence[Mapping[str, Any]],
    dino_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    tree_by_task = {str(tree["task_id"]): tree for tree in trees}
    dino_by_policy = {
        policy: {str(row["task_id"]): row for row in rows}
        for policy, rows in dino_rows.items()
    }
    task_ids = sorted(tree_by_task)
    if any(set(rows) != set(task_ids) for rows in dino_by_policy.values()):
        raise ValueError("Luna and DINO task identities do not pair exactly")
    metrics = ("mean_brier", "mean_log_loss", "accuracy", "mean_truth_probability")
    comparisons = {
        "luna_myopic_minus_dinov2_myopic": ("myopic_width", "dinov2_myopic"),
        "luna_dynamic_minus_dinov2_depth2": ("dynamic_depth2", "dinov2_depth2"),
    }
    return {
        name: {
            metric: baseline.paired_summary(
                [
                    float(
                        tree_by_task[task_id]["policies"][luna_policy][
                            "endpoint"
                        ][metric]
                    )
                    - float(dino_by_policy[dino_policy][task_id][metric])
                    for task_id in task_ids
                ],
                seed=COMPARISON_SEED + comparison_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
        for comparison_index, (name, (luna_policy, dino_policy)) in enumerate(
            comparisons.items()
        )
    }


def build_outcome(
    *,
    stage: str,
    stage_result: Mapping[str, Any],
    tasks: Sequence[bed.VisualTask],
    plan_rows: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
    frozen_inputs: Mapping[str, str],
) -> dict[str, Any]:
    trees = stage_result.get("trees")
    if not isinstance(trees, list) or len(trees) != STAGE_TASK_COUNTS[stage]:
        raise ValueError(f"verified {stage} result lacks complete Luna trees")
    dino = baseline.evaluate_opened_partition(tasks, plan_rows)
    comparisons = _paired_luna_minus_dino(
        trees=trees, dino_rows=dino["rows"]
    )
    finite_values = [
        value
        for policy in dino["pooled"].values()
        for value in policy.values()
    ] + [
        value
        for policy in _pooled_luna_metrics(trees).values()
        for value in policy.values()
    ]
    gates = {
        "stage_authorization_verified_before_label_loading": (
            authorization.get("verified") is True
        ),
        "exact_task_count_and_identity_pairing": (
            {task.task_id for task in tasks}
            == {row["task_id"] for row in plan_rows}
            == {tree["task_id"] for tree in trees}
        ),
        "all_reported_metrics_are_finite": all(
            math.isfinite(float(value)) for value in finite_values
        ),
        "both_frozen_dino_policies_reported": set(dino["pooled"])
        == set(baseline.POLICIES),
        "both_registered_luna_comparisons_reported": set(comparisons)
        == {
            "luna_myopic_minus_dinov2_myopic",
            "luna_dynamic_minus_dinov2_depth2",
        },
    }
    all_gates_pass = all(gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "classical_comparator_complete" if all_gates_pass else "fail",
        "stage": stage,
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "candidate_labels_accessed_after_stage_authorization": True,
        "endpoint_labels_accessed_after_stage_authorization": True,
        "stage_authorization": dict(authorization),
        "frozen_inputs": dict(frozen_inputs),
        "stage_result_status": stage_result.get("status"),
        "stage_claim_tier": stage_result.get("claim_tier"),
        "dino": dino,
        "luna_pooled": _pooled_luna_metrics(trees),
        "paired_luna_minus_dino": comparisons,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "interpretation": (
            "Lower paired differences favor Luna for Brier/log loss; higher paired "
            "differences favor Luna for accuracy/truth probability. The comparator "
            "must be reported regardless of direction."
        ),
    }


def run_outcome(
    *,
    stage: str,
    result_path: Path,
    output_path: Path,
    block_results: Sequence[Path] = (),
    wrapper_result: Path | None = None,
    authorizer: Callable[..., dict[str, Any]] = authorize_stage,
    task_loader: Callable[[str], list[bed.VisualTask]] = load_opened_tasks,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown DINO outcome stage {stage!r}")
    if output_path.exists():
        raise FileExistsError(output_path)
    frozen_inputs = verify_frozen_inputs()
    plan_rows = load_plan_rows(stage)
    authorization = authorizer(
        stage=stage,
        result_path=result_path,
        block_results=block_results,
        wrapper_result=wrapper_result,
    )
    # This ordering is the privacy boundary: opened labels load only after replay.
    tasks = task_loader(stage)
    stage_result = _load(result_path)
    result = build_outcome(
        stage=stage,
        stage_result=stage_result,
        tasks=tasks,
        plan_rows=plan_rows,
        authorization=authorization,
        frozen_inputs=frozen_inputs,
    )
    result["stage_result_path"] = str(result_path)
    result["stage_result_sha256"] = baseline.sha256_file(result_path)
    checkpoint(output_path, result)
    return result


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
    result = run_outcome(
        stage=args.stage,
        result_path=args.result_path,
        output_path=args.output_path,
        block_results=args.block_result,
        wrapper_result=args.wrapper_result,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
