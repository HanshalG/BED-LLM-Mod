#!/usr/bin/env python3
"""Freeze endpoint-sealed SigLIP myopic and depth-two Bongard plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_dinov2_classical_baseline as shared
from scripts import bongard_openworld_siglip_embedding_extract as embedding_extract
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-siglip-classical-baseline-1"
SHARED_PLANNER_SHA256 = (
    "ab30cb7add964eecfdb85194bd591c5be3dc6f78cd6a54d07c46868740a8bbfc"
)
EMBEDDING_DIR = (
    REPO_ROOT
    / "results/nonmyopic/bongard_openworld_siglip_classical_baseline/"
    "embedding-extract-20260809"
)
EMBEDDING_MANIFEST_SHA256 = (
    "9c3aa45197730421697c4eaa8cff6d395beeda624b188665af53f24dbd7946ba"
)
EMBEDDING_INDEX_SHA256 = "9f3a1de76dc7124c0a1a8872949b52e46fb7b85a5ee26267032aa548516e2733"
EMBEDDINGS_SHA256 = "338151bc734b9ea4259a57ccabacd6a499df5fe8afcf8cefdcdd3a4c6b13f4d1"
POLICIES = ("siglip_myopic", "siglip_depth2")
SCALE_GRID = (
    -64.0,
    -32.0,
    -16.0,
    -8.0,
    -4.0,
    -2.0,
    -1.0,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
)
SHARED_POLICIES = {
    "siglip_myopic": "dinov2_myopic",
    "siglip_depth2": "dinov2_depth2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_shared_planner() -> None:
    observed = sha256_file(Path(shared.__file__).resolve())
    if observed != SHARED_PLANNER_SHA256:
        raise ValueError(
            "shared classical planner changed: expected "
            f"{SHARED_PLANNER_SHA256}, observed {observed}"
        )


def policy_plan(
    task: bed.VisualTask,
    embeddings: Mapping[str, np.ndarray],
    policy: str,
    *,
    scale: float,
) -> dict[str, Any]:
    if policy not in SHARED_POLICIES:
        raise ValueError(f"unknown SigLIP policy {policy!r}")
    plan = shared.policy_plan(
        task,
        embeddings,
        SHARED_POLICIES[policy],
        scale=scale,
    )
    plan["policy"] = policy
    return plan


def load_bound_embeddings(
    tasks_by_partition: Mapping[str, Sequence[bed.VisualTask]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    manifest_path = EMBEDDING_DIR / "MANIFEST.json"
    index_path = EMBEDDING_DIR / "INDEX.json"
    embeddings_path = EMBEDDING_DIR / "EMBEDDINGS.npy"
    hashes = {
        "manifest": sha256_file(manifest_path),
        "index": sha256_file(index_path),
        "embeddings": sha256_file(embeddings_path),
    }
    expected_hashes = {
        "manifest": EMBEDDING_MANIFEST_SHA256,
        "index": EMBEDDING_INDEX_SHA256,
        "embeddings": EMBEDDINGS_SHA256,
    }
    if hashes != expected_hashes:
        raise ValueError(
            f"SigLIP embedding artifact changed: expected {expected_hashes}, "
            f"observed {hashes}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("all_gates_pass") is not True
        or manifest.get("endpoint_labels_accessed") is not False
        or manifest.get("candidate_labels_accessed") is not False
    ):
        raise ValueError("SigLIP embedding manifest is not the frozen label-free pass")
    rows = json.loads(index_path.read_text(encoding="utf-8"))["rows"]
    values = np.load(embeddings_path, allow_pickle=False)
    if len(rows) != len(values):
        raise ValueError("SigLIP index and embedding row counts differ")
    tasks = {
        task.task_id: task
        for partition_tasks in tasks_by_partition.values()
        for task in partition_tasks
    }
    by_task: dict[str, dict[str, np.ndarray]] = {
        task_id: {} for task_id in tasks
    }
    for row, value in zip(rows, values, strict=True):
        task = tasks.get(row["task_id"])
        if task is None or row["image_id"] not in task.image_bytes:
            raise ValueError("SigLIP index contains an unknown task image")
        observed_sha256 = hashlib.sha256(
            task.image_bytes[row["image_id"]]
        ).hexdigest()
        if observed_sha256 != row["image_sha256"]:
            raise ValueError("SigLIP index image hash does not match the frozen task")
        by_task[task.task_id][row["image_id"]] = value.astype(float)
    if any(
        set(by_task[task_id]) != set(task.image_ids)
        for task_id, task in tasks.items()
    ):
        raise ValueError("SigLIP embedding support is incomplete")
    return by_task, {"manifest": manifest, "hashes": hashes}


def build_plans(
    tasks_by_partition: Mapping[str, Sequence[bed.VisualTask]],
    embeddings_by_task: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    samples = shared.calibration_samples(tasks_by_partition, embeddings_by_task)
    losses = {
        str(scale): shared.calibration_log_loss(samples, scale)
        for scale in SCALE_GRID
    }
    scale = min(
        SCALE_GRID,
        key=lambda candidate: (
            losses[str(candidate)],
            abs(candidate),
            candidate,
        ),
    )
    tasks = []
    for partition in embedding_extract.PARTITION_SIZES:
        for task in tasks_by_partition[partition]:
            task_embeddings = embeddings_by_task[task.task_id]
            tasks.append(
                {
                    "partition": partition,
                    "task_id": task.task_id,
                    "policies": {
                        policy: policy_plan(
                            task, task_embeddings, policy, scale=scale
                        )
                        for policy in POLICIES
                    },
                }
            )
    return {
        "interface_version": INTERFACE_VERSION,
        "calibration": {
            "partitions": ["mechanics", "development"],
            "source": "leave-one-out initial-history labels only",
            "samples": len(samples),
            "scale_grid": list(SCALE_GRID),
            "log_loss_by_scale": losses,
            "selected_scale": scale,
            "bias": 0.0,
        },
        "tasks": tasks,
    }


def plan_summary(plans: Mapping[str, Any]) -> dict[str, Any]:
    summary = {}
    for partition in embedding_extract.PARTITION_SIZES:
        tasks = [row for row in plans["tasks"] if row["partition"] == partition]
        changed = [
            row
            for row in tasks
            if row["policies"]["siglip_myopic"]["first_image_id"]
            != row["policies"]["siglip_depth2"]["first_image_id"]
        ]
        robust = [
            row
            for row in changed
            if min(
                row["policies"][policy]["first_score_margin"]
                for policy in POLICIES
            )
            > shared.TIE_TOLERANCE
        ]
        summary[partition] = {
            "tasks": len(tasks),
            "changed_first_actions": len(changed),
            "robust_changed_first_actions": len(robust),
            "median_myopic_margin": float(
                np.median(
                    [
                        row["policies"]["siglip_myopic"]["first_score_margin"]
                        for row in tasks
                    ]
                )
            ),
            "median_depth2_margin": float(
                np.median(
                    [
                        row["policies"]["siglip_depth2"]["first_score_margin"]
                        for row in tasks
                    ]
                )
            ),
        }
    return summary


def evaluate_opened_partition(
    tasks: Sequence[bed.VisualTask],
    plan_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plans_by_task = {row["task_id"]: row for row in plan_rows}
    if set(plans_by_task) != {task.task_id for task in tasks}:
        raise ValueError("opened partition tasks do not exactly match frozen plans")
    rows = {
        policy: [
            shared.realized_policy_row(
                task, plans_by_task[task.task_id]["policies"][policy]
            )
            for task in tasks
        ]
        for policy in POLICIES
    }
    metrics = (
        "mean_brier",
        "mean_log_loss",
        "accuracy",
        "mean_truth_probability",
    )
    pooled = {
        policy: {
            metric: sum(row[metric] for row in policy_rows) / len(policy_rows)
            for metric in metrics
        }
        for policy, policy_rows in rows.items()
    }
    paired = {
        metric: [
            depth2[metric] - myopic[metric]
            for myopic, depth2 in zip(
                rows["siglip_myopic"], rows["siglip_depth2"], strict=True
            )
        ]
        for metric in metrics
    }
    return {
        "rows": rows,
        "pooled": pooled,
        "paired_depth2_minus_myopic": {
            metric: shared.paired_summary(
                values, seed=shared.BOOTSTRAP_SEED + 100 + metric_index
            )
            for metric_index, (metric, values) in enumerate(paired.items())
        },
    }


def run_protocol(*, output_dir: Path) -> dict[str, Any]:
    verify_shared_planner()
    tasks = embedding_extract.default_tasks()
    embeddings, binding = load_bound_embeddings(tasks)
    plans = build_plans(tasks, embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans_path = output_dir / "PLANS.json"
    manifest_path = output_dir / "MANIFEST.json"
    for path in (plans_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)
    plans_path.write_text(bed.canonical_json(plans) + "\n", encoding="utf-8")
    summary = plan_summary(plans)
    gates = {
        "bound_label_free_embedding_artifact_matches": bool(binding),
        "shared_planner_implementation_matches": True,
        "scale_selected_from_exact_initial_history_samples": (
            plans["calibration"]["samples"] == 272
            and plans["calibration"]["selected_scale"] in SCALE_GRID
        ),
        "all_partition_plans_are_complete": all(
            summary[partition]["tasks"] == size
            for partition, size in embedding_extract.PARTITION_SIZES.items()
        ),
        "both_policies_have_complete_counterfactual_branches": all(
            set(row["policies"]) == set(POLICIES)
            and all(
                set(plan["branches"]) == set(bed.LABELS.values())
                and all(
                    set(branch["outcomes"]) == set(bed.LABELS.values())
                    for branch in plan["branches"].values()
                )
                for plan in row["policies"].values()
            )
            for row in plans["tasks"]
        ),
        "depth_two_changes_a_robust_first_action": sum(
            row["robust_changed_first_actions"] for row in summary.values()
        )
        > 0,
    }
    all_gates_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "classical_plans_frozen" if all_gates_pass else "fail",
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "endpoint_labels_accessed": False,
        "candidate_labels_accessed": False,
        "embedding_binding": binding["hashes"],
        "shared_planner_sha256": SHARED_PLANNER_SHA256,
        "plans_path": str(plans_path.relative_to(REPO_ROOT)),
        "plans_sha256": sha256_file(plans_path),
        "calibration": plans["calibration"],
        "summary": summary,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "interpretation": (
            "These are endpoint-sealed classical SigLIP counterfactual plans, "
            "not an endpoint result. They become a performance baseline only "
            "after an independently authorized stage opens the corresponding outcomes."
        ),
    }
    checkpoint(manifest_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/bongard_openworld_siglip_classical_baseline/"
            "plans-20260809"
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = run_protocol(output_dir=parse_args().output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
