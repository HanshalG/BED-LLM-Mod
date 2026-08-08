#!/usr/bin/env python3
"""Freeze endpoint-sealed DINOv2 myopic and depth-two Bongard plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_dinov2_embedding_extract as embedding_extract
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-dinov2-classical-baseline-1"
EMBEDDING_DIR = (
    REPO_ROOT
    / "results/nonmyopic/bongard_openworld_dinov2_classical_baseline/"
    "embedding-extract-20260808"
)
EMBEDDING_MANIFEST_SHA256 = (
    "011b2e99ea1f1e0e603bcf1fa01f9abad35ccf0e14ca39d133c28551ff3e1caf"
)
EMBEDDING_INDEX_SHA256 = "9f3a1de76dc7124c0a1a8872949b52e46fb7b85a5ee26267032aa548516e2733"
EMBEDDINGS_SHA256 = "79b678777e2bec8506d59a17c7bf3a8199882f72b3c77178554ccb252c4f08d9"
SCALE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
POLICIES = ("dinov2_myopic", "dinov2_depth2")
TIE_TOLERANCE = 1e-8
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260808


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(value: float) -> float:
    clipped = min(max(value, -60.0), 60.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def binary_entropy(probability: float) -> float:
    probability = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log1p(-probability)
    )


def normalized_mean(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("prototype requires at least one embedding")
    mean = np.mean(np.stack(values), axis=0)
    norm = np.linalg.norm(mean)
    if not math.isfinite(float(norm)) or norm <= 0:
        raise ValueError("prototype mean has invalid norm")
    return mean / norm


def similarity_score(
    embedding: np.ndarray,
    positive: Sequence[np.ndarray],
    negative: Sequence[np.ndarray],
) -> float:
    return float(
        embedding @ normalized_mean(positive)
        - embedding @ normalized_mean(negative)
    )


def positive_probability(
    embedding: np.ndarray,
    positive: Sequence[np.ndarray],
    negative: Sequence[np.ndarray],
    *,
    scale: float,
) -> float:
    return sigmoid(scale * similarity_score(embedding, positive, negative))


def updated_prototypes(
    positive: Sequence[np.ndarray],
    negative: Sequence[np.ndarray],
    embedding: np.ndarray,
    label: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if label:
        return [*positive, embedding], list(negative)
    return list(positive), [*negative, embedding]


def endpoint_entropy(
    embeddings: Mapping[str, np.ndarray],
    endpoint_ids: Sequence[str],
    positive: Sequence[np.ndarray],
    negative: Sequence[np.ndarray],
    *,
    scale: float,
) -> float:
    return sum(
        binary_entropy(
            positive_probability(
                embeddings[image_id], positive, negative, scale=scale
            )
        )
        for image_id in endpoint_ids
    )


def expected_entropy_after_query(
    embeddings: Mapping[str, np.ndarray],
    endpoint_ids: Sequence[str],
    positive: Sequence[np.ndarray],
    negative: Sequence[np.ndarray],
    query_id: str,
    *,
    scale: float,
) -> float:
    query = embeddings[query_id]
    probability = positive_probability(query, positive, negative, scale=scale)
    positive_branch = updated_prototypes(positive, negative, query, True)
    negative_branch = updated_prototypes(positive, negative, query, False)
    return (
        probability
        * endpoint_entropy(
            embeddings, endpoint_ids, *positive_branch, scale=scale
        )
        + (1.0 - probability)
        * endpoint_entropy(
            embeddings, endpoint_ids, *negative_branch, scale=scale
        )
    )


def myopic_scores(
    embeddings: Mapping[str, np.ndarray],
    candidate_ids: Sequence[str],
    endpoint_ids: Sequence[str],
    positive: Sequence[np.ndarray],
    negative: Sequence[np.ndarray],
    *,
    scale: float,
) -> dict[str, float]:
    root_entropy = endpoint_entropy(
        embeddings, endpoint_ids, positive, negative, scale=scale
    )
    return {
        query_id: root_entropy
        - expected_entropy_after_query(
            embeddings,
            endpoint_ids,
            positive,
            negative,
            query_id,
            scale=scale,
        )
        for query_id in candidate_ids
    }


def depth_two_scores(
    embeddings: Mapping[str, np.ndarray],
    candidate_ids: Sequence[str],
    endpoint_ids: Sequence[str],
    positive: Sequence[np.ndarray],
    negative: Sequence[np.ndarray],
    *,
    scale: float,
) -> dict[str, float]:
    root_entropy = endpoint_entropy(
        embeddings, endpoint_ids, positive, negative, scale=scale
    )
    scores = {}
    for first_id in candidate_ids:
        first = embeddings[first_id]
        probability = positive_probability(
            first, positive, negative, scale=scale
        )
        terminal_entropy = 0.0
        for label, outcome_probability in (
            (True, probability),
            (False, 1.0 - probability),
        ):
            branch_positive, branch_negative = updated_prototypes(
                positive, negative, first, label
            )
            remaining = [
                candidate_id
                for candidate_id in candidate_ids
                if candidate_id != first_id
            ]
            terminal_entropy += outcome_probability * min(
                expected_entropy_after_query(
                    embeddings,
                    endpoint_ids,
                    branch_positive,
                    branch_negative,
                    second_id,
                    scale=scale,
                )
                for second_id in remaining
            )
        scores[first_id] = root_entropy - terminal_entropy
    return scores


def score_margin(scores: Mapping[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        raise ValueError("score margin requires at least two actions")
    return ordered[0] - ordered[1]


def history_prototypes(
    task: bed.VisualTask, embeddings: Mapping[str, np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    positive = [
        embeddings[image_id]
        for image_id, label in task.initial_history
        if label
    ]
    negative = [
        embeddings[image_id]
        for image_id, label in task.initial_history
        if not label
    ]
    if len(positive) != 2 or len(negative) != 2:
        raise ValueError("DINO baseline requires two initial examples per class")
    return positive, negative


def endpoint_probabilities(
    task: bed.VisualTask,
    embeddings: Mapping[str, np.ndarray],
    positive: Sequence[np.ndarray],
    negative: Sequence[np.ndarray],
    *,
    scale: float,
) -> dict[str, float]:
    return {
        endpoint_id: positive_probability(
            embeddings[endpoint_id], positive, negative, scale=scale
        )
        for endpoint_id in task.endpoint_ids
    }


def policy_plan(
    task: bed.VisualTask,
    embeddings: Mapping[str, np.ndarray],
    policy: str,
    *,
    scale: float,
) -> dict[str, Any]:
    positive, negative = history_prototypes(task, embeddings)
    root_scores = (
        myopic_scores(
            embeddings,
            task.candidate_ids,
            task.endpoint_ids,
            positive,
            negative,
            scale=scale,
        )
        if policy == "dinov2_myopic"
        else depth_two_scores(
            embeddings,
            task.candidate_ids,
            task.endpoint_ids,
            positive,
            negative,
            scale=scale,
        )
    )
    first_id = bed.select_best(root_scores)
    first_embedding = embeddings[first_id]
    first_probability = positive_probability(
        first_embedding, positive, negative, scale=scale
    )
    branches = {}
    for first_label in (False, True):
        branch_positive, branch_negative = updated_prototypes(
            positive, negative, first_embedding, first_label
        )
        remaining = [
            candidate_id
            for candidate_id in task.candidate_ids
            if candidate_id != first_id
        ]
        second_scores = myopic_scores(
            embeddings,
            remaining,
            task.endpoint_ids,
            branch_positive,
            branch_negative,
            scale=scale,
        )
        second_id = bed.select_best(second_scores)
        second_embedding = embeddings[second_id]
        second_probability = positive_probability(
            second_embedding,
            branch_positive,
            branch_negative,
            scale=scale,
        )
        outcomes = {}
        for second_label in (False, True):
            final_positive, final_negative = updated_prototypes(
                branch_positive,
                branch_negative,
                second_embedding,
                second_label,
            )
            outcomes[bed.LABELS[second_label]] = {
                "endpoint_positive_probabilities": endpoint_probabilities(
                    task,
                    embeddings,
                    final_positive,
                    final_negative,
                    scale=scale,
                )
            }
        branches[bed.LABELS[first_label]] = {
            "second_image_id": second_id,
            "second_positive_probability": second_probability,
            "second_scores": second_scores,
            "second_score_margin": score_margin(second_scores),
            "outcomes": outcomes,
        }
    return {
        "policy": policy,
        "first_image_id": first_id,
        "first_positive_probability": first_probability,
        "first_scores": root_scores,
        "first_score_margin": score_margin(root_scores),
        "branches": branches,
    }


def calibration_samples(
    tasks_by_partition: Mapping[str, Sequence[bed.VisualTask]],
    embeddings_by_task: Mapping[str, Mapping[str, np.ndarray]],
) -> list[tuple[float, bool]]:
    samples = []
    for partition in ("mechanics", "development"):
        for task in tasks_by_partition[partition]:
            task_embeddings = embeddings_by_task[task.task_id]
            history = list(task.initial_history)
            for held_index, (held_id, held_label) in enumerate(history):
                positive = [
                    task_embeddings[image_id]
                    for index, (image_id, label) in enumerate(history)
                    if index != held_index and label
                ]
                negative = [
                    task_embeddings[image_id]
                    for index, (image_id, label) in enumerate(history)
                    if index != held_index and not label
                ]
                samples.append(
                    (
                        similarity_score(
                            task_embeddings[held_id], positive, negative
                        ),
                        held_label,
                    )
                )
    return samples


def calibration_log_loss(
    samples: Sequence[tuple[float, bool]], scale: float
) -> float:
    return -sum(
        math.log(
            positive_probability
            if label
            else 1.0 - positive_probability
        )
        for score, label in samples
        for positive_probability in [
            min(max(sigmoid(scale * score), 1e-12), 1.0 - 1e-12)
        ]
    ) / len(samples)


def select_scale(
    samples: Sequence[tuple[float, bool]],
) -> tuple[float, dict[str, float]]:
    losses = {
        str(scale): calibration_log_loss(samples, scale)
        for scale in SCALE_GRID
    }
    selected = min(SCALE_GRID, key=lambda scale: (losses[str(scale)], scale))
    return selected, losses


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
            f"DINO embedding artifact changed: expected {expected_hashes}, observed {hashes}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("all_gates_pass") is not True
        or manifest.get("endpoint_labels_accessed") is not False
        or manifest.get("candidate_labels_accessed") is not False
    ):
        raise ValueError("DINO embedding manifest is not the frozen label-free pass")
    rows = json.loads(index_path.read_text(encoding="utf-8"))["rows"]
    values = np.load(embeddings_path, allow_pickle=False)
    if len(rows) != len(values):
        raise ValueError("DINO index and embedding row counts differ")
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
            raise ValueError("DINO index contains an unknown task image")
        observed_sha256 = hashlib.sha256(
            task.image_bytes[row["image_id"]]
        ).hexdigest()
        if observed_sha256 != row["image_sha256"]:
            raise ValueError("DINO index image hash does not match the frozen task")
        by_task[task.task_id][row["image_id"]] = value.astype(float)
    if any(set(by_task[task_id]) != set(task.image_ids) for task_id, task in tasks.items()):
        raise ValueError("DINO embedding support is incomplete")
    return by_task, {"manifest": manifest, "hashes": hashes}


def build_plans(
    tasks_by_partition: Mapping[str, Sequence[bed.VisualTask]],
    embeddings_by_task: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    samples = calibration_samples(tasks_by_partition, embeddings_by_task)
    scale, losses = select_scale(samples)
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
            if row["policies"]["dinov2_myopic"]["first_image_id"]
            != row["policies"]["dinov2_depth2"]["first_image_id"]
        ]
        robust = [
            row
            for row in changed
            if min(
                row["policies"][policy]["first_score_margin"]
                for policy in POLICIES
            )
            > TIE_TOLERANCE
        ]
        summary[partition] = {
            "tasks": len(tasks),
            "changed_first_actions": len(changed),
            "robust_changed_first_actions": len(robust),
            "median_myopic_margin": float(
                np.median(
                    [
                        row["policies"]["dinov2_myopic"]["first_score_margin"]
                        for row in tasks
                    ]
                )
            ),
            "median_depth2_margin": float(
                np.median(
                    [
                        row["policies"]["dinov2_depth2"]["first_score_margin"]
                        for row in tasks
                    ]
                )
            ),
        }
    return summary


def realized_policy_row(
    task: bed.VisualTask, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Interpret one frozen plan only after its partition outcomes are opened."""
    first_id = str(plan["first_image_id"])
    first_label = task.actual_labels[first_id]
    branch = plan["branches"][bed.LABELS[first_label]]
    second_id = str(branch["second_image_id"])
    second_label = task.actual_labels[second_id]
    endpoint_probabilities = branch["outcomes"][bed.LABELS[second_label]][
        "endpoint_positive_probabilities"
    ]
    endpoint_rows = []
    for endpoint_id in task.endpoint_ids:
        label = task.actual_labels[endpoint_id]
        probability = float(endpoint_probabilities[endpoint_id])
        truth_probability = probability if label else 1.0 - probability
        endpoint_rows.append(
            {
                "image_id": endpoint_id,
                "label": bed.LABELS[label],
                "positive_probability": probability,
                "truth_probability": truth_probability,
                "brier": (float(label) - probability) ** 2,
                "log_loss": -math.log(max(truth_probability, 1e-12)),
                "correct": (probability >= 0.5) == label,
            }
        )
    return {
        "task_id": task.task_id,
        "policy": plan["policy"],
        "first_image_id": first_id,
        "first_label": bed.LABELS[first_label],
        "second_image_id": second_id,
        "second_label": bed.LABELS[second_label],
        "mean_brier": sum(row["brier"] for row in endpoint_rows)
        / len(endpoint_rows),
        "mean_log_loss": sum(row["log_loss"] for row in endpoint_rows)
        / len(endpoint_rows),
        "accuracy": sum(row["correct"] for row in endpoint_rows)
        / len(endpoint_rows),
        "mean_truth_probability": sum(
            row["truth_probability"] for row in endpoint_rows
        )
        / len(endpoint_rows),
        "endpoints": endpoint_rows,
    }


def paired_summary(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("paired summary requires finite values")
    rng = random.Random(seed)
    count = len(values)
    draws = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_DRAWS)
    )
    return {
        "values": list(values),
        "mean": sum(values) / count,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed,
        "bootstrap_95pct_ci": [
            draws[int(0.025 * BOOTSTRAP_DRAWS)],
            draws[int(0.975 * BOOTSTRAP_DRAWS) - 1],
        ],
    }


def evaluate_opened_partition(
    tasks: Sequence[bed.VisualTask],
    plan_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate frozen plans; callers must independently authorize label access."""
    plans_by_task = {row["task_id"]: row for row in plan_rows}
    if set(plans_by_task) != {task.task_id for task in tasks}:
        raise ValueError("opened partition tasks do not exactly match frozen plans")
    rows = {
        policy: [
            realized_policy_row(
                task, plans_by_task[task.task_id]["policies"][policy]
            )
            for task in tasks
        ]
        for policy in POLICIES
    }
    metrics = ("mean_brier", "mean_log_loss", "accuracy", "mean_truth_probability")
    pooled = {
        policy: {
            metric: sum(row[metric] for row in policy_rows) / len(policy_rows)
            for metric in metrics
        }
        for policy, policy_rows in rows.items()
    }
    paired_depth2_minus_myopic = {
        metric: [
            depth2[metric] - myopic[metric]
            for myopic, depth2 in zip(
                rows["dinov2_myopic"], rows["dinov2_depth2"], strict=True
            )
        ]
        for metric in metrics
    }
    return {
        "rows": rows,
        "pooled": pooled,
        "paired_depth2_minus_myopic": {
            metric: paired_summary(
                values, seed=BOOTSTRAP_SEED + metric_index
            )
            for metric_index, (metric, values) in enumerate(
                paired_depth2_minus_myopic.items()
            )
        },
    }


def run_protocol(*, output_dir: Path) -> dict[str, Any]:
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
        "plans_path": str(plans_path.relative_to(REPO_ROOT)),
        "plans_sha256": sha256_file(plans_path),
        "calibration": plans["calibration"],
        "summary": summary,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "interpretation": (
            "These are endpoint-sealed classical counterfactual plans, not an "
            "endpoint result. They become a performance baseline only after an "
            "independently authorized stage opens the corresponding outcomes."
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
            / "results/nonmyopic/bongard_openworld_dinov2_classical_baseline/"
            "plans-20260808"
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = run_protocol(output_dir=parse_args().output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
