#!/usr/bin/env python3
"""Audit frozen SigLIP calibration on observed initial-history labels only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_dinov2_classical_baseline as shared
from scripts import bongard_openworld_siglip_classical_baseline as siglip
from scripts import bongard_openworld_siglip_embedding_extract as extract
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-siglip-calibration-audit-1"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_SIGLIP_CLASSICAL_BASELINE_PROTOCOL_20260809.md"
)
PROTOCOL_SHA256 = "e4b126b9b95f18d62a8968959c67b1b5e2c9e8b01c7794b0dc23e7b8a80f5b86"
FINAL_PLAN_DIR = (
    REPO_ROOT
    / "results/nonmyopic/bongard_openworld_siglip_classical_baseline/"
    "plans-20260809"
)
FINAL_PLAN_MANIFEST_SHA256 = (
    "95c00d948ee77589995c3434e3eb01b0a908b496fe2d984b859dbae4032d057d"
)
FINAL_PLANS_SHA256 = "a41cc3b01d18fa9008f67d6f60a3f113b8f3194b73e932b4e2c3cfc83215f587"
SIGLIP_IMPLEMENTATION_SHA256 = (
    "4fd19b2e9fabca09ef3d20e31d163dd7133784e5992d1f9afdbd77e495e84211"
)
FROZEN_SCALE = -2.0
ZERO_SCALE = 0.0
POSITIVE_DIAGNOSTIC_SCALE = 0.25
BOOTSTRAP_SEED = 20263809
PARTITIONS = ("mechanics", "development", "confirmation")


def verify_frozen_inputs() -> dict[str, str]:
    observed = {
        "protocol": siglip.sha256_file(PROTOCOL),
        "embedding_manifest": siglip.sha256_file(
            siglip.EMBEDDING_DIR / "MANIFEST.json"
        ),
        "embedding_index": siglip.sha256_file(siglip.EMBEDDING_DIR / "INDEX.json"),
        "embeddings": siglip.sha256_file(siglip.EMBEDDING_DIR / "EMBEDDINGS.npy"),
        "plan_manifest": siglip.sha256_file(FINAL_PLAN_DIR / "MANIFEST.json"),
        "plans": siglip.sha256_file(FINAL_PLAN_DIR / "PLANS.json"),
        "siglip_implementation": siglip.sha256_file(
            REPO_ROOT / "scripts/bongard_openworld_siglip_classical_baseline.py"
        ),
    }
    expected = {
        "protocol": PROTOCOL_SHA256,
        "embedding_manifest": siglip.EMBEDDING_MANIFEST_SHA256,
        "embedding_index": siglip.EMBEDDING_INDEX_SHA256,
        "embeddings": siglip.EMBEDDINGS_SHA256,
        "plan_manifest": FINAL_PLAN_MANIFEST_SHA256,
        "plans": FINAL_PLANS_SHA256,
        "siglip_implementation": SIGLIP_IMPLEMENTATION_SHA256,
    }
    if observed != expected:
        raise ValueError(
            f"frozen SigLIP audit inputs changed: expected {expected}, observed {observed}"
        )
    plan_manifest = json.loads(
        (FINAL_PLAN_DIR / "MANIFEST.json").read_text(encoding="utf-8")
    )
    if (
        plan_manifest.get("all_gates_pass") is not True
        or plan_manifest.get("candidate_labels_accessed") is not False
        or plan_manifest.get("endpoint_labels_accessed") is not False
        or plan_manifest.get("calibration", {}).get("selected_scale") != FROZEN_SCALE
    ):
        raise ValueError("registered SigLIP plans are not the frozen signed-scale pass")
    return observed


def task_samples(
    task: bed.VisualTask,
    embeddings: Mapping[str, np.ndarray],
) -> list[tuple[float, bool]]:
    history = list(task.initial_history)
    samples = []
    for held_index, (held_id, held_label) in enumerate(history):
        positive = [
            embeddings[image_id]
            for index, (image_id, label) in enumerate(history)
            if index != held_index and label
        ]
        negative = [
            embeddings[image_id]
            for index, (image_id, label) in enumerate(history)
            if index != held_index and not label
        ]
        samples.append(
            (
                shared.similarity_score(
                    embeddings[held_id], positive, negative
                ),
                held_label,
            )
        )
    return samples


def sample_brier(samples: Sequence[tuple[float, bool]], scale: float) -> float:
    return sum(
        (float(label) - shared.sigmoid(scale * score)) ** 2
        for score, label in samples
    ) / len(samples)


def raw_score_auc(samples: Sequence[tuple[float, bool]]) -> float:
    positive = [score for score, label in samples if label]
    negative = [score for score, label in samples if not label]
    if not positive or not negative:
        raise ValueError("AUC requires both classes")
    return sum(
        float(pos > neg) + 0.5 * float(pos == neg)
        for pos in positive
        for neg in negative
    ) / (len(positive) * len(negative))


def partition_metrics(
    tasks: Sequence[bed.VisualTask],
    embeddings_by_task: Mapping[str, Mapping[str, np.ndarray]],
    *,
    seed: int,
) -> dict[str, Any]:
    by_task = {
        task.task_id: task_samples(task, embeddings_by_task[task.task_id])
        for task in tasks
    }
    samples = [sample for task in tasks for sample in by_task[task.task_id]]
    positive_scores = [score for score, label in samples if label]
    negative_scores = [score for score, label in samples if not label]
    scales = (FROZEN_SCALE, ZERO_SCALE, POSITIVE_DIAGNOSTIC_SCALE)
    log_loss = {
        str(scale): shared.calibration_log_loss(samples, scale)
        for scale in scales
    }
    brier = {str(scale): sample_brier(samples, scale) for scale in scales}
    per_task_log_delta = [
        shared.calibration_log_loss(by_task[task.task_id], FROZEN_SCALE)
        - shared.calibration_log_loss(by_task[task.task_id], ZERO_SCALE)
        for task in tasks
    ]
    per_task_brier_delta = [
        sample_brier(by_task[task.task_id], FROZEN_SCALE)
        - sample_brier(by_task[task.task_id], ZERO_SCALE)
        for task in tasks
    ]
    return {
        "tasks": len(tasks),
        "samples": len(samples),
        "raw_score_auc": raw_score_auc(samples),
        "frozen_probability_auc": 1.0 - raw_score_auc(samples),
        "mean_raw_score_positive": float(np.mean(positive_scores)),
        "mean_raw_score_negative": float(np.mean(negative_scores)),
        "log_loss_by_scale": log_loss,
        "brier_by_scale": brier,
        "paired_frozen_minus_zero_log_loss": shared.paired_summary(
            per_task_log_delta, seed=seed
        ),
        "paired_frozen_minus_zero_brier": shared.paired_summary(
            per_task_brier_delta, seed=seed + 1
        ),
    }


def run_audit(*, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    frozen_inputs = verify_frozen_inputs()
    tasks = extract.default_tasks()
    embeddings, embedding_binding = siglip.load_bound_embeddings(tasks)
    partitions = {
        partition: partition_metrics(
            tasks[partition],
            embeddings,
            seed=BOOTSTRAP_SEED + 10 * partition_index,
        )
        for partition_index, partition in enumerate(PARTITIONS)
    }
    confirmation = partitions["confirmation"]
    finite_values = [
        value
        for metrics in partitions.values()
        for value in (
            metrics["raw_score_auc"],
            metrics["frozen_probability_auc"],
            metrics["mean_raw_score_positive"],
            metrics["mean_raw_score_negative"],
            *metrics["log_loss_by_scale"].values(),
            *metrics["brier_by_scale"].values(),
        )
    ]
    gates = {
        "frozen_inputs_match": bool(frozen_inputs) and bool(embedding_binding),
        "exact_partition_task_counts": all(
            partitions[partition]["tasks"] == extract.PARTITION_SIZES[partition]
            for partition in PARTITIONS
        ),
        "exact_four_initial_samples_per_task": all(
            partitions[partition]["samples"]
            == 4 * extract.PARTITION_SIZES[partition]
            for partition in PARTITIONS
        ),
        "all_reported_values_are_finite": all(
            math.isfinite(float(value)) for value in finite_values
        ),
        "candidate_and_endpoint_labels_not_accessed": True,
    }
    all_gates_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "calibration_audit_complete" if all_gates_pass else "fail",
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "candidate_labels_accessed": False,
        "endpoint_labels_accessed": False,
        "frozen_scale": FROZEN_SCALE,
        "zero_scale": ZERO_SCALE,
        "positive_diagnostic_scale": POSITIVE_DIAGNOSTIC_SCALE,
        "frozen_inputs": frozen_inputs,
        "partitions": partitions,
        "confirmation_descriptive_checks": {
            "frozen_log_loss_below_zero_scale": (
                confirmation["log_loss_by_scale"][str(FROZEN_SCALE)]
                < confirmation["log_loss_by_scale"][str(ZERO_SCALE)]
            ),
            "frozen_brier_below_zero_scale": (
                confirmation["brier_by_scale"][str(FROZEN_SCALE)]
                < confirmation["brier_by_scale"][str(ZERO_SCALE)]
            ),
            "raw_similarity_is_inverted": confirmation["raw_score_auc"] < 0.5,
        },
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "interpretation": (
            "This audit uses only each task's four already-observed initial labels. "
            "Confirmation is a held-out calibration diagnostic, not a candidate or "
            "endpoint efficacy result, and it cannot authorize or stop paid work."
        ),
    }
    checkpoint(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/bongard_openworld_siglip_calibration_audit/"
            "AUDIT.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = run_audit(output_path=parse_args().output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
