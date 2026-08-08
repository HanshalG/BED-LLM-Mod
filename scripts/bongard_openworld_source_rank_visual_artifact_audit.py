#!/usr/bin/env python3
"""Audit whether low-level visible image features reveal Bongard endpoint role."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence
import warnings

import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_prompt_role_blindness_audit as role_audit
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-source-rank-visual-artifact-audit-1"
ROLE_BLINDNESS_MANIFEST = (
    REPO_ROOT
    / "results/nonmyopic/bongard_openworld_prompt_role_blindness_audit/"
    "bongard-openworld-prompt-role-blindness-audit-20260808/MANIFEST.json"
)
ROLE_BLINDNESS_MANIFEST_SHA256 = (
    "40077db8f1caccdd9eafd0f6b0d714b32bebf8319d43dd773b3b08e188bc6857"
)
L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
NUM_FOLDS = 5
MAX_CONFIRMATION_AUC = 0.60
MAX_CONFIRMATION_BRIER_REGRET = 0.005
MAX_CONFIRMATION_TOP2_PRECISION = 0.30
MAX_CONFIRMATION_PROBABILITY = 0.40
FEATURE_NAMES = (
    "log_bytes",
    "log_width",
    "log_height",
    "log_aspect_ratio",
    "gray_mean",
    "gray_std",
    "gray_q10",
    "gray_q50",
    "gray_q90",
    "red_mean",
    "green_mean",
    "blue_mean",
    "red_std",
    "green_std",
    "blue_std",
    "saturation_mean",
    "saturation_std",
    "edge_mean",
    "edge_std",
    "edge_q90",
    "gray_histogram_entropy",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_features(data: bytes) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Palette images with Transparency expressed in bytes.*",
        )
        with Image.open(BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            pixels = np.asarray(
                image.resize((64, 64), Image.Resampling.BILINEAR),
                dtype=np.float64,
            ) / 255.0
    gray = (
        0.299 * pixels[:, :, 0]
        + 0.587 * pixels[:, :, 1]
        + 0.114 * pixels[:, :, 2]
    )
    maximum = pixels.max(axis=2)
    minimum = pixels.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    horizontal_edges = np.abs(np.diff(gray, axis=1)).ravel()
    vertical_edges = np.abs(np.diff(gray, axis=0)).ravel()
    edges = np.concatenate((horizontal_edges, vertical_edges))
    histogram = np.histogram(gray, bins=32, range=(0.0, 1.0))[0].astype(float)
    histogram /= histogram.sum()
    histogram_entropy = -sum(
        probability * math.log(probability)
        for probability in histogram
        if probability > 0
    )
    values = np.array(
        [
            math.log1p(len(data)),
            math.log(width),
            math.log(height),
            math.log(width / height),
            gray.mean(),
            gray.std(),
            *np.quantile(gray, (0.1, 0.5, 0.9)),
            *pixels.mean(axis=(0, 1)),
            *pixels.std(axis=(0, 1)),
            saturation.mean(),
            saturation.std(),
            edges.mean(),
            edges.std(),
            np.quantile(edges, 0.9),
            histogram_entropy,
        ],
        dtype=float,
    )
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("image feature extraction produced an invalid vector")
    return values


def partition_rows(partition: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = bed.load_validation_partition_tasks(
        partition, include_endpoint_labels=False
    )
    for task in tasks:
        raw = np.stack(
            [image_features(task.image_bytes[image_id]) for image_id in task.image_ids]
        )
        task_mean = raw.mean(axis=0)
        task_sd = raw.std(axis=0)
        task_sd[task_sd < 1e-9] = 1.0
        relative = (raw - task_mean) / task_sd
        for position, image_id in enumerate(task.image_ids):
            role = role_audit.image_role(task, image_id)
            if role == "initial":
                continue
            rows.append(
                {
                    "task_id": task.task_id,
                    "image_id": image_id,
                    "features": np.concatenate((raw[position], relative[position])),
                    "endpoint": int(role == "endpoint"),
                }
            )
    return rows


def standardized(
    training: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = training.mean(axis=0)
    sd = training.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (training - mean) / sd, (evaluation - mean) / sd


def fit_logistic_ridge(
    features: np.ndarray, labels: np.ndarray, l2: float
) -> np.ndarray:
    design = np.column_stack((np.ones(len(features)), features))

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ weights
        loss = (
            np.logaddexp(0.0, logits).sum()
            - labels @ logits
            + 0.5 * l2 * (weights[1:] @ weights[1:])
        )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        gradient = design.T @ (probabilities - labels)
        gradient[1:] += l2 * weights[1:]
        return float(loss), gradient

    endpoint_rate = float(labels.mean())
    initial = np.zeros(design.shape[1])
    initial[0] = math.log(endpoint_rate / (1.0 - endpoint_rate))
    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"ridge logistic fit failed: {result.message}")
    return result.x


def predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(features)), features))
    logits = np.clip(design @ weights, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log1p(-probabilities)
        )
    )


def fold_for_task(task_id: str) -> int:
    return int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % NUM_FOLDS


def select_l2(rows: Sequence[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    features = np.stack([row["features"] for row in rows])
    labels = np.array([row["endpoint"] for row in rows], dtype=float)
    folds = np.array([fold_for_task(row["task_id"]) for row in rows])
    losses: dict[str, float] = {}
    for l2 in L2_GRID:
        fold_losses = []
        for fold in range(NUM_FOLDS):
            training = folds != fold
            validation = ~training
            scaled_training, scaled_validation = standardized(
                features[training], features[validation]
            )
            weights = fit_logistic_ridge(
                scaled_training, labels[training], l2
            )
            fold_losses.append(
                log_loss(labels[validation], predict(scaled_validation, weights))
            )
        losses[str(l2)] = sum(fold_losses) / len(fold_losses)
    return min(L2_GRID, key=lambda value: (losses[str(value)], value)), losses


def evaluate(
    training_rows: Sequence[dict[str, Any]],
    confirmation_rows: Sequence[dict[str, Any]],
    l2: float,
) -> dict[str, Any]:
    training_features = np.stack([row["features"] for row in training_rows])
    training_labels = np.array(
        [row["endpoint"] for row in training_rows], dtype=float
    )
    confirmation_features = np.stack(
        [row["features"] for row in confirmation_rows]
    )
    confirmation_labels = np.array(
        [row["endpoint"] for row in confirmation_rows], dtype=float
    )
    scaled_training, scaled_confirmation = standardized(
        training_features, confirmation_features
    )
    weights = fit_logistic_ridge(scaled_training, training_labels, l2)
    probabilities = predict(scaled_confirmation, weights)
    base_rate = float(confirmation_labels.mean())
    baseline = np.full(len(confirmation_labels), base_rate)

    top2_true_positives = 0
    for task_id in sorted({row["task_id"] for row in confirmation_rows}):
        indices = [
            index
            for index, row in enumerate(confirmation_rows)
            if row["task_id"] == task_id
        ]
        selected = sorted(indices, key=lambda index: (-probabilities[index], index))[:2]
        top2_true_positives += int(confirmation_labels[selected].sum())

    feature_names = (
        *(f"raw_{name}" for name in FEATURE_NAMES),
        *(f"relative_{name}" for name in FEATURE_NAMES),
    )
    largest_coefficients = sorted(
        (
            {"feature": name, "coefficient": float(coefficient)}
            for name, coefficient in zip(feature_names, weights[1:], strict=True)
        ),
        key=lambda row: (-abs(row["coefficient"]), row["feature"]),
    )[:10]
    endpoint_rows = int(confirmation_labels.sum())
    return {
        "rows": len(confirmation_rows),
        "endpoint_rows": endpoint_rows,
        "endpoint_base_rate": base_rate,
        "auc": role_audit.binary_auc(
            confirmation_labels.astype(int).tolist(), probabilities.tolist()
        ),
        "brier": float(np.mean((confirmation_labels - probabilities) ** 2)),
        "constant_base_rate_brier": float(
            np.mean((confirmation_labels - baseline) ** 2)
        ),
        "brier_regret": float(
            np.mean((confirmation_labels - probabilities) ** 2)
            - np.mean((confirmation_labels - baseline) ** 2)
        ),
        "log_loss": log_loss(confirmation_labels, probabilities),
        "constant_base_rate_log_loss": log_loss(confirmation_labels, baseline),
        "minimum_probability": float(probabilities.min()),
        "maximum_probability": float(probabilities.max()),
        "top2_true_positives": top2_true_positives,
        "top2_precision": top2_true_positives / (2 * 96),
        "top2_recall": top2_true_positives / endpoint_rows,
        "largest_standardized_coefficients": largest_coefficients,
    }


def run_audit(*, output_path: Path) -> dict[str, Any]:
    observed_role_manifest_sha256 = sha256_file(ROLE_BLINDNESS_MANIFEST)
    role_manifest = json.loads(ROLE_BLINDNESS_MANIFEST.read_text(encoding="utf-8"))
    partitions = {
        name: partition_rows(name)
        for name in ("mechanics", "development", "confirmation")
    }
    training = [*partitions["mechanics"], *partitions["development"]]
    selected_l2, cross_validation = select_l2(training)
    confirmation = evaluate(
        training, partitions["confirmation"], selected_l2
    )
    gates = {
        "bound_prompt_role_blindness_audit_matches": (
            observed_role_manifest_sha256 == ROLE_BLINDNESS_MANIFEST_SHA256
            and role_manifest.get("all_gates_pass") is True
            and role_manifest.get("authorizes_paid_calls") is False
        ),
        "partition_row_counts_match_endpoint_sealed_protocol": (
            {name: len(rows) for name, rows in partitions.items()}
            == {"mechanics": 40, "development": 640, "confirmation": 960}
        ),
        "regularization_selected_without_confirmation_labels": (
            selected_l2 in L2_GRID and len(cross_validation) == len(L2_GRID)
        ),
        "held_out_visual_feature_auc_is_below_0_60": (
            confirmation["auc"] < MAX_CONFIRMATION_AUC
        ),
        "held_out_visual_feature_brier_regret_is_negligible": (
            confirmation["brier_regret"] <= MAX_CONFIRMATION_BRIER_REGRET
        ),
        "held_out_visual_feature_top2_precision_is_below_0_30": (
            confirmation["top2_precision"] < MAX_CONFIRMATION_TOP2_PRECISION
        ),
        "held_out_visual_feature_predictions_remain_below_0_40": (
            confirmation["maximum_probability"] < MAX_CONFIRMATION_PROBABILITY
        ),
    }
    all_gates_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "source_rank_visual_artifact_pass" if all_gates_pass else "fail"
        ),
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "bound_prompt_role_blindness_manifest_sha256": (
            observed_role_manifest_sha256
        ),
        "feature_protocol": {
            "raw_features": list(FEATURE_NAMES),
            "task_relative_features": list(FEATURE_NAMES),
            "total_features": 2 * len(FEATURE_NAMES),
            "training_partitions": ["mechanics", "development"],
            "evaluation_partition": "confirmation",
            "folds": NUM_FOLDS,
            "l2_grid": list(L2_GRID),
            "selected_l2": selected_l2,
            "cross_validation_log_loss": cross_validation,
        },
        "confirmation": confirmation,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "limitations": [
            "This audit tests handcrafted low-level image and compression features, not semantic image understanding.",
            "It cannot rule out memorization of public Bongard-OpenWorld images or concepts by a pretrained VLM.",
            "Passing does not test belief quality, calibration, planning efficacy, or scientific endpoints.",
        ],
    }
    checkpoint(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/bongard_openworld_source_rank_visual_artifact_audit/"
            "bongard-openworld-source-rank-visual-artifact-audit-20260808/"
            "MANIFEST.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = run_audit(output_path=parse_args().output_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "authorizes_paid_calls": result["authorizes_paid_calls"],
                "feature_protocol": result["feature_protocol"],
                "confirmation": result["confirmation"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
