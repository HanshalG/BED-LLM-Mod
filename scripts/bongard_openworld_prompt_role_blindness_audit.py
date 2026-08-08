#!/usr/bin/env python3
"""Audit whether public Bongard prompt metadata predicts hidden image roles."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-prompt-role-blindness-audit-1"
PARTITION_SIZES = {"mechanics": 4, "development": 64, "confirmation": 96}
PUBLIC_IMAGE_IDS = tuple(f"image-{index:02d}" for index in range(bed.NUM_IMAGES))
MAX_CONFIRMATION_AUC = 0.60
MAX_CONFIRMATION_BRIER_REGRET = 0.005
MAX_CONFIRMATION_TOP2_PRECISION = 0.30


def image_role(task: bed.VisualTask, image_id: str) -> str:
    observed = {observed_id for observed_id, _ in task.initial_history}
    if image_id in observed:
        return "initial"
    if image_id in set(task.endpoint_ids):
        return "endpoint"
    if image_id in set(task.candidate_ids):
        return "candidate"
    raise ValueError(f"image {image_id!r} has no protocol role")


def fit_position_predictor(
    tasks: Sequence[bed.VisualTask],
) -> tuple[float, ...]:
    """Estimate P(endpoint | public slot, unobserved) with a Beta(1, 4) prior."""
    endpoint_counts = Counter()
    unobserved_counts = Counter()
    for task in tasks:
        for position, image_id in enumerate(task.image_ids):
            role = image_role(task, image_id)
            if role == "initial":
                continue
            unobserved_counts[position] += 1
            endpoint_counts[position] += int(role == "endpoint")
    return tuple(
        (endpoint_counts[position] + 1.0)
        / (unobserved_counts[position] + 5.0)
        for position in range(bed.NUM_IMAGES)
    )


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        raise ValueError("AUC requires both endpoint and candidate rows")
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def predictor_evaluation(
    tasks: Sequence[bed.VisualTask], probabilities: Sequence[float]
) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    top2_true_positives = 0
    for task in tasks:
        unobserved = [
            (position, image_id)
            for position, image_id in enumerate(task.image_ids)
            if image_role(task, image_id) != "initial"
        ]
        predicted_endpoints = {
            image_id
            for position, image_id in sorted(
                unobserved,
                key=lambda item: (probabilities[item[0]], -item[0]),
                reverse=True,
            )[:2]
        }
        top2_true_positives += len(predicted_endpoints & set(task.endpoint_ids))
        for position, image_id in unobserved:
            labels.append(int(image_role(task, image_id) == "endpoint"))
            scores.append(float(probabilities[position]))

    base_rate = sum(labels) / len(labels)
    brier = sum((label - score) ** 2 for label, score in zip(labels, scores, strict=True)) / len(labels)
    baseline_brier = sum((label - base_rate) ** 2 for label in labels) / len(labels)
    log_loss = -sum(
        label * math.log(score) + (1 - label) * math.log1p(-score)
        for label, score in zip(labels, scores, strict=True)
    ) / len(labels)
    baseline_log_loss = -(
        base_rate * math.log(base_rate)
        + (1.0 - base_rate) * math.log1p(-base_rate)
    )
    top2_total = 2 * len(tasks)
    return {
        "rows": len(labels),
        "endpoint_rows": sum(labels),
        "endpoint_base_rate": base_rate,
        "auc": binary_auc(labels, scores),
        "brier": brier,
        "constant_base_rate_brier": baseline_brier,
        "brier_regret": brier - baseline_brier,
        "log_loss": log_loss,
        "constant_base_rate_log_loss": baseline_log_loss,
        "top2_true_positives": top2_true_positives,
        "top2_precision": top2_true_positives / top2_total,
        "top2_recall": top2_true_positives / sum(labels),
    }


def partition_summary(tasks: Sequence[bed.VisualTask]) -> dict[str, Any]:
    role_by_position = {
        position: Counter(
            image_role(task, task.image_ids[position]) for task in tasks
        )
        for position in range(bed.NUM_IMAGES)
    }
    endpoint_pairs = Counter(tuple(sorted(task.endpoint_ids)) for task in tasks)
    return {
        "tasks": len(tasks),
        "role_counts": dict(
            sorted(
                Counter(
                    image_role(task, image_id)
                    for task in tasks
                    for image_id in task.image_ids
                ).items()
            )
        ),
        "endpoint_counts_by_public_position": [
            role_by_position[position]["endpoint"]
            for position in range(bed.NUM_IMAGES)
        ],
        "initial_counts_by_public_position": [
            role_by_position[position]["initial"]
            for position in range(bed.NUM_IMAGES)
        ],
        "unique_endpoint_id_pairs": len(endpoint_pairs),
        "maximum_endpoint_pair_repetitions": max(endpoint_pairs.values()),
    }


def public_prompt_errors(task: bed.VisualTask) -> list[str]:
    messages = bed.build_belief_messages(task, task.initial_history)
    errors = bed.prompt_hidden_state_errors(task, task.initial_history, messages)
    request = bed.request_payload(messages)
    if request.get("image_order") != list(task.image_ids):
        errors.append("nonexact_image_order")
    if set(request) != {
        "task",
        "task_id",
        "image_order",
        "observed_labels",
        "requirements",
    }:
        errors.append("unexpected_public_request_field")
    return sorted(set(errors))


def run_audit(*, output_path: Path) -> dict[str, Any]:
    partitions = {
        name: bed.load_validation_partition_tasks(
            name, include_endpoint_labels=False
        )
        for name in PARTITION_SIZES
    }
    all_tasks = [task for tasks in partitions.values() for task in tasks]
    training_tasks = [*partitions["mechanics"], *partitions["development"]]
    probabilities = fit_position_predictor(training_tasks)
    confirmation = predictor_evaluation(
        partitions["confirmation"], probabilities
    )
    role_sets = [
        {
            image_role(task, task.image_ids[position]) for task in all_tasks
        }
        for position in range(bed.NUM_IMAGES)
    ]
    prompt_errors = {
        name: public_prompt_errors(tasks[0])
        for name, tasks in partitions.items()
    }

    gates = {
        "partition_sizes_match_frozen_protocol": all(
            len(partitions[name]) == size
            for name, size in PARTITION_SIZES.items()
        ),
        "task_ids_are_unique": (
            len({task.task_id for task in all_tasks}) == len(all_tasks)
        ),
        "all_tasks_use_the_exact_opaque_public_id_set": all(
            task.image_ids == PUBLIC_IMAGE_IDS for task in all_tasks
        ),
        "all_tasks_have_balanced_disjoint_roles": all(
            len(task.initial_history) == 4
            and len(task.candidate_ids) == 8
            and len(task.endpoint_ids) == 2
            and len(
                {image_id for image_id, _ in task.initial_history}
                | set(task.candidate_ids)
                | set(task.endpoint_ids)
            )
            == bed.NUM_IMAGES
            for task in all_tasks
        ),
        "endpoint_truth_is_sealed_in_every_loaded_task": all(
            not (set(task.endpoint_ids) & set(task.actual_labels))
            for task in all_tasks
        ),
        "every_public_position_spans_every_protocol_role": all(
            roles == {"initial", "candidate", "endpoint"}
            for roles in role_sets
        ),
        "sampled_public_prompts_have_no_hidden_state_or_role_errors": not any(
            prompt_errors.values()
        ),
        "held_out_position_predictor_auc_is_below_0_60": (
            confirmation["auc"] < MAX_CONFIRMATION_AUC
        ),
        "held_out_position_predictor_has_negligible_brier_regret": (
            confirmation["brier_regret"]
            <= MAX_CONFIRMATION_BRIER_REGRET
        ),
        "held_out_top2_position_precision_is_below_0_30": (
            confirmation["top2_precision"]
            < MAX_CONFIRMATION_TOP2_PRECISION
        ),
    }
    all_gates_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "prompt_role_blindness_pass" if all_gates_pass else "fail",
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "partitions": {
            name: partition_summary(tasks)
            for name, tasks in partitions.items()
        },
        "position_predictor": {
            "training_partitions": ["mechanics", "development"],
            "evaluation_partition": "confirmation",
            "beta_prior": {"endpoint": 1.0, "candidate": 4.0},
            "endpoint_probabilities_by_public_position": list(probabilities),
            "confirmation": confirmation,
        },
        "public_prompt_errors": prompt_errors,
        "role_sets_by_public_position": [sorted(roles) for roles in role_sets],
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "limitations": [
            "The audit tests public IDs, prompt order, and request metadata; it cannot rule out memorization of a public benchmark image.",
            "The source protocol holds out class index 6 as the endpoint, but source paths and source positions are absent from the public prompt.",
            "Passing does not test semantic belief quality or non-myopic planning efficacy.",
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
            / "results/nonmyopic/bongard_openworld_prompt_role_blindness_audit/"
            "bongard-openworld-prompt-role-blindness-audit-20260808/"
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
                "position_predictor": result["position_predictor"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
