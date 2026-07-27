#!/usr/bin/env python3
"""Audit fixed-persona non-myopic opportunities in WorldValuesBench."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


SOURCE_REPOSITORY = "https://github.com/yw3453/adaptive-query-ai-persona-priors"
SOURCE_COMMIT = "fbd8e19eed6af960b64e3afae13e3bcf4020b73f"
SOURCE_RELATIVE_PATH = Path(
    "data/WorldValuesBench/worldvalues_simulated.csv"
)
SOURCE_SHA256 = (
    "24d5d7b3a9bdf94894952dc7bfff39d409f8f78c8131ccd7de87fb292068638e"
)
QUESTION_ID_HASH = (
    "1309815471ee599723f58af7a394676b42027178bd715787eef78a2d11f11971"
)
TASK_SPEC_HASH = (
    "da1ae2fdc0d1f0495e8e6c01c78a1c656dab0416076419c65f2e3d5e52d5602d"
)

SCHEMA_VERSION = 1
SEED = 24711
NUM_TASKS = 20
NUM_TARGET_QUESTIONS = 8
NUM_CANDIDATE_QUESTIONS = 24
EXPECTED_PERSONAS = 2058
EXPECTED_QUESTIONS = 91
EXPECTED_CATEGORIES = 4
TOLERANCE = 1e-10

MIN_INITIAL_TARGET_ENTROPY = 0.5
MIN_DYNAMIC_SCORE_TASKS = 20
MIN_ROOT_CHANGES = 5
MIN_STRICT_TRADEOFFS = 4
MIN_MEAN_FINAL_ADVANTAGE = 0.001
MIN_STRICT_TOTAL_FINAL_ADVANTAGE = 0.01
MIN_MEAN_STRICT_IMMEDIATE_SACRIFICE = 0.002


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _question_hash(question_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(question_ids).encode("utf-8")).hexdigest()


def frozen_task_specs(question_ids: Sequence[str]) -> list[dict[str, Any]]:
    if len(question_ids) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"expected {EXPECTED_QUESTIONS} questions, got {len(question_ids)}"
        )
    rng = np.random.default_rng(SEED)
    specs = []
    for task_index in range(NUM_TASKS):
        permutation = rng.permutation(len(question_ids))
        specs.append(
            {
                "task_index": task_index,
                "target_questions": [
                    question_ids[index]
                    for index in permutation[:NUM_TARGET_QUESTIONS]
                ],
                "candidate_questions": [
                    question_ids[index]
                    for index in permutation[
                        NUM_TARGET_QUESTIONS : (
                            NUM_TARGET_QUESTIONS
                            + NUM_CANDIDATE_QUESTIONS
                        )
                    ]
                ],
            }
        )
    return specs


def load_persona_probabilities(
    source_path: Path,
) -> tuple[list[str], list[str], np.ndarray]:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("persona response matrix SHA-256 does not match")

    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        question_ids = [str(value) for value in header[1:]]
        persona_ids: list[str] = []
        rows: list[list[list[float]]] = []
        for raw_row in reader:
            if len(raw_row) != len(header):
                raise ValueError("persona response row has unexpected width")
            persona_ids.append(str(raw_row[0]))
            parsed_row = []
            for raw_cell in raw_row[1:]:
                values = json.loads(raw_cell)
                if not isinstance(values, list):
                    raise ValueError("persona response cell is not a list")
                parsed_row.append([float(value) for value in values])
            rows.append(parsed_row)

    probabilities = np.asarray(rows, dtype=np.float64)
    expected_shape = (
        EXPECTED_PERSONAS,
        EXPECTED_QUESTIONS,
        EXPECTED_CATEGORIES,
    )
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"expected probability shape {expected_shape}, "
            f"got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("persona response matrix contains non-finite values")
    if (probabilities < 0.0).any():
        raise ValueError("persona response matrix contains negative values")
    totals = probabilities.sum(axis=2, keepdims=True)
    if (totals <= 0.0).any():
        raise ValueError("persona response matrix contains zero-mass rows")
    probabilities = probabilities / totals

    if len(set(persona_ids)) != len(persona_ids):
        raise ValueError("persona IDs are not unique")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("question IDs are not unique")
    if _question_hash(question_ids) != QUESTION_ID_HASH:
        raise ValueError("ordered question-ID hash does not reproduce")
    return persona_ids, question_ids, probabilities


def categorical_entropy(probabilities: np.ndarray) -> np.ndarray:
    terms = np.zeros_like(probabilities, dtype=np.float64)
    positive = probabilities > 0.0
    terms[positive] = (
        -probabilities[positive] * np.log(probabilities[positive])
    )
    return terms.sum(axis=-1)


def target_entropy(
    weights: np.ndarray,
    target_probabilities: np.ndarray,
) -> float:
    predictive = np.einsum(
        "n,ntk->tk", weights, target_probabilities, optimize=True
    )
    return float(categorical_entropy(predictive).mean())


def posterior_branches(
    weights: np.ndarray,
    question_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outcome_probabilities = weights @ question_probabilities
    weighted = weights[:, None] * question_probabilities
    posteriors = np.zeros(
        (question_probabilities.shape[1], len(weights)), dtype=np.float64
    )
    for outcome_index, outcome_probability in enumerate(
        outcome_probabilities
    ):
        if outcome_probability > TOLERANCE:
            posteriors[outcome_index] = (
                weighted[:, outcome_index] / outcome_probability
            )
        else:
            posteriors[outcome_index] = weights
    return outcome_probabilities, posteriors


def expected_target_entropy_after_question(
    weights: np.ndarray,
    question_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
) -> float:
    outcome_probabilities, posteriors = posterior_branches(
        weights, question_probabilities
    )
    predictive = np.einsum(
        "on,ntk->otk", posteriors, target_probabilities, optimize=True
    )
    branch_entropies = categorical_entropy(predictive).mean(axis=1)
    return float(outcome_probabilities @ branch_entropies)


def analyze_task(
    task_spec: dict[str, Any],
    question_ids: Sequence[str],
    probabilities: np.ndarray,
) -> dict[str, Any]:
    question_index = {
        question_id: index for index, question_id in enumerate(question_ids)
    }
    target_indices = [
        question_index[question_id]
        for question_id in task_spec["target_questions"]
    ]
    candidate_indices = [
        question_index[question_id]
        for question_id in task_spec["candidate_questions"]
    ]
    target_probabilities = probabilities[:, target_indices, :]
    candidate_probabilities = probabilities[:, candidate_indices, :]
    weights = np.full(
        probabilities.shape[0],
        1.0 / float(probabilities.shape[0]),
        dtype=np.float64,
    )
    initial_entropy = target_entropy(weights, target_probabilities)

    immediate_costs = np.asarray(
        [
            expected_target_entropy_after_question(
                weights,
                candidate_probabilities[:, root_index, :],
                target_probabilities,
            )
            for root_index in range(len(candidate_indices))
        ],
        dtype=np.float64,
    )

    final_costs = np.zeros(len(candidate_indices), dtype=np.float64)
    branch_followup_indices: list[list[int]] = []
    branch_followup_costs: list[list[float]] = []
    for root_index in range(len(candidate_indices)):
        outcome_probabilities, posteriors = posterior_branches(
            weights, candidate_probabilities[:, root_index, :]
        )
        selected_indices = []
        selected_costs = []
        for outcome_index, branch_weights in enumerate(posteriors):
            costs = np.full(len(candidate_indices), np.inf, dtype=np.float64)
            for followup_index in range(len(candidate_indices)):
                if followup_index == root_index:
                    continue
                costs[followup_index] = (
                    expected_target_entropy_after_question(
                        branch_weights,
                        candidate_probabilities[:, followup_index, :],
                        target_probabilities,
                    )
                )
            selected_index = int(np.argmin(costs))
            selected_indices.append(selected_index)
            selected_costs.append(float(costs[selected_index]))
        final_costs[root_index] = float(
            outcome_probabilities @ np.asarray(selected_costs)
        )
        branch_followup_indices.append(selected_indices)
        branch_followup_costs.append(selected_costs)

    myopic_root_index = int(np.argmin(immediate_costs))
    adaptive_root_index = int(np.argmin(final_costs))
    immediate_sacrifice = float(
        immediate_costs[adaptive_root_index]
        - immediate_costs[myopic_root_index]
    )
    final_advantage = float(
        final_costs[myopic_root_index] - final_costs[adaptive_root_index]
    )
    root_changed = adaptive_root_index != myopic_root_index
    strict_tradeoff = (
        root_changed
        and immediate_sacrifice > TOLERANCE
        and final_advantage > TOLERANCE
    )

    candidate_questions = list(task_spec["candidate_questions"])
    return {
        "task_index": int(task_spec["task_index"]),
        "target_questions": list(task_spec["target_questions"]),
        "candidate_questions": candidate_questions,
        "initial_target_entropy": initial_entropy,
        "myopic_root_index": myopic_root_index,
        "myopic_root_question": candidate_questions[myopic_root_index],
        "adaptive_d2_root_index": adaptive_root_index,
        "adaptive_d2_root_question": candidate_questions[adaptive_root_index],
        "myopic_immediate_cost": float(immediate_costs[myopic_root_index]),
        "adaptive_d2_immediate_cost": float(
            immediate_costs[adaptive_root_index]
        ),
        "myopic_root_final_cost": float(final_costs[myopic_root_index]),
        "adaptive_d2_final_cost": float(final_costs[adaptive_root_index]),
        "immediate_sacrifice": immediate_sacrifice,
        "final_advantage": final_advantage,
        "root_changed": root_changed,
        "strict_tradeoff": strict_tradeoff,
        "immediate_cost_range": float(
            immediate_costs.max() - immediate_costs.min()
        ),
        "final_cost_range": float(final_costs.max() - final_costs.min()),
        "root_scores": [
            {
                "question": candidate_questions[index],
                "immediate_cost": float(immediate_costs[index]),
                "adaptive_d2_final_cost": float(final_costs[index]),
                "branch_followup_questions": [
                    candidate_questions[followup_index]
                    for followup_index in branch_followup_indices[index]
                ],
                "branch_followup_costs": branch_followup_costs[index],
            }
            for index in range(len(candidate_questions))
        ],
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict = [record for record in records if record["strict_tradeoff"]]
    summary = {
        "num_tasks": len(records),
        "mean_initial_target_entropy": _mean(
            [float(record["initial_target_entropy"]) for record in records]
        ),
        "dynamic_immediate_score_task_count": sum(
            int(float(record["immediate_cost_range"]) > TOLERANCE)
            for record in records
        ),
        "dynamic_final_score_task_count": sum(
            int(float(record["final_cost_range"]) > TOLERANCE)
            for record in records
        ),
        "root_change_count": sum(
            int(record["root_changed"]) for record in records
        ),
        "strict_tradeoff_count": len(strict),
        "mean_final_advantage": _mean(
            [float(record["final_advantage"]) for record in records]
        ),
        "strict_total_final_advantage": sum(
            float(record["final_advantage"]) for record in strict
        ),
        "mean_strict_final_advantage": _mean(
            [float(record["final_advantage"]) for record in strict]
        ),
        "mean_strict_immediate_sacrifice": _mean(
            [float(record["immediate_sacrifice"]) for record in strict]
        ),
    }
    gates = {
        "exactly_20_tasks": summary["num_tasks"] == NUM_TASKS,
        "mean_initial_target_entropy_at_least_0_5": (
            summary["mean_initial_target_entropy"]
            >= MIN_INITIAL_TARGET_ENTROPY
        ),
        "all_immediate_scores_dynamic": (
            summary["dynamic_immediate_score_task_count"]
            >= MIN_DYNAMIC_SCORE_TASKS
        ),
        "all_final_scores_dynamic": (
            summary["dynamic_final_score_task_count"]
            >= MIN_DYNAMIC_SCORE_TASKS
        ),
        "root_changes_at_least_5": (
            summary["root_change_count"] >= MIN_ROOT_CHANGES
        ),
        "strict_tradeoffs_at_least_4": (
            summary["strict_tradeoff_count"] >= MIN_STRICT_TRADEOFFS
        ),
        "mean_final_advantage_at_least_0_001": (
            summary["mean_final_advantage"] >= MIN_MEAN_FINAL_ADVANTAGE
        ),
        "strict_total_final_advantage_at_least_0_01": (
            summary["strict_total_final_advantage"]
            >= MIN_STRICT_TOTAL_FINAL_ADVANTAGE
        ),
        "mean_strict_immediate_sacrifice_at_least_0_002": (
            summary["mean_strict_immediate_sacrifice"]
            >= MIN_MEAN_STRICT_IMMEDIATE_SACRIFICE
        ),
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_audit(source_root: Path) -> dict[str, Any]:
    source_path = source_root / SOURCE_RELATIVE_PATH
    persona_ids, question_ids, probabilities = load_persona_probabilities(
        source_path
    )
    task_specs = frozen_task_specs(question_ids)
    if _canonical_hash(task_specs) != TASK_SPEC_HASH:
        raise AssertionError("frozen task-spec hash does not reproduce")

    records = []
    for task_spec in task_specs:
        record = analyze_task(task_spec, question_ids, probabilities)
        records.append(record)
        print(
            f"completed={len(records)}/{len(task_specs)} "
            f"strict={int(record['strict_tradeoff'])}",
            file=sys.stderr,
        )
    summary = summarize(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "opportunity_passed"
            if summary["gates"]["all_pass"]
            else "opportunity_failed"
        ),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "relative_path": str(SOURCE_RELATIVE_PATH),
            "sha256": SOURCE_SHA256,
            "persona_count": len(persona_ids),
            "question_count": len(question_ids),
            "category_count": probabilities.shape[2],
        },
        "protocol": {
            "seed": SEED,
            "task_spec_hash": TASK_SPEC_HASH,
            "question_id_hash": QUESTION_ID_HASH,
            "num_tasks": NUM_TASKS,
            "num_target_questions": NUM_TARGET_QUESTIONS,
            "num_candidate_questions": NUM_CANDIDATE_QUESTIONS,
            "prior": "uniform_over_all_personas",
            "objective": "mean_target_predictive_entropy_nats",
            "myopic_policy": "minimum_expected_entropy_after_root",
            "adaptive_d2_policy": (
                "minimum_expected_entropy_after_root_and_"
                "answer_conditioned_best_followup"
            ),
        },
        "summary": summary,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_audit(args.source_root)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"status={payload['status']}")
    print(f"output={args.output_path}")
    return 0 if payload["status"] == "opportunity_passed" else 1


if __name__ == "__main__":
    sys.exit(main())
