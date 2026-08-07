#!/usr/bin/env python3
"""Audit per-draw RegretBench ranking fidelity with zero model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_BRANCH_DRAW_FIDELITY_PROTOCOL_20260807.md"
)
BINDING = REPO_ROOT / (
    "results/nonmyopic/regretbench_branch_draw_fidelity/EXECUTION_BINDING.json"
)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 202608410000
ALLOWED_INTERFACES = {
    "regretbench-deepseek-dynamic-depth2-policy-1": (
        "dynamic_depth2",
        "myopic_refresh_brier",
    ),
    "regretbench-deepseek-dynamic-depth2-confirmation-1": (
        "dynamic_depth2",
        "myopic_refresh_brier",
    ),
    "regretbench-deepseek-smc-dynamic-depth2-experiment-1": (
        "smc_dynamic_depth2",
        "smc_myopic_refresh_brier",
    ),
    "regretbench-deepseek-smc-confirmation-1": (
        "smc_dynamic_depth2",
        "smc_myopic_refresh_brier",
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"nonnumeric value: {label}")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"nonfinite value: {label}")
    return output


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if float(np.std(left_rank)) == 0.0 or float(np.std(right_rank)) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _point_metrics(predicted: np.ndarray, realized: np.ndarray) -> dict[str, Any]:
    nonzero = (np.abs(predicted) > 1e-12) & (np.abs(realized) > 1e-12)
    error = predicted - realized
    return {
        "spearman": spearman(predicted, realized),
        "sign_accuracy": (
            float(np.mean(np.sign(predicted[nonzero]) == np.sign(realized[nonzero])))
            if np.any(nonzero)
            else None
        ),
        "sign_accuracy_n": int(np.sum(nonzero)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def _bootstrap(
    draws: Sequence[np.ndarray],
    realized: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    correlations: list[list[float]] = []
    for indexes in rng.integers(0, realized.size, size=(samples, realized.size)):
        values = [spearman(draw[indexes], realized[indexes]) for draw in draws]
        if all(value is not None for value in values):
            correlations.append([float(value) for value in values])
    if not correlations:
        return {
            "retained_samples": 0,
            "requested_samples": samples,
            "seed": seed,
            "correlations": [
                {"ci95": [None, None], "probability_positive": None}
                for _ in draws
            ],
            "ensemble_exceeds_draw0_probability": None,
            "ensemble_exceeds_draw1_probability": None,
            "ensemble_exceeds_both_probability": None,
        }
    array = np.asarray(correlations, dtype=float)
    ensemble = array[:, 2]
    return {
        "retained_samples": int(array.shape[0]),
        "requested_samples": samples,
        "seed": seed,
        "correlations": [
            {
                "ci95": [float(value) for value in np.quantile(array[:, index], [0.025, 0.975])],
                "probability_positive": float(np.mean(array[:, index] > 0.0)),
            }
            for index in range(array.shape[1])
        ],
        "ensemble_exceeds_draw0_probability": float(np.mean(ensemble > array[:, 0])),
        "ensemble_exceeds_draw1_probability": float(np.mean(ensemble > array[:, 1])),
        "ensemble_exceeds_both_probability": float(
            np.mean(ensemble > np.maximum(array[:, 0], array[:, 1]))
        ),
    }


def validate_binding() -> dict[str, Any]:
    binding = _load(BINDING)
    if binding.get("status") != "frozen_before_any_regretbench_policy_response":
        raise ValueError("branch-draw fidelity binding status changed")
    expected = {
        "protocol": PROTOCOL,
        "script": Path(__file__).resolve(),
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)):
            raise ValueError(f"branch-draw fidelity {name} path changed")
        if row.get("sha256") != sha256_file(path):
            raise ValueError(f"branch-draw fidelity {name} hash changed")
    return binding


def _validated_inputs(run_dir: Path) -> tuple[dict[str, Any], str, str]:
    result_path = run_dir / "RESULT.json"
    verification_path = run_dir / "VERIFICATION.json"
    result = _load(result_path)
    verification = _load(verification_path)
    result_sha = sha256_file(result_path)
    verification_sha = sha256_file(verification_path)
    checks = verification.get("checks") or {}
    artifacts = verification.get("artifact_sha256") or {}
    if (
        verification.get("status") != "verified"
        or verification.get("mismatches") != []
        or checks.get("reported_result_matches_replay") is not True
        or artifacts.get("RESULT.json") != result_sha
        or verification.get("result_status") != result.get("status")
    ):
        raise ValueError("result is not bound to a clean independent verification")
    if result.get("interface_version") not in ALLOWED_INTERFACES:
        raise ValueError("unsupported RegretBench result interface")
    if len(result.get("tasks") or []) != 64:
        raise ValueError("branch-draw fidelity requires exactly 64 tasks")
    return result, result_sha, verification_sha


def analyze(
    result: Mapping[str, Any],
    *,
    result_sha256: str,
    verification_sha256: str,
    binding_sha256: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    interface = str(result.get("interface_version"))
    dynamic, refresh = ALLOWED_INTERFACES[interface]
    mechanics = result.get("mechanics_gates") or {}
    if result.get("status") == "mechanics_failed" or mechanics.get("all_pass") is not True:
        return {
            "schema_version": 1,
            "interface_version": "regretbench-branch-draw-fidelity-1",
            "status": "unavailable_mechanics_failed",
            "source_interface": interface,
            "result_sha256": result_sha256,
            "verification_sha256": verification_sha256,
            "execution_binding_sha256": binding_sha256,
            "metrics": None,
            "can_change_status_authorization_or_claim_tier": False,
            "model_calls": 0,
            "cost_usd": 0.0,
        }

    draw_values = [[], []]
    ensemble_values = []
    realized_values = []
    task_ids = []
    for task in result["tasks"]:
        selected = task.get("selected_roots") or {}
        dynamic_root = int(selected[dynamic])
        refresh_root = int(selected[refresh])
        if dynamic_root == refresh_root:
            continue
        per_draw = task.get("conditioned_draw_root_risks") or []
        averaged = task.get("conditioned_root_risks") or []
        if len(per_draw) != 2 or any(len(row) != 4 for row in per_draw) or len(averaged) != 4:
            raise ValueError("stored branch-draw risk shape changed")
        predictions = []
        for draw in range(2):
            value = _finite(per_draw[draw][refresh_root]["brier"], "draw refresh risk") - _finite(
                per_draw[draw][dynamic_root]["brier"], "draw dynamic risk"
            )
            draw_values[draw].append(value)
            predictions.append(value)
        ensemble = sum(predictions) / 2.0
        stored_ensemble = _finite(averaged[refresh_root]["brier"], "averaged refresh risk") - _finite(
            averaged[dynamic_root]["brier"], "averaged dynamic risk"
        )
        if not math.isclose(ensemble, stored_ensemble, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("stored averaged risk is not the mean of both draws")
        policies = task.get("policies") or {}
        realized = _finite(policies[refresh]["brier"], "realized refresh Brier") - _finite(
            policies[dynamic]["brier"], "realized dynamic Brier"
        )
        ensemble_values.append(ensemble)
        realized_values.append(realized)
        task_ids.append(str(task.get("task_id")))
    if len(task_ids) < 2:
        raise ValueError("fewer than two changed-root tasks are available")

    draws = [np.asarray(row, dtype=float) for row in draw_values]
    ensemble = np.asarray(ensemble_values, dtype=float)
    realized = np.asarray(realized_values, dtype=float)
    bootstrap = _bootstrap([draws[0], draws[1], ensemble], realized, samples=samples, seed=seed)
    labels = ("draw0", "draw1", "ensemble")
    point = [_point_metrics(values, realized) for values in (*draws, ensemble)]
    for index, row in enumerate(point):
        row.update(bootstrap["correlations"][index])
    gap = draws[0] - draws[1]
    metrics = {
        "changed_root_task_count": len(task_ids),
        "changed_root_task_ids": task_ids,
        "predicted_to_realized": dict(zip(labels, point, strict=True)),
        "draw_prediction_agreement": {
            "pearson": _pearson(draws[0], draws[1]),
            "mean_absolute_gap": float(np.mean(np.abs(gap))),
            "root_mean_square_gap": float(np.sqrt(np.mean(gap**2))),
        },
        "bootstrap": {
            key: value
            for key, value in bootstrap.items()
            if key != "correlations"
        },
    }
    return {
        "schema_version": 1,
        "interface_version": "regretbench-branch-draw-fidelity-1",
        "status": "complete_descriptive_non_gating",
        "source_interface": interface,
        "source_result_status": result.get("status"),
        "result_sha256": result_sha256,
        "verification_sha256": verification_sha256,
        "execution_binding_sha256": binding_sha256,
        "metrics": metrics,
        "interpretation_boundary": (
            "descriptive variance-versus-shared-error evidence; no causal decomposition"
        ),
        "can_change_status_authorization_or_claim_tier": False,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def run(run_dir: Path) -> dict[str, Any]:
    validate_binding()
    result, result_sha, verification_sha = _validated_inputs(run_dir)
    return analyze(
        result,
        result_sha256=result_sha,
        verification_sha256=verification_sha,
        binding_sha256=sha256_file(BINDING),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.run_dir.resolve())
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
