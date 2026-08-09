#!/usr/bin/env python3
"""Evaluate frozen DINO and SigLIP plans after a Bongard stage is verified."""

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

from scripts import bongard_openworld_dinov2_outcome as dino_outcome
from scripts import bongard_openworld_siglip_classical_baseline as siglip
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-classical-suite-outcome-1"
SIGLIP_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_SIGLIP_CLASSICAL_BASELINE_PROTOCOL_20260809.md"
)
SIGLIP_PROTOCOL_SHA256 = (
    "e4b126b9b95f18d62a8968959c67b1b5e2c9e8b01c7794b0dc23e7b8a80f5b86"
)
SIGLIP_PLANS_DIR = (
    REPO_ROOT
    / "results/nonmyopic/bongard_openworld_siglip_classical_baseline/"
    "plans-20260809"
)
SIGLIP_PLANS_MANIFEST_SHA256 = (
    "95c00d948ee77589995c3434e3eb01b0a908b496fe2d984b859dbae4032d057d"
)
SIGLIP_PLANS_SHA256 = "a41cc3b01d18fa9008f67d6f60a3f113b8f3194b73e932b4e2c3cfc83215f587"
SIGLIP_IMPLEMENTATION_SHA256 = (
    "4fd19b2e9fabca09ef3d20e31d163dd7133784e5992d1f9afdbd77e495e84211"
)
COMPARISON_SEED = 20262808


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_frozen_inputs() -> dict[str, Any]:
    dino = dino_outcome.verify_frozen_inputs()
    observed_siglip = {
        "protocol": siglip.sha256_file(SIGLIP_PROTOCOL),
        "plans_manifest": siglip.sha256_file(SIGLIP_PLANS_DIR / "MANIFEST.json"),
        "plans": siglip.sha256_file(SIGLIP_PLANS_DIR / "PLANS.json"),
        "baseline_implementation": siglip.sha256_file(
            REPO_ROOT / "scripts/bongard_openworld_siglip_classical_baseline.py"
        ),
    }
    expected_siglip = {
        "protocol": SIGLIP_PROTOCOL_SHA256,
        "plans_manifest": SIGLIP_PLANS_MANIFEST_SHA256,
        "plans": SIGLIP_PLANS_SHA256,
        "baseline_implementation": SIGLIP_IMPLEMENTATION_SHA256,
    }
    if observed_siglip != expected_siglip:
        raise ValueError(
            "frozen SigLIP outcome inputs changed: expected "
            f"{expected_siglip}, observed {observed_siglip}"
        )
    manifest = _load(SIGLIP_PLANS_DIR / "MANIFEST.json")
    if (
        manifest.get("all_gates_pass") is not True
        or manifest.get("endpoint_labels_accessed") is not False
        or manifest.get("candidate_labels_accessed") is not False
        or manifest.get("plans_sha256") != SIGLIP_PLANS_SHA256
    ):
        raise ValueError("SigLIP plans manifest is not the endpoint-sealed pass")
    return {"dino": dino, "siglip": observed_siglip}


def load_siglip_plan_rows(stage: str) -> list[dict[str, Any]]:
    plans = _load(SIGLIP_PLANS_DIR / "PLANS.json")
    rows = [row for row in plans.get("tasks", []) if row.get("partition") == stage]
    expected = dino_outcome.STAGE_TASK_COUNTS[stage]
    if len(rows) != expected or len({row.get("task_id") for row in rows}) != len(rows):
        raise ValueError(f"frozen SigLIP {stage} plans are incomplete")
    return rows


def _paired_luna_minus_siglip(
    *,
    trees: Sequence[Mapping[str, Any]],
    siglip_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    tree_by_task = {str(tree["task_id"]): tree for tree in trees}
    siglip_by_policy = {
        policy: {str(row["task_id"]): row for row in rows}
        for policy, rows in siglip_rows.items()
    }
    task_ids = sorted(tree_by_task)
    if any(set(rows) != set(task_ids) for rows in siglip_by_policy.values()):
        raise ValueError("Luna and SigLIP task identities do not pair exactly")
    metrics = (
        "mean_brier",
        "mean_log_loss",
        "accuracy",
        "mean_truth_probability",
    )
    comparisons = {
        "luna_myopic_minus_siglip_myopic": ("myopic_width", "siglip_myopic"),
        "luna_dynamic_minus_siglip_depth2": (
            "dynamic_depth2",
            "siglip_depth2",
        ),
    }
    return {
        name: {
            metric: siglip.shared.paired_summary(
                [
                    float(
                        tree_by_task[task_id]["policies"][luna_policy][
                            "endpoint"
                        ][metric]
                    )
                    - float(siglip_by_policy[siglip_policy][task_id][metric])
                    for task_id in task_ids
                ],
                seed=COMPARISON_SEED + comparison_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
        for comparison_index, (name, (luna_policy, siglip_policy)) in enumerate(
            comparisons.items()
        )
    }


def build_outcome(
    *,
    stage: str,
    stage_result: Mapping[str, Any],
    tasks: Sequence[bed.VisualTask],
    dino_plan_rows: Sequence[Mapping[str, Any]],
    siglip_plan_rows: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
    frozen_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    trees = stage_result.get("trees")
    if not isinstance(trees, list) or len(trees) != dino_outcome.STAGE_TASK_COUNTS[stage]:
        raise ValueError(f"verified {stage} result lacks complete Luna trees")
    dino_result = dino_outcome.build_outcome(
        stage=stage,
        stage_result=stage_result,
        tasks=tasks,
        plan_rows=dino_plan_rows,
        authorization=authorization,
        frozen_inputs=frozen_inputs["dino"],
    )
    siglip_result = siglip.evaluate_opened_partition(tasks, siglip_plan_rows)
    siglip_comparisons = _paired_luna_minus_siglip(
        trees=trees,
        siglip_rows=siglip_result["rows"],
    )
    finite_values = [
        value
        for policy in siglip_result["pooled"].values()
        for value in policy.values()
    ]
    task_ids = {task.task_id for task in tasks}
    gates = {
        "stage_authorization_verified_before_label_loading": (
            authorization.get("verified") is True
        ),
        "dino_comparator_complete": (
            dino_result.get("all_gates_pass") is True
        ),
        "exact_task_count_and_identity_pairing": (
            task_ids
            == {row["task_id"] for row in dino_plan_rows}
            == {row["task_id"] for row in siglip_plan_rows}
            == {tree["task_id"] for tree in trees}
        ),
        "all_siglip_metrics_are_finite": all(
            math.isfinite(float(value)) for value in finite_values
        ),
        "both_frozen_siglip_policies_reported": set(siglip_result["pooled"])
        == set(siglip.POLICIES),
        "both_registered_luna_siglip_comparisons_reported": set(
            siglip_comparisons
        )
        == {
            "luna_myopic_minus_siglip_myopic",
            "luna_dynamic_minus_siglip_depth2",
        },
    }
    all_gates_pass = all(gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "classical_suite_complete" if all_gates_pass else "fail",
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
        "dino": dino_result["dino"],
        "siglip": siglip_result,
        "luna_pooled": dino_result["luna_pooled"],
        "paired_luna_minus_dino": dino_result["paired_luna_minus_dino"],
        "paired_luna_minus_siglip": siglip_comparisons,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "interpretation": (
            "Lower paired differences favor Luna for Brier/log loss; higher paired "
            "differences favor Luna for accuracy/truth probability. DINO and SigLIP "
            "must both be reported regardless of direction."
        ),
    }


def run_outcome(
    *,
    stage: str,
    result_path: Path,
    output_path: Path,
    block_results: Sequence[Path] = (),
    wrapper_result: Path | None = None,
    authorizer: Callable[..., dict[str, Any]] = dino_outcome.authorize_stage,
    task_loader: Callable[[str], list[bed.VisualTask]] = dino_outcome.load_opened_tasks,
) -> dict[str, Any]:
    if stage not in dino_outcome.STAGES:
        raise ValueError(f"unknown classical outcome stage {stage!r}")
    if output_path.exists():
        raise FileExistsError(output_path)
    frozen_inputs = verify_frozen_inputs()
    dino_plan_rows = dino_outcome.load_plan_rows(stage)
    siglip_plan_rows = load_siglip_plan_rows(stage)
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
        dino_plan_rows=dino_plan_rows,
        siglip_plan_rows=siglip_plan_rows,
        authorization=authorization,
        frozen_inputs=frozen_inputs,
    )
    result["stage_result_path"] = str(result_path)
    result["stage_result_sha256"] = siglip.sha256_file(result_path)
    checkpoint(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=dino_outcome.STAGES, required=True)
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
