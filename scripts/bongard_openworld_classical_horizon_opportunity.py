#!/usr/bin/env python3
"""Freeze and evaluate endpoint-blind Bongard horizon-opportunity strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_compute_matched_control as compute_control
from scripts import bongard_openworld_dinov2_outcome as stage_outcome
from scripts import bongard_openworld_luna_vlm_development as development


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-classical-horizon-opportunity-1"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_CLASSICAL_HORIZON_OPPORTUNITY_STRATUM_PROTOCOL_20260809.md"
)
PROTOCOL_SHA256 = (
    "144c93f148e90f73dc62b1983e24ab59565c168800225375f3aecbda9554ebbf"
)
DINO_DIR = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_dinov2_classical_baseline/"
    "plans-20260808"
)
SIGLIP_DIR = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_siglip_classical_baseline/"
    "plans-20260809"
)
PLAN_INPUTS = {
    "dinov2": {
        "plans": (
            DINO_DIR / "PLANS.json",
            "58154f052424f1632c302f9b40af030c172fc4d626908c9f977211c1e8299847",
        ),
        "manifest": (
            DINO_DIR / "MANIFEST.json",
            "56e9d538ded501f4518dc662d8403f3f15ba254c8cedb132e4bd09d10949e504",
        ),
        "depth2": "dinov2_depth2",
        "myopic": "dinov2_myopic",
    },
    "siglip": {
        "plans": (
            SIGLIP_DIR / "PLANS.json",
            "a41cc3b01d18fa9008f67d6f60a3f113b8f3194b73e932b4e2c3cfc83215f587",
        ),
        "manifest": (
            SIGLIP_DIR / "MANIFEST.json",
            "95c00d948ee77589995c3434e3eb01b0a908b496fe2d984b859dbae4032d057d",
        ),
        "depth2": "siglip_depth2",
        "myopic": "siglip_myopic",
    },
}
MANIFEST_PATH = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_CLASSICAL_HORIZON_OPPORTUNITY_MANIFEST_20260809.json"
)
MANIFEST_SHA256 = "bad3ced617de71dc52ffa1d27d8b06bbda72ab0a56478391fc02acf0af4069a0"
PARTITION_COUNTS = {"mechanics": 4, "development": 64, "confirmation": 96}
DISAGREEMENT_COUNTS = {"mechanics": 1, "development": 27, "confirmation": 49}
ANALYSIS_STAGES = ("development", "confirmation")
STRATA = ("classical_horizon_disagreement", "classical_horizon_agreement")
CONTROL = "compute_matched_myopic_ensemble"
METRICS = ("mean_brier", "mean_log_loss")
BOOTSTRAP_SEED = 2026280902


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_plan_inputs() -> dict[str, Any]:
    if _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("classical-horizon opportunity protocol changed")
    observed: dict[str, Any] = {"protocol": PROTOCOL_SHA256, "encoders": {}}
    for encoder, config in PLAN_INPUTS.items():
        plan_path, plan_hash = config["plans"]
        manifest_path, manifest_hash = config["manifest"]
        if _sha256(plan_path) != plan_hash or _sha256(manifest_path) != manifest_hash:
            raise ValueError(f"frozen {encoder} opportunity inputs changed")
        manifest = _load(manifest_path)
        if (
            manifest.get("all_gates_pass") is not True
            or manifest.get("endpoint_labels_accessed") is not False
            or manifest.get("candidate_labels_accessed") is not False
            or manifest.get("plans_sha256") != plan_hash
        ):
            raise ValueError(f"{encoder} opportunity plan is not endpoint blind")
        observed["encoders"][encoder] = {
            "plans_path": str(plan_path.relative_to(REPO_ROOT)),
            "plans_sha256": plan_hash,
            "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "manifest_sha256": manifest_hash,
        }
    return observed


def _selection(row: Mapping[str, Any], policy: str) -> str:
    policies = row.get("policies")
    selected = (
        policies.get(policy, {}).get("first_image_id")
        if isinstance(policies, Mapping)
        else None
    )
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"classical plan omitted {policy} first query")
    return selected


def build_manifest() -> dict[str, Any]:
    inputs = verify_plan_inputs()
    by_encoder: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    for encoder, config in PLAN_INPUTS.items():
        plans = _load(config["plans"][0])
        partitioned: dict[str, dict[str, Mapping[str, Any]]] = {}
        for partition, expected in PARTITION_COUNTS.items():
            rows = [
                row
                for row in plans.get("tasks", [])
                if row.get("partition") == partition
            ]
            indexed = {row.get("task_id"): row for row in rows}
            if (
                len(rows) != expected
                or len(indexed) != expected
                or not all(isinstance(key, str) for key in indexed)
            ):
                raise ValueError(
                    f"{encoder} {partition} opportunity plan is incomplete"
                )
            partitioned[partition] = indexed
        by_encoder[encoder] = partitioned

    partitions: dict[str, Any] = {}
    for partition, expected in PARTITION_COUNTS.items():
        task_ids = set(by_encoder["dinov2"][partition])
        if task_ids != set(by_encoder["siglip"][partition]):
            raise ValueError(f"classical {partition} task supports differ")
        rows = []
        for task_id in sorted(task_ids):
            encoders = {}
            changed = False
            for encoder, config in PLAN_INPUTS.items():
                source = by_encoder[encoder][partition][task_id]
                depth2 = _selection(source, config["depth2"])
                myopic = _selection(source, config["myopic"])
                encoder_changed = depth2 != myopic
                changed = changed or encoder_changed
                encoders[encoder] = {
                    "depth2_first_image_id": depth2,
                    "myopic_first_image_id": myopic,
                    "changed_first_query": encoder_changed,
                }
            rows.append(
                {
                    "task_id": task_id,
                    "stratum": STRATA[0] if changed else STRATA[1],
                    "encoders": encoders,
                }
            )
        disagreement = sum(row["stratum"] == STRATA[0] for row in rows)
        if len(rows) != expected or disagreement != DISAGREEMENT_COUNTS[partition]:
            raise ValueError(f"frozen {partition} opportunity counts changed")
        partitions[partition] = {
            "task_count": expected,
            "classical_horizon_disagreement_count": disagreement,
            "classical_horizon_agreement_count": expected - disagreement,
            "rows": rows,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "endpoint_blind_classical_horizon_opportunity_manifest",
        "definition": "dinov2_depth2_or_siglip_depth2_changes_own_myopic_first_query",
        "inputs": inputs,
        "partitions": partitions,
        "candidate_labels_accessed": False,
        "endpoint_labels_accessed": False,
        "luna_responses_accessed": False,
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
    }


def freeze_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = build_manifest()
    _write_once(path, manifest)
    return manifest


def load_frozen_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if MANIFEST_SHA256 == "TO_BE_FROZEN" or _sha256(path) != MANIFEST_SHA256:
        raise ValueError("classical-horizon opportunity manifest changed")
    manifest = _load(path)
    if _canonical(manifest) != _canonical(build_manifest()):
        raise ValueError("classical-horizon opportunity manifest does not replay")
    return manifest


def _paired(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    summary = development.paired_summary(values, seed=seed)
    summary["standard_error"] = summary["sample_sd"] / math.sqrt(summary["n"])
    summary["bootstrap_draws"] = development.BOOTSTRAP_REPLICATES
    summary["negative_favors"] = "dynamic_depth2"
    return summary


def _stratum_summary(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("opportunity stratum is too small for paired analysis")
    comparisons = {
        metric: _paired(
            [float(row["dynamic_minus_control"][metric]) for row in rows],
            seed=seed + index,
        )
        for index, metric in enumerate(METRICS)
    }
    dynamic_brier = statistics.fmean(float(row["dynamic_brier"]) for row in rows)
    control_brier = statistics.fmean(float(row["control_brier"]) for row in rows)
    return {
        "task_count": len(rows),
        "task_ids": [str(row["task_id"]) for row in rows],
        "dynamic_compute_matched_first_query_changes": sum(
            row["dynamic_first_image_id"] != row["control_first_image_id"]
            for row in rows
        ),
        "dynamic_compute_matched_final_history_changes": sum(
            row["dynamic_final_history_key"] != row["control_final_history_key"]
            for row in rows
        ),
        "dynamic_mean_brier": dynamic_brier,
        "compute_matched_myopic_mean_brier": control_brier,
        "dynamic_relative_brier_gain": (
            (control_brier - dynamic_brier) / control_brier
            if control_brier > 0
            else -math.inf
        ),
        "dynamic_mean_spearman": statistics.fmean(
            float(row["dynamic_spearman"]) for row in rows
        ),
        "compute_matched_myopic_mean_spearman": statistics.fmean(
            float(row["control_spearman"]) for row in rows
        ),
        "paired": comparisons,
    }


def build_report(
    *,
    stage: str,
    stage_result: Mapping[str, Any],
    authorization: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if stage not in ANALYSIS_STAGES or authorization.get("verified") is not True:
        raise ValueError("opportunity stage is not authorized")
    trees = stage_result.get("trees")
    expected = PARTITION_COUNTS[stage]
    if not isinstance(trees, list) or len(trees) != expected:
        raise ValueError("opportunity stage task count changed")
    manifest_rows = manifest.get("partitions", {}).get(stage, {}).get("rows")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != expected:
        raise ValueError("opportunity manifest stage is incomplete")
    manifest_by_id = {row.get("task_id"): row for row in manifest_rows}
    tree_by_id = {
        tree.get("task_id"): tree
        for tree in trees
        if isinstance(tree, Mapping)
    }
    if set(manifest_by_id) != set(tree_by_id) or len(tree_by_id) != expected:
        raise ValueError("opportunity stage identities differ from the frozen manifest")

    rows = []
    for task_id in sorted(tree_by_id):
        tree = tree_by_id[task_id]
        replay = compute_control._task_row(tree)
        dynamic = replay["endpoint"]["dynamic_depth2"]
        control = replay["endpoint"][CONTROL]
        ranking = tree.get("ranking_fidelity")
        if not isinstance(ranking, Mapping):
            raise ValueError("opportunity tree omitted ranking fidelity")
        dynamic_rank = ranking.get("dynamic_depth2")
        control_rank = ranking.get(CONTROL)
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in (dynamic_rank, control_rank)
        ):
            raise ValueError("opportunity ranking fidelity is not finite")
        rows.append(
            {
                "task_id": task_id,
                "stratum": manifest_by_id[task_id]["stratum"],
                "classical_plan": manifest_by_id[task_id]["encoders"],
                "dynamic_first_image_id": replay["selections"]["dynamic_depth2"][
                    "first_image_id"
                ],
                "control_first_image_id": replay["selections"][CONTROL][
                    "first_image_id"
                ],
                "dynamic_final_history_key": replay["selections"][
                    "dynamic_depth2"
                ]["final_history_key"],
                "control_final_history_key": replay["selections"][CONTROL][
                    "final_history_key"
                ],
                "dynamic_brier": dynamic["mean_brier"],
                "control_brier": control["mean_brier"],
                "dynamic_minus_control": {
                    metric: dynamic[metric] - control[metric] for metric in METRICS
                },
                "dynamic_spearman": float(dynamic_rank),
                "control_spearman": float(control_rank),
            }
        )
    strata = {
        stratum: _stratum_summary(
            [row for row in rows if row["stratum"] == stratum],
            seed=BOOTSTRAP_SEED + index * 100,
        )
        for index, stratum in enumerate(STRATA)
    }
    if strata[STRATA[0]]["task_count"] != DISAGREEMENT_COUNTS[stage]:
        raise ValueError("opportunity report stratum count changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "classical_horizon_opportunity_stratum_complete",
        "stage": stage,
        "stage_status": stage_result.get("status"),
        "stage_claim_tier": stage_result.get("claim_tier"),
        "stage_authorization": dict(authorization),
        "task_count": expected,
        "stratum_definition": manifest.get("definition"),
        "strict_control": CONTROL,
        "compute_contract_exact": True,
        "strata": strata,
        "rows": rows,
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "changes_primary_gates": False,
        "changes_claim_tier": False,
        "interpretation_scope": (
            "Prospective secondary diagnostic only; it distinguishes classical "
            "horizon-disagreement from agreement tasks without rescuing the primary."
        ),
    }


def run_report(
    *,
    stage: str,
    result_path: Path,
    output_path: Path,
    block_results: Sequence[Path] = (),
    authorizer: Callable[..., dict[str, Any]] = stage_outcome.authorize_stage,
    result_loader: Callable[[Path], dict[str, Any]] = _load,
) -> dict[str, Any]:
    if stage not in ANALYSIS_STAGES:
        raise ValueError(f"unknown opportunity stage {stage!r}")
    if output_path.exists():
        raise FileExistsError(output_path)
    manifest = load_frozen_manifest()
    authorization = authorizer(
        stage=stage,
        result_path=result_path,
        block_results=block_results,
        wrapper_result=None,
    )
    if authorization.get("verified") is not True:
        raise ValueError("opportunity stage authorization failed")
    result = result_loader(result_path)
    report = build_report(
        stage=stage,
        stage_result=result,
        authorization=authorization,
        manifest=manifest,
    )
    report["protocol"] = {"path": str(PROTOCOL), "sha256": PROTOCOL_SHA256}
    report["manifest"] = {"path": str(MANIFEST_PATH), "sha256": MANIFEST_SHA256}
    report["stage_result_path"] = str(result_path)
    report["stage_result_sha256"] = _sha256(result_path)
    report["block_result_sha256"] = [_sha256(path) for path in block_results]
    _write_once(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", action="store_true")
    parser.add_argument("--stage", choices=ANALYSIS_STAGES)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--block-result", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_manifest:
        if any(
            value is not None
            for value in (args.stage, args.result_path, args.output_path)
        ):
            raise SystemExit(
                "--freeze-manifest cannot be combined with report arguments"
            )
        result = freeze_manifest()
    else:
        if args.stage is None or args.result_path is None or args.output_path is None:
            raise SystemExit(
                "report mode requires --stage, --result-path, and --output-path"
            )
        result = run_report(
            stage=args.stage,
            result_path=args.result_path,
            output_path=args.output_path,
            block_results=args.block_result,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
