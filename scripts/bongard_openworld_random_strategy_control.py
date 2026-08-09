#!/usr/bin/env python3
"""Report paired Bongard effects against the frozen random strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_dinov2_outcome as stage_outcome
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-random-strategy-control-1"
BOOTSTRAP_SEED = 2_026_080_902
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_RANDOM_STRATEGY_CONTROL_PROTOCOL_20260809.md"
)
PROTOCOL_SHA256 = (
    "03e91353a04cb1133cc79f9b38b3dc4d04111492137634100091be5de122adad"
)
STAGES = stage_outcome.STAGES
METRICS = ("mean_brier", "mean_log_loss")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _number(value: Any, *, context: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{context} must be finite")
    return float(value)


def _expected_random_choices(
    *, task_id: str, candidates: Sequence[str]
) -> tuple[str, str]:
    canonical = sorted(str(candidate) for candidate in candidates)
    if len(canonical) < 2 or len(set(canonical)) != len(canonical):
        raise ValueError("random control requires distinct candidate IDs")
    offset = int.from_bytes(
        hashlib.sha256(task_id.encode()).digest()[:8], "big"
    )
    rng = random.Random(mechanics.RANDOM_SEED + offset)
    first, second = rng.sample(canonical, 2)
    return first, second


def _policy_endpoint(
    policies: Mapping[str, Any], *, policy_name: str
) -> tuple[Mapping[str, Any], dict[str, float]]:
    policy = policies.get(policy_name)
    if not isinstance(policy, Mapping):
        raise ValueError(f"random audit is missing policy {policy_name}")
    endpoint = policy.get("endpoint")
    if not isinstance(endpoint, Mapping):
        raise ValueError(f"random audit is missing {policy_name} endpoint")
    return policy, {
        metric: _number(
            endpoint.get(metric), context=f"{policy_name} {metric}"
        )
        for metric in METRICS
    }


def _task_row(tree: Mapping[str, Any]) -> dict[str, Any]:
    task_id = tree.get("task_id")
    root_scores = tree.get("root_scores")
    policies = tree.get("policies")
    if (
        not isinstance(task_id, str)
        or not task_id
        or not isinstance(root_scores, Mapping)
        or not isinstance(policies, Mapping)
    ):
        raise ValueError("random-control tree structure is incomplete")
    dynamic_scores = root_scores.get("dynamic_depth2")
    if not isinstance(dynamic_scores, Mapping) or len(dynamic_scores) < 2:
        raise ValueError("random-control candidate support is incomplete")
    candidates = sorted(str(candidate) for candidate in dynamic_scores)
    expected_first, expected_second = _expected_random_choices(
        task_id=task_id, candidates=candidates
    )
    dynamic, dynamic_endpoint = _policy_endpoint(
        policies, policy_name="dynamic_depth2"
    )
    random_policy, random_endpoint = _policy_endpoint(
        policies, policy_name="random"
    )
    random_first = random_policy.get("first_image_id")
    random_second = random_policy.get("second_image_id")
    if (
        random_first != expected_first
        or random_second != expected_second
        or random_first == random_second
        or random_first not in candidates
        or random_second not in candidates
        or random_policy.get("first_score") is not None
        or random_policy.get("first_score_margin") is not None
        or random_policy.get("second_scores") is not None
        or random_policy.get("second_score_margin") is not None
    ):
        raise ValueError("stored random policy does not replay exactly")
    dynamic_first = dynamic.get("first_image_id")
    dynamic_second = dynamic.get("second_image_id")
    if (
        dynamic_first not in candidates
        or dynamic_second not in candidates
        or dynamic_first == dynamic_second
    ):
        raise ValueError("stored dynamic policy has invalid query choices")
    dynamic_history = dynamic.get("final_history_key")
    random_history = random_policy.get("final_history_key")
    if (
        not isinstance(dynamic_history, str)
        or not dynamic_history
        or not isinstance(random_history, str)
        or not random_history
    ):
        raise ValueError("random-control final history is missing")
    return {
        "task_id": task_id,
        "candidate_count": len(candidates),
        "random_draw_replayed": True,
        "selections": {
            "dynamic_depth2": {
                "first_image_id": dynamic_first,
                "second_image_id": dynamic_second,
                "final_history_key": dynamic_history,
            },
            "random": {
                "first_image_id": random_first,
                "second_image_id": random_second,
                "final_history_key": random_history,
            },
        },
        "endpoint": {
            "dynamic_depth2": dynamic_endpoint,
            "random": random_endpoint,
        },
        "dynamic_minus_random": {
            metric: dynamic_endpoint[metric] - random_endpoint[metric]
            for metric in METRICS
        },
    }


def _paired(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    summary = development.paired_summary(values, seed=seed)
    summary["standard_error"] = summary["sample_sd"] / math.sqrt(summary["n"])
    summary["bootstrap_draws"] = development.BOOTSTRAP_REPLICATES
    summary["negative_favors"] = "dynamic_depth2"
    return summary


def build_report(
    *, stage: str, stage_result: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in STAGES or authorization.get("verified") is not True:
        raise ValueError("random-control stage is not authorized")
    if mechanics.RANDOM_SEED != 2_026_081_022:
        raise ValueError("frozen random-policy seed changed")
    trees = stage_result.get("trees")
    expected = stage_outcome.STAGE_TASK_COUNTS[stage]
    if not isinstance(trees, list) or len(trees) != expected:
        raise ValueError("random-control stage task count changed")
    rows = sorted((_task_row(tree) for tree in trees), key=lambda row: row["task_id"])
    if len({row["task_id"] for row in rows}) != expected:
        raise ValueError("random-control task identities are not unique")
    comparisons = {
        metric: _paired(
            [float(row["dynamic_minus_random"][metric]) for row in rows],
            seed=BOOTSTRAP_SEED + metric_index,
        )
        for metric_index, metric in enumerate(METRICS)
    }
    first_query_changes = sum(
        row["selections"]["dynamic_depth2"]["first_image_id"]
        != row["selections"]["random"]["first_image_id"]
        for row in rows
    )
    final_history_changes = sum(
        row["selections"]["dynamic_depth2"]["final_history_key"]
        != row["selections"]["random"]["final_history_key"]
        for row in rows
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "random_strategy_control_audit_complete",
        "stage": stage,
        "stage_status": stage_result.get("status"),
        "stage_claim_tier": stage_result.get("claim_tier"),
        "stage_authorization": dict(authorization),
        "task_count": expected,
        "random_policy_seed": mechanics.RANDOM_SEED,
        "random_policy_draws_without_replacement": True,
        "random_draws_replayed_exactly": True,
        "comparisons": comparisons,
        "first_query_changes": first_query_changes,
        "final_history_changes": final_history_changes,
        "rows": rows,
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "changes_claim_tier": False,
        "interpretation_scope": (
            "Descriptive all-task paired sanity baseline only. Random selection "
            "is not compute matched and cannot establish non-myopia."
        ),
    }


def run_report(
    *,
    stage: str,
    result_path: Path,
    output_path: Path,
    block_results: Sequence[Path] = (),
    wrapper_result: Path | None = None,
    authorizer: Callable[..., dict[str, Any]] = stage_outcome.authorize_stage,
    result_loader: Callable[[Path], dict[str, Any]] = _load,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown random-control stage {stage!r}")
    if output_path.exists():
        raise FileExistsError(output_path)
    if _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("random-strategy control protocol changed")
    authorization = authorizer(
        stage=stage,
        result_path=result_path,
        block_results=block_results,
        wrapper_result=wrapper_result,
    )
    if authorization.get("verified") is not True:
        raise ValueError("random-control stage authorization failed")
    stage_result = result_loader(result_path)
    report = build_report(
        stage=stage,
        stage_result=stage_result,
        authorization=authorization,
    )
    report["protocol"] = {"path": str(PROTOCOL), "sha256": PROTOCOL_SHA256}
    report["stage_result_path"] = str(result_path)
    report["stage_result_sha256"] = _sha256(result_path)
    report["block_result_sha256"] = [_sha256(path) for path in block_results]
    _write_once(output_path, report)
    return report


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
    result = run_report(
        stage=args.stage,
        result_path=args.result_path,
        output_path=args.output_path,
        block_results=args.block_result,
        wrapper_result=args.wrapper_result,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
