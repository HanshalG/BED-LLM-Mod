#!/usr/bin/env python3
"""Report paired Bongard effects against frozen compute-matched controls."""

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

from scripts import bongard_openworld_dinov2_outcome as stage_outcome
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_vlm_bed as bed


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-compute-matched-control-2"
BOOTSTRAP_SEED = 2026280901
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_COMPUTE_MATCHED_CONTROL_AUDIT_PROTOCOL_20260809.md"
)
PROTOCOL_SHA256 = (
    "994be46dfabe30ca564ad9befbd47e6e84935f39c389c33db287b2c85b417c6f"
)
STAGES = stage_outcome.STAGES
CONTROLS = (
    "compute_matched_myopic_ensemble",
    "shuffled_dynamic_depth2",
    "history_blind_depth2",
    "myopic_width",
)
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


def _score_map(value: Any, *, context: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or len(value) < 2:
        raise ValueError(f"{context} must be a nontrivial score map")
    return {
        str(key): _number(score, context=f"{context}[{key}]")
        for key, score in value.items()
    }


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _task_row(tree: Mapping[str, Any]) -> dict[str, Any]:
    task_id = tree.get("task_id")
    root_scores = tree.get("root_scores")
    continuations = tree.get("continuation_values")
    mapping = tree.get("shuffled_branch_mapping")
    policies = tree.get("policies")
    if (
        not isinstance(task_id, str)
        or not task_id
        or not isinstance(root_scores, Mapping)
        or not isinstance(continuations, Mapping)
        or not isinstance(mapping, Mapping)
        or not isinstance(policies, Mapping)
    ):
        raise ValueError("compute-matched tree structure is incomplete")

    myopic_scores = _score_map(
        root_scores.get("myopic_width"), context="myopic root scores"
    )
    dynamic_scores = _score_map(
        root_scores.get("dynamic_depth2"), context="dynamic root scores"
    )
    shuffled_scores = _score_map(
        root_scores.get("shuffled_dynamic_depth2"),
        context="shuffled root scores",
    )
    dynamic_future = _score_map(
        continuations.get("dynamic_expected_continuation_utility"),
        context="dynamic continuation values",
    )
    shuffled_future = _score_map(
        continuations.get("shuffled_expected_continuation_utility"),
        context="shuffled continuation values",
    )
    candidates = set(myopic_scores)
    if any(
        set(values) != candidates
        for values in (
            dynamic_scores,
            shuffled_scores,
            dynamic_future,
            shuffled_future,
        )
    ):
        raise ValueError("compute-matched score supports differ")
    canonical_mapping = {str(key): str(value) for key, value in mapping.items()}
    if (
        set(canonical_mapping) != candidates
        or set(canonical_mapping.values()) != candidates
        or any(key == value for key, value in canonical_mapping.items())
    ):
        raise ValueError("shuffled continuation mapping is not a derangement")
    for candidate in sorted(candidates):
        if not _same(
            dynamic_scores[candidate],
            myopic_scores[candidate] + dynamic_future[candidate],
        ):
            raise ValueError("dynamic score does not replay from root plus future")
        if not _same(
            shuffled_scores[candidate],
            myopic_scores[candidate] + shuffled_future[candidate],
        ):
            raise ValueError("shuffled score does not replay from root plus future")
        if not _same(
            shuffled_future[candidate],
            dynamic_future[canonical_mapping[candidate]],
        ):
            raise ValueError("shuffled continuation is not the recorded permutation")
    if sorted(dynamic_future.values()) != sorted(shuffled_future.values()):
        raise ValueError("shuffled continuation multiset changed")

    score_maps = {
        "dynamic_depth2": dynamic_scores,
        "compute_matched_myopic_ensemble": _score_map(
            root_scores.get("compute_matched_myopic_ensemble"),
            context="compute-matched myopic ensemble root scores",
        ),
        "shuffled_dynamic_depth2": shuffled_scores,
        "history_blind_depth2": _score_map(
            root_scores.get("history_blind_depth2"),
            context="history-blind root scores",
        ),
        "myopic_width": myopic_scores,
    }
    if any(
        set(score_maps[name]) != candidates
        for name in (
            "compute_matched_myopic_ensemble",
            "history_blind_depth2",
        )
    ):
        raise ValueError("matched control score support differs")
    endpoints: dict[str, dict[str, float]] = {}
    selections: dict[str, dict[str, str]] = {}
    for policy_name, scores in score_maps.items():
        policy = policies.get(policy_name)
        if not isinstance(policy, Mapping):
            raise ValueError(f"missing policy {policy_name}")
        first = policy.get("first_image_id")
        if first != bed.select_best(scores):
            raise ValueError(f"{policy_name} first query is not its score argmax")
        endpoint = policy.get("endpoint")
        if not isinstance(endpoint, Mapping):
            raise ValueError(f"{policy_name} endpoint is missing")
        endpoints[policy_name] = {
            metric: _number(
                endpoint.get(metric), context=f"{policy_name} {metric}"
            )
            for metric in METRICS
        }
        final_history = policy.get("final_history_key")
        if not isinstance(final_history, str) or not final_history:
            raise ValueError(f"{policy_name} final history is missing")
        selections[policy_name] = {
            "first_image_id": str(first),
            "final_history_key": final_history,
        }

    return {
        "task_id": task_id,
        "compute_contract_exact": True,
        "selections": selections,
        "endpoint": endpoints,
        "dynamic_minus_control": {
            control: {
                metric: endpoints["dynamic_depth2"][metric]
                - endpoints[control][metric]
                for metric in METRICS
            }
            for control in CONTROLS
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
        raise ValueError("compute-matched stage is not authorized")
    trees = stage_result.get("trees")
    expected = stage_outcome.STAGE_TASK_COUNTS[stage]
    if not isinstance(trees, list) or len(trees) != expected:
        raise ValueError("compute-matched stage task count changed")
    rows = [_task_row(tree) for tree in trees]
    task_ids = [row["task_id"] for row in rows]
    if len(set(task_ids)) != expected:
        raise ValueError("compute-matched stage task identities are not unique")
    rows.sort(key=lambda row: row["task_id"])

    comparisons = {}
    for control_index, control in enumerate(CONTROLS):
        comparisons[control] = {}
        for metric_index, metric in enumerate(METRICS):
            comparisons[control][metric] = _paired(
                [
                    float(row["dynamic_minus_control"][control][metric])
                    for row in rows
                ],
                seed=BOOTSTRAP_SEED + control_index * 10 + metric_index,
            )
        comparisons[control]["first_query_changes"] = sum(
            row["selections"]["dynamic_depth2"]["first_image_id"]
            != row["selections"][control]["first_image_id"]
            for row in rows
        )
        comparisons[control]["final_history_changes"] = sum(
            row["selections"]["dynamic_depth2"]["final_history_key"]
            != row["selections"][control]["final_history_key"]
            for row in rows
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "compute_matched_control_audit_complete",
        "stage": stage,
        "stage_status": stage_result.get("status"),
        "stage_claim_tier": stage_result.get("claim_tier"),
        "stage_authorization": dict(authorization),
        "task_count": expected,
        "strict_compute_matched_myopic_control": (
            "compute_matched_myopic_ensemble"
        ),
        "strict_continuation_compute_matched_control": (
            "shuffled_dynamic_depth2"
        ),
        "matched_request_count_control": "history_blind_depth2",
        "online_regeneration_greedy_control": "myopic_width",
        "compute_contract_exact": True,
        "comparisons": comparisons,
        "rows": rows,
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "changes_claim_tier": False,
        "interpretation_scope": (
            "Descriptive all-task paired effects only. The myopic ensemble uses "
            "the root plus all 16 answer-free root-prompt draws, matching the "
            "dynamic planner's root plus 16 answer-conditioned branches. Shuffled "
            "continuation is branch-bank-compute matched; history blind is "
            "branch-request-count matched; myopic is the online greedy baseline."
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
        raise ValueError(f"unknown compute-matched stage {stage!r}")
    if output_path.exists():
        raise FileExistsError(output_path)
    if _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("compute-matched control protocol changed")
    authorization = authorizer(
        stage=stage,
        result_path=result_path,
        block_results=block_results,
        wrapper_result=wrapper_result,
    )
    if authorization.get("verified") is not True:
        raise ValueError("compute-matched stage authorization failed")
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
