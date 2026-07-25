#!/usr/bin/env python3
"""Run manifest-bound ClariQ multisample likelihood holdout confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts.analyze_clariq_multisample_v2_stability import _selected
from scripts.clariq_multisample_likelihood_development import (
    DevelopmentExecutionError,
    DeterministicFixtureModel,
    _build_model,
    _checkpoint,
    run_development,
)


INTERFACE_VERSION = "clariq-multisample-likelihood-holdout-1"
MANIFEST_SHA256 = (
    "889a9a952945fea0c5cd910d559c1e12c9e40f5a9500f65810b0ce6a0eb82cfa"
)
SELECTED_TOPIC_IDS = (
    "115",
    "10",
    "104",
    "14",
    "135",
    "122",
    "105",
    "131",
    "113",
    "137",
    "119",
    "116",
)
VALID_ROOT_COUNTS = (15, 14, 12, 12, 12, 13, 10, 14, 12, 15, 12, 10)
EXPECTED_REQUESTS = 755
MAX_COST_USD = 1.25


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("ClariQ holdout manifest hash does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "passed":
        raise ValueError("ClariQ holdout manifest did not pass")
    if tuple(payload["selected_topic_ids"]) != SELECTED_TOPIC_IDS:
        raise ValueError("ClariQ holdout topic IDs changed")
    tasks = {
        task["topic_id"]: task for task in payload["selected_topics"]
    }
    if tuple(tasks) != SELECTED_TOPIC_IDS:
        raise ValueError("ClariQ holdout task order changed")
    if tuple(len(task["questions"]) for task in tasks.values()) != (
        VALID_ROOT_COUNTS
    ):
        raise ValueError("ClariQ holdout root counts changed")
    if payload["expected_likelihood_requests"] != EXPECTED_REQUESTS:
        raise ValueError("ClariQ holdout request count changed")
    return tasks


def exact_sign_flip_p(values: Sequence[float]) -> float:
    nonzero = [value for value in values if abs(value) > 1e-12]
    if not nonzero:
        return 1.0
    observed = sum(nonzero)
    signed_sums = [0.0]
    for value in nonzero:
        signed_sums = [
            current + sign * value
            for current in signed_sums
            for sign in (-1.0, 1.0)
        ]
    return sum(
        value >= observed - 1e-12 for value in signed_sums
    ) / len(signed_sums)


def apply_holdout_gates(payload: dict[str, Any]) -> dict[str, Any]:
    rows = {row["topic_id"]: row for row in payload["rows"]}
    stability = []
    for topic_id, maps in payload["maps"].items():
        full = payload["policies"][topic_id]["depth_two_question_id"]
        myopic = payload["policies"][topic_id]["myopic_question_id"]
        sample_count = len(next(iter(maps.values())))
        leave_one_out = [
            _selected(
                maps,
                [
                    index
                    for index in range(sample_count)
                    if index != omitted
                ],
            )
            for omitted in range(sample_count)
        ]
        stability.append(
            {
                "topic_id": topic_id,
                "root_changed": full != myopic,
                "full_depth_two_question_id": full,
                "myopic_question_id": myopic,
                "leave_one_out_selections": leave_one_out,
                "leave_one_out_agreement_count": leave_one_out.count(full),
            }
        )

    endpoint_complete = all(
        row["myopic_oracle_tail"] is not None
        and row["depth_two_oracle_tail"] is not None
        and row["random_oracle_tail"] is not None
        for row in rows.values()
    )
    paired_differences = {
        topic_id: (
            rows[topic_id]["depth_two_oracle_tail"]
            - rows[topic_id]["myopic_oracle_tail"]
            if rows[topic_id]["depth_two_oracle_tail"] is not None
            and rows[topic_id]["myopic_oracle_tail"] is not None
            else None
        )
        for topic_id in SELECTED_TOPIC_IDS
    }
    differences = [
        value for value in paired_differences.values() if value is not None
    ]
    exact_p = exact_sign_flip_p(differences) if endpoint_complete else 1.0
    topic_count = len(rows)
    changed_threshold = max(4, math.ceil(0.25 * topic_count))
    usage = payload["usage"]
    metrics = payload["metrics"]
    gates = {
        "manifest_topic_count_between_8_and_12": 8 <= topic_count <= 12,
        "exact_physical_requests": (
            usage["physical_requests"] == EXPECTED_REQUESTS
        ),
        "exact_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_maps_parse": all(
            len(samples) == 5
            for topic in payload["maps"].values()
            for samples in topic.values()
        ),
        "three_modal_partitions_per_topic": all(
            row["distinct_modal_partition_count"] >= 3
            for row in rows.values()
        ),
        "eig_range_at_least_0_05_per_topic": all(
            row["myopic_eig_range"] >= 0.05 for row in rows.values()
        ),
        "enough_changed_roots": (
            metrics["depth_two_root_change_count"] >= changed_threshold
        ),
        "all_changed_roots_leave_one_out_stable": all(
            row["leave_one_out_agreement_count"] >= 3
            for row in stability
            if row["root_changed"]
        ),
        "all_selected_roots_have_endpoints": endpoint_complete,
        "depth_two_wins_at_least_4": (
            metrics["depth_two_wins_over_myopic"] >= 4
        ),
        "depth_two_losses_at_most_2": (
            metrics["depth_two_losses_to_myopic"] <= 2
        ),
        "mean_depth_two_gain_at_least_0_003": (
            metrics["mean_depth_two_gain_over_myopic"] >= 0.003
        ),
        "exact_one_sided_p_at_most_0_10": exact_p <= 0.10,
        "mean_depth_two_gain_over_random_nonnegative": (
            metrics["mean_depth_two_gain_over_random"] >= 0.0
        ),
        "depth_two_spearman_at_least_0_20": (
            metrics["pooled_depth_two_score_oracle_tail_spearman"] >= 0.20
        ),
        "cost_at_most_1_25": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    payload["status"] = "passed" if gates["all_pass"] else "gate_failed"
    payload["gates"] = gates
    payload["holdout_stability"] = stability
    payload["holdout_statistics"] = {
        "topic_count": topic_count,
        "changed_root_threshold": changed_threshold,
        "paired_differences": paired_differences,
        "exact_one_sided_sign_flip_p": exact_p,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tasks = load_manifest(args.manifest)
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.60
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 128
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model = DeterministicFixtureModel() if args.dry_run else _build_model(config)
    try:
        payload = run_development(
            config,
            source_root=args.source_root,
            raw_path=raw_path,
            model=model,
            tasks_override=tasks,
            expected_requests=EXPECTED_REQUESTS,
            interface_version=INTERFACE_VERSION,
        )
        payload = apply_holdout_gates(payload)
        payload["protocol"].pop(
            "development_endpoints_loaded_after_scores_froze",
            None,
        )
        payload["protocol"]["holdout_endpoints_loaded"] = True
        payload["protocol"][
            "holdout_endpoints_loaded_after_scores_froze"
        ] = True
        payload["protocol"]["manifest_sha256"] = MANIFEST_SHA256
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "manifest_sha256": MANIFEST_SHA256,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, DevelopmentExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "HOLDOUT_FAILURE.json", failure)
        raise
    output = args.output_dir / "HOLDOUT.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "metrics": payload["metrics"],
                "holdout_statistics": payload["holdout_statistics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
