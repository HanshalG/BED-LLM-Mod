#!/usr/bin/env python3
"""Independently replay a HiddenBench semantic-query serving result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import hiddenbench_semantic_query_serving as serving
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


INTERFACE_VERSION = "hiddenbench-semantic-query-serving-verifier-v1"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def infer_option_ids(root_raw: str) -> tuple[str, ...]:
    value = json.loads(root_raw)
    prior = value.get("prior") if isinstance(value, dict) else None
    if not isinstance(prior, list) or len(prior) not in (3, 4):
        raise RuntimeError("raw root does not expose three or four opaque options")
    ids = tuple(item.get("id") for item in prior if isinstance(item, dict))
    expected = tuple(f"O{index + 1}" for index in range(len(prior)))
    if ids != expected:
        raise RuntimeError("raw root option IDs are not canonical")
    return ids


def verify(run_dir: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    result = load_object(run_dir / "RESULT.json")
    raw = load_object(run_dir / "private/RAW_RESPONSES.json")
    if set(raw) != {"root", "world", "router"}:
        raise RuntimeError("raw response bank has the wrong components")
    if {key: len(raw[key]) for key in raw} != {"root": 4, "world": 4, "router": 2}:
        raise RuntimeError("raw response bank is incomplete")

    roots = []
    worlds = []
    option_ids_by_task = []
    for root_raw in raw["root"]:
        option_ids = infer_option_ids(root_raw)
        option_ids_by_task.append(option_ids)
        roots.append(serving.parse_root(root_raw, option_ids))
    for world_raw, option_ids in zip(raw["world"], option_ids_by_task, strict=True):
        worlds.append(serving.parse_world(world_raw, option_ids))
    routers = [
        serving.parse_router(response, ("F1", "F2", "F3", "F4"))
        for response in raw["router"]
    ]

    scores = [
        serving.score_model(root["prior"], world, option_ids)
        for root, world, option_ids in zip(
            roots, worlds, option_ids_by_task, strict=True
        )
    ]
    tv_counts = []
    followup_counts = []
    for world, option_ids in zip(worlds, option_ids_by_task, strict=True):
        tv_counts.append(
            sum(
                max(
                    0.5
                    * sum(
                        abs(left_value - right_value)
                        for left_value, right_value in zip(
                            world[query_id]["likelihoods"][left],
                            world[query_id]["likelihoods"][right],
                            strict=True,
                        )
                    )
                    for index, left in enumerate(option_ids)
                    for right in option_ids[index + 1 :]
                )
                >= serving.MIN_TV
                for query_id in serving.QUERY_IDS
            )
        )
        followup_counts.append(
            sum(
                len(set(world[query_id]["followups"].values())) >= 2
                for query_id in serving.QUERY_IDS
            )
        )

    changed = sum(
        score["greedy_query_id"] != score["depth_two_query_id"]
        for score in scores
    )
    semantic_gates = {
        "option_sensitive_worlds": all(value >= 2 for value in tv_counts),
        "response_dependent_followups": all(
            value >= 2 for value in followup_counts
        ),
        "nondegenerate_planner": all(
            score["max_one_step_eig"] >= serving.MIN_MAX_EIG
            and score["one_step_range"] >= serving.MIN_EIG_RANGE
            and score["greedy_margin"] >= serving.MIN_WIN_MARGIN
            and score["depth_two_margin"] >= serving.MIN_WIN_MARGIN
            for score in scores
        ),
        "depth_two_changed_first_request": changed >= 2,
        "exact_distinct_fact_routing": (
            len(routers) == 2
            and all(item["addressed"] for item in routers)
            and len({item["fact_id"] for item in routers}) == 2
        ),
    }
    expected_aggregate = {
        "tasks": 4,
        "parsed_responses": 10,
        "option_sensitive_query_counts": tv_counts,
        "response_dependent_query_counts": followup_counts,
        "changed_first_request_tasks": changed,
    }
    public_gates = result.get("gates") or {}
    gates = {
        "exact_interface": result.get("interface_version") == serving.INTERFACE_VERSION,
        "exact_protocol": result.get("protocol_sha256") == serving.PROTOCOL_SHA256,
        "raw_bank_complete_and_parseable": True,
        "aggregate_replays_exactly": result.get("aggregate") == expected_aggregate,
        "scores_replay_exactly": result.get("scores") == scores,
        "semantic_gates_replay_exactly": all(
            public_gates.get(name) is value
            for name, value in semantic_gates.items()
        ),
        "transport_gate_is_strict": public_gates.get("exact_transport") is True,
        "schema_gate_is_strict": public_gates.get("strict_schema_all_ten") is True,
        "budget_gate_is_strict": public_gates.get("within_run_budget") is True,
        "public_status_matches_conjunction": (
            result.get("status")
            == ("serving_pass" if all(public_gates.values()) else "serving_failed_closed")
        ),
        "public_authority_is_bounded": result.get("authorizes")
        in {"separately_frozen_opportunity_mechanics_only", "nothing"},
    }
    verification = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "verification_pass" if all(gates.values()) else "verification_failed",
        "gates": gates,
        "replayed": {
            "scores": scores,
            "aggregate": expected_aggregate,
            "semantic_gates": semantic_gates,
        },
        "source_task_values_loaded": False,
        "registered_answers_loaded": False,
        "model_calls_made": 0,
    }
    if output_path is not None:
        checkpoint(output_path, verification)
    if verification["status"] != "verification_pass":
        raise RuntimeError("HiddenBench serving verification failed")
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir.resolve(), output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
