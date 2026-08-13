#!/usr/bin/env python3
"""Score HiddenBench V3 policies after sealed endpoint extraction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.hiddenbench_dynamic_belief_v3_codec import parse_refresh, parse_root
from scripts.hiddenbench_dynamic_belief_v3_math import (
    CHANNEL_IDS,
    QUERY_IDS,
    best_query,
    endpoint_metrics,
    fixed_depth_two,
    update,
)


RANDOM_SEED = 202608135100


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def matrices(root: dict[str, Any], option_ids: tuple[str, ...]) -> dict[str, list[list[float]]]:
    return {
        query_id: [root["queries"][query_id]["likelihoods"][option_id] for option_id in option_ids]
        for query_id in QUERY_IDS
    }


def second_query(
    belief: list[float], likelihoods: dict[str, list[list[float]]], first_id: str
) -> str:
    candidates = {query_id: matrix for query_id, matrix in likelihoods.items() if query_id != first_id}
    return best_query(belief, candidates)[0]


def execute_regenerated(
    *,
    first_id: str,
    root: dict[str, Any],
    refreshed: dict[str, dict[str, list[float]]],
    likelihoods: dict[str, list[list[float]]],
    routing: dict[str, dict[str, str]],
) -> dict[str, Any]:
    first_channel_id = routing[first_id]["channel_id"]
    branch = list(refreshed[first_id][first_channel_id])
    second_id = second_query(branch, likelihoods, first_id)
    second_channel_id = routing[second_id]["channel_id"]
    terminal = update(branch, likelihoods[second_id], CHANNEL_IDS.index(second_channel_id))
    return {
        "first_query_id": first_id,
        "first_channel_id": first_channel_id,
        "second_query_id": second_id,
        "second_channel_id": second_channel_id,
        "terminal_belief": terminal,
    }


def execute_fixed(
    *,
    first_id: str,
    root: dict[str, Any],
    likelihoods: dict[str, list[list[float]]],
    routing: dict[str, dict[str, str]],
) -> dict[str, Any]:
    first_channel_id = routing[first_id]["channel_id"]
    branch = update(root["prior"], likelihoods[first_id], CHANNEL_IDS.index(first_channel_id))
    second_id = second_query(branch, likelihoods, first_id)
    second_channel_id = routing[second_id]["channel_id"]
    terminal = update(branch, likelihoods[second_id], CHANNEL_IDS.index(second_channel_id))
    return {
        "first_query_id": first_id,
        "first_channel_id": first_channel_id,
        "second_query_id": second_id,
        "second_channel_id": second_channel_id,
        "terminal_belief": terminal,
    }


def score(
    *, raw: dict[str, Any], label_free: dict[str, Any], endpoints: dict[str, Any]
) -> dict[str, Any]:
    if label_free.get("status") != "serving_pass" or label_free.get("authorizes") != "endpoint_only" or not all((label_free.get("gates") or {}).values()):
        raise RuntimeError("label-free result does not authorize endpoint scoring")
    endpoint_rows = endpoints.get("endpoints")
    if not isinstance(endpoint_rows, list) or len(endpoint_rows) != 4:
        raise RuntimeError("endpoint rows are incomplete")
    endpoint_by_slot = {row["slot"]: row["correct_option_id"] for row in endpoint_rows}
    if set(endpoint_by_slot) != {"T1", "T2", "T3", "T4"}:
        raise RuntimeError("endpoint slot coverage is incomplete")
    roots = []
    refreshes = []
    for raw_root, raw_refresh in zip(raw["roots"], raw["refreshes"], strict=True):
        value = json.loads(raw_root)
        option_ids = tuple(f"O{index + 1}" for index in range(len(value["prior"])))
        root = parse_root(raw_root, option_ids)
        roots.append((option_ids, root))
        refreshes.append(parse_refresh(raw_refresh, option_ids))
    routing = label_free.get("consensus_routing")
    if not isinstance(routing, dict) or set(routing) != {"T1", "T2", "T3", "T4"}:
        raise RuntimeError("consensus routing is absent")

    task_rows = []
    for task_index, ((option_ids, root), refreshed) in enumerate(zip(roots, refreshes, strict=True)):
        slot = f"T{task_index + 1}"
        correct_id = endpoint_by_slot[slot]
        if correct_id not in option_ids:
            raise RuntimeError("correct option ID is invalid")
        correct_index = option_ids.index(correct_id)
        likelihoods = matrices(root, option_ids)
        public_task = label_free["semantic"]["tasks"][task_index]
        dynamic_first = public_task["dynamic"]["first_query_id"]
        myopic_first = public_task["myopic_first_query_id"]
        fixed_first = public_task["fixed"]["first_query_id"]
        random_first = QUERY_IDS[(RANDOM_SEED + task_index) % len(QUERY_IDS)]
        trajectories = {
            "dynamic": execute_regenerated(first_id=dynamic_first, root=root, refreshed=refreshed, likelihoods=likelihoods, routing=routing[slot]),
            "matched_myopic": execute_regenerated(first_id=myopic_first, root=root, refreshed=refreshed, likelihoods=likelihoods, routing=routing[slot]),
            "fixed_depth_two": execute_fixed(first_id=fixed_first, root=root, likelihoods=likelihoods, routing=routing[slot]),
            "random": execute_regenerated(first_id=random_first, root=root, refreshed=refreshed, likelihoods=likelihoods, routing=routing[slot]),
        }
        task_rows.append(
            {
                "slot": slot,
                "correct_option_id": correct_id,
                "policies": {
                    name: {**trajectory, **endpoint_metrics(trajectory["terminal_belief"], correct_index)}
                    for name, trajectory in trajectories.items()
                },
            }
        )
    means = {
        policy: {
            metric: sum(task["policies"][policy][metric] for task in task_rows) / 4
            for metric in ("brier", "log_loss", "correct_probability")
        }
        for policy in ("dynamic", "matched_myopic", "fixed_depth_two", "random")
    }
    dynamic_brier_wins = sum(
        task["policies"]["dynamic"]["brier"] < task["policies"]["matched_myopic"]["brier"]
        for task in task_rows
    )
    first_disagreements = sum(
        task["policies"]["dynamic"]["first_query_id"] != task["policies"]["matched_myopic"]["first_query_id"]
        for task in task_rows
    )
    gates = {
        "dynamic_changes_first_query": first_disagreements >= 2,
        "dynamic_brier_wins_three_of_four": dynamic_brier_wins >= 3,
        "dynamic_mean_brier_gain_at_least_001": means["matched_myopic"]["brier"] - means["dynamic"]["brier"] >= 0.01,
        "dynamic_log_loss_better_than_myopic": means["dynamic"]["log_loss"] < means["matched_myopic"]["log_loss"],
        "dynamic_nonworse_than_fixed": means["dynamic"]["brier"] <= means["fixed_depth_two"]["brier"] + 1e-9 and means["dynamic"]["log_loss"] <= means["fixed_depth_two"]["log_loss"] + 1e-9,
        "dynamic_better_than_random": means["dynamic"]["brier"] < means["random"]["brier"] and means["dynamic"]["log_loss"] < means["random"]["log_loss"],
        "all_rows_finite_nonsaturated": all(math.isfinite(task["policies"][policy][metric]) and 0 < task["policies"][policy]["correct_probability"] < 1 for task in task_rows for policy in task["policies"] for metric in ("brier", "log_loss", "correct_probability")),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "interface_version": "hiddenbench-dynamic-belief-v3-endpoint-score-v1",
        "status": "mechanics_pass" if passed else "mechanics_null",
        "decision": "development_protocol_authorized" if passed else "close_exact_v3_route",
        "authorizes": "separately_frozen_development_protocol_only" if passed else "nothing",
        "tasks": task_rows,
        "means": means,
        "summary": {"first_query_disagreements": first_disagreements, "dynamic_brier_wins": dynamic_brier_wins},
        "gates": gates,
        "model_calls_made": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--label-free", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(raw=load_object(args.raw), label_free=load_object(args.label_free), endpoints=load_object(args.endpoints))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "mechanics_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
