#!/usr/bin/env python3
"""Replay and verify a frozen ChemBench policy-ladder mechanics result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.mechanics import (
    BankedProposer,
    PolicyLadderPlanner,
    ProposalCache,
)
from environments.chembench_mopen.source import build_mixed_version_responses
from scripts.chembench_mopen_mechanics import (
    INITIAL_SUPPORT_NAMES,
    PRACTICAL_TIE_TOLERANCE,
    build_bank,
    comparison_at_tolerance,
)
from scripts.chembench_mopen_nonmyopic_opportunity import (
    EXECUTION_BUDGET,
    ValidationSlice,
    active_domains,
    comparison,
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-policy-ladder-mechanics-v4"
PROTOCOLS = {
    Path("results/nonmyopic/CHEMBENCH_SPECULATIVE_PARTICLE_MECHANICS_V3_PROTOCOL_20260814.md"):
        "370bd776e2f7b47a5cc92182ec759ea290c1395679c09a4a5c29e316f65580e1",
    Path("results/nonmyopic/CHEMBENCH_MDA_SECOND_PASS_ARCHITECTURE_REVIEW_20260814.md"):
        "a42458ee96fa7f89c6283ce9503778d8e9b0d5fd80db0361dba2a1b15ef0d38e",
    Path("results/nonmyopic/CHEMBENCH_POLICY_LADDER_MECHANICS_V4_PROTOCOL_20260814.md"):
        "68221f5ec63d3b04c31a1926ee739723ee710058f144ab4ca90707c0ff08e3fa",
}
SEALED_SLICES = (
    ValidationSlice("easy", "v4", 2026081701),
    ValidationSlice("medium", "v4", 2026081702),
    ValidationSlice("hard", "v4", 2026081703),
)
OPENED_SLICES = (
    ValidationSlice("easy", "v3", 2026081601),
    ValidationSlice("medium", "v3", 2026081602),
    ValidationSlice("hard", "v3", 2026081603),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(slice_results: Sequence[dict[str, Any]], level: int) -> list[float]:
    return [
        value
        for item in slice_results
        for value in item["policy_levels"][f"d{level}"]["truth_losses"]
    ]


def recompute_gate(
    slice_results: Sequence[dict[str, Any]], runtime_checks: Mapping[str, bool]
) -> dict[str, Any]:
    d1 = aggregate(slice_results, 1)
    d2 = aggregate(slice_results, 2)
    d3 = aggregate(slice_results, 3)
    d2_vs_d1 = comparison_at_tolerance(d1, d2, PRACTICAL_TIE_TOLERANCE)
    d3_vs_d2 = comparison_at_tolerance(d2, d3, PRACTICAL_TIE_TOLERANCE)
    calibrated = all(
        abs(
            item["policy_levels"][f"d{level}"]["planned_value"]
            - item["policy_levels"][f"d{level}"]["expected_terminal_mse"]
        )
        <= 1e-10
        for item in slice_results
        for level in (1, 2, 3)
    )
    per_slice_monotonic = all(
        item["policy_levels"]["d2"]["planned_value"]
        <= item["policy_levels"]["d1"]["planned_value"] + 1e-12
        and item["policy_levels"]["d3"]["planned_value"]
        <= item["policy_levels"]["d2"]["planned_value"] + 1e-12
        for item in slice_results
    )
    d2_root_differences = sum(
        item["policy_levels"]["d1"]["root_action_index"]
        != item["policy_levels"]["d2"]["root_action_index"]
        for item in slice_results
    )
    d3_root_differences = sum(
        item["policy_levels"]["d2"]["root_action_index"]
        != item["policy_levels"]["d3"]["root_action_index"]
        for item in slice_results
    )
    call_matched = all(
        item["call_matched_d1"]["matches_primary_d1"]
        and item["call_matched_d1"]["cache_misses_before"]
        == item["call_matched_d1"]["cache_misses_after"]
        for item in slice_results
    )
    conditions = {
        "runtime_checks_pass": all(runtime_checks.values()),
        "planned_truth_replay_calibrated": calibrated,
        "per_slice_model_risk_nonincreasing": per_slice_monotonic,
        "d2_mean_reduction_at_least_5pct": d2_vs_d1["relative_reduction"] >= 0.05,
        "d3_mean_reduction_at_least_5pct": d3_vs_d2["relative_reduction"] >= 0.05,
        "d2_truth_cell_majority": d2_vs_d1["wins"] > d2_vs_d1["losses"],
        "d3_truth_cell_majority": d3_vs_d2["wins"] > d3_vs_d2["losses"],
        "d2_root_differs_on_at_least_one_slice": d2_root_differences >= 1,
        "d3_root_differs_on_at_least_one_slice": d3_root_differences >= 1,
        "call_matched_d1_replay_exact": call_matched,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "d2_vs_d1": d2_vs_d1,
        "d3_vs_d2": d3_vs_d2,
        "d3_vs_d1": comparison_at_tolerance(d1, d3, PRACTICAL_TIE_TOLERANCE),
        "exact_descriptive": {
            "d2_vs_d1": comparison(d1, d2),
            "d3_vs_d2": comparison(d2, d3),
            "d3_vs_d1": comparison(d1, d3),
        },
        "d2_root_differences": d2_root_differences,
        "d3_root_differences": d3_root_differences,
    }


def verify(
    result_path: Path,
    transition_bank_path: Path,
    source_root: Path,
    *,
    expected_slices: Sequence[ValidationSlice],
) -> dict[str, Any]:
    for path, expected in PROTOCOLS.items():
        if sha256(path) != expected:
            raise RuntimeError(f"protocol hash mismatch: {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    transition_bank = json.loads(transition_bank_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result schema mismatch")
    if result.get("model_calls") != 0 or result.get("cost_usd") != 0.0:
        raise ValueError("mechanics result records nonzero calls or cost")
    if result["transition_bank"]["sha256"] != sha256(transition_bank_path):
        raise ValueError("transition-bank hash mismatch")
    if transition_bank.get("source_mode") != "registry_oracle":
        raise ValueError("transition bank source mode mismatch")
    source_binding = verify_source(source_root)
    if source_binding["commit"] != result["source"]["commit"]:
        raise ValueError("source binding mismatch")
    source = load_source(source_root)
    domains = active_domains(source)
    if list(domains) != result["active_domains"]:
        raise ValueError("active domain order mismatch")
    if result["slices"] != [item.__dict__ for item in expected_slices]:
        raise ValueError("validation slices mismatch")
    assays = frozen_assays()
    action_names = tuple(item.name for item in assays)
    replayed = []
    for slice_index, item in enumerate(expected_slices):
        mixed = build_mixed_version_responses(
            source,
            domains,
            INITIAL_SUPPORT_NAMES,
            difficulty=item.difficulty,
            initial_version="v2",
            truth_version=item.version,
            query_seed=item.query_seed,
            assays=assays,
        )
        bank = build_bank(
            mixed.observation_means,
            mixed.target_log_rates,
            domains,
            action_names,
        )
        record = transition_bank["slices"][item.name]
        seed = 2026081700 + slice_index
        if record["planner_seed"] != seed:
            raise ValueError(f"planner seed mismatch for {item.name}")
        cache = ProposalCache(
            BankedProposer("registry_oracle", record["records"]),
            source_mode="registry_oracle",
        )
        policy_levels = {}
        for level in (3, 2, 1):
            planner = PolicyLadderPlanner(bank, cache, mixed.truth_indices, seed=seed)
            policy_levels[f"d{level}"] = planner.evaluate_policy_level(
                level, execution_budget=EXECUTION_BUDGET
            )
        misses_before = cache.misses
        d1_replay = PolicyLadderPlanner(
            bank, cache, mixed.truth_indices, seed=seed
        ).evaluate_policy_level(1, execution_budget=EXECUTION_BUDGET)
        replayed_item = {
            "slice": item.name,
            "query_seed": item.query_seed,
            "num_domains": bank.num_models,
            "num_truths": len(mixed.truth_indices),
            "truth_domains": [domains[index] for index in mixed.truth_indices],
            "version_map_sha256": mixed.version_map_sha256,
            "policy_levels": policy_levels,
            "call_matched_d1": {
                "result": d1_replay,
                "matches_primary_d1": d1_replay == policy_levels["d1"],
                "cache_misses_before": misses_before,
                "cache_misses_after": cache.misses,
                "cache_hits_after": cache.hits,
            },
            "cache": {
                "entries": len(cache.records),
                "hits": cache.hits,
                "misses": cache.misses,
                "proposed_candidates": sum(len(value) for value in cache.records.values()),
            },
        }
        replayed.append(replayed_item)
    if replayed != result["slice_results"]:
        raise AssertionError("banked policy replay differs from producer result")
    gate = recompute_gate(replayed, result["runtime_checks"])
    if gate != result["gate"]:
        raise AssertionError("independently recomputed gate differs from producer gate")
    return {
        "schema_version": f"{SCHEMA_VERSION}-verification",
        "status": "verified" if gate["passed"] else "verified_failed_closed",
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "transition_bank_path": str(transition_bank_path),
        "transition_bank_sha256": sha256(transition_bank_path),
        "source": source_binding,
        "slices": [item.__dict__ for item in expected_slices],
        "gate": gate,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--transition-bank", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen-opened-v3", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output}")
    expected_slices = OPENED_SLICES if args.screen_opened_v3 else SEALED_SLICES
    verification = verify(
        args.result,
        args.transition_bank,
        args.source_root,
        expected_slices=expected_slices,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": verification["status"],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
