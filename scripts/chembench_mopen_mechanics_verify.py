#!/usr/bin/env python3
"""Producer-independent replay verifier for ChemBench M-open mechanics V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.mechanics import (
    BankedProposer,
    DynamicPlanner,
    ModelBank,
    ProposalCache,
)
from environments.chembench_mopen.source import build_mixed_version_responses
from scripts.chembench_mopen_nonmyopic_opportunity import (
    EXECUTION_BUDGET,
    ValidationSlice,
    active_domains,
    categorical_likelihoods,
    comparison,
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-mopen-mechanics-v2"
PROTOCOLS = {
    Path("results/nonmyopic/CHEMBENCH_MOPEN_MECHANICS_V1_PROTOCOL_20260814.md"):
        "bcf58c405036d16284b8131330d70ae7114524bd961e5681139c70ee0d9e7c43",
    Path("results/nonmyopic/CHEMBENCH_MOPEN_MECHANICS_V1_DEVELOPMENT_AMENDMENT_20260814.md"):
        "932c1a789eabb1acccaad7e7129443fe933d4410f33412e23784bde6633b5c01",
    Path("results/nonmyopic/CHEMBENCH_NONMYOPIC_MOPEN_ARCHITECTURE_PROTOCOL_20260814.md"):
        "552639280609b0c179304fb3dd784b9dfc8d0960e444bc4e312055762e7bd7c6",
    Path("results/nonmyopic/CHEMBENCH_MOPEN_MECHANICS_V1_TERMINAL_20260814.md"):
        "d69843882b9c97f7c6b325b38b2aa5ca8e29a9e2de439146ccf085774220f91a",
    Path("results/nonmyopic/CHEMBENCH_MOPEN_MECHANICS_V2_PROTOCOL_20260814.md"):
        "3fd3e9ba0c1bb80645c0687ddf8e5b09985f5a2c4d136e2f8c3bc17dd99d0279",
}
INITIAL_SUPPORT_NAMES = (
    "c0_michaelis_menten",
    "c1_competitive_inhibition",
    "c2_product_inhibition",
    "c3_arrhenius_temperature",
    "c5_pingpong_bisubstrate",
    "c6_uncompetitive_inhibition",
    "c7_substrate_inhibition",
    "c8_hill_cooperativity",
    "c9_noncompetitive_inhibition",
)
EXPECTED_SLICES = (
    ValidationSlice("easy", "v3", 2026081601),
    ValidationSlice("medium", "v3", 2026081602),
    ValidationSlice("hard", "v3", 2026081603),
)
OPENED_SCREEN_SLICES = (
    ValidationSlice("easy", "v2", 2026081502),
    ValidationSlice("medium", "v2", 2026081505),
    ValidationSlice("hard", "v2", 2026081508),
)
PRACTICAL_TIE_TOLERANCE = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_groups(names: Sequence[str]) -> tuple[str | None, ...]:
    groups = []
    for name in names:
        if name == "baseline":
            groups.append(None)
        elif name.startswith("C_A="):
            groups.append("C_A")
        elif name.startswith("C_I="):
            groups.append("C_I")
        elif name.startswith("C_B="):
            groups.append("C_B")
        elif name.startswith("C_P="):
            groups.append("C_P")
        elif name.startswith("T="):
            groups.append("T")
        elif name.startswith("pH="):
            groups.append("pH")
        else:
            raise ValueError(f"unknown assay name: {name}")
    return tuple(groups)


def _build_bank(
    means: np.ndarray,
    targets: np.ndarray,
    domains: Sequence[str],
    action_names: Sequence[str],
) -> ModelBank:
    index = {name: item for item, name in enumerate(domains)}
    return ModelBank(
        categorical_likelihoods(means),
        targets,
        model_names=domains,
        action_names=action_names,
        action_groups=_action_groups(action_names),
        initial_support=tuple(index[name] for name in INITIAL_SUPPORT_NAMES),
        outside_prior=0.35,
        live_cap=12,
        reserve_cap=12,
    )


def _compare(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    difference = left_values - right_values
    wins = int(np.sum(difference > PRACTICAL_TIE_TOLERANCE))
    losses = int(np.sum(difference < -PRACTICAL_TIE_TOLERANCE))
    ties = int(len(difference) - wins - losses)
    left_mean = float(np.mean(left_values))
    right_mean = float(np.mean(right_values))
    return {
        "tolerance": PRACTICAL_TIE_TOLERANCE,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "absolute_reduction": left_mean - right_mean,
        "relative_reduction": ((left_mean - right_mean) / left_mean if left_mean > 0 else 0.0),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _aggregate(slice_results: Sequence[dict[str, Any]], depth: int) -> list[float]:
    return [
        value
        for item in slice_results
        for value in item["modes"]["oracle"]["horizons"][f"d{depth}"]["truth_losses"]
    ]


def _recompute_gate(slice_results: Sequence[dict[str, Any]], runtime_checks: dict[str, bool]) -> dict[str, Any]:
    d1 = _aggregate(slice_results, 1)
    d2 = _aggregate(slice_results, 2)
    d3 = _aggregate(slice_results, 3)
    d2_vs_d1 = _compare(d1, d2)
    d3_vs_d2 = _compare(d2, d3)
    root_differences = sum(
        item["modes"]["oracle"]["horizons"]["d1"]["root_action_index"]
        != item["modes"]["oracle"]["horizons"]["d2"]["root_action_index"]
        for item in slice_results
    )
    call_matched = all(
        item["modes"]["oracle"]["call_matched_myopic"]["matches_dynamic_d1"]
        and item["modes"]["oracle"]["call_matched_myopic"]["cache_misses_before"]
        == item["modes"]["oracle"]["call_matched_myopic"]["cache_misses_after"]
        for item in slice_results
    )
    fixed_empty = all(
        item["modes"]["fixed"]["cache"]["proposed_candidates"] == 0
        and item["modes"]["fixed"]["mode"] == "fixed_support"
        for item in slice_results
    )
    blind_separate = all(
        item["modes"]["history_blind"]["mode"] == "history_blind"
        for item in slice_results
    )
    conditions = {
        "runtime_checks_pass": all(runtime_checks.values()),
        "call_matched_replay_exact": call_matched,
        "d2_mean_reduction_at_least_5pct": d2_vs_d1["relative_reduction"] >= 0.05,
        "d3_mean_reduction_at_least_5pct": d3_vs_d2["relative_reduction"] >= 0.05,
        "d2_truth_cell_majority": d2_vs_d1["wins"] > d2_vs_d1["losses"],
        "d3_truth_cell_majority": d3_vs_d2["wins"] > d3_vs_d2["losses"],
        "d2_root_differs_on_at_least_one_slice": root_differences >= 1,
        "fixed_support_has_no_proposals": fixed_empty,
        "history_blind_is_separate": blind_separate,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "d2_vs_d1": d2_vs_d1,
        "d3_vs_d2": d3_vs_d2,
        "d3_vs_d1": _compare(d1, d3),
        "exact_descriptive": {
            "d2_vs_d1": comparison(d1, d2),
            "d3_vs_d2": comparison(d2, d3),
            "d3_vs_d1": comparison(d1, d3),
        },
        "root_differences": root_differences,
    }


def verify(
    result_path: Path,
    bank_path: Path,
    source_root: Path,
    *,
    expected_slices: Sequence[ValidationSlice] = EXPECTED_SLICES,
) -> dict[str, Any]:
    for path, expected in PROTOCOLS.items():
        if _sha256(path) != expected:
            raise RuntimeError(f"protocol hash mismatch: {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    transition_bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result schema mismatch")
    if result.get("model_calls") != 0 or result.get("cost_usd") != 0.0:
        raise ValueError("mechanics result records nonzero calls or cost")
    if result["transition_bank"]["sha256"] != _sha256(bank_path):
        raise ValueError("transition-bank hash mismatch")
    if transition_bank.get("source_mode") != "registry_oracle":
        raise ValueError("transition bank is not the oracle bank")
    source_binding = verify_source(source_root)
    if source_binding["commit"] != result["source"]["commit"]:
        raise ValueError("source commit differs from recorded result")
    source = load_source(source_root)
    domains = active_domains(source)
    if list(domains) != result["active_domains"]:
        raise ValueError("active domains differ from recorded result")
    if result["slices"] != [item.__dict__ for item in expected_slices]:
        raise ValueError("result does not use the expected slices")
    assays = frozen_assays()
    action_names = tuple(item.name for item in assays)
    replayed_oracle: dict[str, Any] = {}
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
        bank = _build_bank(
            mixed.observation_means,
            mixed.target_log_rates,
            domains,
            action_names,
        )
        truth_indices = mixed.truth_indices
        bank_record = transition_bank["slices"][item.name]
        expected_seed = 2026081600 + slice_index
        if bank_record["planner_seed"] != expected_seed:
            raise ValueError(f"planner seed mismatch for {item.name}")
        proposer = BankedProposer("registry_oracle", bank_record["records"])
        cache = ProposalCache(proposer, source_mode="registry_oracle")
        horizons = {}
        for depth in (3, 2, 1):
            planner = DynamicPlanner(bank, cache, seed=expected_seed)
            horizons[f"d{depth}"] = planner.evaluate_horizon(
                depth,
                execution_budget=EXECUTION_BUDGET,
                truth_indices=truth_indices,
            )
        misses_before = cache.misses
        replay = DynamicPlanner(bank, cache, seed=expected_seed).evaluate_horizon(
            1,
            execution_budget=EXECUTION_BUDGET,
            truth_indices=truth_indices,
        )
        oracle = {
            "mode": "registry_oracle",
            "horizons": horizons,
            "call_matched_myopic": {
                "result": replay,
                "matches_dynamic_d1": replay == horizons["d1"],
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
        recorded_slice = next(value for value in result["slice_results"] if value["slice"] == item.name)
        expected_versions = {
            domain: mixed.version_by_model[index] for index, domain in enumerate(domains)
        }
        if recorded_slice["version_by_model"] != expected_versions:
            raise AssertionError(f"version-map replay mismatch for {item.name}")
        if recorded_slice["version_map_sha256"] != mixed.version_map_sha256:
            raise AssertionError(f"version-map hash mismatch for {item.name}")
        if recorded_slice["truth_domains"] != [domains[index] for index in truth_indices]:
            raise AssertionError(f"truth cohort mismatch for {item.name}")
        if oracle != recorded_slice["modes"]["oracle"]:
            raise AssertionError(f"oracle replay mismatch for {item.name}")
        replayed_oracle[item.name] = oracle
    recomputed_gate = _recompute_gate(result["slice_results"], result["runtime_checks"])
    if recomputed_gate != result["gate"]:
        raise AssertionError("producer-independent gate replay mismatch")
    return {
        "schema_version": f"{SCHEMA_VERSION}-verification",
        "status": "verified" if result["status"] == ("passed" if recomputed_gate["passed"] else "failed_closed") else "invalid",
        "result_sha256": _sha256(result_path),
        "transition_bank_sha256": _sha256(bank_path),
        "source_commit": source_binding["commit"],
        "replayed_slices": sorted(replayed_oracle),
        "gate": recomputed_gate,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--transition-bank", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen-opened-v2", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verification: {args.output}")
    verification = verify(
        args.result,
        args.transition_bank,
        args.source_root,
        expected_slices=(OPENED_SCREEN_SLICES if args.screen_opened_v2 else EXPECTED_SLICES),
    )
    if verification["status"] != "verified":
        raise RuntimeError("result status does not match independently replayed gate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
