#!/usr/bin/env python3
"""Run the frozen zero-call ChemBench dynamic-support mechanics gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.mechanics import (
    DynamicPlanner,
    FixedProposer,
    HistoryBlindProposer,
    ModelBank,
    OracleProposer,
    ProposalCache,
    ScriptedResidualProposer,
)
from scripts.chembench_mopen_nonmyopic_opportunity import (
    EXECUTION_BUDGET,
    ValidationSlice,
    active_domains,
    categorical_likelihoods,
    comparison,
    frozen_assays,
    load_source,
    response_matrices,
    verify_source,
)


SCHEMA_VERSION = "chembench-mopen-mechanics-v1"
PROTOCOL_PATH = Path(
    "results/nonmyopic/CHEMBENCH_MOPEN_MECHANICS_V1_PROTOCOL_20260814.md"
)
PROTOCOL_SHA256 = "bcf58c405036d16284b8131330d70ae7114524bd961e5681139c70ee0d9e7c43"
AMENDMENT_PATH = Path(
    "results/nonmyopic/CHEMBENCH_MOPEN_MECHANICS_V1_DEVELOPMENT_AMENDMENT_20260814.md"
)
AMENDMENT_SHA256 = "932c1a789eabb1acccaad7e7129443fe933d4410f33412e23784bde6633b5c01"
ARCHITECTURE_PATH = Path(
    "results/nonmyopic/CHEMBENCH_NONMYOPIC_MOPEN_ARCHITECTURE_PROTOCOL_20260814.md"
)
ARCHITECTURE_SHA256 = "552639280609b0c179304fb3dd784b9dfc8d0960e444bc4e312055762e7bd7c6"
PRACTICAL_TIE_TOLERANCE = 1e-6
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
MECHANICS_SLICES = (
    ValidationSlice("easy", "v3", 2026081601),
    ValidationSlice("medium", "v3", 2026081602),
    ValidationSlice("hard", "v3", 2026081603),
)
OPENED_SCREEN_SLICES = (
    ValidationSlice("easy", "v2", 2026081502),
    ValidationSlice("medium", "v2", 2026081505),
    ValidationSlice("hard", "v2", 2026081508),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(arguments: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_pushed_commit(required_commit: str) -> str:
    head = _git_value(("rev-parse", "HEAD"))
    resolved = _git_value(("rev-parse", required_commit))
    if head != resolved:
        raise RuntimeError(f"required commit is not HEAD: {resolved} != {head}")
    remote = "origin/codex/location-finding-llmstrategy"
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, remote],
        check=True,
        capture_output=True,
        text=True,
    )
    return head


def verify_protocol_bindings() -> None:
    for path, expected in (
        (PROTOCOL_PATH, PROTOCOL_SHA256),
        (AMENDMENT_PATH, AMENDMENT_SHA256),
        (ARCHITECTURE_PATH, ARCHITECTURE_SHA256),
    ):
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"protocol binding mismatch for {path}: {actual} != {expected}")


def assay_groups(names: Sequence[str]) -> tuple[str | None, ...]:
    result: list[str | None] = []
    for name in names:
        if name == "baseline":
            result.append(None)
        elif name.startswith("C_A="):
            result.append("C_A")
        elif name.startswith("C_I="):
            result.append("C_I")
        elif name.startswith("C_B="):
            result.append("C_B")
        elif name.startswith("C_P="):
            result.append("C_P")
        elif name.startswith("T="):
            result.append("T")
        elif name.startswith("pH="):
            result.append("pH")
        else:
            raise ValueError(f"unrecognized frozen assay name: {name}")
    if set(item for item in result if item is not None) != {"C_A", "C_I", "C_B", "C_P", "T", "pH"}:
        raise AssertionError("frozen assays do not cover all six action groups")
    return tuple(result)


def _ids(names: Sequence[str], model_index: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(model_index[name] for name in names if name in model_index)


def scripted_dictionary(model_index: Mapping[str, int]) -> dict[tuple[str, int], tuple[int, ...]]:
    names: dict[tuple[str, int], tuple[str, ...]] = {
        ("C_A", 0): ("c7_substrate_inhibition", "c68_anticoop_hill", "c69_fractal_kinetics", "c48_sinh_competitive"),
        ("C_A", 1): ("c0_michaelis_menten", "c8_hill_cooperativity", "c33_hill_competitive", "c37_hill_arrhenius"),
        ("C_A", 2): ("c8_hill_cooperativity", "c67_allosteric_act", "c73_metal_activation", "c74_product_activation"),
        ("C_I", 0): ("c1_competitive_inhibition", "c6_uncompetitive_inhibition", "c9_noncompetitive_inhibition", "c70_mixed_inhibition"),
        ("C_I", 1): ("c33_hill_competitive", "c34_hill_uncompetitive", "c35_hill_noncompetitive", "c71_coop_inhibition"),
        ("C_I", 2): ("c67_allosteric_act", "c73_metal_activation", "c74_product_activation", "c92_allosteric_act_feedback"),
        ("C_B", 0): ("c75_two_substrate_inhibition", "c98_two_sub_inh_arrhenius", "c65_ordered_bi_bi", "c5_pingpong_bisubstrate"),
        ("C_B", 1): ("c5_pingpong_bisubstrate", "c65_ordered_bi_bi", "c25_pingpong_competitive", "c26_pingpong_noncompetitive"),
        ("C_B", 2): ("c65_ordered_bi_bi", "c76_ordered_bi_bi_arrhenius", "c89_ordered_bi_bi_competitive", "c90_ordered_bi_bi_noncomp"),
        ("C_P", 0): ("c2_product_inhibition", "c36_hill_product", "c51_sinh_product", "c61_sinh_product_arrhenius"),
        ("C_P", 1): ("c16_mm_product_arrhenius", "c47_hill_product_arrhenius", "c74_product_activation", "c92_allosteric_act_feedback"),
        ("C_P", 2): ("c74_product_activation", "c92_allosteric_act_feedback", "c67_allosteric_act", "c73_metal_activation"),
        ("T", 0): ("c3_arrhenius_temperature", "c10_mm_competitive_arrhenius", "c23_pingpong_arrhenius", "c37_hill_arrhenius"),
        ("T", 1): ("c52_sinh_arrhenius", "c76_ordered_bi_bi_arrhenius", "c78_allosteric_act_arrhenius", "c79_anticoop_arrhenius"),
        ("T", 2): ("c80_mixed_inh_arrhenius", "c81_coop_inh_arrhenius", "c98_two_sub_inh_arrhenius", "c29_pingpong_noncompetitive_arrhenius"),
        ("pH", 0): ("c72_monotonic_ph", "c68_anticoop_hill", "c69_fractal_kinetics", "c70_mixed_inhibition"),
        ("pH", 1): ("c72_monotonic_ph", "c67_allosteric_act", "c73_metal_activation", "c74_product_activation"),
        ("pH", 2): ("c72_monotonic_ph", "c78_allosteric_act_arrhenius", "c92_allosteric_act_feedback", "c94_metal_act_competitive"),
    }
    return {key: _ids(value, model_index) for key, value in names.items()}


def build_bank(
    means: np.ndarray,
    target_features: np.ndarray,
    domains: Sequence[str],
    action_names: Sequence[str],
) -> ModelBank:
    model_index = {name: index for index, name in enumerate(domains)}
    if any(name not in model_index for name in INITIAL_SUPPORT_NAMES):
        raise ValueError("initial support is absent from active source domains")
    return ModelBank(
        categorical_likelihoods(means),
        target_features,
        model_names=domains,
        action_names=action_names,
        action_groups=assay_groups(action_names),
        initial_support=_ids(INITIAL_SUPPORT_NAMES, model_index),
        outside_prior=0.35,
        live_cap=12,
        reserve_cap=12,
    )


def _evaluate_mode(bank: ModelBank, proposer: Any, *, seed: int) -> tuple[dict[str, Any], ProposalCache]:
    cache = ProposalCache(proposer)
    horizons: dict[str, Any] = {}
    for depth in (3, 2, 1):
        planner = DynamicPlanner(bank, cache, seed=seed)
        horizons[f"d{depth}"] = planner.evaluate_horizon(depth, execution_budget=EXECUTION_BUDGET)
    misses_before_replay = cache.misses
    replay_planner = DynamicPlanner(bank, cache, seed=seed)
    replay = replay_planner.evaluate_horizon(1, execution_budget=EXECUTION_BUDGET)
    call_matched = {
        "result": replay,
        "matches_dynamic_d1": replay == horizons["d1"],
        "cache_misses_before": misses_before_replay,
        "cache_misses_after": cache.misses,
        "cache_hits_after": cache.hits,
    }
    return {
        "mode": cache.source_mode,
        "horizons": horizons,
        "call_matched_myopic": call_matched,
        "cache": {
            "entries": len(cache.records),
            "hits": cache.hits,
            "misses": cache.misses,
            "proposed_candidates": sum(len(value) for value in cache.records.values()),
        },
    }, cache


def _mode_summary(bank: ModelBank, proposer: Any, *, seed: int) -> dict[str, Any]:
    result, _ = _evaluate_mode(bank, proposer, seed=seed)
    return result


def _aggregate(slice_results: Sequence[dict[str, Any]], mode: str, depth: int) -> list[float]:
    return [
        value
        for item in slice_results
        for value in item["modes"][mode]["horizons"][f"d{depth}"]["truth_losses"]
    ]


def comparison_at_tolerance(
    left: Sequence[float], right: Sequence[float], tolerance: float
) -> dict[str, Any]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    difference = left_values - right_values
    wins = int(np.sum(difference > tolerance))
    losses = int(np.sum(difference < -tolerance))
    ties = int(len(difference) - wins - losses)
    left_mean = float(np.mean(left_values))
    right_mean = float(np.mean(right_values))
    return {
        "tolerance": float(tolerance),
        "left_mean": left_mean,
        "right_mean": right_mean,
        "absolute_reduction": left_mean - right_mean,
        "relative_reduction": ((left_mean - right_mean) / left_mean if left_mean > 0 else 0.0),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def apply_mechanics_gate(slice_results: Sequence[dict[str, Any]], runtime_checks: Mapping[str, bool]) -> dict[str, Any]:
    d1 = _aggregate(slice_results, "oracle", 1)
    d2 = _aggregate(slice_results, "oracle", 2)
    d3 = _aggregate(slice_results, "oracle", 3)
    d2_vs_d1 = comparison_at_tolerance(d1, d2, PRACTICAL_TIE_TOLERANCE)
    d3_vs_d2 = comparison_at_tolerance(d2, d3, PRACTICAL_TIE_TOLERANCE)
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
        "d3_vs_d1": comparison_at_tolerance(d1, d3, PRACTICAL_TIE_TOLERANCE),
        "exact_descriptive": {
            "d2_vs_d1": comparison(d1, d2),
            "d3_vs_d2": comparison(d2, d3),
            "d3_vs_d1": comparison(d1, d3),
        },
        "root_differences": root_differences,
    }


def run_mechanics(
    source_root: Path,
    *,
    slices: Sequence[ValidationSlice],
    implementation_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_protocol_bindings()
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    domains = active_domains(source)
    assays = frozen_assays()
    action_names = tuple(item.name for item in assays)
    slice_results = []
    transition_bank: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}-transition-bank",
        "source_mode": "registry_oracle",
        "slices": {},
    }
    runtime_checks = {
        "model_calls_zero": True,
        "cost_zero": True,
        "policy_state_has_no_truth_id": True,
        "invalid_proposal_parent_immutable": True,
        "finite_and_normalized": True,
    }
    model_index = {name: index for index, name in enumerate(domains)}
    dictionary = scripted_dictionary(model_index)
    noninitial = tuple(index for index, name in enumerate(domains) if name not in INITIAL_SUPPORT_NAMES)
    for slice_index, item in enumerate(slices):
        means, target_features = response_matrices(source, domains, item, assays)
        bank = build_bank(means, target_features, domains, action_names)
        parent = bank.initial_state()
        parent_snapshot = parent
        invalid_child = bank.transition(parent, 0, 0, (-1, bank.num_models, bank.initial_support[0]))
        runtime_checks["invalid_proposal_parent_immutable"] &= parent == parent_snapshot
        runtime_checks["invalid_proposal_parent_immutable"] &= invalid_child.discovered == parent.discovered
        runtime_checks["policy_state_has_no_truth_id"] &= "truth" not in json.dumps(parent.public_key()).lower()
        runtime_checks["finite_and_normalized"] &= math.isclose(
            sum(parent.represented_mass) + parent.outside_mass,
            1.0,
            abs_tol=1e-12,
            rel_tol=0.0,
        )

        seed = 2026081600 + slice_index
        oracle_result, oracle_cache = _evaluate_mode(bank, OracleProposer(bank), seed=seed)
        scripted_result = _mode_summary(
            bank, ScriptedResidualProposer(bank, dictionary), seed=seed
        )
        fixed_result = _mode_summary(bank, FixedProposer(), seed=seed)
        blind_result = _mode_summary(bank, HistoryBlindProposer(noninitial), seed=seed)
        for mode_result in (oracle_result, scripted_result, fixed_result, blind_result):
            for horizon in mode_result["horizons"].values():
                runtime_checks["finite_and_normalized"] &= math.isfinite(
                    horizon["expected_terminal_mse"]
                )
        slice_results.append(
            {
                "slice": item.name,
                "query_seed": item.query_seed,
                "num_domains": bank.num_models,
                "num_queries": target_features.shape[1],
                "modes": {
                    "oracle": oracle_result,
                    "scripted": scripted_result,
                    "fixed": fixed_result,
                    "history_blind": blind_result,
                },
            }
        )
        transition_bank["slices"][item.name] = {
            "planner_seed": seed,
            "records": oracle_cache.records,
        }

    gate = apply_mechanics_gate(slice_results, runtime_checks)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gate["passed"] else "failed_closed",
        "model_calls": 0,
        "cost_usd": 0.0,
        "implementation_commit": implementation_commit,
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": _sha256(PROTOCOL_PATH),
        },
        "development_amendment": {
            "path": str(AMENDMENT_PATH),
            "sha256": _sha256(AMENDMENT_PATH),
        },
        "architecture_protocol": {
            "path": str(ARCHITECTURE_PATH),
            "sha256": _sha256(ARCHITECTURE_PATH),
        },
        "source": source_binding,
        "active_domains": list(domains),
        "initial_support": list(INITIAL_SUPPORT_NAMES),
        "slices": [item.__dict__ for item in slices],
        "runtime_checks": runtime_checks,
        "slice_results": slice_results,
        "gate": gate,
    }
    return result, transition_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition-bank", type=Path, required=True)
    parser.add_argument("--required-commit")
    parser.add_argument("--screen-opened-v2", action="store_true")
    args = parser.parse_args()
    for path in (args.output, args.transition_bank):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    if args.screen_opened_v2:
        if args.required_commit:
            raise ValueError("development screen must not bind a required commit")
        implementation_commit = _git_value(("rev-parse", "HEAD")) + "+working-tree-screen"
        slices = OPENED_SCREEN_SLICES
    else:
        if not args.required_commit:
            raise ValueError("exact v3 mechanics requires --required-commit")
        implementation_commit = require_pushed_commit(args.required_commit)
        slices = MECHANICS_SLICES
    result, bank = run_mechanics(
        args.source_root,
        slices=slices,
        implementation_commit=implementation_commit,
    )
    args.transition_bank.parent.mkdir(parents=True, exist_ok=True)
    args.transition_bank.write_text(
        json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["transition_bank"] = {
        "path": str(args.transition_bank),
        "sha256": _sha256(args.transition_bank),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate": result["gate"],
                "output": str(args.output),
                "output_sha256": _sha256(args.output),
                "transition_bank_sha256": result["transition_bank"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
