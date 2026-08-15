#!/usr/bin/env python3
"""Run the frozen zero-call staged-compound ChemBench opportunity gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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

from environments.chembench_mopen.compositional import (
    AtomicStructureOracleProposer,
    AuditedCompositionalPolicyPlanner,
    CompoundSignature,
    CompositionalEditCompiler,
    StructureParticleIndex,
    replay_proposer,
)
from environments.chembench_mopen.mechanics import FixedProposer, ModelBank, ProposalCache
from environments.chembench_mopen.source import _query_assays
from scripts.chembench_factored_mopen_oracle import sha256
from scripts.chembench_mopen_mechanics import INITIAL_SUPPORT_NAMES, assay_groups
from scripts.chembench_mopen_nonmyopic_opportunity import (
    categorical_likelihoods,
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-staged-compound-corridor-v1"
PROTOCOL_PATH = Path(
    "results/nonmyopic/CHEMBENCH_STAGED_COMPOUND_CORRIDOR_PROTOCOL_20260815.md"
)
PROTOCOL_SHA256 = "22562fb887717ad1b5478fd2d447fccb0abfde91247fddad9a0f60a591a263a7"
INFERENCE_VERSIONS = ("v0", "v1", "v2")
TRUTH_VERSION = "v3"
DIFFICULTIES = ("easy", "medium", "hard")
QUERY_SEEDS = {"easy": 2026084001, "medium": 2026084002, "hard": 2026084003}
NUM_QUERIES = 512
EXECUTION_BUDGET = 4
OUTSIDE_PRIOR = 0.35
PRACTICAL_TIE = 1e-8
TRUTH_STRUCTURES = (
    "c19_mm_competitive_arrhenius_ph",
    "c20_mm_uncompetitive_arrhenius_ph",
    "c21_mm_noncompetitive_arrhenius_ph",
    "c22_mm_product_arrhenius_ph",
    "c32_pingpong_competitive_arrhenius_ph",
    "c44_hill_competitive_arrhenius_ph",
    "c45_hill_noncompetitive_arrhenius_ph",
    "c59_sinh_competitive_arrhenius_ph",
    "c60_sinh_noncompetitive_arrhenius_ph",
)
PRIMITIVE_SIGNATURES = {
    "c0_michaelis_menten": CompoundSignature("mm"),
    "c1_competitive_inhibition": CompoundSignature("mm", "competitive"),
    "c2_product_inhibition": CompoundSignature("mm", "product"),
    "c3_arrhenius_temperature": CompoundSignature("mm", temperature="arrhenius"),
    "c4_ph_activity": CompoundSignature("mm", ph_dependence="bell_curve"),
    "c5_pingpong_bisubstrate": CompoundSignature("pingpong"),
    "c6_uncompetitive_inhibition": CompoundSignature("mm", "uncompetitive"),
    "c7_substrate_inhibition": CompoundSignature("substrate_inh"),
    "c8_hill_cooperativity": CompoundSignature("hill"),
    "c9_noncompetitive_inhibition": CompoundSignature("mm", "noncompetitive"),
}


def git_value(arguments: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_pushed_commit(required_commit: str) -> str:
    head = git_value(("rev-parse", "HEAD"))
    resolved = git_value(("rev-parse", required_commit))
    if head != resolved:
        raise RuntimeError(f"required commit is not HEAD: {resolved} != {head}")
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            head,
            "origin/codex/location-finding-llmstrategy",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return head


def standard_structure_names(source: Any) -> tuple[str, ...]:
    names = tuple(
        name
        for name in source.CHEM_DOMAIN_REGISTRY
        if name.startswith("c")
        and name.split("_", 1)[0][1:].isdigit()
        and int(name.split("_", 1)[0][1:]) <= 64
    )
    expected = tuple(f"c{index}" for index in range(65))
    actual = tuple(name.split("_", 1)[0] for name in names)
    if actual != expected:
        raise ValueError("source c0-c64 structure sequence is incomplete or reordered")
    return names


def source_signatures(source: Any) -> dict[str, CompoundSignature]:
    compound = importlib.import_module("autoscilab.oracle.compound_domains")
    signatures = dict(PRIMITIVE_SIGNATURES)
    for name, substrate, inhibitor, temperature, ph_dependence in compound.COMPOUND_DOMAIN_SPECS:
        if int(name.split("_", 1)[0][1:]) > 64:
            continue
        signatures[name] = CompoundSignature(
            substrate=substrate,
            inhibitor=inhibitor,
            temperature=temperature,
            ph_dependence=ph_dependence,
        )
    names = standard_structure_names(source)
    if set(signatures) != set(names):
        raise ValueError("public signatures do not exactly cover source c0-c64")
    return {name: signatures[name] for name in names}


def _evaluate_particle(
    source: Any,
    structure: str,
    difficulty: str,
    version: str,
    assays: Sequence[Any],
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parameters = source._PARAMS[structure][difficulty][version]
    rate_fn = source._RATE_FNS[structure]
    means = []
    for assay in assays:
        value = float(rate_fn(parameters, *assay.values))
        value *= float(source._secondary_effects(assay.values[5], assay.values[6]))
        means.append(max(value, 0.0))
    targets = []
    for query in queries:
        value = float(rate_fn(parameters, *query))
        value *= float(source._secondary_effects(query[5], query[6]))
        targets.append(math.log1p(max(value, 0.0)))
    return np.asarray(means, dtype=float), np.asarray(targets, dtype=float)


def build_particle_arrays(
    source: Any, difficulty: str
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[int, ...]],
    tuple[int, ...],
    str,
]:
    structures = standard_structure_names(source)
    assays = frozen_assays()
    queries = _query_assays(
        QUERY_SEEDS[difficulty],
        source.CHEM_INPUT_BOUNDS,
        source.CHEM_LOG_VARS,
        NUM_QUERIES,
    )
    means = []
    targets = []
    names = []
    particle_structure = []
    inference_particles: dict[str, tuple[int, ...]] = {}
    for structure in structures:
        group = []
        for version in INFERENCE_VERSIONS:
            particle_means, particle_targets = _evaluate_particle(
                source, structure, difficulty, version, assays, queries
            )
            group.append(len(names))
            means.append(particle_means)
            targets.append(particle_targets)
            names.append(f"{structure}@{version}")
            particle_structure.append(structure)
        inference_particles[structure] = tuple(group)
    truth_particles = []
    for structure in TRUTH_STRUCTURES:
        particle_means, particle_targets = _evaluate_particle(
            source, structure, difficulty, TRUTH_VERSION, assays, queries
        )
        truth_particles.append(len(names))
        means.append(particle_means)
        targets.append(particle_targets)
        names.append(f"{structure}@{TRUTH_VERSION}:heldout")
        particle_structure.append(structure)
    arrays = (np.asarray(means, dtype=float), np.asarray(targets, dtype=float))
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(array.astype(np.float64).tobytes(order="C"))
    return (
        arrays[0],
        arrays[1],
        tuple(names),
        tuple(particle_structure),
        inference_particles,
        tuple(truth_particles),
        digest.hexdigest(),
    )


def make_bank(
    source: Any, difficulty: str, *, full_support: bool = False
) -> tuple[
    ModelBank,
    CompositionalEditCompiler,
    StructureParticleIndex,
    tuple[int, ...],
    str,
]:
    signatures = source_signatures(source)
    (
        means,
        targets,
        model_names,
        particle_structure,
        inference_particles,
        truth_particles,
        response_sha256,
    ) = build_particle_arrays(source, difficulty)
    initial_structures = (
        tuple(signatures)
        if full_support
        else tuple(name for name in INITIAL_SUPPORT_NAMES if name in signatures)
    )
    initial_particles = tuple(
        particle
        for structure in initial_structures
        for particle in inference_particles[structure]
    )
    assays = frozen_assays()
    bank = ModelBank(
        categorical_likelihoods(means),
        targets,
        model_names,
        tuple(assay.name for assay in assays),
        assay_groups(tuple(assay.name for assay in assays)),
        initial_particles,
        outside_prior=OUTSIDE_PRIOR,
        live_cap=len(model_names),
        reserve_cap=0,
    )
    particles = StructureParticleIndex(
        particle_structure,
        inference_particles=inference_particles,
        truth_particles=truth_particles,
    )
    return bank, CompositionalEditCompiler(signatures), particles, truth_particles, response_sha256


def _merge_audit(target: dict[str, dict[str, Any]], source: Mapping[str, dict[str, Any]]) -> None:
    for key, value in source.items():
        if key in target and target[key] != value:
            raise AssertionError("transition audit mismatch")
        target[key] = value


def evaluate_levels(
    bank: ModelBank,
    compiler: CompositionalEditCompiler,
    particles: StructureParticleIndex,
    truth_particles: Sequence[int],
    proposer: Any,
    *,
    seed: int,
    source_mode: str | None = None,
    levels: Sequence[int] = (3, 2, 1),
) -> tuple[dict[str, Any], ProposalCache]:
    cache = ProposalCache(proposer, source_mode=source_mode)
    results = {}
    audit: dict[str, dict[str, Any]] = {}
    for level in levels:
        planner = AuditedCompositionalPolicyPlanner(
            bank,
            cache,
            truth_particles,
            compiler=compiler,
            particles=particles,
            seed=seed,
        )
        results[f"d{level}"] = planner.evaluate_policy_level(
            level, execution_budget=EXECUTION_BUDGET
        )
        _merge_audit(audit, planner.transition_audit)
    return {
        "policy_levels": results,
        "cache": {
            "source_mode": cache.source_mode,
            "entries": len(cache.records),
            "hits": cache.hits,
            "misses": cache.misses,
        },
        "transition_audit": {key: audit[key] for key in sorted(audit)},
    }, cache


def comparison(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    difference = left_values - right_values
    left_mean = float(np.mean(left_values))
    right_mean = float(np.mean(right_values))
    return {
        "left_mean": left_mean,
        "right_mean": right_mean,
        "absolute_reduction": left_mean - right_mean,
        "relative_reduction": (
            (left_mean - right_mean) / left_mean if left_mean > 0.0 else 0.0
        ),
        "wins": int(np.sum(difference > PRACTICAL_TIE)),
        "ties": int(np.sum(np.abs(difference) <= PRACTICAL_TIE)),
        "losses": int(np.sum(difference < -PRACTICAL_TIE)),
    }


def apply_gate(slices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = {
        level: [
            value
            for item in slices
            for value in item["primary"]["policy_levels"][level]["truth_losses"]
        ]
        for level in ("d1", "d2", "d3")
    }
    d2_vs_d1 = comparison(aggregate["d1"], aggregate["d2"])
    d3_vs_d2 = comparison(aggregate["d2"], aggregate["d3"])
    d3_vs_d1 = comparison(aggregate["d1"], aggregate["d3"])
    per_difficulty = []
    for item in slices:
        risk = {
            level: float(item["primary"]["policy_levels"][level]["expected_terminal_mse"])
            for level in aggregate
        }
        per_difficulty.append({"difficulty": item["difficulty"], "risk": risk})
    atomic_records = [
        record
        for item in slices
        for record in item["primary"]["transition_audit"].values()
        if record["proposal"]
    ]
    conditions = {
        "finite_normalized": all(
            math.isfinite(value) for values in aggregate.values() for value in values
        ),
        "immutable_replay_exact": all(item["replay_exact"] for item in slices),
        "all_proposals_are_complete_atomic_three_particle_edits": bool(atomic_records)
        and all(
            len(record["proposal"]) == 3
            and len(record["proposal_structures"]) == 1
            and record["edit"] is not None
            for record in atomic_records
        ),
        "no_truth_particle_proposed": all(
            not record["truth_particle_proposed"] for record in atomic_records
        ),
        "d2_mean_reduction_at_least_5pct": d2_vs_d1["relative_reduction"] >= 0.05,
        "d3_mean_reduction_at_least_5pct": d3_vs_d2["relative_reduction"] >= 0.05,
        "d2_paired_wins_exceed_losses": d2_vs_d1["wins"] > d2_vs_d1["losses"],
        "d3_paired_wins_exceed_losses": d3_vs_d2["wins"] > d3_vs_d2["losses"],
        "d3_beats_d1_on_at_least_21_cells": d3_vs_d1["wins"] >= 21,
        "d2_improves_at_least_two_difficulties": sum(
            item["risk"]["d2"] < item["risk"]["d1"] - PRACTICAL_TIE
            for item in per_difficulty
        )
        >= 2,
        "d3_improves_all_difficulties": all(
            item["risk"]["d3"] < item["risk"]["d2"] - PRACTICAL_TIE
            for item in per_difficulty
        ),
        "d1_d2_root_differs": any(
            item["primary"]["policy_levels"]["d1"]["root_action_index"]
            != item["primary"]["policy_levels"]["d2"]["root_action_index"]
            for item in slices
        ),
        "d2_d3_root_differs": any(
            item["primary"]["policy_levels"]["d2"]["root_action_index"]
            != item["primary"]["policy_levels"]["d3"]["root_action_index"]
            for item in slices
        ),
        "d2_risk_above_numerical_zero": float(np.mean(aggregate["d2"])) > 1e-8,
        "dynamic_d3_at_least_20pct_better_than_fixed_d1": float(np.mean(aggregate["d3"]))
        <= 0.8 * float(np.mean([item["fixed"]["expected_terminal_mse"] for item in slices])),
        "dynamic_d3_within_15x_full_support_d1": float(np.mean(aggregate["d3"]))
        <= 1.5 * float(
            np.mean([item["full_support"]["expected_terminal_mse"] for item in slices])
        ),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "d2_vs_d1": d2_vs_d1,
        "d3_vs_d2": d3_vs_d2,
        "d3_vs_d1": d3_vs_d1,
        "per_difficulty": per_difficulty,
    }


def run(source_root: Path, *, implementation_commit: str) -> dict[str, Any]:
    if sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("staged-compound protocol binding mismatch")
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    slices = []
    for difficulty_index, difficulty in enumerate(DIFFICULTIES):
        bank, compiler, particles, truths, response_sha = make_bank(source, difficulty)
        seed = 2026084100 + difficulty_index
        primary, cache = evaluate_levels(
            bank,
            compiler,
            particles,
            truths,
            AtomicStructureOracleProposer(bank, compiler, particles),
            seed=seed,
        )
        replay, _ = evaluate_levels(
            bank,
            compiler,
            particles,
            truths,
            replay_proposer(cache.source_mode, cache.records),
            source_mode=cache.source_mode,
            seed=seed,
        )
        fixed, _ = evaluate_levels(
            bank,
            compiler,
            particles,
            truths,
            FixedProposer(),
            seed=seed,
            levels=(1,),
        )
        full_bank, full_compiler, full_particles, full_truths, full_sha = make_bank(
            source, difficulty, full_support=True
        )
        if full_sha != response_sha or full_truths != truths:
            raise AssertionError("full-support control changed the response bank")
        full, _ = evaluate_levels(
            full_bank,
            full_compiler,
            full_particles,
            full_truths,
            FixedProposer(),
            seed=seed,
            levels=(1,),
        )
        slices.append(
            {
                "difficulty": difficulty,
                "query_seed": QUERY_SEEDS[difficulty],
                "response_sha256": response_sha,
                "num_models": bank.num_models,
                "num_inference_structures": len(particles.inference_particles),
                "num_truths": len(truths),
                "primary": primary,
                "replay_exact": replay == primary,
                "fixed": fixed["policy_levels"]["d1"],
                "full_support": full["policy_levels"]["d1"],
            }
        )
    gate = apply_gate(slices)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gate["passed"] else "failed_closed",
        "implementation_commit": implementation_commit,
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": PROTOCOL_SHA256},
        "source": source_binding,
        "settings": {
            "truth_structures": list(TRUTH_STRUCTURES),
            "difficulties": list(DIFFICULTIES),
            "inference_versions": list(INFERENCE_VERSIONS),
            "truth_version": TRUTH_VERSION,
            "num_queries": NUM_QUERIES,
            "query_seeds": QUERY_SEEDS,
            "execution_budget": EXECUTION_BUDGET,
            "outside_prior": OUTSIDE_PRIOR,
            "practical_tie": PRACTICAL_TIE,
        },
        "slices": slices,
        "gate": gate,
        "model_calls": 0,
        "network_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-commit", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    implementation_commit = require_pushed_commit(args.required_commit)
    result = run(args.source_root, implementation_commit=implementation_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate": result["gate"],
                "output": str(args.output),
                "output_sha256": sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
