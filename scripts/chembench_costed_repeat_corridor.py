#!/usr/bin/env python3
"""Run the frozen stochastic costed-repeat ChemBench corridor gate."""

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

from environments.chembench_mopen.compositional import (
    AtomicStructureOracleProposer,
    CompositionalEditCompiler,
    StructureParticleIndex,
    replay_proposer,
)
from environments.chembench_mopen.costed import (
    CostedAction,
    CostedCompositionalPolicyPlanner,
    RandomCostedCompositionalPolicyPlanner,
)
from environments.chembench_mopen.mechanics import FixedProposer, ModelBank, ProposalCache
from environments.chembench_mopen.source import _query_assays
from scripts.chembench_factored_mopen_oracle import sha256
from scripts.chembench_mopen_mechanics import INITIAL_SUPPORT_NAMES
from scripts.chembench_mopen_nonmyopic_opportunity import frozen_assays, load_source, verify_source
from scripts.chembench_staged_compound_corridor import (
    DIFFICULTIES,
    INFERENCE_VERSIONS,
    OUTSIDE_PRIOR,
    PRACTICAL_TIE,
    TRUTH_STRUCTURES,
    _evaluate_particle,
    comparison,
    require_pushed_commit,
    source_signatures,
    standard_structure_names,
)


SCHEMA_VERSION = "chembench-costed-repeat-corridor-v1"
PROTOCOL_PATH = Path(
    "results/nonmyopic/CHEMBENCH_COSTED_REPEAT_CORRIDOR_PROTOCOL_20260815.md"
)
PROTOCOL_SHA256 = "6c16c4488f4ef0fc5fdb91304e3d831b1735e5fdbec0858c90e2f3d5a064981a"
CONTROL_PATH = Path(
    "results/nonmyopic/CHEMBENCH_COSTED_REPEAT_CORRIDOR_CONTROL_CLARIFICATION_20260815.md"
)
CONTROL_SHA256 = "ec664cd6da2f7207832d305429bb6ad421ecf853cacb5788af85f62b11a77125"
CRN_PATH = Path(
    "results/nonmyopic/CHEMBENCH_COSTED_REPEAT_CORRIDOR_CRN_EVALUATION_AMENDMENT_20260815.md"
)
CRN_SHA256 = "bc61ce8d9e0886726b11a6f73949c2350ab30f803cf5b06b71f57a5cb70f379a"
TRUTH_VERSION = "v4"
QUERY_SEEDS = {"easy": 2026084201, "medium": 2026084202, "hard": 2026084203}
NUM_QUERIES = 512
WELL_BUDGET = 8
REPEAT_COUNTS = (1, 2, 4)
LOG_NOISE = math.sqrt(math.log1p(0.10**2))
BASE_ASSAY_NAMES = (
    "C_A=0.02",
    "C_A=100",
    "C_I=50,C_A=0.1",
    "C_I=50,C_A=100",
    "C_B=0.01",
    "C_B=100",
    "C_P=20",
    "T=278",
    "pH=4",
    "pH=10",
)
RANDOM_SEED_BASE = 2026084400
RANDOM_REPLICATES = 8
NUM_SCENARIOS = 128
CRN_SEEDS = {"easy": 2026084501, "medium": 2026084502, "hard": 2026084503}


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def costed_likelihoods(
    base_means: np.ndarray,
    *,
    num_inference_particles: int,
) -> tuple[np.ndarray, np.ndarray, tuple[CostedAction, ...]]:
    means = np.asarray(base_means, dtype=float)
    if (
        means.ndim != 2
        or not 0 < num_inference_particles <= len(means)
        or np.any(means < 0)
        or not np.isfinite(means).all()
    ):
        raise ValueError("costed response means are invalid")
    log_means = np.log1p(means)
    thresholds = np.quantile(
        log_means[:num_inference_particles], (1.0 / 3.0, 2.0 / 3.0), axis=0
    ).T
    actions = tuple(
        CostedAction(
            name=f"{base_name}|r={repeats}",
            base_name=base_name,
            base_index=base_index,
            repeats=repeats,
        )
        for base_index, base_name in enumerate(BASE_ASSAY_NAMES)
        for repeats in REPEAT_COUNTS
    )
    likelihoods = np.empty((len(means), len(actions), 3), dtype=float)
    for action_index, action in enumerate(actions):
        low, high = thresholds[action.base_index]
        sigma = LOG_NOISE / math.sqrt(action.repeats)
        for particle in range(len(means)):
            value = float(log_means[particle, action.base_index])
            low_mass = _normal_cdf((float(low) - value) / sigma)
            high_mass = 1.0 - _normal_cdf((float(high) - value) / sigma)
            likelihoods[particle, action_index] = (
                low_mass,
                max(0.0, 1.0 - low_mass - high_mass),
                high_mass,
            )
    likelihoods = np.maximum(likelihoods, 1e-300)
    likelihoods /= likelihoods.sum(axis=2, keepdims=True)
    if not np.allclose(likelihoods.sum(axis=2), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("costed likelihoods do not normalize")
    return likelihoods, thresholds, actions


def build_costed_arrays(
    source: Any, difficulty: str
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[int, ...]],
    tuple[int, ...],
    np.ndarray,
    tuple[CostedAction, ...],
    str,
]:
    structures = standard_structure_names(source)
    assay_by_name = {assay.name: assay for assay in frozen_assays()}
    if any(name not in assay_by_name for name in BASE_ASSAY_NAMES):
        raise ValueError("one or more frozen base assays are absent")
    assays = tuple(assay_by_name[name] for name in BASE_ASSAY_NAMES)
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
    base_means = np.asarray(means, dtype=float)
    targets_array = np.asarray(targets, dtype=float)
    likelihoods, thresholds, actions = costed_likelihoods(
        base_means,
        num_inference_particles=len(structures) * len(INFERENCE_VERSIONS),
    )
    digest = hashlib.sha256()
    for array in (base_means, targets_array, thresholds, likelihoods):
        digest.update(str(array.shape).encode())
        digest.update(np.asarray(array, dtype=np.float64).tobytes(order="C"))
    return (
        likelihoods,
        targets_array,
        tuple(names),
        tuple(particle_structure),
        inference_particles,
        tuple(truth_particles),
        thresholds,
        actions,
        digest.hexdigest(),
    )


def make_costed_bank(
    source: Any, difficulty: str, *, full_support: bool = False
) -> tuple[
    ModelBank,
    CompositionalEditCompiler,
    StructureParticleIndex,
    tuple[int, ...],
    tuple[CostedAction, ...],
    np.ndarray,
    str,
]:
    signatures = source_signatures(source)
    (
        likelihoods,
        targets,
        model_names,
        particle_structure,
        inference_particles,
        truth_particles,
        thresholds,
        actions,
        response_sha,
    ) = build_costed_arrays(source, difficulty)
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
    bank = ModelBank(
        likelihoods,
        targets,
        model_names,
        tuple(action.name for action in actions),
        tuple(action.base_name for action in actions),
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
    return (
        bank,
        CompositionalEditCompiler(signatures),
        particles,
        truth_particles,
        actions,
        thresholds,
        response_sha,
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _audit_summary(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    proposed = [record for record in records.values() if record["proposal"]]
    return {
        "records": len(records),
        "records_sha256": _canonical_hash(records),
        "proposals": len(proposed),
        "all_complete_three_particle_edits": bool(proposed)
        and all(
            len(record["proposal"]) == 3
            and len(record["proposal_structures"]) == 1
            and record["edit"] is not None
            for record in proposed
        ),
        "truth_particles_proposed": sum(
            bool(record["truth_particle_proposed"]) for record in proposed
        ),
    }


def _merge_audit(target: dict[str, dict[str, Any]], source: Mapping[str, dict[str, Any]]) -> None:
    for key, value in source.items():
        if key in target and target[key] != value:
            raise AssertionError("costed transition audit mismatch")
        target[key] = value


def _planner(
    planner_type: type[CostedCompositionalPolicyPlanner],
    bank: ModelBank,
    cache: ProposalCache,
    truths: Sequence[int],
    compiler: CompositionalEditCompiler,
    particles: StructureParticleIndex,
    actions: Sequence[CostedAction],
    *,
    seed: int,
    cost_aware: bool = True,
) -> CostedCompositionalPolicyPlanner:
    return planner_type(
        bank,
        cache,
        truths,
        compiler=compiler,
        particles=particles,
        actions=actions,
        well_budget=WELL_BUDGET,
        cost_aware=cost_aware,
        seed=seed,
    )


def evaluate_dynamic_suite(
    bank: ModelBank,
    compiler: CompositionalEditCompiler,
    particles: StructureParticleIndex,
    truths: Sequence[int],
    actions: Sequence[CostedAction],
    proposer: Any,
    *,
    difficulty_index: int,
    scenario_uniforms: np.ndarray,
    source_mode: str | None = None,
) -> tuple[dict[str, Any], ProposalCache]:
    cache = ProposalCache(proposer, source_mode=source_mode)
    audit: dict[str, dict[str, Any]] = {}
    primary = {}
    seed = 2026084300 + difficulty_index
    for level in (3, 2, 1):
        planner = _planner(
            CostedCompositionalPolicyPlanner,
            bank,
            cache,
            truths,
            compiler,
            particles,
            actions,
            seed=seed,
        )
        primary[f"d{level}"] = planner.evaluate_policy_level(
            level, scenario_uniforms=scenario_uniforms
        )
        _merge_audit(audit, planner.transition_audit)

    cost_blind_planner = _planner(
        CostedCompositionalPolicyPlanner,
        bank,
        cache,
        truths,
        compiler,
        particles,
        actions,
        seed=seed,
        cost_aware=False,
    )
    cost_blind = cost_blind_planner.evaluate_policy_level(
        3, scenario_uniforms=scenario_uniforms
    )
    _merge_audit(audit, cost_blind_planner.transition_audit)

    random_results = []
    for replicate in range(RANDOM_REPLICATES):
        random_seed = RANDOM_SEED_BASE + difficulty_index * RANDOM_REPLICATES + replicate
        random_planner = _planner(
            RandomCostedCompositionalPolicyPlanner,
            bank,
            cache,
            truths,
            compiler,
            particles,
            actions,
            seed=random_seed,
        )
        random_results.append(
            random_planner.evaluate_policy_level(
                1, scenario_uniforms=scenario_uniforms
            )
        )
        _merge_audit(audit, random_planner.transition_audit)
    random_truth_losses = np.mean(
        [item["truth_losses"] for item in random_results], axis=0
    )
    return {
        "primary": primary,
        "cost_blind_d3": cost_blind,
        "random_replicates": random_results,
        "random_mean_truth_losses": [float(value) for value in random_truth_losses],
        "random_mean_terminal_mse": float(np.mean(random_truth_losses)),
        "transition_audit": _audit_summary(audit),
        "cache_records_sha256": _canonical_hash(cache.records),
        "cache_entries": len(cache.records),
    }, cache


def evaluate_fixed(
    bank: ModelBank,
    compiler: CompositionalEditCompiler,
    particles: StructureParticleIndex,
    truths: Sequence[int],
    actions: Sequence[CostedAction],
    *,
    seed: int,
    scenario_uniforms: np.ndarray,
) -> dict[str, Any]:
    planner = _planner(
        CostedCompositionalPolicyPlanner,
        bank,
        ProposalCache(FixedProposer()),
        truths,
        compiler,
        particles,
        actions,
        seed=seed,
    )
    return planner.evaluate_policy_level(1, scenario_uniforms=scenario_uniforms)


def apply_gate(slices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = {
        level: [
            loss
            for item in slices
            for loss in item["suite"]["primary"][level]["truth_losses"]
        ]
        for level in ("d1", "d2", "d3")
    }
    d2_vs_d1 = comparison(aggregate["d1"], aggregate["d2"])
    d3_vs_d2 = comparison(aggregate["d2"], aggregate["d3"])
    d3_vs_d1 = comparison(aggregate["d1"], aggregate["d3"])
    random_losses = [
        loss for item in slices for loss in item["suite"]["random_mean_truth_losses"]
    ]
    cost_blind_losses = [
        loss
        for item in slices
        for loss in item["suite"]["cost_blind_d3"]["truth_losses"]
    ]
    fixed_losses = [
        loss for item in slices for loss in item["fixed"]["truth_losses"]
    ]
    full_losses = [
        loss for item in slices for loss in item["full_support"]["truth_losses"]
    ]
    per_difficulty = []
    for item in slices:
        risk = {
            level: float(item["suite"]["primary"][level]["expected_terminal_mse"])
            for level in aggregate
        }
        per_difficulty.append({"difficulty": item["difficulty"], "risk": risk})
    all_execution_audits = [
        result["execution_audit"]
        for item in slices
        for result in (
            *item["suite"]["primary"].values(),
            item["suite"]["cost_blind_d3"],
            *item["suite"]["random_replicates"],
            item["fixed"],
            item["full_support"],
        )
    ]
    conditions = {
        "finite_normalized": all(
            math.isfinite(value) for values in aggregate.values() for value in values
        ),
        "thresholds_inference_only": all(item["thresholds_inference_only"] for item in slices),
        "all_proposals_complete_atomic_three_particle": all(
            item["suite"]["transition_audit"]["all_complete_three_particle_edits"]
            for item in slices
        ),
        "no_truth_particle_proposed": all(
            item["suite"]["transition_audit"]["truth_particles_proposed"] == 0
            for item in slices
        ),
        "immutable_replay_exact": all(item["replay_exact"] for item in slices),
        "budget_and_base_integrity": all(
            audit["within_budget_no_base_reuse"] for audit in all_execution_audits
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
            item["suite"]["primary"]["d1"]["root_action_index"]
            != item["suite"]["primary"]["d2"]["root_action_index"]
            for item in slices
        ),
        "d2_d3_root_differs": any(
            item["suite"]["primary"]["d2"]["root_action_index"]
            != item["suite"]["primary"]["d3"]["root_action_index"]
            for item in slices
        ),
        "root_repeat_count_changes": any(
            len(
                {
                    item["suite"]["primary"][level]["root_repeat_count"]
                    for level in ("d1", "d2", "d3")
                }
            )
            > 1
            for item in slices
        ),
        "d2_risk_above_numerical_zero": float(np.mean(aggregate["d2"])) > 1e-8,
        "dynamic_d3_at_least_20pct_better_than_fixed": float(np.mean(aggregate["d3"]))
        <= 0.8 * float(np.mean(fixed_losses)),
        "dynamic_d3_within_15x_full_support": float(np.mean(aggregate["d3"]))
        <= 1.5 * float(np.mean(full_losses)),
        "dynamic_d3_at_least_3pct_better_than_cost_blind": float(np.mean(aggregate["d3"]))
        <= 0.97 * float(np.mean(cost_blind_losses)),
        "random_worse_than_dynamic_d3": float(np.mean(random_losses))
        > float(np.mean(aggregate["d3"])),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "d2_vs_d1": d2_vs_d1,
        "d3_vs_d2": d3_vs_d2,
        "d3_vs_d1": d3_vs_d1,
        "cost_blind_vs_dynamic_d3": comparison(cost_blind_losses, aggregate["d3"]),
        "random_vs_dynamic_d3": comparison(random_losses, aggregate["d3"]),
        "per_difficulty": per_difficulty,
    }


def run(source_root: Path, *, implementation_commit: str) -> dict[str, Any]:
    if (
        sha256(PROTOCOL_PATH) != PROTOCOL_SHA256
        or sha256(CONTROL_PATH) != CONTROL_SHA256
        or sha256(CRN_PATH) != CRN_SHA256
    ):
        raise RuntimeError("costed-repeat protocol binding mismatch")
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    slices = []
    for difficulty_index, difficulty in enumerate(DIFFICULTIES):
        (
            bank,
            compiler,
            particles,
            truths,
            actions,
            thresholds,
            response_sha,
        ) = make_costed_bank(source, difficulty)
        scenario_uniforms = np.random.default_rng(CRN_SEEDS[difficulty]).random(
            (len(truths), NUM_SCENARIOS, WELL_BUDGET)
        )
        scenario_sha = hashlib.sha256(
            np.asarray(scenario_uniforms, dtype=np.float64).tobytes(order="C")
        ).hexdigest()
        suite, cache = evaluate_dynamic_suite(
            bank,
            compiler,
            particles,
            truths,
            actions,
            AtomicStructureOracleProposer(bank, compiler, particles),
            difficulty_index=difficulty_index,
            scenario_uniforms=scenario_uniforms,
        )
        replay, _ = evaluate_dynamic_suite(
            bank,
            compiler,
            particles,
            truths,
            actions,
            replay_proposer(cache.source_mode, cache.records),
            source_mode=cache.source_mode,
            difficulty_index=difficulty_index,
            scenario_uniforms=scenario_uniforms,
        )
        seed = 2026084300 + difficulty_index
        fixed = evaluate_fixed(
            bank,
            compiler,
            particles,
            truths,
            actions,
            seed=seed,
            scenario_uniforms=scenario_uniforms,
        )
        full_bank, full_compiler, full_particles, full_truths, full_actions, full_thresholds, full_sha = (
            make_costed_bank(source, difficulty, full_support=True)
        )
        if (
            full_sha != response_sha
            or full_truths != truths
            or full_actions != actions
            or not np.array_equal(full_thresholds, thresholds)
        ):
            raise AssertionError("full-support control changed costed responses")
        full = evaluate_fixed(
            full_bank,
            full_compiler,
            full_particles,
            full_truths,
            full_actions,
            seed=seed,
            scenario_uniforms=scenario_uniforms,
        )
        slices.append(
            {
                "difficulty": difficulty,
                "query_seed": QUERY_SEEDS[difficulty],
                "crn_seed": CRN_SEEDS[difficulty],
                "scenario_uniforms_sha256": scenario_sha,
                "response_sha256": response_sha,
                "thresholds_sha256": hashlib.sha256(
                    np.asarray(thresholds, dtype=np.float64).tobytes(order="C")
                ).hexdigest(),
                "thresholds_inference_only": True,
                "num_models": bank.num_models,
                "num_truths": len(truths),
                "suite": suite,
                "replay_exact": replay == suite,
                "fixed": fixed,
                "full_support": full,
            }
        )
    gate = apply_gate(slices)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gate["passed"] else "failed_closed",
        "implementation_commit": implementation_commit,
        "protocols": {
            str(PROTOCOL_PATH): {"sha256": PROTOCOL_SHA256},
            str(CONTROL_PATH): {"sha256": CONTROL_SHA256},
            str(CRN_PATH): {"sha256": CRN_SHA256},
        },
        "source": source_binding,
        "settings": {
            "truth_structures": list(TRUTH_STRUCTURES),
            "difficulties": list(DIFFICULTIES),
            "inference_versions": list(INFERENCE_VERSIONS),
            "truth_version": TRUTH_VERSION,
            "query_seeds": QUERY_SEEDS,
            "num_queries": NUM_QUERIES,
            "log_noise": LOG_NOISE,
            "repeat_counts": list(REPEAT_COUNTS),
            "base_assays": list(BASE_ASSAY_NAMES),
            "well_budget": WELL_BUDGET,
            "random_seed_base": RANDOM_SEED_BASE,
            "random_replicates": RANDOM_REPLICATES,
            "num_scenarios": NUM_SCENARIOS,
            "crn_seeds": CRN_SEEDS,
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
    commit = require_pushed_commit(args.required_commit)
    result = run(args.source_root, implementation_commit=commit)
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
