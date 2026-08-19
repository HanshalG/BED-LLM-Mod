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
)
from environments.chembench_mopen.costed import (
    CostedAction,
    CostedCompositionalPolicyPlanner,
    RandomCostedCompositionalPolicyPlanner,
    TranscriptReplayCostedPolicyPlanner,
)
from environments.chembench_mopen.mechanics import (
    FixedProposer,
    ModelBank,
    ProposalCache,
    proposal_key,
)
from environments.chembench_mopen.source import _query_assays
from scripts.chembench_factored_mopen_oracle import sha256
from scripts.chembench_costed_repeat_disk_runtime import (
    DiskRuntimeStores,
    canonical_mapping_items,
)
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
SCIENTIFIC_IMPLEMENTATION_COMMIT = "d9559a2f03d415966200d9f74c3bd84bbe12f021"
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
EFFICIENCY_PATH = Path(
    "results/nonmyopic/"
    "CHEMBENCH_COSTED_REPEAT_CORRIDOR_EVALUATOR_EFFICIENCY_AMENDMENT_20260816.md"
)
EFFICIENCY_SHA256 = "102ce68e4ca4316d2650bc8096356baf3defc6580735f5d24cb165c32ccec783"
SHARDING_PATH = Path(
    "results/nonmyopic/"
    "CHEMBENCH_COSTED_REPEAT_CORRIDOR_DIFFICULTY_SHARDING_CLARIFICATION_20260816.md"
)
SHARDING_SHA256 = "2b87d325e3265ba485046469da50b5e2de432904c43cb69bbeb8b94c018a4454"
DISK_RUNTIME_PATH = Path(
    "results/nonmyopic/"
    "CHEMBENCH_COSTED_REPEAT_CORRIDOR_DISK_AUDIT_RUNTIME_AMENDMENT_20260818.md"
)
DISK_RUNTIME_SHA256 = "a830ce66cd9ffd7824b2f14523d4d2a0789055189a6af9a8c7f2de5613e60d6d"
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
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"))
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode())
    return digest.hexdigest()


def _sorted_mapping_hash(records: Mapping[str, Any]) -> str:
    """Hash a large sorted mapping without copying or materializing it as JSON."""
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"))
    digest.update(b"{")
    for index, (key, canonical_value) in enumerate(canonical_mapping_items(records)):
        if index:
            digest.update(b",")
        digest.update(encoder.encode(key).encode())
        digest.update(b":")
        digest.update(canonical_value)
    digest.update(b"}")
    return digest.hexdigest()


def _proposal_records_hash(records: Mapping[str, Sequence[int]]) -> str:
    return _sorted_mapping_hash(records)


def _audit_summary(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counters = getattr(records, "audit_counters", None)
    if counters is not None:
        return {
            **counters(),
            "records_sha256": _sorted_mapping_hash(records),
        }
    proposal_count = 0
    all_complete = True
    truth_particles_proposed = 0
    for record in records.values():
        if not record["proposal"]:
            continue
        proposal_count += 1
        all_complete = all_complete and (
            len(record["proposal"]) == 3
            and len(record["proposal_structures"]) == 1
            and record["edit"] is not None
        )
        truth_particles_proposed += bool(record["truth_particle_proposed"])
    return {
        "records": len(records),
        "records_sha256": _sorted_mapping_hash(records),
        "proposals": proposal_count,
        "all_complete_three_particle_edits": proposal_count > 0 and all_complete,
        "truth_particles_proposed": truth_particles_proposed,
    }


def _combined_audit(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proposals = sum(int(item["proposals"]) for item in summaries)
    return {
        "components": len(summaries),
        "component_sha256": [str(item["records_sha256"]) for item in summaries],
        "records": sum(int(item["records"]) for item in summaries),
        "proposals": proposals,
        "all_complete_three_particle_edits": proposals > 0
        and all(
            int(item["proposals"]) == 0
            or bool(item["all_complete_three_particle_edits"])
            for item in summaries
        ),
        "truth_particles_proposed": sum(
            int(item["truth_particles_proposed"]) for item in summaries
        ),
    }


class _ImmutableRecordProposer:
    mode = "immutable-reference"

    def __init__(
        self, source_mode: str, records: Mapping[str, tuple[int, ...]]
    ) -> None:
        self.source_mode = source_mode
        self.records = records

    def get_by_key(self, key: str) -> tuple[int, ...]:
        if key not in self.records:
            raise KeyError(f"proposal key is absent from immutable records: {key}")
        return tuple(self.records[key])

    def propose(self, state: Any, action: int, outcome: int, seed: int) -> tuple[int, ...]:
        key = proposal_key(self.source_mode, state, action, outcome, seed)
        return self.get_by_key(key)


def _proposal_record_replay(cache: ProposalCache) -> dict[str, Any]:
    records = cache.frozen_records
    proposer = _ImmutableRecordProposer(cache.source_mode, records)
    validator = getattr(records, "validate_canonical_values", None)
    if validator is not None:
        exact = bool(validator())
    else:
        exact = True
        for key in sorted(records):
            if proposer.get_by_key(key) != tuple(records[key]):
                exact = False
                break
    return {
        "exact": exact,
        "records": len(records),
        "records_sha256": _proposal_records_hash(records),
    }


def _policy_records_hash(
    records: Mapping[tuple[Any, tuple[int, ...], int, int], int]
) -> str:
    row_hashes = []
    for (state, available, remaining, level), action in records.items():
        row_hashes.append(
            _canonical_hash(
                {
                    "state": state.inference.public_key(),
                    "particle_weight": list(state.particle_weight),
                    "available": list(available),
                    "remaining_wells": int(remaining),
                    "level": int(level),
                    "action": int(action),
                }
            )
        )
    return _canonical_hash(sorted(row_hashes))


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
    well_budget: int = WELL_BUDGET,
    policy_records: Mapping[tuple[Any, tuple[int, ...], int, int], int] | None = None,
    transition_audit: Any | None = None,
) -> CostedCompositionalPolicyPlanner:
    extra: dict[str, Any] = {}
    if policy_records is not None:
        extra["policy_records"] = policy_records
    planner = planner_type(
        bank,
        cache,
        truths,
        compiler=compiler,
        particles=particles,
        actions=actions,
        well_budget=well_budget,
        cost_aware=cost_aware,
        seed=seed,
        **extra,
    )
    if transition_audit is not None:
        planner.transition_audit = transition_audit
    return planner


def _replay_execution(
    source_planner: CostedCompositionalPolicyPlanner,
    source_cache: ProposalCache,
    expected_result: Mapping[str, Any],
    bank: ModelBank,
    truths: Sequence[int],
    compiler: CompositionalEditCompiler,
    particles: StructureParticleIndex,
    actions: Sequence[CostedAction],
    *,
    level: int,
    seed: int,
    cost_aware: bool,
    scenario_uniforms: np.ndarray,
) -> dict[str, Any]:
    if source_planner.last_scenario_losses is None:
        raise AssertionError("source policy did not retain scenario losses")
    policy_records = source_planner.last_execution_policy_records
    immutable = _ImmutableRecordProposer(
        source_cache.source_mode, source_cache.frozen_records
    )
    replay_cache = ProposalCache(immutable, source_mode=source_cache.source_mode)
    replay_planner = _planner(
        TranscriptReplayCostedPolicyPlanner,
        bank,
        replay_cache,
        truths,
        compiler,
        particles,
        actions,
        seed=seed,
        cost_aware=cost_aware,
        well_budget=source_planner.well_budget,
        policy_records=policy_records,
    )
    replay_losses, replay_execution_audit = replay_planner.simulate_truth_losses(
        level, scenario_uniforms
    )
    expected_losses = source_planner.last_scenario_losses
    expected_loss_sha = hashlib.sha256(
        np.asarray(expected_losses, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    replay_loss_sha = hashlib.sha256(
        np.asarray(replay_losses, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    policy_exact = replay_planner.last_execution_policy_records == policy_records
    losses_exact = np.array_equal(replay_losses, expected_losses)
    execution_exact = replay_execution_audit == expected_result["execution_audit"]
    unused = replay_planner.unused_policy_record_count
    return {
        "exact": policy_exact and losses_exact and execution_exact and unused == 0,
        "policy_records": len(policy_records),
        "policy_records_sha256": _policy_records_hash(policy_records),
        "unused_policy_records": unused,
        "scenario_losses_exact": losses_exact,
        "scenario_losses_sha256": replay_loss_sha,
        "expected_scenario_losses_sha256": expected_loss_sha,
        "execution_audit_exact": execution_exact,
        "transition_audit": _audit_summary(replay_planner.transition_audit),
        "proposal_cache_entries": len(replay_cache.frozen_records),
        "proposal_cache_sha256": _proposal_records_hash(replay_cache.frozen_records),
    }


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
    runtime: DiskRuntimeStores | None = None,
) -> tuple[dict[str, Any], Any]:
    cache = (
        ProposalCache(proposer, source_mode=source_mode)
        if runtime is None
        else runtime.proposal_cache(proposer, source_mode=source_mode)
    )
    audit_summaries: list[dict[str, Any]] = []
    primary = {}
    seed = 2026084300 + difficulty_index
    primary_planner = _planner(
        CostedCompositionalPolicyPlanner,
        bank,
        cache,
        truths,
        compiler,
        particles,
        actions,
        seed=seed,
        transition_audit=(
            None if runtime is None else runtime.transition_audit("primary")
        ),
    )
    for level in (3, 2, 1):
        result = primary_planner.evaluate_policy_level(
            level, scenario_uniforms=scenario_uniforms
        )
        result["replay_audit"] = _replay_execution(
            primary_planner,
            cache,
            result,
            bank,
            truths,
            compiler,
            particles,
            actions,
            level=level,
            seed=seed,
            cost_aware=True,
            scenario_uniforms=scenario_uniforms,
        )
        primary[f"d{level}"] = result
        print(
            f"difficulty_index={difficulty_index} primary=d{level} complete",
            file=sys.stderr,
            flush=True,
        )
    audit_summaries.append(_audit_summary(primary_planner.transition_audit))
    if runtime is not None:
        runtime.close_store(primary_planner.transition_audit)
    del primary_planner

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
        transition_audit=(
            None if runtime is None else runtime.transition_audit("cost-blind")
        ),
    )
    cost_blind = cost_blind_planner.evaluate_policy_level(
        3, scenario_uniforms=scenario_uniforms
    )
    cost_blind["replay_audit"] = _replay_execution(
        cost_blind_planner,
        cache,
        cost_blind,
        bank,
        truths,
        compiler,
        particles,
        actions,
        level=3,
        seed=seed,
        cost_aware=False,
        scenario_uniforms=scenario_uniforms,
    )
    audit_summaries.append(_audit_summary(cost_blind_planner.transition_audit))
    if runtime is not None:
        runtime.close_store(cost_blind_planner.transition_audit)
    print(
        f"difficulty_index={difficulty_index} cost_blind=d3 complete",
        file=sys.stderr,
        flush=True,
    )
    del cost_blind_planner

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
        random_result = random_planner.evaluate_policy_level(
            1, scenario_uniforms=scenario_uniforms
        )
        random_result["replay_audit"] = _replay_execution(
            random_planner,
            cache,
            random_result,
            bank,
            truths,
            compiler,
            particles,
            actions,
            level=1,
            seed=random_seed,
            cost_aware=True,
            scenario_uniforms=scenario_uniforms,
        )
        random_results.append(random_result)
        audit_summaries.append(_audit_summary(random_planner.transition_audit))
        del random_planner
        print(
            f"difficulty_index={difficulty_index} random={replicate + 1}/{RANDOM_REPLICATES} complete",
            file=sys.stderr,
            flush=True,
        )
    random_truth_losses = np.mean(
        [item["truth_losses"] for item in random_results], axis=0
    )
    proposal_replay = _proposal_record_replay(cache)
    replay_components = [
        result["replay_audit"]
        for result in (*primary.values(), cost_blind, *random_results)
    ]
    return {
        "primary": primary,
        "cost_blind_d3": cost_blind,
        "random_replicates": random_results,
        "random_mean_truth_losses": [float(value) for value in random_truth_losses],
        "random_mean_terminal_mse": float(np.mean(random_truth_losses)),
        "transition_audit": _combined_audit(audit_summaries),
        "replay_audit": {
            "exact": proposal_replay["exact"]
            and all(item["exact"] for item in replay_components),
            "proposal_records": proposal_replay,
            "execution_components": len(replay_components),
        },
        "cache_records_sha256": proposal_replay["records_sha256"],
        "cache_entries": proposal_replay["records"],
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
    cache = ProposalCache(FixedProposer())
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
    result = planner.evaluate_policy_level(1, scenario_uniforms=scenario_uniforms)
    result["replay_audit"] = _replay_execution(
        planner,
        cache,
        result,
        bank,
        truths,
        compiler,
        particles,
        actions,
        level=1,
        seed=seed,
        cost_aware=True,
        scenario_uniforms=scenario_uniforms,
    )
    result["transition_audit"] = _audit_summary(planner.transition_audit)
    result["proposal_record_replay"] = _proposal_record_replay(cache)
    result["replay_exact"] = (
        result["replay_audit"]["exact"]
        and result["proposal_record_replay"]["exact"]
    )
    return result


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


def _protocol_bindings() -> dict[str, dict[str, str]]:
    return {
        str(PROTOCOL_PATH): {"sha256": PROTOCOL_SHA256},
        str(CONTROL_PATH): {"sha256": CONTROL_SHA256},
        str(CRN_PATH): {"sha256": CRN_SHA256},
        str(EFFICIENCY_PATH): {"sha256": EFFICIENCY_SHA256},
        str(SHARDING_PATH): {"sha256": SHARDING_SHA256},
    }


def _verify_protocol_bindings(*, include_disk_runtime: bool = False) -> None:
    for path, binding in _protocol_bindings().items():
        if sha256(Path(path)) != binding["sha256"]:
            raise RuntimeError(f"costed-repeat protocol binding mismatch: {path}")
    if include_disk_runtime and sha256(DISK_RUNTIME_PATH) != DISK_RUNTIME_SHA256:
        raise RuntimeError(
            f"costed-repeat disk runtime binding mismatch: {DISK_RUNTIME_PATH}"
        )


def _frozen_settings() -> dict[str, Any]:
    return {
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
    }


def evaluate_difficulty(
    source: Any,
    difficulty_index: int,
    *,
    runtime: DiskRuntimeStores | None = None,
) -> dict[str, Any]:
    if not 0 <= difficulty_index < len(DIFFICULTIES):
        raise ValueError("difficulty index is outside the frozen set")
    difficulty = DIFFICULTIES[difficulty_index]
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
        runtime=runtime,
    )
    close_cache = getattr(cache, "close", None)
    if close_cache is not None:
        close_cache()
    del cache
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
    (
        full_bank,
        full_compiler,
        full_particles,
        full_truths,
        full_actions,
        full_thresholds,
        full_sha,
    ) = make_costed_bank(source, difficulty, full_support=True)
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
    return {
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
        "replay_exact": suite["replay_audit"]["exact"]
        and fixed["replay_exact"]
        and full["replay_exact"],
        "fixed": fixed,
        "full_support": full,
    }


def _binding(
    source_binding: Mapping[str, Any], implementation_commit: str
) -> dict[str, Any]:
    settings = _frozen_settings()
    return {
        "implementation_commit": implementation_commit,
        "protocols": _protocol_bindings(),
        "source": dict(source_binding),
        "source_sha256": _canonical_hash(source_binding),
        "settings": settings,
        "settings_sha256": _canonical_hash(settings),
    }


def _final_result(
    binding: Mapping[str, Any],
    slices: Sequence[Mapping[str, Any]],
    *,
    shard_bindings: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    gate = apply_gate(slices)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gate["passed"] else "failed_closed",
        **dict(binding),
        "shards": list(shard_bindings),
        "slices": list(slices),
        "gate": gate,
        "model_calls": 0,
        "network_calls": 0,
        "cost_usd": 0.0,
    }


def run(source_root: Path, *, implementation_commit: str) -> dict[str, Any]:
    _verify_protocol_bindings()
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    slices = [evaluate_difficulty(source, index) for index in range(len(DIFFICULTIES))]
    return _final_result(_binding(source_binding, implementation_commit), slices)


def _runtime_equivalence_binding(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    required_true = (
        "canonical_hash_exact",
        "duplicate_detection_exact",
        "policy_results_exact",
        "proposal_replay_exact",
        "crn_trajectories_exact",
        "bounded_memory_passed",
    )
    expected_files = {
        "environments/chembench_mopen/compositional.py",
        "scripts/chembench_costed_repeat_corridor.py",
        "scripts/chembench_costed_repeat_disk_runtime.py",
        "scripts/chembench_costed_repeat_disk_runtime_verify.py",
        "tests/test_chembench_costed_repeat_corridor.py",
    }
    if (
        payload.get("schema_version") != "chembench-costed-repeat-disk-runtime-equivalence-v1"
        or payload.get("status") != "passed"
        or payload.get("scientific_implementation_commit")
        != SCIENTIFIC_IMPLEMENTATION_COMMIT
        or payload.get("runtime_mode") != DiskRuntimeStores.mode
        or payload.get("protocol_sha256") != DISK_RUNTIME_SHA256
        or not all(payload.get("conditions", {}).get(key) is True for key in required_true)
        or set(payload.get("file_sha256", {})) != expected_files
    ):
        raise ValueError("disk runtime equivalence manifest failed validation")
    for file_name, expected_sha in payload["file_sha256"].items():
        if sha256(Path(file_name)) != expected_sha:
            raise ValueError(f"disk runtime equivalence file mismatch: {file_name}")
    return {
        "file": str(path),
        "sha256": sha256(path),
        "conditions": {key: True for key in required_true},
    }


def _disk_runtime_binding(
    *, runtime_implementation_commit: str, equivalence_path: Path
) -> dict[str, Any]:
    return {
        "runtime_mode": DiskRuntimeStores.mode,
        "runtime_implementation_commit": runtime_implementation_commit,
        "amendment": {
            "file": str(DISK_RUNTIME_PATH),
            "sha256": DISK_RUNTIME_SHA256,
        },
        "equivalence": _runtime_equivalence_binding(equivalence_path),
    }


def run_shard(
    source_root: Path,
    *,
    implementation_commit: str,
    scientific_implementation_commit: str,
    difficulty: str,
    runtime: DiskRuntimeStores | None = None,
    runtime_equivalence_path: Path | None = None,
) -> dict[str, Any]:
    disk_runtime = runtime is not None
    _verify_protocol_bindings(include_disk_runtime=disk_runtime)
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty is outside the frozen set")
    if disk_runtime:
        if scientific_implementation_commit != SCIENTIFIC_IMPLEMENTATION_COMMIT:
            raise ValueError("disk runtime scientific commit differs from frozen commit")
        if difficulty not in {"medium", "hard"}:
            raise ValueError("disk runtime is authorized only for an incomplete shard")
        if runtime_equivalence_path is None:
            raise ValueError("disk runtime requires an equivalence manifest")
        runtime_binding = _disk_runtime_binding(
            runtime_implementation_commit=implementation_commit,
            equivalence_path=runtime_equivalence_path,
        )
    elif scientific_implementation_commit != implementation_commit:
        raise ValueError("legacy runtime requires identical scientific and runtime commits")
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    difficulty_index = DIFFICULTIES.index(difficulty)
    try:
        result = {
            "schema_version": (
                f"{SCHEMA_VERSION}-difficulty-shard-v2"
                if disk_runtime
                else f"{SCHEMA_VERSION}-difficulty-shard-v1"
            ),
            "status": "slice_complete",
            "difficulty_index": difficulty_index,
            "difficulty": difficulty,
            "binding": _binding(source_binding, scientific_implementation_commit),
            "slice": evaluate_difficulty(
                source, difficulty_index, runtime=runtime
            ),
            "model_calls": 0,
            "network_calls": 0,
            "cost_usd": 0.0,
        }
        if disk_runtime:
            result["runtime_binding"] = runtime_binding
        return result
    finally:
        if runtime is not None:
            runtime.close()


def _validated_shard_slice(
    payload: Mapping[str, Any],
    *,
    expected_binding: Mapping[str, Any],
    expected_index: int,
    difficulty: str,
) -> Mapping[str, Any]:
    if payload.get("schema_version") != f"{SCHEMA_VERSION}-difficulty-shard-v1":
        raise ValueError("invalid difficulty shard schema")
    if (
        payload.get("status") != "slice_complete"
        or payload.get("difficulty_index") != expected_index
        or payload.get("difficulty") != difficulty
        or payload.get("binding") != expected_binding
        or payload.get("model_calls") != 0
        or payload.get("network_calls") != 0
        or payload.get("cost_usd") != 0.0
        or payload.get("slice", {}).get("difficulty") != difficulty
    ):
        raise ValueError("difficulty shard failed binding validation")
    return payload["slice"]


def _validated_disk_runtime_shard_slice(
    payload: Mapping[str, Any],
    *,
    expected_binding: Mapping[str, Any],
    expected_index: int,
    difficulty: str,
    runtime_implementation_commit: str,
) -> Mapping[str, Any]:
    if payload.get("schema_version") != f"{SCHEMA_VERSION}-difficulty-shard-v2":
        raise ValueError("invalid disk runtime difficulty shard schema")
    equivalence_file = (
        payload.get("runtime_binding", {}).get("equivalence", {}).get("file")
    )
    if not isinstance(equivalence_file, str) or not equivalence_file:
        raise ValueError("disk runtime shard lacks equivalence provenance")
    expected_runtime = _disk_runtime_binding(
        runtime_implementation_commit=runtime_implementation_commit,
        equivalence_path=Path(equivalence_file),
    )
    if payload.get("runtime_binding") != expected_runtime:
        raise ValueError("difficulty shard failed disk runtime validation")
    legacy_view = dict(payload)
    legacy_view["schema_version"] = f"{SCHEMA_VERSION}-difficulty-shard-v1"
    legacy_view.pop("runtime_binding", None)
    return _validated_shard_slice(
        legacy_view,
        expected_binding=expected_binding,
        expected_index=expected_index,
        difficulty=difficulty,
    )


def assemble_shards(
    source_root: Path,
    *,
    implementation_commit: str,
    scientific_implementation_commit: str,
    shard_paths: Sequence[Path],
    allow_disk_runtime: bool = False,
) -> dict[str, Any]:
    _verify_protocol_bindings(include_disk_runtime=allow_disk_runtime)
    if len(shard_paths) != len(DIFFICULTIES):
        raise ValueError("assembler requires exactly three difficulty shards")
    if allow_disk_runtime:
        if scientific_implementation_commit != SCIENTIFIC_IMPLEMENTATION_COMMIT:
            raise ValueError("mixed-runtime assembler scientific commit mismatch")
    elif scientific_implementation_commit != implementation_commit:
        raise ValueError("legacy assembler requires identical commits")
    source_binding = verify_source(source_root)
    expected_binding = _binding(source_binding, scientific_implementation_commit)
    slices = []
    shard_bindings = []
    for expected_index, (difficulty, path) in enumerate(zip(DIFFICULTIES, shard_paths)):
        payload = json.loads(path.read_text())
        if allow_disk_runtime and difficulty == "hard":
            slices.append(
                _validated_disk_runtime_shard_slice(
                    payload,
                    expected_binding=expected_binding,
                    expected_index=expected_index,
                    difficulty=difficulty,
                    runtime_implementation_commit=implementation_commit,
                )
            )
        else:
            slices.append(
                _validated_shard_slice(
                    payload,
                    expected_binding=expected_binding,
                    expected_index=expected_index,
                    difficulty=difficulty,
                )
            )
        shard_bindings.append(
            {"difficulty": difficulty, "sha256": sha256(path), "file": path.name}
        )
    final_binding = dict(expected_binding)
    if allow_disk_runtime:
        final_binding["runtime_provenance"] = {
            "assembler_implementation_commit": implementation_commit,
            "disk_runtime_amendment_sha256": DISK_RUNTIME_SHA256,
            "hard_runtime_binding": json.loads(shard_paths[2].read_text())["runtime_binding"],
        }
    return _final_result(final_binding, slices, shard_bindings=shard_bindings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--scientific-commit")
    parser.add_argument(
        "--runtime-mode", choices=("in_memory", DiskRuntimeStores.mode), default="in_memory"
    )
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--runtime-equivalence", type=Path)
    parser.add_argument("--allow-disk-runtime-shards", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--difficulty", choices=DIFFICULTIES)
    mode.add_argument("--assemble-shards", nargs=3, type=Path, metavar=("EASY", "MEDIUM", "HARD"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    commit = require_pushed_commit(args.required_commit)
    scientific_commit = args.scientific_commit or commit
    runtime = None
    if args.runtime_mode == DiskRuntimeStores.mode:
        if args.difficulty is None or args.runtime_dir is None:
            raise ValueError("disk runtime requires one difficulty and --runtime-dir")
        runtime = DiskRuntimeStores(args.runtime_dir)
    elif args.runtime_dir is not None or args.runtime_equivalence is not None:
        raise ValueError("runtime paths require the disk runtime mode")
    if args.allow_disk_runtime_shards and args.assemble_shards is None:
        raise ValueError("disk runtime shard allowance is assembler-only")
    if args.difficulty is not None:
        result = run_shard(
            args.source_root,
            implementation_commit=commit,
            scientific_implementation_commit=scientific_commit,
            difficulty=args.difficulty,
            runtime=runtime,
            runtime_equivalence_path=args.runtime_equivalence,
        )
    elif args.assemble_shards is not None:
        result = assemble_shards(
            args.source_root,
            implementation_commit=commit,
            scientific_implementation_commit=scientific_commit,
            shard_paths=args.assemble_shards,
            allow_disk_runtime=args.allow_disk_runtime_shards,
        )
    else:
        result = run(args.source_root, implementation_commit=commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        "status": result["status"],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    if "gate" in result:
        summary["gate"] = result["gate"]
    else:
        summary["difficulty"] = result["difficulty"]
    print(
        json.dumps(summary, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
