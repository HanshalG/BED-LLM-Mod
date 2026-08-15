#!/usr/bin/env python3
"""Run the opened-v4 continuous ChemBench policy-ladder development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.continuous import (
    ContinuousOracleProposer,
    ContinuousParameterBank,
    ScenarioPolicyLadderPlanner,
)
from environments.chembench_mopen.mechanics import ProposalCache
from environments.chembench_mopen.source import build_empirical_parameter_responses
from scripts.chembench_mopen_mechanics import INITIAL_SUPPORT_NAMES, assay_groups
from scripts.chembench_mopen_nonmyopic_opportunity import (
    active_domains,
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-continuous-policy-screen-v1"
DIFFICULTY_QUERY_SEEDS = {
    "easy": 2026081701,
    "medium": 2026081702,
    "hard": 2026081703,
}
INFERENCE_PARTICLE_SEED = 2026081900
PLANNING_PARTICLE_SEED = 2026081901
PLANNER_SEED_BASE = 2026082700


def array_sha256(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array, dtype=np.float64)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_planner(
    source_root: Path,
    difficulty: str,
    *,
    weight_resolution: float,
    predictive_resolution: float,
    risk_resolution: float,
) -> tuple[ScenarioPolicyLadderPlanner, dict[str, Any]]:
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    domains = active_domains(source)
    assays = frozen_assays()
    action_names = tuple(item.name for item in assays)
    action_inputs = np.asarray([item.values for item in assays], dtype=float)
    response_kwargs = {
        "source": source,
        "domains": domains,
        "initial_names": INITIAL_SUPPORT_NAMES,
        "difficulty": difficulty,
        "initial_candidate_versions": ("v0", "v1", "v2"),
        "outside_candidate_versions": ("v0", "v1", "v2", "v3"),
        "truth_version": "v4",
        "query_seed": DIFFICULTY_QUERY_SEEDS[difficulty],
        "assays": assays,
        "num_queries": 1_000,
        "broadened_num_particles": 16,
        "broadened_expansion_factor": 1.5,
    }
    inference = build_empirical_parameter_responses(
        **response_kwargs,
        broadened_seed=INFERENCE_PARTICLE_SEED,
    )
    planning = build_empirical_parameter_responses(
        **response_kwargs,
        broadened_seed=PLANNING_PARTICLE_SEED,
    )
    model_index = {name: index for index, name in enumerate(domains)}
    initial_support = tuple(model_index[name] for name in INITIAL_SUPPORT_NAMES)
    bank = ContinuousParameterBank(
        inference.particle_observation_means,
        inference.particle_target_log_rates,
        model_names=domains,
        action_names=action_names,
        action_groups=assay_groups(action_names),
        action_inputs=action_inputs,
        initial_support=initial_support,
        parameter_kernel_scale=1.0,
        outside_prior=0.35,
        live_cap=12,
        reserve_cap=12,
    )
    difficulty_index = tuple(DIFFICULTY_QUERY_SEEDS).index(difficulty)
    planner = ScenarioPolicyLadderPlanner(
        bank=bank,
        proposal_cache=ProposalCache(ContinuousOracleProposer(bank)),
        speculative_models=inference.truth_indices,
        truth_observation_means=inference.truth_observation_means,
        truth_target_features=inference.truth_target_log_rates,
        speculative_particle_observation_means=planning.particle_observation_means,
        speculative_particle_target_features=planning.particle_target_log_rates,
        truth_quadrature_order=1,
        scenario_counts_by_remaining=(2, 3, 4, 96),
        action_widths_by_remaining=(2, 2, 3, 6),
        policy_improvement_replicates=3,
        minimum_improvement_fraction=0.05,
        use_policy_abstraction=True,
        policy_signature_mode="predictive",
        policy_signature_top_k=3,
        policy_weight_resolution=weight_resolution,
        policy_predictive_resolution=predictive_resolution,
        policy_risk_resolution=risk_resolution,
        policy_table_max_size=10_000,
        freeze_predecessor_on_miss=False,
        policy_cover_paths_by_level=(0, 0, 0),
        policy_nearest_max_distance=0.75,
        seed=PLANNER_SEED_BASE + difficulty_index,
    )
    provenance = {
        "source": source_binding,
        "difficulty": difficulty,
        "truth_version": "v4_opened_development",
        "query_seed": DIFFICULTY_QUERY_SEEDS[difficulty],
        "inference_particle_seed": INFERENCE_PARTICLE_SEED,
        "planning_particle_seed": PLANNING_PARTICLE_SEED,
        "inference_version_plan_sha256": inference.version_plan_sha256,
        "planning_version_plan_sha256": planning.version_plan_sha256,
        "inference_particles_sha256": array_sha256(
            (*inference.particle_observation_means, *inference.particle_target_log_rates)
        ),
        "planning_particles_sha256": array_sha256(
            (*planning.particle_observation_means, *planning.particle_target_log_rates)
        ),
        "truth_sha256": array_sha256(
            (inference.truth_observation_means, inference.truth_target_log_rates)
        ),
        "num_structures": len(domains),
        "num_truths": len(inference.truth_indices),
        "num_actions": len(assays),
        "num_queries": inference.truth_target_log_rates.shape[1],
    }
    return planner, provenance


def run(
    source_root: Path,
    difficulty: str,
    levels: Sequence[int],
    *,
    weight_resolution: float = 0.1,
    predictive_resolution: float = 0.25,
    risk_resolution: float = 0.1,
) -> dict[str, Any]:
    planner, provenance = build_planner(
        source_root,
        difficulty,
        weight_resolution=weight_resolution,
        predictive_resolution=predictive_resolution,
        risk_resolution=risk_resolution,
    )
    level_results = {}
    for level in levels:
        started = time.perf_counter()
        result = planner.evaluate_policy_level(
            level,
            execution_budget=4,
            evaluate_execution_model_risk=False,
            clear_before_truth=True,
        )
        result["elapsed_seconds"] = time.perf_counter() - started
        level_results[f"d{level}"] = result
    comparisons = {}
    for lower, upper in zip(levels, levels[1:]):
        lower_value = float(level_results[f"d{lower}"]["expected_truth_mse"])
        upper_value = float(level_results[f"d{upper}"]["expected_truth_mse"])
        comparisons[f"d{upper}_vs_d{lower}"] = {
            "relative_reduction": (
                (lower_value - upper_value) / lower_value if lower_value > 0 else math.nan
            ),
            "monotonic": upper_value <= lower_value,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "settings": {
            "execution_budget": 4,
            "parameter_particles_per_structure": 16,
            "parameter_kernel_scale": 1.0,
            "scenario_counts_by_remaining": [2, 3, 4, 96],
            "action_widths_by_remaining": [2, 2, 3, 6],
            "policy_improvement_replicates": 3,
            "minimum_improvement_fraction": 0.05,
            "policy_signature_mode": "predictive",
            "policy_weight_resolution": weight_resolution,
            "policy_predictive_resolution": predictive_resolution,
            "policy_risk_resolution": risk_resolution,
            "policy_table_max_size": 10_000,
            "freeze_predecessor_on_miss": False,
            "policy_cover_paths_by_level": [0, 0, 0],
            "policy_nearest_max_distance": 0.75,
            "truth_quadrature_order": 1,
        },
        "levels": level_results,
        "comparisons": comparisons,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def parse_levels(value: str) -> tuple[int, ...]:
    levels = tuple(int(item) for item in value.split(",") if item)
    if levels not in {(1,), (1, 2), (1, 2, 3)}:
        raise argparse.ArgumentTypeError("levels must be a contiguous prefix of 1,2,3")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTY_QUERY_SEEDS), required=True)
    parser.add_argument("--levels", type=parse_levels, default=(1, 2, 3))
    parser.add_argument("--weight-resolution", type=float, default=0.1)
    parser.add_argument("--predictive-resolution", type=float, default=0.25)
    parser.add_argument("--risk-resolution", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        args.source_root,
        args.difficulty,
        args.levels,
        weight_resolution=args.weight_resolution,
        predictive_resolution=args.predictive_resolution,
        risk_resolution=args.risk_resolution,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
