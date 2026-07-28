#!/usr/bin/env python3
"""Replay a robust clipped generated-support blend on open endpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_discoverphysics_dark_matter_structured_replication import (
    stratified_bootstrap_interval,
    weighted_mean,
)
from scripts.discoverphysics_dark_matter_balanced_modular_replication import (
    hidden_map_family as balanced_hidden_map_family,
    modular_component_posteriors,
)
from scripts.discoverphysics_dark_matter_fixed_initial_branch_replication import (
    BOOTSTRAP_SEED as FIXED_BRANCH_BOOTSTRAP_SEED,
    MAP_SEEDS as FIXED_BRANCH_MAP_SEEDS,
    NOISE_SEED as FIXED_BRANCH_NOISE_SEED,
)
from scripts.discoverphysics_dark_matter_grounded_policy import (
    LOOKAHEAD_ROOT_ID,
    OBSERVATION_NOISE_STD,
    compile_support,
    load_executor_class,
    root_by_id,
    simulate_maps,
)
from scripts.discoverphysics_dark_matter_retained_support_confirmation import (
    CONFIRMATION_BOOTSTRAP_SEED,
    CONFIRMATION_CONTINUATION_SAMPLES,
    CONFIRMATION_NOISE_SEED,
    CONFIRMATION_ROOT_SAMPLES,
    confirmation_hidden_map_family,
)
from scripts.discoverphysics_dark_matter_structured_replication import (
    REFRESH_COMPONENT_MASS,
    REPLICATION_CONTINUATION_SAMPLES,
    REPLICATION_NOISE_SEED,
    REPLICATION_ROOT_SAMPLES,
    compile_refresh_models,
    replication_hidden_map_family,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-uncertainty-clipped-blend-1"
EPSILON = 1e-15
MIN_MEAN_REDUCTION = 0.005
MIN_FULL_TREE_REDUCTION = 0.01


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    source_dir: Path
    result_path: Path
    hidden_family: Callable[
        [], tuple[np.ndarray, list[str], np.ndarray]
    ]
    noise_seed: int
    bootstrap_seed: int
    root_samples: int
    continuation_samples: int
    independently_generated_full_tree: bool


def uncertainty_clip_factors(
    *,
    initial_variance: np.ndarray,
    displacement_mse: np.ndarray,
) -> np.ndarray:
    ratio = initial_variance / np.maximum(displacement_mse, EPSILON)
    return np.minimum(1.0, np.sqrt(np.maximum(ratio, 0.0)))


def uncertainty_clipped_predictions(
    *,
    initial_prediction: np.ndarray,
    refresh_prediction: np.ndarray,
    initial_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = refresh_prediction - initial_prediction
    displacement_mse = np.mean(displacement**2, axis=1)
    factors = uncertainty_clip_factors(
        initial_variance=initial_variance,
        displacement_mse=displacement_mse,
    )
    prediction = initial_prediction + (
        REFRESH_COMPONENT_MASS
        * factors[:, None]
        * displacement
    )
    return prediction, factors


def fixed_branch_hidden_family(
) -> tuple[np.ndarray, list[str], np.ndarray]:
    return balanced_hidden_map_family(map_seeds=FIXED_BRANCH_MAP_SEEDS)


def default_endpoint_specs(root: Path) -> tuple[EndpointSpec, ...]:
    original_source = (
        root
        / "results/nonmyopic/discoverphysics_dark_matter_grounded_policy"
        / "discoverphysics-dark-matter-grounded-policy-20260727T020000Z"
    )
    structured_source = (
        root
        / "results/nonmyopic/discoverphysics_dark_matter_structured_replication_v3"
        / "discoverphysics-dark-matter-structured-replication-v3-20260728T063000Z"
    )
    fixed_branch_source = (
        root
        / "results/nonmyopic/discoverphysics_fixed_initial_branch_replication"
        / "discoverphysics-fixed-initial-branch-replication-20260728T093000Z"
    )
    return (
        EndpointSpec(
            name="original_tree",
            source_dir=original_source,
            result_path=(
                root
                / "results/nonmyopic/discoverphysics_dark_matter_retained_support_confirmation.json"
            ),
            hidden_family=confirmation_hidden_map_family,
            noise_seed=CONFIRMATION_NOISE_SEED,
            bootstrap_seed=CONFIRMATION_BOOTSTRAP_SEED,
            root_samples=CONFIRMATION_ROOT_SAMPLES,
            continuation_samples=CONFIRMATION_CONTINUATION_SAMPLES,
            independently_generated_full_tree=True,
        ),
        EndpointSpec(
            name="structured_v3_tree",
            source_dir=structured_source,
            result_path=structured_source / "RESULT.json",
            hidden_family=replication_hidden_map_family,
            noise_seed=REPLICATION_NOISE_SEED,
            bootstrap_seed=24717,
            root_samples=REPLICATION_ROOT_SAMPLES,
            continuation_samples=REPLICATION_CONTINUATION_SAMPLES,
            independently_generated_full_tree=True,
        ),
        EndpointSpec(
            name="fixed_initial_fresh_branches",
            source_dir=fixed_branch_source,
            result_path=fixed_branch_source / "RESULT.json",
            hidden_family=fixed_branch_hidden_family,
            noise_seed=FIXED_BRANCH_NOISE_SEED,
            bootstrap_seed=FIXED_BRANCH_BOOTSTRAP_SEED,
            root_samples=8,
            continuation_samples=4,
            independently_generated_full_tree=False,
        ),
    )


def replay_endpoint(
    *,
    executor_class: type,
    spec: EndpointSpec,
) -> dict[str, Any]:
    frozen = json.loads((spec.source_dir / "MODEL_FROZEN.json").read_text())
    result = json.loads(spec.result_path.read_text())
    endpoint = result["fresh_hidden_endpoint"]
    initial_support = frozen["initial_support"]
    initial_maps = compile_support(initial_support)
    branches = frozen["branches"][LOOKAHEAD_ROOT_ID]
    refresh_models = compile_refresh_models(
        executor_class=executor_class,
        refreshes=frozen["refreshes"],
    )
    root_action = root_by_id(LOOKAHEAD_ROOT_ID)["action_id"]
    continuations = sorted(
        {
            model["continuation_action"]
            for model in refresh_models[LOOKAHEAD_ROOT_ID]
        }
    )
    initial_action_means, initial_heldout = simulate_maps(
        executor_class,
        initial_maps,
        action_ids=[root_action, *continuations],
    )
    hidden_maps, hidden_regions, hidden_prior = spec.hidden_family()
    hidden_action_means, hidden_heldout = simulate_maps(
        executor_class,
        hidden_maps,
        action_ids=[root_action, *continuations],
    )
    centers = np.asarray(
        [branch["representative_final_coordinate"] for branch in branches]
    )
    num_truths = len(hidden_maps)
    rng = np.random.default_rng(spec.noise_seed)
    root_noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(num_truths, spec.root_samples, 2),
    )
    continuation_noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(
            num_truths,
            spec.root_samples,
            spec.continuation_samples,
            2,
        ),
    )
    fixed_errors = np.empty(num_truths)
    standard_cut_errors = np.empty(num_truths)
    clipped_errors = np.empty(num_truths)
    clip_factors = np.empty(num_truths)
    linear_terms = np.empty(num_truths)
    quadratic_terms = np.empty(num_truths)

    for truth_index in range(num_truths):
        fixed_events = []
        standard_events = []
        clipped_events = []
        factors = []
        linear_events = []
        quadratic_events = []
        for root_sample in range(spec.root_samples):
            root_observation = (
                hidden_action_means[root_action][truth_index]
                + root_noise[truth_index, root_sample]
            )
            branch_index = int(
                np.argmin(
                    np.sum(
                        (centers - root_observation[None, :]) ** 2,
                        axis=-1,
                    )
                )
            )
            branch = branches[branch_index]
            model = refresh_models[LOOKAHEAD_ROOT_ID][branch_index]
            continuation = model["continuation_action"]
            observations = (
                hidden_action_means[continuation][truth_index]
                + continuation_noise[truth_index, root_sample]
            )
            initial_posterior, refresh_posterior = (
                modular_component_posteriors(
                    initial_branch_prior=np.asarray(
                        branch["posterior_probabilities"]
                    ),
                    refresh_branch_prior=model["prior"],
                    representative_observation=centers[branch_index],
                    actual_root_observation=root_observation,
                    initial_root_means=initial_action_means[root_action],
                    refresh_root_means=model["root_means"],
                    continuation_observations=observations,
                    initial_continuation_means=initial_action_means[
                        continuation
                    ],
                    refresh_continuation_means=model[
                        "continuation_means"
                    ],
                )
            )
            initial_prediction = initial_posterior @ initial_heldout
            refresh_prediction = refresh_posterior @ model["heldout"]
            squared_residuals = np.mean(
                (
                    initial_heldout[None, :, :]
                    - initial_prediction[:, None, :]
                )
                ** 2,
                axis=2,
            )
            initial_variance = np.sum(
                initial_posterior * squared_residuals,
                axis=1,
            )
            clipped_prediction, event_factors = (
                uncertainty_clipped_predictions(
                    initial_prediction=initial_prediction,
                    refresh_prediction=refresh_prediction,
                    initial_variance=initial_variance,
                )
            )
            standard_prediction = (
                (1.0 - REFRESH_COMPONENT_MASS) * initial_prediction
                + REFRESH_COMPONENT_MASS * refresh_prediction
            )
            truth = hidden_heldout[truth_index]
            clipped_direction = (
                event_factors[:, None]
                * (refresh_prediction - initial_prediction)
            )
            initial_residual = initial_prediction - truth[None, :]
            fixed_events.extend(
                np.mean(
                    (initial_prediction - truth[None, :]) ** 2,
                    axis=1,
                ).tolist()
            )
            standard_events.extend(
                np.mean(
                    (standard_prediction - truth[None, :]) ** 2,
                    axis=1,
                ).tolist()
            )
            clipped_events.extend(
                np.mean(
                    (clipped_prediction - truth[None, :]) ** 2,
                    axis=1,
                ).tolist()
            )
            factors.extend(event_factors.tolist())
            linear_events.extend(
                (
                    2.0
                    * np.mean(
                        initial_residual * clipped_direction,
                        axis=1,
                    )
                ).tolist()
            )
            quadratic_events.extend(
                np.mean(clipped_direction**2, axis=1).tolist()
            )
        fixed_errors[truth_index] = float(np.mean(fixed_events))
        standard_cut_errors[truth_index] = float(np.mean(standard_events))
        clipped_errors[truth_index] = float(np.mean(clipped_events))
        clip_factors[truth_index] = float(np.mean(factors))
        linear_terms[truth_index] = float(np.mean(linear_events))
        quadratic_terms[truth_index] = float(
            np.mean(quadratic_events)
        )

    expected_fixed = np.asarray(endpoint["per_map_fixed_support_mse"])
    fixed_risk = weighted_mean(fixed_errors, hidden_prior)
    standard_risk = weighted_mean(standard_cut_errors, hidden_prior)
    clipped_risk = weighted_mean(clipped_errors, hidden_prior)
    difference = fixed_errors - clipped_errors
    interval = stratified_bootstrap_interval(
        difference,
        hidden_regions,
        seed=spec.bootstrap_seed,
    )
    reduction = (fixed_risk - clipped_risk) / fixed_risk
    region_array = np.asarray(hidden_regions)
    by_region = {}
    for region in ("NE", "NW", "SW", "SE"):
        mask = region_array == region
        by_region[region] = {
            "fixed_mse": float(np.mean(fixed_errors[mask])),
            "standard_cut_mse": float(
                np.mean(standard_cut_errors[mask])
            ),
            "uncertainty_clipped_mse": float(
                np.mean(clipped_errors[mask])
            ),
            "mean_clip_factor": float(np.mean(clip_factors[mask])),
            "fraction_maps_clipped": float(
                np.mean(clip_factors[mask] < 1.0)
            ),
        }
    return {
        "source_status": result["status"],
        "independently_generated_full_tree": (
            spec.independently_generated_full_tree
        ),
        "num_maps": num_truths,
        "reproduction": {
            "fixed_max_abs_error": float(
                np.max(np.abs(fixed_errors - expected_fixed))
            )
        },
        "fixed_support_mse": fixed_risk,
        "standard_modular_cut_mse": standard_risk,
        "uncertainty_clipped_mse": clipped_risk,
        "clipped_vs_fixed_reduction": reduction,
        "fixed_minus_clipped_ci95": list(interval),
        "mean_clip_factor": weighted_mean(clip_factors, hidden_prior),
        "fraction_maps_clipped": weighted_mean(
            (clip_factors < 1.0).astype(float),
            hidden_prior,
        ),
        "by_region": by_region,
        "regions": hidden_regions,
        "per_map_fixed_mse": fixed_errors.tolist(),
        "per_map_clipped_mse": clipped_errors.tolist(),
        "per_map_linear_term": linear_terms.tolist(),
        "per_map_quadratic_term": quadratic_terms.tolist(),
    }


def analyze(
    *,
    discoverphysics_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    executor_class = load_executor_class(discoverphysics_root)
    endpoints = {
        spec.name: replay_endpoint(
            executor_class=executor_class,
            spec=spec,
        )
        for spec in default_endpoint_specs(repo_root)
    }
    reductions = [
        endpoint["clipped_vs_fixed_reduction"]
        for endpoint in endpoints.values()
    ]
    full_tree_reductions = [
        endpoint["clipped_vs_fixed_reduction"]
        for endpoint in endpoints.values()
        if endpoint["independently_generated_full_tree"]
    ]
    gates = {
        "all_fixed_reproductions_within_1e-10": all(
            endpoint["reproduction"]["fixed_max_abs_error"] <= 1e-10
            for endpoint in endpoints.values()
        ),
        "all_three_clipped_risks_below_fixed": all(
            reduction > 0.0 for reduction in reductions
        ),
        "all_three_paired_lower_bounds_positive": all(
            endpoint["fixed_minus_clipped_ci95"][0] > 0.0
            for endpoint in endpoints.values()
        ),
        "at_least_one_full_tree_gain_at_least_1_percent": (
            max(full_tree_reductions) >= MIN_FULL_TREE_REDUCTION
        ),
        "mean_gain_at_least_half_percent": (
            float(np.mean(reductions)) >= MIN_MEAN_REDUCTION
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "development_pass" if all(gates.values()) else "development_null",
        "method": {
            "refresh_component_mass": REFRESH_COMPONENT_MASS,
            "clip_threshold_initial_posterior_rms": 1.0,
            "epsilon": EPSILON,
        },
        "endpoints": endpoints,
        "mean_relative_reduction": float(np.mean(reductions)),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "new_model_calls": 0,
        "openrouter_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        discoverphysics_root=args.discoverphysics_root.resolve(),
    )
    checkpoint(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
