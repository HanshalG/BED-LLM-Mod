#!/usr/bin/env python3
"""Diagnose refreshed-component calibration in a frozen replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_dark_matter_grounded_policy import (
    LOOKAHEAD_ROOT_ID,
    OBSERVATION_NOISE_STD,
    compile_support,
    load_executor_class,
    root_by_id,
    simulate_maps,
)
from scripts.discoverphysics_dark_matter_retained_support_replay import (
    retained_support_full_history_posterior,
)
from scripts.discoverphysics_dark_matter_retained_support_confirmation import (
    CONFIRMATION_BOOTSTRAP_SEED,
    CONFIRMATION_CONTINUATION_SAMPLES,
    CONFIRMATION_NOISE_SEED,
    CONFIRMATION_ROOT_SAMPLES,
    confirmation_hidden_map_family,
)
from scripts.discoverphysics_dark_matter_structured_replication import (
    INITIAL_COMPONENT_MASS,
    REFRESH_COMPONENT_MASS,
    REPLICATION_CONTINUATION_SAMPLES,
    REPLICATION_NOISE_SEED,
    REPLICATION_ROOT_SAMPLES,
    compile_refresh_models,
    replication_hidden_map_family,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = (
    "discoverphysics-dark-matter-structured-replication-analysis-1"
)
CONSERVATIVE_REFRESH_MASS_CAP = REFRESH_COMPONENT_MASS


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_correlation(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> float:
    left_mean = weighted_mean(left, weights)
    right_mean = weighted_mean(right, weights)
    left_centered = left - left_mean
    right_centered = right - right_mean
    covariance = weighted_mean(left_centered * right_centered, weights)
    left_variance = weighted_mean(left_centered**2, weights)
    right_variance = weighted_mean(right_centered**2, weights)
    denominator = np.sqrt(left_variance * right_variance)
    return float(covariance / denominator) if denominator > 0.0 else 0.0


def stratified_bootstrap_interval(
    differences: np.ndarray,
    regions: list[str],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    region_array = np.asarray(regions)
    estimates = np.empty(samples)
    for sample_index in range(samples):
        estimate = 0.0
        for region, weight in {
            "NE": 0.4,
            "NW": 0.3,
            "SW": 0.2,
            "SE": 0.1,
        }.items():
            indices = np.flatnonzero(region_array == region)
            sampled = rng.choice(indices, size=len(indices), replace=True)
            estimate += weight * float(np.mean(differences[sampled]))
        estimates[sample_index] = estimate
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def analyze(
    *,
    discoverphysics_root: Path,
    source_dir: Path,
    endpoint_kind: str = "structured_v3",
    endpoint_result_path: Path | None = None,
) -> dict[str, Any]:
    frozen = json.loads((source_dir / "MODEL_FROZEN.json").read_text())
    if endpoint_result_path is None:
        endpoint_result_path = source_dir / "RESULT.json"
    result = json.loads(endpoint_result_path.read_text())
    executor_class = load_executor_class(discoverphysics_root)
    if endpoint_kind == "structured_v3":
        hidden_family = replication_hidden_map_family
        noise_seed = REPLICATION_NOISE_SEED
        bootstrap_seed = 24717
        root_samples = REPLICATION_ROOT_SAMPLES
        continuation_samples = REPLICATION_CONTINUATION_SAMPLES
    elif endpoint_kind == "original_confirmation":
        hidden_family = confirmation_hidden_map_family
        noise_seed = CONFIRMATION_NOISE_SEED
        bootstrap_seed = CONFIRMATION_BOOTSTRAP_SEED
        root_samples = CONFIRMATION_ROOT_SAMPLES
        continuation_samples = CONFIRMATION_CONTINUATION_SAMPLES
    else:
        raise ValueError(f"unsupported endpoint kind: {endpoint_kind}")

    initial_support = frozen["initial_support"]
    initial_maps = compile_support(initial_support)
    root_action = root_by_id(LOOKAHEAD_ROOT_ID)["action_id"]
    refresh_models = compile_refresh_models(
        executor_class=executor_class,
        refreshes=frozen["refreshes"],
    )
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
    hidden_maps, hidden_regions, hidden_prior = hidden_family()
    hidden_action_means, hidden_heldout = simulate_maps(
        executor_class,
        hidden_maps,
        action_ids=[root_action, *continuations],
    )

    branches = frozen["branches"][LOOKAHEAD_ROOT_ID]
    centers = np.asarray(
        [branch["representative_final_coordinate"] for branch in branches]
    )
    num_truths = len(hidden_maps)
    rng = np.random.default_rng(noise_seed)
    root_noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(num_truths, root_samples, 2),
    )
    continuation_noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(
            num_truths,
            root_samples,
            continuation_samples,
            2,
        ),
    )

    retained_errors = np.empty(num_truths)
    initial_component_errors = np.empty(num_truths)
    refresh_component_errors = np.empty(num_truths)
    capped_errors = np.empty(num_truths)
    refresh_masses = np.empty(num_truths)
    refresh_better_fractions = np.empty(num_truths)
    branch_one_fractions = np.empty(num_truths)
    event_masses = []
    event_advantages = []
    event_weights = []

    for truth_index in range(num_truths):
        retained = []
        initial_only = []
        refresh_only = []
        masses = []
        refresh_better = []
        branch_indices = []
        capped = []
        for root_sample in range(root_samples):
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
            branch_indices.append(branch_index)
            branch = branches[branch_index]
            model = refresh_models[LOOKAHEAD_ROOT_ID][branch_index]
            continuation = model["continuation_action"]
            continuation_observations = (
                hidden_action_means[continuation][truth_index]
                + continuation_noise[truth_index, root_sample]
            )
            posterior = retained_support_full_history_posterior(
                initial_branch_prior=np.asarray(
                    branch["posterior_probabilities"]
                ),
                refresh_branch_prior=model["prior"],
                representative_observation=centers[branch_index],
                actual_root_observation=root_observation,
                initial_root_means=initial_action_means[root_action],
                refresh_root_means=model["root_means"],
                continuation_observations=continuation_observations,
                initial_continuation_means=initial_action_means[continuation],
                refresh_continuation_means=model["continuation_means"],
                initial_component_mass=INITIAL_COMPONENT_MASS,
                refresh_component_mass=REFRESH_COMPONENT_MASS,
            )
            split = len(initial_support)
            initial_probabilities = posterior[:, :split]
            refresh_probabilities = posterior[:, split:]
            refresh_mass = refresh_probabilities.sum(axis=1)
            initial_conditional = retained_support_full_history_posterior(
                initial_branch_prior=np.asarray(
                    branch["posterior_probabilities"]
                ),
                refresh_branch_prior=model["prior"],
                representative_observation=centers[branch_index],
                actual_root_observation=root_observation,
                initial_root_means=initial_action_means[root_action],
                refresh_root_means=model["root_means"],
                continuation_observations=continuation_observations,
                initial_continuation_means=initial_action_means[continuation],
                refresh_continuation_means=model["continuation_means"],
                initial_component_mass=1.0,
                refresh_component_mass=0.0,
            )[:, :split]
            refresh_conditional = retained_support_full_history_posterior(
                initial_branch_prior=np.asarray(
                    branch["posterior_probabilities"]
                ),
                refresh_branch_prior=model["prior"],
                representative_observation=centers[branch_index],
                actual_root_observation=root_observation,
                initial_root_means=initial_action_means[root_action],
                refresh_root_means=model["root_means"],
                continuation_observations=continuation_observations,
                initial_continuation_means=initial_action_means[continuation],
                refresh_continuation_means=model["continuation_means"],
                initial_component_mass=0.0,
                refresh_component_mass=1.0,
            )[:, split:]
            initial_prediction = initial_conditional @ initial_heldout
            refresh_prediction = refresh_conditional @ model["heldout"]
            retained_prediction = (
                initial_probabilities @ initial_heldout
                + refresh_probabilities @ model["heldout"]
            )
            capped_mass = np.minimum(
                refresh_mass,
                CONSERVATIVE_REFRESH_MASS_CAP,
            )
            capped_prediction = (
                (1.0 - capped_mass[:, None]) * initial_prediction
                + capped_mass[:, None] * refresh_prediction
            )
            truth = hidden_heldout[truth_index]
            initial_error = np.mean(
                (initial_prediction - truth[None, :]) ** 2,
                axis=1,
            )
            refresh_error = np.mean(
                (refresh_prediction - truth[None, :]) ** 2,
                axis=1,
            )
            retained_error = np.mean(
                (retained_prediction - truth[None, :]) ** 2,
                axis=1,
            )
            capped_error = np.mean(
                (capped_prediction - truth[None, :]) ** 2,
                axis=1,
            )
            retained.extend(retained_error.tolist())
            initial_only.extend(initial_error.tolist())
            refresh_only.extend(refresh_error.tolist())
            capped.extend(capped_error.tolist())
            masses.extend(refresh_mass.tolist())
            refresh_better.extend((refresh_error < initial_error).tolist())
            event_masses.extend(refresh_mass.tolist())
            event_advantages.extend((initial_error - refresh_error).tolist())
            event_weights.extend(
                [hidden_prior[truth_index]]
                * continuation_samples
            )
        retained_errors[truth_index] = float(np.mean(retained))
        initial_component_errors[truth_index] = float(np.mean(initial_only))
        refresh_component_errors[truth_index] = float(np.mean(refresh_only))
        capped_errors[truth_index] = float(np.mean(capped))
        refresh_masses[truth_index] = float(np.mean(masses))
        refresh_better_fractions[truth_index] = float(
            np.mean(refresh_better)
        )
        branch_one_fractions[truth_index] = float(
            np.mean(np.asarray(branch_indices) == 1)
        )

    expected_retained = np.asarray(
        result["fresh_hidden_endpoint"]["per_map_retained_mse"][
            LOOKAHEAD_ROOT_ID
        ]
    )
    expected_fixed = np.asarray(
        result["fresh_hidden_endpoint"]["per_map_fixed_support_mse"]
    )
    event_masses_array = np.asarray(event_masses)
    event_advantages_array = np.asarray(event_advantages)
    event_weights_array = np.asarray(event_weights)
    better = event_advantages_array > 0.0

    def summary(mask: np.ndarray) -> dict[str, float]:
        weights = hidden_prior[mask]
        return {
            "retained_mse": weighted_mean(retained_errors[mask], weights),
            "initial_component_mse": weighted_mean(
                initial_component_errors[mask], weights
            ),
            "refresh_component_mse": weighted_mean(
                refresh_component_errors[mask], weights
            ),
            "capped_at_prior_mse": weighted_mean(
                capped_errors[mask], weights
            ),
            "mean_refresh_posterior_mass": weighted_mean(
                refresh_masses[mask], weights
            ),
            "refresh_component_better_fraction": weighted_mean(
                refresh_better_fractions[mask], weights
            ),
            "branch_one_fraction": weighted_mean(
                branch_one_fractions[mask], weights
            ),
        }

    regions = np.asarray(hidden_regions)
    all_mask = np.ones(num_truths, dtype=bool)
    fixed_risk = weighted_mean(expected_fixed, hidden_prior)
    capped_risk = weighted_mean(capped_errors, hidden_prior)
    capped_difference = expected_fixed - capped_errors
    capped_ci = stratified_bootstrap_interval(
        capped_difference,
        hidden_regions,
        seed=bootstrap_seed,
    )
    capped_reduction = (fixed_risk - capped_risk) / fixed_risk
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "posthoc_descriptive",
        "source_status": result["status"],
        "endpoint_kind": endpoint_kind,
        "reproduction": {
            "retained_max_abs_error": float(
                np.max(np.abs(retained_errors - expected_retained))
            ),
            "fixed_max_abs_error": float(
                np.max(np.abs(initial_component_errors - expected_fixed))
            ),
        },
        "overall": summary(all_mask),
        "by_region": {
            region: summary(regions == region)
            for region in ("NE", "NW", "SW", "SE")
        },
        "event_level": {
            "refresh_mass_advantage_correlation": weighted_correlation(
                event_masses_array,
                event_advantages_array,
                event_weights_array,
            ),
            "refresh_better_fraction": weighted_mean(
                better.astype(float),
                event_weights_array,
            ),
            "mean_refresh_mass_when_better": weighted_mean(
                event_masses_array[better],
                event_weights_array[better],
            ),
            "mean_refresh_mass_when_worse": weighted_mean(
                event_masses_array[~better],
                event_weights_array[~better],
            ),
        },
        "conservative_cap": {
            "refresh_mass_cap": CONSERVATIVE_REFRESH_MASS_CAP,
            "fixed_support_mse": fixed_risk,
            "capped_mse": capped_risk,
            "capped_vs_fixed_reduction": capped_reduction,
            "fixed_minus_capped_ci95": list(capped_ci),
            "reduction_at_least_1_percent": capped_reduction >= 0.01,
            "paired_lower_bound_positive": capped_ci[0] > 0.0,
        },
        "new_model_calls": 0,
        "openrouter_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint-kind",
        choices=("structured_v3", "original_confirmation"),
        default="structured_v3",
    )
    parser.add_argument("--endpoint-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        discoverphysics_root=args.discoverphysics_root.resolve(),
        source_dir=args.source_dir.resolve(),
        endpoint_kind=args.endpoint_kind,
        endpoint_result_path=(
            args.endpoint_result.resolve()
            if args.endpoint_result is not None
            else None
        ),
    )
    checkpoint(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
