#!/usr/bin/env python3
"""Replay the frozen dark-matter policy while retaining initial support."""

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

from scripts.discoverphysics_dark_matter_full_history_replay import (
    DISCOVERPHYSICS_COMMIT,
    SOURCE_MODEL_FROZEN_SHA256,
    SOURCE_POLICY_SHA256,
    _log_likelihood,
    _sha256,
)
from scripts.discoverphysics_dark_matter_grounded_policy import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    HIDDEN_CONTINUATION_SAMPLES,
    HIDDEN_MAP_SEEDS,
    HIDDEN_NOISE_SEED,
    HIDDEN_ROOT_SAMPLES,
    INTERNAL_CONTINUATION_SAMPLES,
    INTERNAL_ROOT_SAMPLES,
    LOOKAHEAD_ROOT_ID,
    MIN_FIXED_SUPPORT_RISK_REDUCTION,
    MIN_HIDDEN_RISK_REDUCTION,
    MIN_INTERNAL_RISK_REDUCTION,
    MIN_RANDOM_RISK_REDUCTION,
    MYOPIC_ROOT_ID,
    OBSERVATION_NOISE_STD,
    POLICY_NOISE_SEED,
    RANDOM_ROOT_ID,
    ROOTS,
    action_table,
    compile_support,
    evaluate_fixed_support_root,
    hidden_map_family,
    immediate_eig,
    load_executor_class,
    relative_reduction,
    root_by_id,
    simulate_maps,
    stratified_bootstrap_difference,
    support_prior,
    weighted_mean,
)
from scripts.discoverphysics_dark_matter_opportunity import (
    verify_discoverphysics,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-retained-support-replay-1"
INITIAL_COMPONENT_MASS = 0.5
REFRESH_COMPONENT_MASS = 0.5


def retained_support_full_history_posterior(
    *,
    initial_branch_prior: np.ndarray,
    refresh_branch_prior: np.ndarray,
    representative_observation: np.ndarray,
    actual_root_observation: np.ndarray,
    initial_root_means: np.ndarray,
    refresh_root_means: np.ndarray,
    continuation_observations: np.ndarray,
    initial_continuation_means: np.ndarray,
    refresh_continuation_means: np.ndarray,
    initial_component_mass: float = INITIAL_COMPONENT_MASS,
    refresh_component_mass: float = REFRESH_COMPONENT_MASS,
) -> np.ndarray:
    """Update an equal branch-level mixture of retained and refreshed support."""

    if initial_component_mass < 0.0 or refresh_component_mass < 0.0:
        raise ValueError("component masses must be non-negative")
    if initial_component_mass + refresh_component_mass <= 0.0:
        raise ValueError("at least one component mass must be positive")
    component_total = initial_component_mass + refresh_component_mass
    initial_component_mass /= component_total
    refresh_component_mass /= component_total

    def component_logits(
        *,
        component_mass: float,
        branch_prior: np.ndarray,
        root_means: np.ndarray,
        continuation_means: np.ndarray,
    ) -> np.ndarray:
        representative_ll = _log_likelihood(
            representative_observation,
            root_means,
        )
        actual_root_ll = _log_likelihood(
            actual_root_observation,
            root_means,
        )
        continuation_ll = (
            -0.5
            * np.sum(
                (
                    continuation_observations[:, None, :]
                    - continuation_means[None, :, :]
                )
                ** 2,
                axis=-1,
            )
            / OBSERVATION_NOISE_STD**2
        )
        return (
            (np.log(component_mass) if component_mass > 0.0 else -np.inf)
            + np.log(branch_prior[None, :] + 1e-300)
            + actual_root_ll[None, :]
            - representative_ll[None, :]
            + continuation_ll
        )

    logits = np.concatenate(
        [
            component_logits(
                component_mass=initial_component_mass,
                branch_prior=initial_branch_prior,
                root_means=initial_root_means,
                continuation_means=initial_continuation_means,
            ),
            component_logits(
                component_mass=refresh_component_mass,
                branch_prior=refresh_branch_prior,
                root_means=refresh_root_means,
                continuation_means=refresh_continuation_means,
            ),
        ],
        axis=-1,
    )
    logits -= logits.max(axis=-1, keepdims=True)
    posterior = np.exp(logits)
    return posterior / posterior.sum(axis=-1, keepdims=True)


def evaluate_retained_root(
    *,
    root_id: str,
    true_action_means: dict[str, np.ndarray],
    true_heldout: np.ndarray,
    initial_action_means: dict[str, np.ndarray],
    initial_heldout: np.ndarray,
    branches: dict[str, list[dict[str, Any]]],
    refresh_models: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
    root_samples: int,
    continuation_samples: int,
    initial_component_mass: float = INITIAL_COMPONENT_MASS,
    refresh_component_mass: float = REFRESH_COMPONENT_MASS,
) -> np.ndarray:
    num_truths = len(true_heldout)
    per_truth = np.zeros(num_truths)
    root_action = root_by_id(root_id)["action_id"]
    centers = np.asarray(
        [
            branch["representative_final_coordinate"]
            for branch in branches[root_id]
        ]
    )
    root_noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(num_truths, root_samples, 2),
    )
    continuation_noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(num_truths, root_samples, continuation_samples, 2),
    )
    for truth_index in range(num_truths):
        errors = []
        for root_sample in range(root_samples):
            root_observation = (
                true_action_means[root_action][truth_index]
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
            branch = branches[root_id][branch_index]
            model = refresh_models[root_id][branch_index]
            continuation = model["continuation_action"]
            continuation_observations = (
                true_action_means[continuation][truth_index]
                + continuation_noise[truth_index, root_sample]
            )
            posterior = retained_support_full_history_posterior(
                initial_branch_prior=np.asarray(
                    branch["posterior_probabilities"],
                ),
                refresh_branch_prior=model["prior"],
                representative_observation=centers[branch_index],
                actual_root_observation=root_observation,
                initial_root_means=initial_action_means[root_action],
                refresh_root_means=model["root_means"],
                continuation_observations=continuation_observations,
                initial_continuation_means=initial_action_means[continuation],
                refresh_continuation_means=model["continuation_means"],
                initial_component_mass=initial_component_mass,
                refresh_component_mass=refresh_component_mass,
            )
            combined_heldout = np.concatenate(
                [initial_heldout, model["heldout"]],
                axis=0,
            )
            predictions = posterior @ combined_heldout
            errors.extend(
                np.mean(
                    (predictions - true_heldout[truth_index][None, :]) ** 2,
                    axis=-1,
                ).tolist()
            )
        per_truth[truth_index] = float(np.mean(errors))
    return per_truth


def retained_coverage_risk(
    *,
    root_id: str,
    hidden_action_means: dict[str, np.ndarray],
    hidden_heldout: np.ndarray,
    hidden_prior: np.ndarray,
    initial_heldout: np.ndarray,
    branches: dict[str, list[dict[str, Any]]],
    refresh_models: dict[str, list[dict[str, Any]]],
) -> tuple[float, float]:
    root_action = root_by_id(root_id)["action_id"]
    centers = np.asarray(
        [
            branch["representative_final_coordinate"]
            for branch in branches[root_id]
        ]
    )
    initial_errors = []
    retained_errors = []
    for index, truth in enumerate(hidden_heldout):
        branch_index = int(
            np.argmin(
                np.sum(
                    (
                        centers
                        - hidden_action_means[root_action][index][None, :]
                    )
                    ** 2,
                    axis=-1,
                )
            )
        )
        refreshed = refresh_models[root_id][branch_index]["heldout"]
        initial_error = float(
            np.min(np.mean((initial_heldout - truth[None, :]) ** 2, axis=-1))
        )
        refreshed_error = float(
            np.min(np.mean((refreshed - truth[None, :]) ** 2, axis=-1))
        )
        initial_errors.append(initial_error)
        retained_errors.append(min(initial_error, refreshed_error))
    return (
        weighted_mean(np.asarray(initial_errors), hidden_prior),
        weighted_mean(np.asarray(retained_errors), hidden_prior),
    )


def run_replay(
    *,
    discoverphysics_root: Path,
    source_dir: Path,
) -> dict[str, Any]:
    commit = verify_discoverphysics(discoverphysics_root)
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError("DiscoverPhysics commit changed")
    policy_path = source_dir / "POLICY.json"
    frozen_path = source_dir / "MODEL_FROZEN.json"
    if _sha256(policy_path) != SOURCE_POLICY_SHA256:
        raise ValueError("source policy hash changed")
    if _sha256(frozen_path) != SOURCE_MODEL_FROZEN_SHA256:
        raise ValueError("source frozen-model hash changed")

    frozen = json.loads(frozen_path.read_text())
    initial_support = frozen["initial_support"]
    branches = frozen["branches"]
    refreshes = frozen["refreshes"]
    executor_class = load_executor_class(discoverphysics_root)

    initial_maps = compile_support(initial_support)
    initial_prior = support_prior(initial_support)
    initial_action_means, initial_heldout = simulate_maps(
        executor_class,
        initial_maps,
        action_ids=list(action_table()),
    )
    refresh_models: dict[str, list[dict[str, Any]]] = {}
    for root in ROOTS:
        models = []
        for refresh in refreshes[root["id"]]:
            maps = compile_support(refresh["hypotheses"])
            means, heldout = simulate_maps(
                executor_class,
                maps,
                action_ids=[
                    root["action_id"],
                    refresh["continuation_action"],
                ],
            )
            models.append(
                {
                    **refresh,
                    "prior": support_prior(refresh["hypotheses"]),
                    "root_means": means[root["action_id"]],
                    "continuation_means": means[
                        refresh["continuation_action"]
                    ],
                    "heldout": heldout,
                }
            )
        refresh_models[root["id"]] = models

    immediate_values = {
        root["id"]: immediate_eig(
            initial_action_means[root["action_id"]],
            initial_prior,
            rng=np.random.default_rng(POLICY_NOISE_SEED),
            samples_per_hypothesis=INTERNAL_ROOT_SAMPLES,
        )
        for root in ROOTS
    }
    internal_risks = {}
    for root in ROOTS:
        per_truth = evaluate_retained_root(
            root_id=root["id"],
            true_action_means=initial_action_means,
            true_heldout=initial_heldout,
            initial_action_means=initial_action_means,
            initial_heldout=initial_heldout,
            branches=branches,
            refresh_models=refresh_models,
            rng=np.random.default_rng(POLICY_NOISE_SEED),
            root_samples=INTERNAL_ROOT_SAMPLES,
            continuation_samples=INTERNAL_CONTINUATION_SAMPLES,
        )
        internal_risks[root["id"]] = weighted_mean(per_truth, initial_prior)
    myopic_root = min(
        immediate_values,
        key=lambda root_id: (-immediate_values[root_id], root_id),
    )
    lookahead_root = min(
        internal_risks,
        key=lambda root_id: (internal_risks[root_id], root_id),
    )

    hidden_maps, hidden_regions, hidden_prior = hidden_map_family()
    required_hidden_actions = {
        root["action_id"] for root in ROOTS
    } | {
        refresh["continuation_action"]
        for root_refreshes in refreshes.values()
        for refresh in root_refreshes
    }
    hidden_action_means, hidden_heldout = simulate_maps(
        executor_class,
        hidden_maps,
        action_ids=sorted(required_hidden_actions),
    )
    retained_per_map = {}
    for root_id in {MYOPIC_ROOT_ID, LOOKAHEAD_ROOT_ID, RANDOM_ROOT_ID}:
        retained_per_map[root_id] = evaluate_retained_root(
            root_id=root_id,
            true_action_means=hidden_action_means,
            true_heldout=hidden_heldout,
            initial_action_means=initial_action_means,
            initial_heldout=initial_heldout,
            branches=branches,
            refresh_models=refresh_models,
            rng=np.random.default_rng(HIDDEN_NOISE_SEED),
            root_samples=HIDDEN_ROOT_SAMPLES,
            continuation_samples=HIDDEN_CONTINUATION_SAMPLES,
        )
    fixed_per_map = evaluate_fixed_support_root(
        root_id=LOOKAHEAD_ROOT_ID,
        true_action_means=hidden_action_means,
        true_heldout=hidden_heldout,
        initial_action_means=initial_action_means,
        initial_heldout=initial_heldout,
        initial_prior=initial_prior,
        branches=branches,
        refresh_models=refresh_models,
        rng=np.random.default_rng(HIDDEN_NOISE_SEED),
        root_samples=HIDDEN_ROOT_SAMPLES,
        continuation_samples=HIDDEN_CONTINUATION_SAMPLES,
    )
    hidden_risks = {
        root_id: weighted_mean(values, hidden_prior)
        for root_id, values in retained_per_map.items()
    }
    fixed_risk = weighted_mean(fixed_per_map, hidden_prior)
    bootstrap_ci = stratified_bootstrap_difference(
        retained_per_map[MYOPIC_ROOT_ID]
        - retained_per_map[LOOKAHEAD_ROOT_ID],
        hidden_regions,
    )
    initial_coverage, retained_coverage = retained_coverage_risk(
        root_id=LOOKAHEAD_ROOT_ID,
        hidden_action_means=hidden_action_means,
        hidden_heldout=hidden_heldout,
        hidden_prior=hidden_prior,
        initial_heldout=initial_heldout,
        branches=branches,
        refresh_models=refresh_models,
    )
    internal_reduction = relative_reduction(
        internal_risks[MYOPIC_ROOT_ID],
        internal_risks[LOOKAHEAD_ROOT_ID],
    )
    hidden_reduction = relative_reduction(
        hidden_risks[MYOPIC_ROOT_ID],
        hidden_risks[LOOKAHEAD_ROOT_ID],
    )
    random_reduction = relative_reduction(
        hidden_risks[RANDOM_ROOT_ID],
        hidden_risks[LOOKAHEAD_ROOT_ID],
    )
    fixed_reduction = relative_reduction(
        fixed_risk,
        hidden_risks[LOOKAHEAD_ROOT_ID],
    )
    coverage_reduction = relative_reduction(
        initial_coverage,
        retained_coverage,
    )
    gates = {
        "myopic_root_is_D": myopic_root == MYOPIC_ROOT_ID,
        "lookahead_root_is_B": lookahead_root == LOOKAHEAD_ROOT_ID,
        "internal_risk_reduction_at_least_10_percent": (
            internal_reduction >= MIN_INTERNAL_RISK_REDUCTION
        ),
        "hidden_risk_reduction_at_least_10_percent": (
            hidden_reduction >= MIN_HIDDEN_RISK_REDUCTION
        ),
        "paired_bootstrap_lower_bound_positive": bootstrap_ci[0] > 0.0,
        "gain_vs_random_at_least_5_percent": (
            random_reduction >= MIN_RANDOM_RISK_REDUCTION
        ),
        "gain_vs_fixed_support_at_least_5_percent": (
            fixed_reduction >= MIN_FIXED_SUPPORT_RISK_REDUCTION
        ),
        "retained_coverage_gain_at_least_5_percent": (
            coverage_reduction >= 0.05
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_pass" if all(gates.values()) else "diagnostic_null",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "discoverphysics_commit": commit,
            "source_policy_sha256": SOURCE_POLICY_SHA256,
            "source_model_frozen_sha256": SOURCE_MODEL_FROZEN_SHA256,
            "initial_component_mass": INITIAL_COMPONENT_MASS,
            "refresh_component_mass": REFRESH_COMPONENT_MASS,
            "policy_noise_seed": POLICY_NOISE_SEED,
            "hidden_map_seeds": list(HIDDEN_MAP_SEEDS),
            "hidden_noise_seed": HIDDEN_NOISE_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "new_model_calls": 0,
            "openrouter_cost_usd": 0.0,
        },
        "selection": {
            "immediate_eig_nats": immediate_values,
            "retained_internal_trajectory_risk": internal_risks,
            "myopic_root": myopic_root,
            "lookahead_root": lookahead_root,
            "internal_risk_reduction": internal_reduction,
        },
        "hidden_endpoint": {
            "retained_dynamic_trajectory_mse": hidden_risks,
            "fixed_support_center_trajectory_mse": fixed_risk,
            "lookahead_vs_myopic_risk_reduction": hidden_reduction,
            "lookahead_vs_random_risk_reduction": random_reduction,
            "lookahead_vs_fixed_support_risk_reduction": fixed_reduction,
            "paired_difference_ci95": list(bootstrap_ci),
            "initial_nearest_support_risk": initial_coverage,
            "retained_nearest_support_risk": retained_coverage,
            "coverage_risk_reduction": coverage_reduction,
            "per_map_retained_mse": {
                root_id: values.tolist()
                for root_id, values in retained_per_map.items()
            },
            "per_map_fixed_support_mse": fixed_per_map.tolist(),
            "regions": hidden_regions,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "new_model_calls": 0,
        "openrouter_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    payload = run_replay(
        discoverphysics_root=args.discoverphysics_root.resolve(),
        source_dir=args.source_dir.resolve(),
    )
    checkpoint(args.output_path.resolve(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection": payload["selection"],
                "hidden_endpoint": {
                    key: value
                    for key, value in payload["hidden_endpoint"].items()
                    if not key.startswith("per_map") and key != "regions"
                },
                "gates": payload["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
