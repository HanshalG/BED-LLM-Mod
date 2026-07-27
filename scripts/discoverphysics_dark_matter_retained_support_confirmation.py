#!/usr/bin/env python3
"""Fresh-map confirmation for low-mass regenerated dark-matter support."""

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
    _sha256,
)
from scripts.discoverphysics_dark_matter_grounded_policy import (
    INTERNAL_CONTINUATION_SAMPLES,
    INTERNAL_ROOT_SAMPLES,
    LOOKAHEAD_ROOT_ID,
    MIN_HIDDEN_RISK_REDUCTION,
    MIN_INTERNAL_RISK_REDUCTION,
    MIN_RANDOM_RISK_REDUCTION,
    MYOPIC_ROOT_ID,
    POLICY_NOISE_SEED,
    RANDOM_ROOT_ID,
    REGION_PRIOR,
    ROOTS,
    action_table,
    compile_support,
    evaluate_fixed_support_root,
    hidden_halo_family,
    immediate_eig,
    load_executor_class,
    relative_reduction,
    simulate_maps,
    support_prior,
    weighted_mean,
)
from scripts.discoverphysics_dark_matter_opportunity import (
    verify_discoverphysics,
)
from scripts.discoverphysics_dark_matter_retained_support_replay import (
    evaluate_retained_root,
    retained_coverage_risk,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-retained-confirmation-1"
SOURCE_CURVE_SHA256 = (
    "b8aed4f44e72def92383b2f3171654e03ecf26b8c125ba26f98158308f430fcd"
)
INITIAL_COMPONENT_MASS = 0.95
REFRESH_COMPONENT_MASS = 0.05
CONFIRMATION_MAP_SEEDS = tuple(range(24600, 24616))
CONFIRMATION_NOISE_SEED = 24616
CONFIRMATION_BOOTSTRAP_SEED = 24617
CONFIRMATION_ROOT_SAMPLES = 8
CONFIRMATION_CONTINUATION_SAMPLES = 4
BOOTSTRAP_SAMPLES = 10_000
MIN_FIXED_SUPPORT_RISK_REDUCTION = 0.01
MIN_COVERAGE_RISK_REDUCTION = 0.05


def confirmation_hidden_map_family() -> tuple[np.ndarray, list[str], np.ndarray]:
    maps = []
    regions = []
    for seed in CONFIRMATION_MAP_SEEDS:
        seed_maps, _ = hidden_halo_family(seed=seed)
        maps.append(seed_maps)
        regions.extend(
            region
            for region in ("NE", "NW", "SW", "SE")
            for _ in range(6)
        )
    source_maps = np.concatenate(maps, axis=0)
    prior = np.asarray(
        [
            REGION_PRIOR[region]
            / sum(candidate == region for candidate in regions)
            for region in regions
        ]
    )
    return source_maps, regions, prior


def stratified_bootstrap_interval(
    differences: np.ndarray,
    regions: list[str],
) -> tuple[float, float]:
    rng = np.random.default_rng(CONFIRMATION_BOOTSTRAP_SEED)
    region_array = np.asarray(regions)
    region_indices = {
        region: np.flatnonzero(region_array == region)
        for region in REGION_PRIOR
    }
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        estimate = 0.0
        for region, region_weight in REGION_PRIOR.items():
            indices = region_indices[region]
            sampled = rng.choice(indices, size=len(indices), replace=True)
            estimate += region_weight * float(
                np.mean(differences[sampled])
            )
        estimates[sample_index] = estimate
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _curve_path(source_dir: Path) -> Path:
    return (
        source_dir.parents[1]
        / "discoverphysics_dark_matter_mixture_curve.json"
    )


def _load_frozen_models(
    *,
    executor_class: type,
    frozen: dict[str, Any],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, list[dict[str, Any]]],
]:
    initial_support = frozen["initial_support"]
    initial_prior = support_prior(initial_support)
    initial_action_means, initial_heldout = simulate_maps(
        executor_class,
        compile_support(initial_support),
        action_ids=list(action_table()),
    )
    refresh_models: dict[str, list[dict[str, Any]]] = {}
    for root in ROOTS:
        models = []
        for refresh in frozen["refreshes"][root["id"]]:
            means, heldout = simulate_maps(
                executor_class,
                compile_support(refresh["hypotheses"]),
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
    return (
        initial_prior,
        initial_action_means,
        initial_heldout,
        refresh_models,
    )


def run_confirmation(
    *,
    discoverphysics_root: Path,
    source_dir: Path,
) -> dict[str, Any]:
    commit = verify_discoverphysics(discoverphysics_root)
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError("DiscoverPhysics commit changed")
    policy_path = source_dir / "POLICY.json"
    frozen_path = source_dir / "MODEL_FROZEN.json"
    curve_path = _curve_path(source_dir)
    if _sha256(policy_path) != SOURCE_POLICY_SHA256:
        raise ValueError("source policy hash changed")
    if _sha256(frozen_path) != SOURCE_MODEL_FROZEN_SHA256:
        raise ValueError("source frozen-model hash changed")
    if _sha256(curve_path) != SOURCE_CURVE_SHA256:
        raise ValueError("source mixture curve hash changed")

    frozen = json.loads(frozen_path.read_text())
    branches = frozen["branches"]
    refreshes = frozen["refreshes"]
    executor_class = load_executor_class(discoverphysics_root)
    (
        initial_prior,
        initial_action_means,
        initial_heldout,
        refresh_models,
    ) = _load_frozen_models(
        executor_class=executor_class,
        frozen=frozen,
    )

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
            initial_component_mass=INITIAL_COMPONENT_MASS,
            refresh_component_mass=REFRESH_COMPONENT_MASS,
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

    hidden_maps, hidden_regions, hidden_prior = (
        confirmation_hidden_map_family()
    )
    required_actions = {
        root["action_id"] for root in ROOTS
    } | {
        refresh["continuation_action"]
        for root_refreshes in refreshes.values()
        for refresh in root_refreshes
    }
    hidden_action_means, hidden_heldout = simulate_maps(
        executor_class,
        hidden_maps,
        action_ids=sorted(required_actions),
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
            rng=np.random.default_rng(CONFIRMATION_NOISE_SEED),
            root_samples=CONFIRMATION_ROOT_SAMPLES,
            continuation_samples=CONFIRMATION_CONTINUATION_SAMPLES,
            initial_component_mass=INITIAL_COMPONENT_MASS,
            refresh_component_mass=REFRESH_COMPONENT_MASS,
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
        rng=np.random.default_rng(CONFIRMATION_NOISE_SEED),
        root_samples=CONFIRMATION_ROOT_SAMPLES,
        continuation_samples=CONFIRMATION_CONTINUATION_SAMPLES,
    )
    hidden_risks = {
        root_id: weighted_mean(values, hidden_prior)
        for root_id, values in retained_per_map.items()
    }
    fixed_risk = weighted_mean(fixed_per_map, hidden_prior)
    myopic_difference = (
        retained_per_map[MYOPIC_ROOT_ID]
        - retained_per_map[LOOKAHEAD_ROOT_ID]
    )
    fixed_difference = (
        fixed_per_map - retained_per_map[LOOKAHEAD_ROOT_ID]
    )
    myopic_bootstrap_ci = stratified_bootstrap_interval(
        myopic_difference,
        hidden_regions,
    )
    fixed_bootstrap_ci = stratified_bootstrap_interval(
        fixed_difference,
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
        "myopic_paired_bootstrap_lower_bound_positive": (
            myopic_bootstrap_ci[0] > 0.0
        ),
        "gain_vs_random_at_least_5_percent": (
            random_reduction >= MIN_RANDOM_RISK_REDUCTION
        ),
        "gain_vs_fixed_support_at_least_1_percent": (
            fixed_reduction >= MIN_FIXED_SUPPORT_RISK_REDUCTION
        ),
        "fixed_paired_bootstrap_lower_bound_positive": (
            fixed_bootstrap_ci[0] > 0.0
        ),
        "retained_coverage_gain_at_least_5_percent": (
            coverage_reduction >= MIN_COVERAGE_RISK_REDUCTION
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "confirmation_pass" if all(gates.values()) else "confirmation_null",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "discoverphysics_commit": commit,
            "source_policy_sha256": SOURCE_POLICY_SHA256,
            "source_model_frozen_sha256": SOURCE_MODEL_FROZEN_SHA256,
            "source_curve_sha256": SOURCE_CURVE_SHA256,
            "initial_component_mass": INITIAL_COMPONENT_MASS,
            "refresh_component_mass": REFRESH_COMPONENT_MASS,
            "map_seeds": list(CONFIRMATION_MAP_SEEDS),
            "noise_seed": CONFIRMATION_NOISE_SEED,
            "bootstrap_seed": CONFIRMATION_BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "num_maps": len(hidden_maps),
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
        "fresh_hidden_endpoint": {
            "retained_dynamic_trajectory_mse": hidden_risks,
            "fixed_support_center_trajectory_mse": fixed_risk,
            "lookahead_vs_myopic_risk_reduction": hidden_reduction,
            "lookahead_vs_random_risk_reduction": random_reduction,
            "lookahead_vs_fixed_support_risk_reduction": fixed_reduction,
            "myopic_paired_difference_ci95": list(myopic_bootstrap_ci),
            "fixed_paired_difference_ci95": list(fixed_bootstrap_ci),
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
    payload = run_confirmation(
        discoverphysics_root=args.discoverphysics_root.resolve(),
        source_dir=args.source_dir.resolve(),
    )
    checkpoint(args.output_path.resolve(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection": payload["selection"],
                "fresh_hidden_endpoint": {
                    key: value
                    for key, value in payload["fresh_hidden_endpoint"].items()
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
