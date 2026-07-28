#!/usr/bin/env python3
"""Select a maximin robust generated-support step on open endpoints."""

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

from scripts.analyze_discoverphysics_dark_matter_structured_replication import (
    stratified_bootstrap_interval,
)
from scripts.analyze_discoverphysics_uncertainty_clipped_blend import (
    analyze as replay_clipped_components,
    default_endpoint_specs,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-robust-step-selection-1"
COEFFICIENTS = tuple(float(value) for value in np.arange(0.0, 0.5001, 0.025))
MIN_TREE_REDUCTION = 0.005
MIN_MEAN_REDUCTION = 0.01
MIN_FULL_TREE_REDUCTION = 0.01
REGION_PRIOR = {"NE": 0.4, "NW": 0.3, "SW": 0.2, "SE": 0.1}


def per_map_risk_from_terms(
    *,
    fixed: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    return (
        fixed
        + coefficient * linear
        + coefficient**2 * quadratic
    )


def endpoint_weights(regions: list[str]) -> np.ndarray:
    region_array = np.asarray(regions)
    return np.asarray(
        [
            REGION_PRIOR[region]
            / int(np.sum(region_array == region))
            for region in regions
        ]
    )


def coefficient_metrics(
    *,
    endpoint: dict[str, Any],
    coefficient: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    fixed = np.asarray(endpoint["per_map_fixed_mse"])
    linear = np.asarray(endpoint["per_map_linear_term"])
    quadratic = np.asarray(endpoint["per_map_quadratic_term"])
    candidate = per_map_risk_from_terms(
        fixed=fixed,
        linear=linear,
        quadratic=quadratic,
        coefficient=coefficient,
    )
    weights = endpoint_weights(endpoint["regions"])
    fixed_risk = float(np.sum(weights * fixed))
    candidate_risk = float(np.sum(weights * candidate))
    difference = fixed - candidate
    interval = stratified_bootstrap_interval(
        difference,
        endpoint["regions"],
        seed=bootstrap_seed,
    )
    return {
        "coefficient": coefficient,
        "fixed_support_mse": fixed_risk,
        "candidate_mse": candidate_risk,
        "relative_reduction": (
            (fixed_risk - candidate_risk) / fixed_risk
        ),
        "fixed_minus_candidate_ci95": list(interval),
        "positive_gain": candidate_risk < fixed_risk,
        "paired_lower_bound_positive": interval[0] > 0.0,
    }


def select_maximin(
    curve: list[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = [
        point
        for point in curve
        if point["coefficient"] > 0.0
        and all(
            metrics["positive_gain"]
            and metrics["paired_lower_bound_positive"]
            for metrics in point["endpoints"].values()
        )
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda point: (
            -point["minimum_relative_reduction"],
            point["coefficient"],
        ),
    )


def analyze(
    *,
    discoverphysics_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    replay = replay_clipped_components(
        discoverphysics_root=discoverphysics_root,
        repo_root=repo_root,
    )
    specs = {
        spec.name: spec
        for spec in default_endpoint_specs(repo_root)
    }
    curve = []
    for coefficient in COEFFICIENTS:
        endpoints = {
            name: coefficient_metrics(
                endpoint=endpoint,
                coefficient=coefficient,
                bootstrap_seed=specs[name].bootstrap_seed,
            )
            for name, endpoint in replay["endpoints"].items()
        }
        reductions = [
            metrics["relative_reduction"]
            for metrics in endpoints.values()
        ]
        curve.append(
            {
                "coefficient": coefficient,
                "endpoints": endpoints,
                "minimum_relative_reduction": float(
                    min(reductions)
                ),
                "mean_relative_reduction": float(
                    np.mean(reductions)
                ),
            }
        )
    selected = select_maximin(curve)
    reproduction_gate = all(
        endpoint["reproduction"]["fixed_max_abs_error"] <= 1e-10
        for endpoint in replay["endpoints"].values()
    )
    if selected is None:
        selected_gates = {
            "selected_coefficient_exists": False,
            "minimum_gain_at_least_half_percent": False,
            "mean_gain_at_least_1_percent": False,
            "one_full_tree_gain_at_least_1_percent": False,
        }
    else:
        full_tree_gains = [
            selected["endpoints"][name]["relative_reduction"]
            for name, endpoint in replay["endpoints"].items()
            if endpoint["independently_generated_full_tree"]
        ]
        selected_gates = {
            "selected_coefficient_exists": True,
            "minimum_gain_at_least_half_percent": (
                selected["minimum_relative_reduction"]
                >= MIN_TREE_REDUCTION
            ),
            "mean_gain_at_least_1_percent": (
                selected["mean_relative_reduction"]
                >= MIN_MEAN_REDUCTION
            ),
            "one_full_tree_gain_at_least_1_percent": (
                max(full_tree_gains) >= MIN_FULL_TREE_REDUCTION
            ),
        }
    gates = {
        "all_fixed_reproductions_within_1e-10": reproduction_gate,
        **selected_gates,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "development_pass" if all(gates.values()) else "development_null",
        "coefficients": list(COEFFICIENTS),
        "selection_rule": "maximin_relative_reduction_then_smaller_coefficient",
        "selected": selected,
        "curve": curve,
        "reproduction": {
            name: endpoint["reproduction"]
            for name, endpoint in replay["endpoints"].items()
        },
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
