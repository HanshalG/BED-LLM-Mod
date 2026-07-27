#!/usr/bin/env python3
"""Post-hoc retained-support mixture curve on the open hidden endpoint."""

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
    HIDDEN_CONTINUATION_SAMPLES,
    HIDDEN_MAP_SEEDS,
    HIDDEN_NOISE_SEED,
    HIDDEN_ROOT_SAMPLES,
    LOOKAHEAD_ROOT_ID,
    action_table,
    compile_support,
    evaluate_fixed_support_root,
    hidden_map_family,
    load_executor_class,
    root_by_id,
    simulate_maps,
    support_prior,
    weighted_mean,
)
from scripts.discoverphysics_dark_matter_opportunity import (
    verify_discoverphysics,
)
from scripts.discoverphysics_dark_matter_retained_support_replay import (
    evaluate_retained_root,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-mixture-curve-1"
REFRESH_MASSES = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))


def _regional_means(
    values: np.ndarray,
    regions: list[str],
) -> dict[str, float]:
    region_array = np.asarray(regions)
    return {
        region: float(np.mean(values[region_array == region]))
        for region in ("NE", "NW", "SW", "SE")
    }


def run_curve(
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
    root = root_by_id(LOOKAHEAD_ROOT_ID)

    initial_maps = compile_support(initial_support)
    initial_prior = support_prior(initial_support)
    initial_action_means, initial_heldout = simulate_maps(
        executor_class,
        initial_maps,
        action_ids=list(action_table()),
    )
    refresh_models = {LOOKAHEAD_ROOT_ID: []}
    for refresh in refreshes[LOOKAHEAD_ROOT_ID]:
        maps = compile_support(refresh["hypotheses"])
        means, heldout = simulate_maps(
            executor_class,
            maps,
            action_ids=[
                root["action_id"],
                refresh["continuation_action"],
            ],
        )
        refresh_models[LOOKAHEAD_ROOT_ID].append(
            {
                **refresh,
                "prior": support_prior(refresh["hypotheses"]),
                "root_means": means[root["action_id"]],
                "continuation_means": means[refresh["continuation_action"]],
                "heldout": heldout,
            }
        )

    hidden_maps, hidden_regions, hidden_prior = hidden_map_family()
    required_actions = {
        root["action_id"],
        *(
            refresh["continuation_action"]
            for refresh in refreshes[LOOKAHEAD_ROOT_ID]
        ),
    }
    hidden_action_means, hidden_heldout = simulate_maps(
        executor_class,
        hidden_maps,
        action_ids=sorted(required_actions),
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
    fixed_risk = weighted_mean(fixed_per_map, hidden_prior)

    curve = []
    for refresh_mass in REFRESH_MASSES:
        per_map = evaluate_retained_root(
            root_id=LOOKAHEAD_ROOT_ID,
            true_action_means=hidden_action_means,
            true_heldout=hidden_heldout,
            initial_action_means=initial_action_means,
            initial_heldout=initial_heldout,
            branches=branches,
            refresh_models=refresh_models,
            rng=np.random.default_rng(HIDDEN_NOISE_SEED),
            root_samples=HIDDEN_ROOT_SAMPLES,
            continuation_samples=HIDDEN_CONTINUATION_SAMPLES,
            initial_component_mass=1.0 - refresh_mass,
            refresh_component_mass=refresh_mass,
        )
        risk = weighted_mean(per_map, hidden_prior)
        curve.append(
            {
                "refresh_mass": refresh_mass,
                "initial_mass": 1.0 - refresh_mass,
                "trajectory_mse": risk,
                "relative_improvement_vs_fixed": (
                    (fixed_risk - risk) / fixed_risk
                ),
                "regional_mse": _regional_means(per_map, hidden_regions),
                "per_map_mse": per_map.tolist(),
            }
        )
    best = min(
        curve,
        key=lambda record: (
            record["trajectory_mse"],
            record["refresh_mass"],
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "posthoc_development_only",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "discoverphysics_commit": commit,
            "source_policy_sha256": SOURCE_POLICY_SHA256,
            "source_model_frozen_sha256": SOURCE_MODEL_FROZEN_SHA256,
            "hidden_map_seeds": list(HIDDEN_MAP_SEEDS),
            "hidden_noise_seed": HIDDEN_NOISE_SEED,
            "refresh_mass_grid": list(REFRESH_MASSES),
            "endpoint_was_previously_opened": True,
            "new_model_calls": 0,
            "openrouter_cost_usd": 0.0,
        },
        "fixed_support_trajectory_mse": fixed_risk,
        "fixed_support_regional_mse": _regional_means(
            fixed_per_map,
            hidden_regions,
        ),
        "curve": curve,
        "best_reused_endpoint_point": best,
        "authorizes_claim": False,
        "new_model_calls": 0,
        "openrouter_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    payload = run_curve(
        discoverphysics_root=args.discoverphysics_root.resolve(),
        source_dir=args.source_dir.resolve(),
    )
    checkpoint(args.output_path.resolve(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "fixed_support_trajectory_mse": payload[
                    "fixed_support_trajectory_mse"
                ],
                "best_reused_endpoint_point": payload[
                    "best_reused_endpoint_point"
                ],
                "authorizes_claim": payload["authorizes_claim"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
