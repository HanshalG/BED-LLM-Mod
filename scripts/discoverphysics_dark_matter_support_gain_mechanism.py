#!/usr/bin/env python3
"""Relate regenerated-support coverage to confirmed per-map policy gains."""

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
    LOOKAHEAD_ROOT_ID,
    compile_support,
    load_executor_class,
    root_by_id,
    simulate_maps,
)
from scripts.discoverphysics_dark_matter_opportunity import (
    verify_discoverphysics,
)
from scripts.discoverphysics_dark_matter_retained_support_confirmation import (
    confirmation_hidden_map_family,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-support-gain-mechanism-1"
SOURCE_CONFIRMATION_SHA256 = (
    "aecc11fc2b5efc1e7755137d727a58a7fd69d107ca19035b55d38015145050b9"
)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = _average_ranks(np.asarray(left, dtype=float))
    right_ranks = _average_ranks(np.asarray(right, dtype=float))
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def run_analysis(
    *,
    discoverphysics_root: Path,
    source_dir: Path,
    confirmation_path: Path,
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
    if _sha256(confirmation_path) != SOURCE_CONFIRMATION_SHA256:
        raise ValueError("source confirmation hash changed")

    frozen = json.loads(frozen_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    if confirmation["status"] != "confirmation_pass":
        raise ValueError("source confirmation is not a pass")
    executor_class = load_executor_class(discoverphysics_root)
    root_action = root_by_id(LOOKAHEAD_ROOT_ID)["action_id"]

    _, initial_heldout = simulate_maps(
        executor_class,
        compile_support(frozen["initial_support"]),
        action_ids=[],
    )
    refreshed_heldout = []
    for refresh in frozen["refreshes"][LOOKAHEAD_ROOT_ID]:
        _, heldout = simulate_maps(
            executor_class,
            compile_support(refresh["hypotheses"]),
            action_ids=[],
        )
        refreshed_heldout.append(heldout)

    hidden_maps, regions, hidden_prior = confirmation_hidden_map_family()
    hidden_action_means, hidden_heldout = simulate_maps(
        executor_class,
        hidden_maps,
        action_ids=[root_action],
    )
    centers = np.asarray(
        [
            branch["representative_final_coordinate"]
            for branch in frozen["branches"][LOOKAHEAD_ROOT_ID]
        ]
    )

    initial_nearest = np.empty(len(hidden_maps))
    refreshed_nearest = np.empty(len(hidden_maps))
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
        initial_nearest[index] = float(
            np.min(np.mean((initial_heldout - truth[None, :]) ** 2, axis=-1))
        )
        refreshed_nearest[index] = float(
            np.min(
                np.mean(
                    (
                        refreshed_heldout[branch_index]
                        - truth[None, :]
                    )
                    ** 2,
                    axis=-1,
                )
            )
        )
    union_nearest = np.minimum(initial_nearest, refreshed_nearest)
    coverage_gain = initial_nearest - union_nearest

    endpoint = confirmation["fresh_hidden_endpoint"]
    retained_mse = np.asarray(
        endpoint["per_map_retained_mse"][LOOKAHEAD_ROOT_ID],
    )
    fixed_mse = np.asarray(endpoint["per_map_fixed_support_mse"])
    endpoint_gain = fixed_mse - retained_mse
    refresh_helped = coverage_gain > 0.0
    region_array = np.asarray(regions)

    region_summary = {}
    for region in ("NE", "NW", "SW", "SE"):
        mask = region_array == region
        region_summary[region] = {
            "num_maps": int(mask.sum()),
            "coverage_help_rate": float(np.mean(refresh_helped[mask])),
            "mean_coverage_gain": float(np.mean(coverage_gain[mask])),
            "mean_endpoint_gain": float(np.mean(endpoint_gain[mask])),
            "spearman_coverage_endpoint": spearman_correlation(
                coverage_gain[mask],
                endpoint_gain[mask],
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "posthoc_mechanism_analysis",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "discoverphysics_commit": commit,
            "source_policy_sha256": SOURCE_POLICY_SHA256,
            "source_model_frozen_sha256": SOURCE_MODEL_FROZEN_SHA256,
            "source_confirmation_sha256": SOURCE_CONFIRMATION_SHA256,
            "num_maps": len(hidden_maps),
            "endpoint_was_previously_opened": True,
            "new_model_calls": 0,
            "openrouter_cost_usd": 0.0,
        },
        "aggregate": {
            "coverage_help_rate": float(np.mean(refresh_helped)),
            "weighted_coverage_gain": float(
                np.sum(coverage_gain * hidden_prior)
            ),
            "weighted_endpoint_gain": float(
                np.sum(endpoint_gain * hidden_prior)
            ),
            "spearman_coverage_endpoint": spearman_correlation(
                coverage_gain,
                endpoint_gain,
            ),
            "mean_endpoint_gain_when_coverage_helped": _mean_or_zero(
                endpoint_gain[refresh_helped],
            ),
            "mean_endpoint_gain_when_coverage_did_not_help": _mean_or_zero(
                endpoint_gain[~refresh_helped],
            ),
            "endpoint_win_rate_when_coverage_helped": float(
                np.mean(endpoint_gain[refresh_helped] > 0.0)
            ),
            "endpoint_win_rate_when_coverage_did_not_help": float(
                np.mean(endpoint_gain[~refresh_helped] > 0.0)
            ),
        },
        "by_region": region_summary,
        "per_map": {
            "regions": regions,
            "initial_nearest_mse": initial_nearest.tolist(),
            "refreshed_nearest_mse": refreshed_nearest.tolist(),
            "union_coverage_gain": coverage_gain.tolist(),
            "fixed_minus_retained_endpoint_gain": endpoint_gain.tolist(),
        },
        "authorizes_claim": False,
        "new_model_calls": 0,
        "openrouter_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--confirmation-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    payload = run_analysis(
        discoverphysics_root=args.discoverphysics_root.resolve(),
        source_dir=args.source_dir.resolve(),
        confirmation_path=args.confirmation_path.resolve(),
    )
    checkpoint(args.output_path.resolve(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "aggregate": payload["aggregate"],
                "by_region": payload["by_region"],
                "authorizes_claim": payload["authorizes_claim"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
