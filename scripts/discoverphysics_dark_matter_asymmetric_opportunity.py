#!/usr/bin/env python3
"""Confirm a dark-matter scout advantage under an asymmetric spatial prior."""

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

from scripts.discoverphysics_dark_matter_opportunity import (
    DISCOVERPHYSICS_COMMIT,
    MIN_HELDOUT_RISK_REDUCTION,
    MIN_IMMEDIATE_SACRIFICE_NATS,
    MIN_TOTAL_EIG_GAIN_NATS,
    NUM_HYPOTHESES,
    NUM_VARIANTS,
    OBSERVATION_NOISE_STD,
    OBSERVATION_TIME,
    WORLD,
    active_probe_actions,
    evaluate_roots,
    hidden_halo_family,
    load_executor_class,
    sha256_file,
    simulator_feature_matrices,
    summarize_evaluation,
    verify_discoverphysics,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-asymmetric-opportunity-1"
FAMILY_SEED = 24503
DECISION_NOISE_SEED = 24504
CONFIRMATION_NOISE_SEED = 24505
REGION_PRIOR = np.array([0.40, 0.30, 0.20, 0.10])
ROOT_SAMPLES_PER_HYPOTHESIS = 24
CONTINUATION_SAMPLES_PER_HYPOTHESIS = 8


def hypothesis_prior() -> np.ndarray:
    return np.repeat(REGION_PRIOR / NUM_VARIANTS, NUM_VARIANTS)


def run_opportunity(
    *,
    discoverphysics_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    commit = verify_discoverphysics(discoverphysics_root)
    executor_class = load_executor_class(discoverphysics_root)
    halo_maps, hypothesis_labels = hidden_halo_family(seed=FAMILY_SEED)
    actions, action_labels = active_probe_actions()
    observation_means, heldout_features = simulator_feature_matrices(
        executor_class,
        halo_maps,
        actions,
    )
    prior = hypothesis_prior()
    decision_values = evaluate_roots(
        observation_means,
        heldout_features,
        noise_seed=DECISION_NOISE_SEED,
        prior=prior,
        root_samples_per_hypothesis=ROOT_SAMPLES_PER_HYPOTHESIS,
        continuation_samples_per_hypothesis=(
            CONTINUATION_SAMPLES_PER_HYPOTHESIS
        ),
    )
    confirmation_values = evaluate_roots(
        observation_means,
        heldout_features,
        noise_seed=CONFIRMATION_NOISE_SEED,
        prior=prior,
        root_samples_per_hypothesis=ROOT_SAMPLES_PER_HYPOTHESIS,
        continuation_samples_per_hypothesis=(
            CONTINUATION_SAMPLES_PER_HYPOTHESIS
        ),
    )
    decision = summarize_evaluation(decision_values, action_labels)
    confirmation = summarize_evaluation(
        confirmation_values,
        action_labels,
    )
    gates = {
        "decision_roots_differ": (
            decision["myopic_root_index"]
            != decision["depth_two_root_index"]
        ),
        "confirmation_roots_differ": (
            confirmation["myopic_root_index"]
            != confirmation["depth_two_root_index"]
        ),
        "root_selection_replicates": (
            decision["myopic_root_index"]
            == confirmation["myopic_root_index"]
            and decision["depth_two_root_index"]
            == confirmation["depth_two_root_index"]
        ),
        "depth_two_root_is_center_scout": (
            decision["depth_two_root"] == "center"
            and confirmation["depth_two_root"] == "center"
        ),
        "immediate_sacrifice_at_least_0_03": (
            confirmation["immediate_sacrifice_nats"]
            >= MIN_IMMEDIATE_SACRIFICE_NATS
        ),
        "total_eig_gain_at_least_0_03": (
            confirmation["total_eig_gain_nats"]
            >= MIN_TOTAL_EIG_GAIN_NATS
        ),
        "heldout_risk_reduction_at_least_10_percent": (
            confirmation["heldout_risk_reduction"]
            >= MIN_HELDOUT_RISK_REDUCTION
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "discoverphysics_commit": commit,
            "world": WORLD,
            "family_seed": FAMILY_SEED,
            "decision_noise_seed": DECISION_NOISE_SEED,
            "confirmation_noise_seed": CONFIRMATION_NOISE_SEED,
            "region_prior": REGION_PRIOR.tolist(),
            "hypothesis_prior": prior.tolist(),
            "num_hypotheses": NUM_HYPOTHESES,
            "num_actions": len(actions),
            "observation_time": OBSERVATION_TIME,
            "observation_noise_std": OBSERVATION_NOISE_STD,
            "root_samples_per_hypothesis": (
                ROOT_SAMPLES_PER_HYPOTHESIS
            ),
            "continuation_samples_per_hypothesis": (
                CONTINUATION_SAMPLES_PER_HYPOTHESIS
            ),
            "minimum_immediate_sacrifice_nats": (
                MIN_IMMEDIATE_SACRIFICE_NATS
            ),
            "minimum_total_eig_gain_nats": MIN_TOTAL_EIG_GAIN_NATS,
            "minimum_heldout_risk_reduction": (
                MIN_HELDOUT_RISK_REDUCTION
            ),
            "single_active_probe": True,
            "action_labels": action_labels,
            "hypothesis_labels": hypothesis_labels,
        },
        "decision": decision,
        "confirmation": confirmation,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "llm_requests": 0,
        "openrouter_cost_usd": 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "FEATURES.npz"
    np.savez_compressed(
        feature_path,
        observation_means=observation_means,
        heldout_features=heldout_features,
        halo_maps=halo_maps,
        actions=actions,
        prior=prior,
    )
    payload["features_sha256"] = sha256_file(feature_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run_opportunity(
        discoverphysics_root=args.discoverphysics_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    output_path = args.output_dir / "OPPORTUNITY.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "confirmation": payload["confirmation"],
                "gates": payload["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
