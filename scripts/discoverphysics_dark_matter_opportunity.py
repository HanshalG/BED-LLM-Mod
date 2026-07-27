#!/usr/bin/env python3
"""Test for a non-myopic opportunity in DiscoverPhysics dark matter."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


DISCOVERPHYSICS_COMMIT = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-opportunity-1"
WORLD = "dark_matter"
FAMILY_SEED = 24493
DECISION_NOISE_SEED = 24494
CONFIRMATION_NOISE_SEED = 24495
NUM_REGIONS = 4
NUM_VARIANTS = 6
NUM_HYPOTHESES = NUM_REGIONS * NUM_VARIANTS
OBSERVATION_TIME = 0.5
OBSERVATION_NOISE_STD = 0.075
ROOT_SAMPLES_PER_HYPOTHESIS = 12
CONTINUATION_SAMPLES_PER_HYPOTHESIS = 4
MIN_IMMEDIATE_SACRIFICE_NATS = 0.03
MIN_TOTAL_EIG_GAIN_NATS = 0.03
MIN_HELDOUT_RISK_REDUCTION = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_discoverphysics(root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError(
            f"DiscoverPhysics is at {commit}, expected "
            f"{DISCOVERPHYSICS_COMMIT}"
        )
    return commit


def load_executor_class(root: Path) -> type:
    scienceagent_root = root / "ScienceAgent"
    physics_school_root = root / "PhysicsSchool"
    for path in (scienceagent_root, physics_school_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module("scienceagent.executor")
    return module.NBodyDarkMatterExecutor


def hidden_halo_family(seed: int = FAMILY_SEED) -> tuple[np.ndarray, list[str]]:
    """Return a frozen hierarchical family of 24 hidden source maps."""
    rng = np.random.default_rng(seed)
    centres = np.array(
        [[4.0, 4.0], [-4.0, 4.0], [-4.0, -4.0], [4.0, -4.0]]
    )
    maps: list[np.ndarray] = []
    labels: list[str] = []
    line = np.linspace(-1.1, 1.1, 10)
    phase = np.sin(np.arange(10) * 1.7)
    for region_index, centre in enumerate(centres):
        radial = centre / np.linalg.norm(centre)
        tangential = np.array([-radial[1], radial[0]])
        for variant_index in range(NUM_VARIANTS):
            offset = (
                ((variant_index % 3) - 1) * 1.15 * radial
                + ((variant_index // 3) - 0.5) * 1.2 * tangential
            )
            orientation = radial if variant_index % 2 == 0 else tangential
            perpendicular = np.array([-orientation[1], orientation[0]])
            jitter = rng.normal(0.0, 0.035, size=(10, 2))
            halo = (
                centre
                + offset
                + np.outer(line, orientation)
                + 0.18 * np.outer(phase, perpendicular)
                + jitter
            )
            maps.append(halo)
            labels.append(f"region_{region_index}_variant_{variant_index}")
    return np.asarray(maps), labels


def active_probe_actions() -> tuple[np.ndarray, list[str]]:
    """Return the frozen one-active-probe action bank."""
    points = [np.array([0.0, 0.0])]
    labels = ["center"]
    for radius in (2.5, 4.5, 6.5):
        for angle_index, angle in enumerate(np.arange(8) * math.pi / 4):
            points.append(
                radius * np.array([math.cos(angle), math.sin(angle)])
            )
            labels.append(f"r{radius:g}_a{angle_index}")
    return np.asarray(points), labels


def heldout_experiments() -> list[dict[str, Any]]:
    times = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    return [
        {
            "probe_positions": [
                [2.0, 0.0],
                [0.0, 4.0],
                [-6.0, 0.0],
                [0.0, -8.0],
                [7.0, 7.0],
            ],
            "probe_velocities": [[0.0, 0.0]] * 5,
            "measurement_times": times,
        },
        {
            "probe_positions": [
                [3.5, 3.5],
                [-5.0, 5.0],
                [-6.5, -6.5],
                [8.0, -8.0],
                [0.0, 10.0],
            ],
            "probe_velocities": [
                [0.0, 0.35],
                [-0.35, 0.0],
                [0.25, -0.25],
                [0.0, -0.35],
                [0.3, 0.0],
            ],
            "measurement_times": times,
        },
    ]


def _run_to_times(
    executor: Any,
    *,
    probe_positions: np.ndarray,
    probe_velocities: np.ndarray,
    measurement_times: list[float],
) -> np.ndarray:
    sim, centre = executor._build_sim(
        probe_positions,
        probe_velocities,
        1.0,
    )
    last_time = max(measurement_times)
    n_steps = int(round(last_time / executor.dt))
    trajectory = sim.run(n_steps=n_steps, record_every=1)
    positions = np.asarray(trajectory["positions"])
    probe_indices = np.asarray(executor.PROBES)
    selected = []
    for measurement_time in measurement_times:
        index = int(round(measurement_time / executor.dt))
        selected.append(positions[index, probe_indices] - centre)
    return np.asarray(selected)


def simulator_feature_matrices(
    executor_class: type,
    halo_maps: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    executor = executor_class(noise_std=0.0)
    sentinels = np.array(
        [[-20.0, -20.0], [-20.0, 20.0], [20.0, -20.0], [20.0, 20.0]]
    )
    zero_velocities = np.zeros((5, 2))
    observation_means = np.empty((len(actions), len(halo_maps), 2))
    heldout_features: list[np.ndarray] = []

    for hypothesis_index, halo in enumerate(halo_maps):
        executor._dark_positions_rel = np.asarray(halo)
        for action_index, active_position in enumerate(actions):
            probe_positions = np.vstack([active_position, sentinels])
            result = _run_to_times(
                executor,
                probe_positions=probe_positions,
                probe_velocities=zero_velocities,
                measurement_times=[OBSERVATION_TIME],
            )
            observation_means[action_index, hypothesis_index] = result[0, 0]

        hypothesis_heldout = []
        for experiment in heldout_experiments():
            result = _run_to_times(
                executor,
                probe_positions=np.asarray(experiment["probe_positions"]),
                probe_velocities=np.asarray(
                    experiment["probe_velocities"]
                ),
                measurement_times=experiment["measurement_times"],
            )
            hypothesis_heldout.append(result.reshape(-1))
        heldout_features.append(np.concatenate(hypothesis_heldout))

    return observation_means, np.asarray(heldout_features)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            probabilities > 0.0,
            probabilities * np.log(probabilities),
            0.0,
        )
    return -np.sum(terms, axis=-1)


def posterior_batch(
    priors: np.ndarray,
    observations: np.ndarray,
    means: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    log_likelihoods = (
        -0.5
        * np.sum(
            (observations[:, None, :] - means[None, :, :]) ** 2,
            axis=-1,
        )
        / noise_std**2
    )
    with np.errstate(divide="ignore"):
        logits = np.log(priors[:, None, :]) + log_likelihoods[None, :, :]
    logits -= np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def evaluate_roots(
    observation_means: np.ndarray,
    heldout_features: np.ndarray,
    *,
    noise_seed: int,
    prior: np.ndarray | None = None,
    root_samples_per_hypothesis: int = ROOT_SAMPLES_PER_HYPOTHESIS,
    continuation_samples_per_hypothesis: int = (
        CONTINUATION_SAMPLES_PER_HYPOTHESIS
    ),
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(noise_seed)
    num_actions, num_hypotheses, feature_size = observation_means.shape
    if prior is None:
        prior = np.full(num_hypotheses, 1.0 / num_hypotheses)
    else:
        prior = np.asarray(prior, dtype=float)
        if prior.shape != (num_hypotheses,):
            raise ValueError("prior has the wrong shape")
        if np.any(prior <= 0.0) or not np.isclose(prior.sum(), 1.0):
            raise ValueError("prior must be positive and sum to one")
    prior_entropy = entropy(prior)
    root_noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(
            num_actions,
            num_hypotheses,
            root_samples_per_hypothesis,
            feature_size,
        ),
    )
    continuation_noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(
            num_actions,
            num_hypotheses,
            continuation_samples_per_hypothesis,
            feature_size,
        ),
    )
    immediate_eig = np.empty(num_actions)
    total_eig = np.empty(num_actions)
    heldout_risk = np.empty(num_actions)

    for root_index in range(num_actions):
        root_observations = (
            observation_means[root_index, :, None, :]
            + root_noise[root_index]
        ).reshape(-1, feature_size)
        root_posteriors = posterior_batch(
            prior[None, :],
            root_observations,
            observation_means[root_index],
            OBSERVATION_NOISE_STD,
        )[0]
        root_weights = np.repeat(
            prior / root_samples_per_hypothesis,
            root_samples_per_hypothesis,
        )
        immediate_eig[root_index] = prior_entropy - np.sum(
            root_weights * entropy(root_posteriors)
        )

        expected_entropies = np.empty(
            (num_actions, len(root_posteriors))
        )
        expected_risks = np.empty_like(expected_entropies)
        for continuation_index in range(num_actions):
            continuation_observations = (
                observation_means[continuation_index, :, None, :]
                + continuation_noise[continuation_index]
            ).reshape(-1, feature_size)
            final_posteriors = posterior_batch(
                root_posteriors,
                continuation_observations,
                observation_means[continuation_index],
                OBSERVATION_NOISE_STD,
            )
            outcome_weights = np.repeat(
                root_posteriors / continuation_samples_per_hypothesis,
                continuation_samples_per_hypothesis,
                axis=1,
            )
            expected_entropies[continuation_index] = np.sum(
                outcome_weights * entropy(final_posteriors),
                axis=1,
            )
            predictions = np.einsum(
                "bnh,hf->bnf",
                final_posteriors,
                heldout_features,
            )
            truths = np.repeat(
                heldout_features[:, None, :],
                continuation_samples_per_hypothesis,
                axis=1,
            ).reshape(
                num_hypotheses * continuation_samples_per_hypothesis,
                -1,
            )
            squared_error = np.mean(
                (predictions - truths[None, :, :]) ** 2,
                axis=-1,
            )
            expected_risks[continuation_index] = np.sum(
                outcome_weights * squared_error,
                axis=1,
            )

        best_continuations = np.argmin(expected_entropies, axis=0)
        episode_indices = np.arange(len(root_posteriors))
        total_eig[root_index] = prior_entropy - np.sum(
            root_weights
            * expected_entropies[best_continuations, episode_indices]
        )
        heldout_risk[root_index] = np.sum(
            root_weights
            * expected_risks[best_continuations, episode_indices]
        )

    return {
        "immediate_eig_nats": immediate_eig,
        "total_eig_nats": total_eig,
        "heldout_trajectory_mse": heldout_risk,
    }


def summarize_evaluation(
    values: dict[str, np.ndarray],
    action_labels: list[str],
) -> dict[str, Any]:
    myopic_index = int(np.argmax(values["immediate_eig_nats"]))
    depth_two_index = int(np.argmax(values["total_eig_nats"]))
    myopic_risk = float(values["heldout_trajectory_mse"][myopic_index])
    depth_two_risk = float(values["heldout_trajectory_mse"][depth_two_index])
    return {
        "myopic_root_index": myopic_index,
        "myopic_root": action_labels[myopic_index],
        "depth_two_root_index": depth_two_index,
        "depth_two_root": action_labels[depth_two_index],
        "myopic_immediate_eig_nats": float(
            values["immediate_eig_nats"][myopic_index]
        ),
        "depth_two_immediate_eig_nats": float(
            values["immediate_eig_nats"][depth_two_index]
        ),
        "immediate_sacrifice_nats": float(
            values["immediate_eig_nats"][myopic_index]
            - values["immediate_eig_nats"][depth_two_index]
        ),
        "myopic_total_eig_nats": float(
            values["total_eig_nats"][myopic_index]
        ),
        "depth_two_total_eig_nats": float(
            values["total_eig_nats"][depth_two_index]
        ),
        "total_eig_gain_nats": float(
            values["total_eig_nats"][depth_two_index]
            - values["total_eig_nats"][myopic_index]
        ),
        "myopic_heldout_trajectory_mse": myopic_risk,
        "depth_two_heldout_trajectory_mse": depth_two_risk,
        "heldout_risk_reduction": (
            (myopic_risk - depth_two_risk) / myopic_risk
            if myopic_risk > 0.0
            else 0.0
        ),
    }


def run_opportunity(
    *,
    discoverphysics_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    commit = verify_discoverphysics(discoverphysics_root)
    executor_class = load_executor_class(discoverphysics_root)
    halo_maps, hypothesis_labels = hidden_halo_family()
    actions, action_labels = active_probe_actions()
    observation_means, heldout_features = simulator_feature_matrices(
        executor_class,
        halo_maps,
        actions,
    )
    decision_values = evaluate_roots(
        observation_means,
        heldout_features,
        noise_seed=DECISION_NOISE_SEED,
    )
    confirmation_values = evaluate_roots(
        observation_means,
        heldout_features,
        noise_seed=CONFIRMATION_NOISE_SEED,
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
            "num_hypotheses": NUM_HYPOTHESES,
            "num_actions": len(actions),
            "observation_time": OBSERVATION_TIME,
            "observation_noise_std": OBSERVATION_NOISE_STD,
            "root_samples_per_hypothesis": ROOT_SAMPLES_PER_HYPOTHESIS,
            "continuation_samples_per_hypothesis": (
                CONTINUATION_SAMPLES_PER_HYPOTHESIS
            ),
            "minimum_immediate_sacrifice_nats": (
                MIN_IMMEDIATE_SACRIFICE_NATS
            ),
            "minimum_total_eig_gain_nats": MIN_TOTAL_EIG_GAIN_NATS,
            "minimum_heldout_risk_reduction": MIN_HELDOUT_RISK_REDUCTION,
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
