#!/usr/bin/env python3
"""Zero-call structural search for adaptive force-law experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import special


INTERFACE_VERSION = "discoverphysics-extra-dimensions-opportunity-dev-1"
DISCOVERPHYSICS_COMMIT = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
OBSERVATION_NOISE_STD = 0.075
GROUP_WEIGHTS = np.array([0.40, 0.35, 0.25], dtype=float)
KK_RADII = np.array([0.18, 0.25, 0.35, 0.50, 0.72, 1.05])
YUKAWA_LENGTHS = np.array([0.55, 0.80, 1.15, 1.70, 2.70, 4.50])
RIESZ_ALPHAS = np.array([0.30, 0.42, 0.54, 0.66, 0.78, 0.90])
ACTION_POOL = np.array([0.35, 0.50, 0.75, 1.10, 1.60, 2.40, 3.60, 5.50, 8.00])


@dataclass(frozen=True)
class Candidate:
    calibration_radii: tuple[float, float, float]
    group_offsets: tuple[float, float, float]
    measurement_scale: float
    action_radii: tuple[float, float, float, float]


def entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.maximum(probabilities, 1e-300)
    return -np.sum(np.where(probabilities > 0.0, probabilities * np.log(safe), 0.0), axis=-1)


def _kk_force(radii: np.ndarray, compact_radius: float) -> np.ndarray:
    circumference = 2.0 * math.pi * compact_radius
    image_indices = np.arange(-60, 61, dtype=float)
    image_offsets = image_indices * circumference
    expanded = radii[:, None]
    geometry = np.sum(
        expanded / (expanded**2 + image_offsets**2) ** 1.5,
        axis=1,
    )
    return circumference * geometry / (4.0 * math.pi)


def _yukawa_force(radii: np.ndarray, screening_length: float) -> np.ndarray:
    return (
        special.k1(radii / screening_length)
        / (2.0 * math.pi * screening_length)
    )


def _riesz_force(radii: np.ndarray, alpha: float) -> np.ndarray:
    prefactor = special.gamma(1.0 - alpha) / (
        2.0 ** (2.0 * alpha) * math.pi * special.gamma(alpha)
    )
    return prefactor * (2.0 - 2.0 * alpha) / radii ** (3.0 - 2.0 * alpha)


def raw_force_groups(radii: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray([_kk_force(radii, value) for value in KK_RADII]),
        np.asarray([_yukawa_force(radii, value) for value in YUKAWA_LENGTHS]),
        np.asarray([_riesz_force(radii, value) for value in RIESZ_ALPHAS]),
    ]


def candidate_force_curves(
    candidate: Candidate,
    radii: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return calibrated official-kernel curves, labels, and prior."""
    groups = raw_force_groups(radii)
    poisson = 1.0 / (2.0 * math.pi * radii)
    curves: list[np.ndarray] = []
    labels: list[str] = []
    names_and_values = (
        ("kk", KK_RADII),
        ("yukawa", YUKAWA_LENGTHS),
        ("riesz", RIESZ_ALPHAS),
    )
    for group_index, (name, values) in enumerate(names_and_values):
        calibration_radius = candidate.calibration_radii[group_index]
        target = np.interp(calibration_radius, radii, poisson)
        target *= candidate.group_offsets[group_index]
        for value, raw_curve in zip(values, groups[group_index], strict=True):
            calibration_value = np.interp(
                calibration_radius,
                radii,
                raw_curve,
            )
            curves.append(raw_curve * target / calibration_value)
            labels.append(f"{name}_{value:g}")
    prior = np.repeat(GROUP_WEIGHTS / len(KK_RADII), len(KK_RADII))
    return np.asarray(curves), labels, prior


def action_means(
    candidate: Candidate,
    curves: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    return candidate.measurement_scale * np.asarray(
        [
            [np.log(np.interp(radius, radii, curve)) for curve in curves]
            for radius in candidate.action_radii
        ]
    )


def evaluate_scalar_policy(
    means: np.ndarray,
    prior: np.ndarray,
    *,
    quadrature_points: int = 8,
    heldout_features: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate one-step and adaptive two-step EIG by Gauss-Hermite quadrature."""
    nodes, weights = hermgauss(quadrature_points)
    noise = np.sqrt(2.0) * OBSERVATION_NOISE_STD * nodes
    quadrature_weights = weights / np.sqrt(math.pi)
    num_actions, num_hypotheses = means.shape
    prior_entropy = float(entropy(prior))
    outcome_weights = (
        prior[:, None] * quadrature_weights[None, :]
    ).reshape(-1)
    immediate_eig = np.empty(num_actions)
    total_eig = np.empty(num_actions)
    heldout_risk = (
        np.empty(num_actions) if heldout_features is not None else None
    )
    if heldout_features is not None:
        heldout_features = np.asarray(heldout_features, dtype=float)
        if heldout_features.shape[0] != num_hypotheses:
            raise ValueError("heldout_features must have one row per hypothesis")

    for root_index in range(num_actions):
        root_observations = (
            means[root_index, :, None] + noise[None, :]
        ).reshape(-1)
        root_logits = np.log(prior)[None, :] - 0.5 * (
            (
                root_observations[:, None]
                - means[root_index, None, :]
            )
            / OBSERVATION_NOISE_STD
        ) ** 2
        root_logits -= np.max(root_logits, axis=1, keepdims=True)
        root_posteriors = np.exp(root_logits)
        root_posteriors /= root_posteriors.sum(axis=1, keepdims=True)
        immediate_eig[root_index] = prior_entropy - np.sum(
            outcome_weights * entropy(root_posteriors)
        )

        best_final_entropy = np.full(len(root_observations), np.inf)
        best_final_risk = np.zeros(len(root_observations))
        for continuation_index in range(num_actions):
            continuation_observations = (
                means[continuation_index, :, None] + noise[None, :]
            )
            continuation_log_likelihood = -0.5 * (
                (
                    continuation_observations[:, :, None]
                    - means[continuation_index, None, None, :]
                )
                / OBSERVATION_NOISE_STD
            ) ** 2
            final_logits = (
                np.log(np.maximum(root_posteriors, 1e-300))[:, None, None, :]
                + continuation_log_likelihood[None, :, :, :]
            )
            final_logits -= np.max(final_logits, axis=3, keepdims=True)
            final_posteriors = np.exp(final_logits)
            final_posteriors /= final_posteriors.sum(axis=3, keepdims=True)
            conditional_weights = (
                root_posteriors[:, :, None]
                * quadrature_weights[None, None, :]
            )
            expected_entropy = np.sum(
                conditional_weights * entropy(final_posteriors),
                axis=(1, 2),
            )
            take = expected_entropy < best_final_entropy
            best_final_entropy[take] = expected_entropy[take]
            if heldout_features is not None:
                predictions = np.einsum(
                    "eoqh,hf->eoqf",
                    final_posteriors,
                    heldout_features,
                )
                squared_error = np.mean(
                    (
                        predictions
                        - heldout_features[None, :, None, :]
                    )
                    ** 2,
                    axis=-1,
                )
                expected_risk = np.sum(
                    conditional_weights * squared_error,
                    axis=(1, 2),
                )
                best_final_risk[take] = expected_risk[take]
        total_eig[root_index] = prior_entropy - np.sum(
            outcome_weights * best_final_entropy
        )
        if heldout_risk is not None:
            heldout_risk[root_index] = np.sum(
                outcome_weights * best_final_risk
            )

    result = {
        "immediate_eig_nats": immediate_eig,
        "total_eig_nats": total_eig,
    }
    if heldout_risk is not None:
        result["heldout_prediction_mse"] = heldout_risk
    return result


def candidate_summary(
    candidate: Candidate,
    *,
    quadrature_points: int,
    include_heldout_risk: bool = True,
) -> dict[str, Any]:
    radii = np.geomspace(0.25, 9.0, 120)
    curves, labels, prior = candidate_force_curves(candidate, radii)
    means = action_means(candidate, curves, radii)
    values = evaluate_scalar_policy(
        means,
        prior,
        quadrature_points=quadrature_points,
        heldout_features=(
            candidate.measurement_scale * np.log(curves)
            if include_heldout_risk
            else None
        ),
    )
    myopic_index = int(np.argmax(values["immediate_eig_nats"]))
    depth_two_index = int(np.argmax(values["total_eig_nats"]))
    immediate_sacrifice = float(
        values["immediate_eig_nats"][myopic_index]
        - values["immediate_eig_nats"][depth_two_index]
    )
    total_gain = float(
        values["total_eig_nats"][depth_two_index]
        - values["total_eig_nats"][myopic_index]
    )
    return {
        "candidate": asdict(candidate),
        "hypothesis_labels": labels,
        "prior": prior.tolist(),
        "myopic_index": myopic_index,
        "depth_two_index": depth_two_index,
        "myopic_radius": candidate.action_radii[myopic_index],
        "depth_two_radius": candidate.action_radii[depth_two_index],
        "immediate_sacrifice_nats": immediate_sacrifice,
        "total_eig_gain_nats": total_gain,
        **{key: value.tolist() for key, value in values.items()},
    }


def search_candidates(
    *,
    seed: int,
    num_trials: int,
    quadrature_points: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    passing: list[dict[str, Any]] = []
    for _ in range(num_trials):
        candidate = Candidate(
            calibration_radii=tuple(
                float(value)
                for value in rng.choice(ACTION_POOL, size=3, replace=True)
            ),
            group_offsets=tuple(
                float(value)
                for value in np.exp(rng.normal(0.0, 0.25, size=3))
            ),
            measurement_scale=float(rng.uniform(0.08, 0.60)),
            action_radii=tuple(
                float(value)
                for value in np.sort(
                    rng.choice(ACTION_POOL, size=4, replace=False)
                )
            ),
        )
        summary = candidate_summary(
            candidate,
            quadrature_points=quadrature_points,
            include_heldout_risk=False,
        )
        if (
            summary["myopic_index"] != summary["depth_two_index"]
            and summary["immediate_sacrifice_nats"] >= 0.03
            and summary["total_eig_gain_nats"] >= 0.03
        ):
            passing.append(summary)
    passing.sort(
        key=lambda item: (
            item["total_eig_gain_nats"],
            item["immediate_sacrifice_nats"],
        ),
        reverse=True,
    )
    return passing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=24670)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--quadrature-points", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "interface_version": INTERFACE_VERSION,
        "discoverphysics_commit": DISCOVERPHYSICS_COMMIT,
        "seed": args.seed,
        "num_trials": args.num_trials,
        "quadrature_points": args.quadrature_points,
        "passing": search_candidates(
            seed=args.seed,
            num_trials=args.num_trials,
            quadrature_points=args.quadrature_points,
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
