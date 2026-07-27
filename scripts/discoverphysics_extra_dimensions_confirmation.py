#!/usr/bin/env python3
"""Confirm the frozen extra-dimensions opportunity with official dynamics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import numpy as np
from scipy import special

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_extra_dimensions_opportunity import (
    Candidate,
    GROUP_WEIGHTS,
    KK_RADII,
    OBSERVATION_NOISE_STD,
    RIESZ_ALPHAS,
    YUKAWA_LENGTHS,
    evaluate_scalar_policy,
)


INTERFACE_VERSION = "discoverphysics-extra-dimensions-confirmation-1"
DISCOVERPHYSICS_COMMIT = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
FROZEN_CANDIDATE = Candidate(
    calibration_radii=(0.75, 0.75, 8.0),
    group_offsets=(
        0.7177711921002633,
        0.913222185103666,
        0.9803504883305367,
    ),
    measurement_scale=0.4,
    action_radii=(2.4, 3.6, 5.5, 8.0),
)
DT = 0.005
DURATION = 1.0
SOFTENING = 0.05
KK_IMAGES = 60
HELDOUT_RADII = np.geomspace(0.3, 9.0, 96)
PRIMARY_QUADRATURE_POINTS = 24
STABILITY_QUADRATURE_POINTS = (16, 32)


@dataclass(frozen=True)
class HypothesisSpec:
    family: str
    parameter: float
    strength: float
    label: str


def verify_source(root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError(
            f"DiscoverPhysics is at {commit}, expected {DISCOVERPHYSICS_COMMIT}"
        )
    return commit


def raw_force(
    family: str,
    parameter: float,
    radii: np.ndarray,
) -> np.ndarray:
    radii = np.asarray(radii, dtype=float)
    if family == "kk":
        circumference = 2.0 * math.pi * parameter
        offsets = np.arange(-KK_IMAGES, KK_IMAGES + 1) * circumference
        expanded = radii[..., None]
        geometry = np.sum(
            expanded / (expanded**2 + offsets**2) ** 1.5,
            axis=-1,
        )
        return circumference * geometry / (4.0 * math.pi)
    if family == "yukawa":
        return special.k1(radii / parameter) / (
            2.0 * math.pi * parameter
        )
    if family == "riesz":
        prefactor = special.gamma(1.0 - parameter) / (
            2.0 ** (2.0 * parameter)
            * math.pi
            * special.gamma(parameter)
        )
        return (
            prefactor
            * (2.0 - 2.0 * parameter)
            / radii ** (3.0 - 2.0 * parameter)
        )
    raise ValueError(f"unknown family {family!r}")


def build_hypotheses() -> tuple[list[HypothesisSpec], np.ndarray]:
    definitions = (
        ("kk", KK_RADII),
        ("yukawa", YUKAWA_LENGTHS),
        ("riesz", RIESZ_ALPHAS),
    )
    hypotheses: list[HypothesisSpec] = []
    for group_index, (family, parameters) in enumerate(definitions):
        calibration_radius = FROZEN_CANDIDATE.calibration_radii[group_index]
        target = (
            FROZEN_CANDIDATE.group_offsets[group_index]
            / (2.0 * math.pi * calibration_radius)
        )
        for parameter in parameters:
            unscaled = float(
                raw_force(
                    family,
                    float(parameter),
                    np.array([calibration_radius]),
                )[0]
            )
            hypotheses.append(
                HypothesisSpec(
                    family=family,
                    parameter=float(parameter),
                    strength=target / unscaled,
                    label=f"{family}_{parameter:g}",
                )
            )
    prior = np.repeat(GROUP_WEIGHTS / len(KK_RADII), len(KK_RADII))
    return hypotheses, prior


def force_values(spec: HypothesisSpec, radii: np.ndarray) -> np.ndarray:
    return spec.strength * raw_force(spec.family, spec.parameter, radii)


def load_official_modules(root: Path) -> tuple[type, Any]:
    for path in (root / "ScienceAgent", root / "PhysicsSchool"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    sampler_module = importlib.import_module(
        "physchool.worlds.nbody_sampler"
    )
    force_module = importlib.import_module("physchool.worlds.force_laws")
    return sampler_module.NBodySampler, force_module


def official_force_law(
    spec: HypothesisSpec,
    force_module: Any,
) -> Callable[..., Any]:
    if spec.family == "kk":
        return lambda r, qi, qj, mi, mj: force_module.extra_dimensions_2d_force(
            r,
            qi,
            qj,
            mi,
            mj,
            G=spec.strength,
            R_compact=spec.parameter,
            n_images=KK_IMAGES,
        )
    if spec.family == "yukawa":
        return lambda r, qi, qj, mi, mj: force_module.yukawa_2d_force(
            r,
            qi,
            qj,
            mi,
            mj,
            G=spec.strength,
            lam=spec.parameter,
        )
    if spec.family == "riesz":
        return lambda r, qi, qj, mi, mj: force_module.riesz_2d_force(
            r,
            qi,
            qj,
            mi,
            mj,
            G=spec.strength,
            alpha=spec.parameter,
        )
    raise ValueError(f"unknown family {spec.family!r}")


def official_displacements(
    hypotheses: list[HypothesisSpec],
    sampler_class: type,
    force_module: Any,
) -> np.ndarray:
    action_radii = np.asarray(FROZEN_CANDIDATE.action_radii)
    initial_positions = np.zeros((len(action_radii) + 1, 2))
    initial_positions[1:, 0] = action_radii
    initial_velocities = np.zeros_like(initial_positions)
    masses = np.ones(len(action_radii) + 1)
    masses[0] = 1e15
    source_charges = np.zeros(len(action_radii) + 1)
    source_charges[0] = 1.0
    force_charges = np.ones(len(action_radii) + 1)
    force_charges[0] = 0.0
    result = np.empty((len(action_radii), len(hypotheses)))
    for hypothesis_index, spec in enumerate(hypotheses):
        simulator = sampler_class(
            masses=masses,
            source_charges=source_charges,
            force_charges=force_charges,
            initial_positions=initial_positions,
            initial_velocities=initial_velocities,
            force_law=official_force_law(spec, force_module),
            potential_law=None,
            integrator="yoshida4",
            dt=DT,
            softening=SOFTENING,
            spatial_dimensions=2,
        )
        trajectory = simulator.run(
            n_steps=int(round(DURATION / DT)),
            record_every=1,
        )
        final_positions = np.asarray(trajectory["positions"])[-1, 1:, 0]
        result[:, hypothesis_index] = action_radii - final_positions
    return result


def numpy_displacements(hypotheses: list[HypothesisSpec]) -> np.ndarray:
    action_radii = np.asarray(FROZEN_CANDIDATE.action_radii)
    num_actions = len(action_radii)
    num_hypotheses = len(hypotheses)
    positions = np.zeros((num_actions, num_hypotheses, 2))
    positions[:, :, 0] = action_radii[:, None]
    velocities = np.zeros_like(positions)
    yoshida_weight = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    drift = (
        0.5 * yoshida_weight,
        0.5 * (1.0 - yoshida_weight),
        0.5 * (1.0 - yoshida_weight),
        0.5 * yoshida_weight,
    )
    kick = (
        yoshida_weight,
        1.0 - 2.0 * yoshida_weight,
        yoshida_weight,
    )

    def acceleration(current_positions: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(current_positions, axis=-1)
        effective = np.sqrt(distances**2 + SOFTENING**2)
        magnitudes = np.empty_like(distances)
        for hypothesis_index, spec in enumerate(hypotheses):
            magnitudes[:, hypothesis_index] = force_values(
                spec,
                effective[:, hypothesis_index],
            )
        return (
            -magnitudes[:, :, None]
            * current_positions
            / distances[:, :, None]
        )

    for _ in range(int(round(DURATION / DT))):
        for coefficient_index, kick_coefficient in enumerate(kick):
            positions += (
                drift[coefficient_index] * DT * velocities
            )
            velocities += (
                kick_coefficient * DT * acceleration(positions)
            )
        positions += drift[-1] * DT * velocities
    return action_radii[:, None] - positions[:, :, 0]


def transformed_means(displacements: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(displacements)) or np.any(displacements <= 0.0):
        raise ValueError("all displacements must be finite and positive")
    return FROZEN_CANDIDATE.measurement_scale * np.log(displacements)


def heldout_features(hypotheses: list[HypothesisSpec]) -> np.ndarray:
    return FROZEN_CANDIDATE.measurement_scale * np.log(
        np.asarray(
            [
                force_values(spec, HELDOUT_RADII)
                for spec in hypotheses
            ]
        )
    )


def summarize(
    values: dict[str, np.ndarray],
) -> dict[str, Any]:
    myopic_index = int(np.argmax(values["immediate_eig_nats"]))
    depth_two_index = int(np.argmax(values["total_eig_nats"]))
    myopic_risk = float(values["heldout_prediction_mse"][myopic_index])
    depth_two_risk = float(
        values["heldout_prediction_mse"][depth_two_index]
    )
    return {
        "myopic_index": myopic_index,
        "depth_two_index": depth_two_index,
        "myopic_radius": FROZEN_CANDIDATE.action_radii[myopic_index],
        "depth_two_radius": FROZEN_CANDIDATE.action_radii[depth_two_index],
        "immediate_sacrifice_nats": float(
            values["immediate_eig_nats"][myopic_index]
            - values["immediate_eig_nats"][depth_two_index]
        ),
        "total_eig_gain_nats": float(
            values["total_eig_nats"][depth_two_index]
            - values["total_eig_nats"][myopic_index]
        ),
        "heldout_risk_reduction_fraction": (
            myopic_risk - depth_two_risk
        )
        / myopic_risk,
        **{key: value.tolist() for key, value in values.items()},
    }


def run_confirmation(source_root: Path) -> dict[str, Any]:
    commit = verify_source(source_root)
    hypotheses, prior = build_hypotheses()
    sampler_class, force_module = load_official_modules(source_root)
    official = official_displacements(
        hypotheses,
        sampler_class,
        force_module,
    )
    independent = numpy_displacements(hypotheses)
    max_integrator_error = float(np.max(np.abs(official - independent)))
    means = transformed_means(official)
    heldout = heldout_features(hypotheses)
    evaluations: dict[str, Any] = {}
    for quadrature_points in (
        PRIMARY_QUADRATURE_POINTS,
        *STABILITY_QUADRATURE_POINTS,
    ):
        values = evaluate_scalar_policy(
            means,
            prior,
            quadrature_points=quadrature_points,
            heldout_features=heldout,
        )
        evaluations[str(quadrature_points)] = summarize(values)
    primary = evaluations[str(PRIMARY_QUADRATURE_POINTS)]
    stability = [
        evaluations[str(points)] for points in STABILITY_QUADRATURE_POINTS
    ]
    gates = {
        "finite_positive_displacements": bool(
            np.all(np.isfinite(official)) and np.all(official > 0.0)
        ),
        "myopic_radius_5_5": primary["myopic_radius"] == 5.5,
        "depth_two_radius_2_4": primary["depth_two_radius"] == 2.4,
        "immediate_sacrifice_at_least_0_10": (
            primary["immediate_sacrifice_nats"] >= 0.10
        ),
        "total_gain_at_least_0_05": (
            primary["total_eig_gain_nats"] >= 0.05
        ),
        "risk_reduction_at_least_0_20": (
            primary["heldout_risk_reduction_fraction"] >= 0.20
        ),
        "quadrature_stability": all(
            item["myopic_radius"] == 5.5
            and item["depth_two_radius"] == 2.4
            and item["total_eig_gain_nats"] >= 0.04
            and item["heldout_risk_reduction_fraction"] >= 0.15
            for item in stability
        ),
        "independent_integrator_error_at_most_1e_7": (
            max_integrator_error <= 1e-7
        ),
    }
    return {
        "interface_version": INTERFACE_VERSION,
        "discoverphysics_commit": commit,
        "candidate": asdict(FROZEN_CANDIDATE),
        "observation_noise_std": OBSERVATION_NOISE_STD,
        "hypotheses": [asdict(spec) for spec in hypotheses],
        "prior": prior.tolist(),
        "official_displacements": official.tolist(),
        "independent_displacements": independent.tolist(),
        "max_integrator_absolute_error": max_integrator_error,
        "heldout_radii": HELDOUT_RADII.tolist(),
        "evaluations": evaluations,
        "gates": gates,
        "passed": all(gates.values()),
        "openrouter_calls": 0,
        "cost_usd": 0.0,
        "cluster_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_confirmation(args.source_root)
    text = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
