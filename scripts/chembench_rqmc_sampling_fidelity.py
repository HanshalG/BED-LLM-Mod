#!/usr/bin/env python3
"""Run the frozen ChemBench randomized-QMC sampling fidelity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.smc import _scrambled_sobol_points
from environments.chembench_mopen.source import _query_assays
from scripts.chembench_adaptive_smc_calibration import (
    ABSOLUTE_LOG_NOISE_FLOOR,
    DIFFICULTIES,
    HISTORY_ASSAY_INDICES,
    NOISE_LEVEL,
    OBSERVATION_SEED_BASE,
    QUERY_SEEDS,
    _array_sha256,
    _evaluate_rates,
    _parameter_sha256,
    _prior_for,
    _prior_sha256,
    _sha256,
    _stable_seed,
    _truth_parameters,
)
from scripts.chembench_adaptive_smc_v2 import V1_RESULT_PATH, _load_v1_cases
from scripts.chembench_local_tree_branch_fidelity import (
    PANEL_DOMAINS,
    ROOT_HISTORY_INDICES,
    V3_RESULT_PATH,
    V3_RESULT_SHA256,
    posterior_risks_for_observations,
)
from scripts.chembench_mopen_nonmyopic_opportunity import (
    frozen_assays,
    load_source,
    verify_source,
)
from scripts.chembench_posterior_sampling_fidelity import (
    COMPONENT_RESULT_PATH,
    COMPONENT_RESULT_SHA256,
    NUM_REPLICATES,
    POOLED_RESULT_PATH,
    POOLED_RESULT_SHA256,
    SAMPLE_COUNTS,
    _action_payload,
    _component_case_arrays,
    evaluate_gates,
)
from scripts.chembench_posterior_state_branch_fidelity import _fit_root_bank


SCHEMA_VERSION = "chembench-rqmc-sampling-fidelity-v1"
PROTOCOL_PATH = (
    "results/nonmyopic/CHEMBENCH_RQMC_SAMPLING_FIDELITY_PROTOCOL_20260815.md"
)
IID_RESULT_PATH = "results/nonmyopic/chembench_posterior_sampling_fidelity/result.json"
IID_RESULT_SHA256 = "d71964ff247bc80408b0b5c78c4b168c5117ece372446990c4a5dfd17b42c0f6"
RQMC_SEED_BASE = 2026083900
MAX_SAMPLES = 256
NUM_TARGETS = 128
NORMAL_CLIP = 2.0**-52


def rqmc_coordinates(seed: int, count: int = MAX_SAMPLES) -> np.ndarray:
    """Return a digitally shifted two-dimensional Sobol power-of-two prefix."""

    if count <= 0 or count & (count - 1):
        raise ValueError("RQMC count must be a positive power of two")
    power = int(math.log2(count))
    coordinates = _scrambled_sobol_points(2, power, seed)
    if coordinates.shape != (count, 2) or np.any(coordinates < 0) or np.any(
        coordinates >= 1
    ):
        raise ValueError("RQMC coordinates are invalid")
    return coordinates


def predictive_outcomes_from_coordinates(
    coordinates: np.ndarray,
    predictive_means: np.ndarray,
    predictive_sigmas: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Map shared QMC coordinates to one action's predictive outcomes."""

    points = np.asarray(coordinates, dtype=float)
    means = np.asarray(predictive_means, dtype=float)
    sigmas = np.asarray(predictive_sigmas, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("coordinates must have shape (count, 2)")
    if sigmas.shape != means.shape or probabilities.shape != means.shape:
        raise ValueError("predictive vectors have incompatible shapes")
    if np.any(sigmas <= 0) or np.any(probabilities < 0):
        raise ValueError("predictive scales or weights are invalid")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("weights must sum to one")
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    indices = np.searchsorted(cumulative, points[:, 0], side="right")
    normal = statistics.NormalDist()
    clipped = np.clip(points[:, 1], NORMAL_CLIP, 1.0 - NORMAL_CLIP)
    noises = np.fromiter(
        (normal.inv_cdf(float(value)) for value in clipped),
        dtype=float,
        count=len(clipped),
    )
    outcomes = means[indices] + sigmas[indices] * noises
    if not np.isfinite(outcomes).all():
        raise ValueError("RQMC predictive outcomes are not finite")
    return outcomes


def evaluate_rqmc_case(
    *,
    reference_action_risks: np.ndarray,
    replicate_outcome_risks: np.ndarray,
    action_indices: Sequence[int],
    root_risk: float,
    component_action_risks: Mapping[str, np.ndarray],
    component_root_risks: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate nested prefixes from four independent RQMC scrambles."""

    reference = np.asarray(reference_action_risks, dtype=float)
    risks = np.asarray(replicate_outcome_risks, dtype=float)
    expected_shape = (NUM_REPLICATES, len(action_indices), MAX_SAMPLES)
    if risks.shape != expected_shape:
        raise ValueError(f"replicate risks must have shape {expected_shape}")
    if not np.isfinite(risks).all():
        raise ValueError("RQMC risks must be finite")
    estimates = {}
    for count in SAMPLE_COUNTS:
        estimates[str(count)] = []
        for replicate in range(NUM_REPLICATES):
            estimate = np.mean(risks[replicate, :, :count], axis=1)
            estimates[str(count)].append(
                _action_payload(
                    reference,
                    estimate,
                    action_indices,
                    root_risk,
                    component_action_risks,
                    component_root_risks,
                )
            )
    ensemble_estimate = np.mean(risks, axis=(0, 2))
    return {
        "estimates": estimates,
        "ensemble_1024": _action_payload(
            reference,
            ensemble_estimate,
            action_indices,
            root_risk,
            component_action_risks,
            component_root_risks,
        ),
    }


def run(
    source_root: Path,
    v1_result_path: Path,
    v3_result_path: Path,
    pooled_result_path: Path,
    component_result_path: Path,
    iid_result_path: Path,
    *,
    batch_size: int = 128,
    progress: bool = False,
) -> dict[str, Any]:
    bindings = (
        (v3_result_path, V3_RESULT_SHA256, "V3"),
        (pooled_result_path, POOLED_RESULT_SHA256, "pooled"),
        (component_result_path, COMPONENT_RESULT_SHA256, "component"),
        (iid_result_path, IID_RESULT_SHA256, "IID"),
    )
    for path, expected, name in bindings:
        if _sha256(path) != expected:
            raise ValueError(f"{name} result hash does not match the frozen binding")
    v3_result = json.loads(v3_result_path.read_text())
    if not v3_result["source_only"]["gates"]["pass"]:
        raise ValueError("V3 source-only posterior gate did not pass")
    iid_result = json.loads(iid_result_path.read_text())
    if iid_result["gates"]["pass"]:
        raise ValueError("RQMC successor requires the bound IID gate failure")
    pooled_result = json.loads(pooled_result_path.read_text())
    pooled_cases = {
        (case["difficulty"], case["domain"]): case
        for case in pooled_result["cases"]
    }
    component_result = json.loads(component_result_path.read_text())
    component_cases = {
        (case["difficulty"], case["domain"], case["bank"]): case
        for case in component_result["cases"]
    }
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    _, v1_cases = _load_v1_cases(v1_result_path)
    assays = frozen_assays()
    action_indices = tuple(
        index for index in range(len(assays)) if index not in ROOT_HISTORY_INDICES
    )
    root_designs = np.asarray([assays[index].values for index in ROOT_HISTORY_INDICES])
    action_designs = np.asarray([assays[index].values for index in action_indices])
    full_history_designs = np.asarray(
        [assays[index].values for index in HISTORY_ASSAY_INDICES]
    )
    query_designs = {
        difficulty: _query_assays(
            QUERY_SEEDS[difficulty],
            source.CHEM_INPUT_BOUNDS,
            source.CHEM_LOG_VARS,
            512,
        )
        for difficulty in DIFFICULTIES
    }
    cases = []
    started = time.perf_counter()
    for difficulty in DIFFICULTIES:
        for domain_index, domain in enumerate(PANEL_DOMAINS, start=1):
            case_started = time.perf_counter()
            prior = _prior_for(source, domain, difficulty)
            truth = _truth_parameters(source, prior, domain, difficulty, "source_only")
            truth_means = _evaluate_rates(
                source,
                domain,
                (truth,),
                full_history_designs,
                apply_secondary_effects=True,
            )[0]
            rng = np.random.default_rng(
                _stable_seed(OBSERVATION_SEED_BASE, "source_only", difficulty, domain)
            )
            observations = np.maximum(
                0.0, truth_means * (1.0 + NOISE_LEVEL * rng.normal(size=8))
            )
            compatibility = {
                "prior_sha256": _prior_sha256(prior),
                "truth_sha256": _parameter_sha256(truth),
                "history_sha256": _array_sha256(full_history_designs, observations),
                "query_sha256": _array_sha256(query_designs[difficulty]),
            }
            v1_case = v1_cases[("source_only", difficulty, domain)]
            if any(compatibility[name] != v1_case[name] for name in compatibility):
                raise ValueError(f"V1 compatibility failed for {difficulty}/{domain}")

            bank_1 = _fit_root_bank(
                source,
                domain,
                difficulty,
                prior,
                root_designs,
                observations[:4],
                2026083501,
            )
            bank_2 = _fit_root_bank(
                source,
                domain,
                difficulty,
                prior,
                root_designs,
                observations[:4],
                2026083502,
            )
            particles = np.concatenate((bank_1.particles, bank_2.particles), axis=0)
            weights = np.full(len(particles), 1.0 / len(particles))
            particle_hash = _array_sha256(particles, weights)
            predecessor_case = pooled_cases[(difficulty, domain)]
            if particle_hash != predecessor_case["pooled_particle_sha256"]:
                raise ValueError(f"pooled particle hash failed for {difficulty}/{domain}")
            parameters = prior.decode_many(particles)
            action_rates = _evaluate_rates(
                source,
                domain,
                parameters,
                action_designs,
                apply_secondary_effects=True,
            )
            action_means = np.log1p(action_rates)
            action_sigmas = np.maximum(
                NOISE_LEVEL * action_rates / (1.0 + action_rates),
                ABSOLUTE_LOG_NOISE_FLOOR,
            )
            target_values = np.log1p(
                _evaluate_rates(
                    source,
                    domain,
                    parameters,
                    query_designs[difficulty][:NUM_TARGETS],
                    apply_secondary_effects=False,
                )
            )
            saved_by_action = {
                action["assay_index"]: action["reference_expected_risk"]
                for action in predecessor_case["actions"]
            }
            reference = np.asarray([saved_by_action[index] for index in action_indices])
            replicate_risks = np.empty(
                (NUM_REPLICATES, len(action_indices), MAX_SAMPLES), dtype=float
            )
            coordinate_digest = hashlib.sha256()
            outcome_digest = hashlib.sha256()
            for replicate in range(NUM_REPLICATES):
                coordinates = rqmc_coordinates(
                    _stable_seed(
                        RQMC_SEED_BASE, str(replicate), difficulty, domain
                    )
                )
                coordinate_digest.update(
                    np.asarray(coordinates, dtype=np.float64).tobytes()
                )
                for local_index in range(len(action_indices)):
                    outcomes = predictive_outcomes_from_coordinates(
                        coordinates,
                        action_means[:, local_index],
                        action_sigmas[:, local_index],
                        weights,
                    )
                    outcome_digest.update(
                        np.asarray(outcomes, dtype=np.float64).tobytes()
                    )
                    replicate_risks[replicate, local_index] = (
                        posterior_risks_for_observations(
                            outcomes,
                            action_means[:, local_index],
                            action_sigmas[:, local_index],
                            target_values,
                            weights,
                            batch_size=batch_size,
                        )
                    )
            component_action_risks, component_root_risks = _component_case_arrays(
                component_cases, difficulty, domain, action_indices
            )
            evaluated = evaluate_rqmc_case(
                reference_action_risks=reference,
                replicate_outcome_risks=replicate_risks,
                action_indices=action_indices,
                root_risk=float(predecessor_case["root_risk"]),
                component_action_risks=component_action_risks,
                component_root_risks=component_root_risks,
            )
            case = {
                "domain": domain,
                "difficulty": difficulty,
                "root_risk": predecessor_case["root_risk"],
                "pooled_particle_sha256": particle_hash,
                "coordinate_sha256": coordinate_digest.hexdigest(),
                "outcome_sha256": outcome_digest.hexdigest(),
                "v1_compatibility": compatibility,
                "finite_and_reproducible": True,
                **evaluated,
            }
            cases.append(case)
            if progress:
                rep = case["estimates"]["256"]
                print(
                    f"{difficulty} {domain_index:02d}/12 {domain} "
                    f"rho256={[round(item['spearman'], 3) for item in rep]} "
                    f"elapsed={time.perf_counter() - case_started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
    gates = evaluate_gates(cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {"path": PROTOCOL_PATH, "sha256": _sha256(REPO_ROOT / PROTOCOL_PATH)},
        "v1_result": {"path": V1_RESULT_PATH, "sha256": _sha256(v1_result_path)},
        "v3_result": {"path": V3_RESULT_PATH, "sha256": _sha256(v3_result_path)},
        "pooled_result": {
            "path": POOLED_RESULT_PATH,
            "sha256": _sha256(pooled_result_path),
        },
        "component_result": {
            "path": COMPONENT_RESULT_PATH,
            "sha256": _sha256(component_result_path),
        },
        "iid_result": {"path": IID_RESULT_PATH, "sha256": _sha256(iid_result_path)},
        "source": source_binding,
        "settings": {
            "sample_counts": SAMPLE_COUNTS,
            "num_replicates": NUM_REPLICATES,
            "max_samples": MAX_SAMPLES,
            "num_targets": NUM_TARGETS,
            "rqmc_seed_base": RQMC_SEED_BASE,
            "normal_clip": NORMAL_CLIP,
            "common_coordinates_across_actions": True,
        },
        "gates": gates,
        "cases": cases,
        "elapsed_seconds": time.perf_counter() - started,
        "model_calls": 0,
        "network_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--v1-result", type=Path, default=REPO_ROOT / V1_RESULT_PATH)
    parser.add_argument("--v3-result", type=Path, default=REPO_ROOT / V3_RESULT_PATH)
    parser.add_argument(
        "--pooled-result", type=Path, default=REPO_ROOT / POOLED_RESULT_PATH
    )
    parser.add_argument(
        "--component-result", type=Path, default=REPO_ROOT / COMPONENT_RESULT_PATH
    )
    parser.add_argument("--iid-result", type=Path, default=REPO_ROOT / IID_RESULT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    result = run(
        args.source_root,
        args.v1_result,
        args.v3_result,
        args.pooled_result,
        args.component_result,
        args.iid_result,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
