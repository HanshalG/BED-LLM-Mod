#!/usr/bin/env python3
"""Run the frozen ChemBench moment-shortlist sampling gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    _spearman,
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
)
from scripts.chembench_posterior_sampling_fidelity import (
    POOLED_RESULT_PATH,
    POOLED_RESULT_SHA256,
    SAMPLE_COUNTS,
    _component_case_arrays,
    evaluate_gates as evaluate_sampling_gates,
)
from scripts.chembench_posterior_state_branch_fidelity import _fit_root_bank
from scripts.chembench_rqmc_sampling_fidelity import IID_RESULT_PATH, IID_RESULT_SHA256


SCHEMA_VERSION = "chembench-moment-shortlist-sampling-v1"
PROTOCOL_PATH = (
    "results/nonmyopic/CHEMBENCH_MOMENT_SHORTLIST_SAMPLING_PROTOCOL_20260815.md"
)
RQMC_RESULT_PATH = "results/nonmyopic/chembench_rqmc_sampling_fidelity/result.json"
RQMC_RESULT_SHA256 = "a966d5cf4984c9907649a0dae5d6bb8a19982f942c83f463f7ec61e4e2d439f2"
SHORTLIST_SIZE = 8
MAX_SAMPLES = 256
NUM_TARGETS = 128
SHARED_IID_SEED_BASE = 2026084000


def moment_proxy_gains(
    action_means: np.ndarray,
    action_sigmas: np.ndarray,
    target_values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return linear-Gaussian expected target-variance reductions."""

    means = np.asarray(action_means, dtype=float)
    sigmas = np.asarray(action_sigmas, dtype=float)
    targets = np.asarray(target_values, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    if means.ndim != 2 or sigmas.shape != means.shape or targets.ndim != 2:
        raise ValueError("proxy arrays have incompatible shapes")
    if means.shape[0] != targets.shape[0] or probabilities.shape != (len(means),):
        raise ValueError("proxy particle arrays have incompatible shapes")
    if np.any(sigmas <= 0) or np.any(probabilities < 0):
        raise ValueError("proxy scales or weights are invalid")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("proxy weights must sum to one")
    mean_centered = means - probabilities @ means
    target_centered = targets - probabilities @ targets
    covariance = (probabilities[:, None] * mean_centered).T @ target_centered
    predictive_variance = probabilities @ np.square(mean_centered)
    predictive_variance += probabilities @ np.square(sigmas)
    gains = np.mean(np.square(covariance), axis=1) / predictive_variance
    if not np.isfinite(gains).all() or np.any(gains < 0):
        raise ValueError("moment proxy gains are invalid")
    return gains


def select_shortlist(
    gains: np.ndarray, action_indices: Sequence[int], size: int = SHORTLIST_SIZE
) -> tuple[int, ...]:
    values = np.asarray(gains, dtype=float)
    if values.shape != (len(action_indices),) or not np.isfinite(values).all():
        raise ValueError("shortlist gains have the wrong shape")
    if size <= 0 or size > len(action_indices):
        raise ValueError("shortlist size is invalid")
    order = sorted(
        range(len(action_indices)), key=lambda index: (-values[index], action_indices[index])
    )
    return tuple(order[:size])


def _shortlist_payload(
    full_reference: np.ndarray,
    estimate: np.ndarray,
    shortlist_positions: Sequence[int],
    full_action_indices: Sequence[int],
    root_risk: float,
    component_action_risks: Mapping[str, np.ndarray],
    component_root_risks: Mapping[str, float],
) -> dict[str, Any]:
    positions = np.asarray(shortlist_positions, dtype=int)
    shortlist_reference = full_reference[positions]
    if estimate.shape != shortlist_reference.shape:
        raise ValueError("shortlist estimate has the wrong shape")
    local_selected = int(np.argmin(estimate))
    selected = int(positions[local_selected])
    regret = max(0.0, float(full_reference[selected] - np.min(full_reference)))
    components = {}
    for bank, risks in component_action_risks.items():
        value = max(0.0, float(risks[selected] - np.min(risks)))
        components[bank] = value / component_root_risks[bank]
    return {
        "spearman": _spearman(shortlist_reference, estimate),
        "selected_action_index": int(full_action_indices[selected]),
        "reference_best_action_index": int(
            full_action_indices[int(np.argmin(full_reference))]
        ),
        "top_one_regret": regret,
        "normalized_top_one_regret": regret / root_risk,
        "component_bank_regret": components,
        "estimated_action_risks": estimate.tolist(),
    }


def evaluate_shortlist_case(
    *,
    full_reference: np.ndarray,
    replicate_outcome_risks: np.ndarray,
    shortlist_positions: Sequence[int],
    full_action_indices: Sequence[int],
    root_risk: float,
    component_action_risks: Mapping[str, np.ndarray],
    component_root_risks: Mapping[str, float],
) -> dict[str, Any]:
    risks = np.asarray(replicate_outcome_risks, dtype=float)
    expected_shape = (NUM_REPLICATES, SHORTLIST_SIZE, MAX_SAMPLES)
    if risks.shape != expected_shape or not np.isfinite(risks).all():
        raise ValueError(f"shortlist risks must have finite shape {expected_shape}")
    estimates = {}
    for count in SAMPLE_COUNTS:
        estimates[str(count)] = []
        for replicate in range(NUM_REPLICATES):
            estimates[str(count)].append(
                _shortlist_payload(
                    full_reference,
                    np.mean(risks[replicate, :, :count], axis=1),
                    shortlist_positions,
                    full_action_indices,
                    root_risk,
                    component_action_risks,
                    component_root_risks,
                )
            )
    ensemble = np.mean(risks, axis=(0, 2))
    return {
        "estimates": estimates,
        "ensemble_1024": _shortlist_payload(
            full_reference,
            ensemble,
            shortlist_positions,
            full_action_indices,
            root_risk,
            component_action_risks,
            component_root_risks,
        ),
    }


def evaluate_gates(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sampling = evaluate_sampling_gates(cases)
    included = np.asarray([case["shortlist_oracle"]["full_best_included"] for case in cases])
    regrets = np.asarray(
        [case["shortlist_oracle"]["normalized_top_one_regret"] for case in cases]
    )
    shortlist_summary = {
        "fraction_full_best_included": float(np.mean(included)),
        "fraction_normalized_regret_at_most_003": float(np.mean(regrets <= 0.03)),
        "mean_normalized_top_one_regret": float(np.mean(regrets)),
    }
    shortlist_conditions = {
        "full_best_included_at_least_090": shortlist_summary[
            "fraction_full_best_included"
        ]
        >= 0.90,
        "all_shortlist_oracle_regrets_at_most_003": bool(np.all(regrets <= 0.03)),
        "mean_shortlist_oracle_regret_at_most_0005": shortlist_summary[
            "mean_normalized_top_one_regret"
        ]
        <= 0.005,
    }
    conditions = {
        "sampling_gate_passes": sampling["pass"],
        "shortlist_gate_passes": all(shortlist_conditions.values()),
    }
    return {
        "shortlist": shortlist_summary,
        "shortlist_conditions": shortlist_conditions,
        "sampling": sampling,
        "conditions": conditions,
        "pass": all(conditions.values()),
    }


def run(
    source_root: Path,
    v1_result_path: Path,
    v3_result_path: Path,
    pooled_result_path: Path,
    component_result_path: Path,
    iid_result_path: Path,
    rqmc_result_path: Path,
    *,
    batch_size: int = 128,
    progress: bool = False,
) -> dict[str, Any]:
    bindings = (
        (v3_result_path, V3_RESULT_SHA256, "V3"),
        (pooled_result_path, POOLED_RESULT_SHA256, "pooled"),
        (component_result_path, COMPONENT_RESULT_SHA256, "component"),
        (iid_result_path, IID_RESULT_SHA256, "IID"),
        (rqmc_result_path, RQMC_RESULT_SHA256, "RQMC"),
    )
    for path, expected, name in bindings:
        if _sha256(path) != expected:
            raise ValueError(f"{name} result hash does not match the frozen binding")
    if json.loads(iid_result_path.read_text())["gates"]["pass"]:
        raise ValueError("shortlist successor requires the bound IID failure")
    if json.loads(rqmc_result_path.read_text())["gates"]["pass"]:
        raise ValueError("shortlist successor requires the bound RQMC failure")
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
                source, domain, difficulty, prior, root_designs, observations[:4], 2026083501
            )
            bank_2 = _fit_root_bank(
                source, domain, difficulty, prior, root_designs, observations[:4], 2026083502
            )
            particles = np.concatenate((bank_1.particles, bank_2.particles), axis=0)
            weights = np.full(len(particles), 1.0 / len(particles))
            particle_hash = _array_sha256(particles, weights)
            predecessor_case = pooled_cases[(difficulty, domain)]
            if particle_hash != predecessor_case["pooled_particle_sha256"]:
                raise ValueError(f"pooled particle hash failed for {difficulty}/{domain}")
            parameters = prior.decode_many(particles)
            action_rates = _evaluate_rates(
                source, domain, parameters, action_designs, apply_secondary_effects=True
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
            gains = moment_proxy_gains(action_means, action_sigmas, target_values, weights)
            shortlist_positions = select_shortlist(gains, action_indices)
            saved_by_action = {
                action["assay_index"]: action["reference_expected_risk"]
                for action in predecessor_case["actions"]
            }
            full_reference = np.asarray([saved_by_action[index] for index in action_indices])
            shortlist_reference = full_reference[np.asarray(shortlist_positions)]
            shortlist_best_position = shortlist_positions[int(np.argmin(shortlist_reference))]
            oracle_regret = max(
                0.0,
                float(full_reference[shortlist_best_position] - np.min(full_reference)),
            ) / float(predecessor_case["root_risk"])
            full_best_position = int(np.argmin(full_reference))
            proxy_reference_spearman = _spearman(-gains, full_reference)
            component_action_risks, component_root_risks = _component_case_arrays(
                component_cases, difficulty, domain, action_indices
            )
            replicate_risks = np.empty(
                (NUM_REPLICATES, SHORTLIST_SIZE, MAX_SAMPLES), dtype=float
            )
            stream_digest = hashlib.sha256()
            outcome_digest = hashlib.sha256()
            for replicate in range(NUM_REPLICATES):
                sample_rng = np.random.default_rng(
                    _stable_seed(
                        SHARED_IID_SEED_BASE, str(replicate), difficulty, domain
                    )
                )
                particle_indices = sample_rng.integers(0, len(particles), MAX_SAMPLES)
                noises = sample_rng.normal(size=MAX_SAMPLES)
                stream_digest.update(
                    np.asarray(particle_indices, dtype=np.int64).tobytes()
                )
                stream_digest.update(np.asarray(noises, dtype=np.float64).tobytes())
                for local_index, full_position in enumerate(shortlist_positions):
                    outcomes = action_means[particle_indices, full_position]
                    outcomes = outcomes + action_sigmas[
                        particle_indices, full_position
                    ] * noises
                    outcome_digest.update(
                        np.asarray(outcomes, dtype=np.float64).tobytes()
                    )
                    replicate_risks[replicate, local_index] = (
                        posterior_risks_for_observations(
                            outcomes,
                            action_means[:, full_position],
                            action_sigmas[:, full_position],
                            target_values,
                            weights,
                            batch_size=batch_size,
                        )
                    )
            evaluated = evaluate_shortlist_case(
                full_reference=full_reference,
                replicate_outcome_risks=replicate_risks,
                shortlist_positions=shortlist_positions,
                full_action_indices=action_indices,
                root_risk=float(predecessor_case["root_risk"]),
                component_action_risks=component_action_risks,
                component_root_risks=component_root_risks,
            )
            case = {
                "domain": domain,
                "difficulty": difficulty,
                "root_risk": predecessor_case["root_risk"],
                "pooled_particle_sha256": particle_hash,
                "stream_sha256": stream_digest.hexdigest(),
                "outcome_sha256": outcome_digest.hexdigest(),
                "v1_compatibility": compatibility,
                "proxy_gains": gains.tolist(),
                "proxy_reference_spearman": proxy_reference_spearman,
                "shortlist_positions": list(shortlist_positions),
                "shortlist_action_indices": [action_indices[index] for index in shortlist_positions],
                "shortlist_oracle": {
                    "full_best_included": full_best_position in shortlist_positions,
                    "full_best_action_index": action_indices[full_best_position],
                    "shortlist_best_action_index": action_indices[shortlist_best_position],
                    "normalized_top_one_regret": oracle_regret,
                },
                "finite_and_reproducible": True,
                **evaluated,
            }
            cases.append(case)
            if progress:
                rep = case["estimates"]["256"]
                print(
                    f"{difficulty} {domain_index:02d}/12 {domain} "
                    f"oracle={oracle_regret:.4f} "
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
        "pooled_result": {"path": POOLED_RESULT_PATH, "sha256": _sha256(pooled_result_path)},
        "component_result": {
            "path": COMPONENT_RESULT_PATH,
            "sha256": _sha256(component_result_path),
        },
        "iid_result": {"path": IID_RESULT_PATH, "sha256": _sha256(iid_result_path)},
        "rqmc_result": {"path": RQMC_RESULT_PATH, "sha256": _sha256(rqmc_result_path)},
        "source": source_binding,
        "settings": {
            "shortlist_size": SHORTLIST_SIZE,
            "sample_counts": SAMPLE_COUNTS,
            "num_replicates": NUM_REPLICATES,
            "max_samples": MAX_SAMPLES,
            "num_targets": NUM_TARGETS,
            "shared_iid_seed_base": SHARED_IID_SEED_BASE,
            "common_particle_and_noise_across_actions": True,
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
    parser.add_argument("--pooled-result", type=Path, default=REPO_ROOT / POOLED_RESULT_PATH)
    parser.add_argument(
        "--component-result", type=Path, default=REPO_ROOT / COMPONENT_RESULT_PATH
    )
    parser.add_argument("--iid-result", type=Path, default=REPO_ROOT / IID_RESULT_PATH)
    parser.add_argument("--rqmc-result", type=Path, default=REPO_ROOT / RQMC_RESULT_PATH)
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
        args.rqmc_result,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
