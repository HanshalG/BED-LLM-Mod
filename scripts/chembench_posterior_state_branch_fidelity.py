#!/usr/bin/env python3
"""Run the frozen ChemBench posterior-state branch-fidelity gate."""

from __future__ import annotations

import argparse
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

from environments.chembench_mopen.smc import adaptive_tempered_smc
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
    _make_log_likelihood,
    _parameter_sha256,
    _prior_for,
    _prior_sha256,
    _sha256,
    _spearman,
    _stable_seed,
    _truth_parameters,
)
from scripts.chembench_adaptive_smc_v2 import V1_RESULT_PATH, _load_v1_cases
from scripts.chembench_adaptive_smc_v3 import (
    BANK_1_SEED_BASE,
    BANK_2_SEED_BASE,
    NUM_PARTICLES,
)
from scripts.chembench_local_tree_branch_fidelity import (
    PANEL_DOMAINS,
    ROOT_HISTORY_INDICES,
    V3_RESULT_PATH,
    V3_RESULT_SHA256,
    _draw_reference_outcomes,
    posterior_risks_for_observations,
    quantile_branch_observations,
)
from scripts.chembench_mopen_nonmyopic_opportunity import (
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-posterior-state-branch-fidelity-v1"
PROTOCOL_PATH = (
    "results/nonmyopic/CHEMBENCH_POSTERIOR_STATE_BRANCH_FIDELITY_PROTOCOL_20260815.md"
)
PREDECESSOR_RESULT_PATH = (
    "results/nonmyopic/chembench_local_tree_branch_fidelity/result.json"
)
PREDECESSOR_RESULT_SHA256 = (
    "e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16"
)
REFERENCE_OUTCOMES = 2048
NUM_BRANCHES = 9
NUM_TARGETS = 128
OUTCOME_SEED_BASE = 2026083800
FEATURE_SCALE_FLOOR = 1e-8
MAX_CLUSTER_ITERATIONS = 30


def posterior_moments_for_observations(
    observations: np.ndarray,
    predictive_means: np.ndarray,
    predictive_sigmas: np.ndarray,
    target_values: np.ndarray,
    prior_weights: np.ndarray,
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return child risk, target means, and target variances for outcomes."""

    outcomes = np.asarray(observations, dtype=float)
    means = np.asarray(predictive_means, dtype=float)
    sigmas = np.asarray(predictive_sigmas, dtype=float)
    targets = np.asarray(target_values, dtype=float)
    weights = np.asarray(prior_weights, dtype=float)
    particle_count = len(means)
    if outcomes.ndim != 1 or not len(outcomes):
        raise ValueError("observations must be a nonempty vector")
    if sigmas.shape != (particle_count,) or targets.ndim != 2:
        raise ValueError("predictive arrays have incompatible shapes")
    if targets.shape[0] != particle_count or weights.shape != (particle_count,):
        raise ValueError("particle arrays have incompatible shapes")
    if batch_size <= 0 or np.any(sigmas <= 0) or np.any(weights < 0):
        raise ValueError("posterior moment settings are invalid")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("prior weights must sum to one")
    if not all(np.isfinite(value).all() for value in (outcomes, means, sigmas, targets, weights)):
        raise ValueError("posterior moment inputs must be finite")

    log_prior = np.full(particle_count, -math.inf, dtype=float)
    positive = weights > 0
    log_prior[positive] = np.log(weights[positive])
    target_squares = np.square(targets)
    child_means = np.empty((len(outcomes), targets.shape[1]), dtype=float)
    child_variances = np.empty_like(child_means)
    constant = math.log(2.0 * math.pi)
    for start in range(0, len(outcomes), batch_size):
        stop = min(start + batch_size, len(outcomes))
        residuals = (outcomes[start:stop, None] - means[None, :]) / sigmas[None, :]
        log_values = (
            -0.5 * (np.square(residuals) + constant)
            - np.log(sigmas[None, :])
            + log_prior[None, :]
        )
        maxima = np.max(log_values, axis=1, keepdims=True)
        posterior = np.exp(log_values - maxima)
        posterior /= posterior.sum(axis=1, keepdims=True)
        child_means[start:stop] = posterior @ targets
        second = posterior @ target_squares
        child_variances[start:stop] = np.maximum(
            0.0, second - np.square(child_means[start:stop])
        )
    risks = np.mean(child_variances, axis=1)
    if not all(np.isfinite(value).all() for value in (risks, child_means, child_variances)):
        raise ValueError("posterior moments are not finite")
    return risks, child_means, child_variances


def deterministic_belief_medoids(
    features: np.ndarray,
    cluster_count: int,
    *,
    max_iterations: int = MAX_CLUSTER_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster child beliefs and return labels, medoids, and probabilities."""

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("features must be a finite nonempty matrix")
    if cluster_count <= 0 or cluster_count > len(values) or max_iterations <= 0:
        raise ValueError("clustering settings are invalid")
    centered = values - np.mean(values, axis=0, keepdims=True)
    squared_norms = np.sum(np.square(centered), axis=1)
    center_indices = [int(np.argmax(squared_norms))]
    minimum_distances = np.sum(
        np.square(centered - centered[center_indices[0]][None, :]), axis=1
    )
    for _ in range(1, cluster_count):
        index = int(np.argmax(minimum_distances))
        center_indices.append(index)
        distances = np.sum(np.square(centered - centered[index][None, :]), axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
    centers = centered[np.asarray(center_indices)].copy()
    labels = np.full(len(values), -1, dtype=int)
    for _ in range(max_iterations):
        distances = np.sum(
            np.square(centered[:, None, :] - centers[None, :, :]), axis=2
        )
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(cluster_count):
            members = centered[labels == cluster]
            if len(members):
                centers[cluster] = np.mean(members, axis=0)
    medoids = np.empty(cluster_count, dtype=int)
    probabilities = np.empty(cluster_count, dtype=float)
    for cluster in range(cluster_count):
        indices = np.flatnonzero(labels == cluster)
        if not len(indices):
            medoids[cluster] = int(
                np.argmin(np.sum(np.square(centered - centers[cluster]), axis=1))
            )
            probabilities[cluster] = 0.0
            continue
        distances = np.sum(np.square(centered[indices] - centers[cluster]), axis=1)
        medoids[cluster] = int(indices[int(np.argmin(distances))])
        probabilities[cluster] = len(indices) / len(values)
    return labels, medoids, probabilities


def posterior_state_branch_expected_risk(
    *,
    outcomes: np.ndarray,
    predictive_means: np.ndarray,
    predictive_sigmas: np.ndarray,
    target_values: np.ndarray,
    prior_weights: np.ndarray,
    batch_size: int = 128,
) -> dict[str, Any]:
    risks, child_means, child_variances = posterior_moments_for_observations(
        outcomes,
        predictive_means,
        predictive_sigmas,
        target_values,
        prior_weights,
        batch_size=batch_size,
    )
    root_mean = prior_weights @ target_values
    root_variance = np.maximum(
        0.0, prior_weights @ np.square(target_values) - np.square(root_mean)
    )
    mean_scale = np.maximum(np.sqrt(root_variance), FEATURE_SCALE_FLOOR)
    variance_scale = np.maximum(root_variance, FEATURE_SCALE_FLOOR)
    features = np.concatenate(
        (
            (child_means - root_mean[None, :]) / mean_scale[None, :],
            (child_variances - root_variance[None, :]) / variance_scale[None, :],
        ),
        axis=1,
    )
    _, medoids, probabilities = deterministic_belief_medoids(features, NUM_BRANCHES)
    representatives = np.asarray(outcomes)[medoids]
    posterior_state_risk = float(np.dot(probabilities, risks[medoids]))

    raw_representatives, raw_probabilities = quantile_branch_observations(
        outcomes, NUM_BRANCHES
    )
    raw_risks = posterior_risks_for_observations(
        raw_representatives,
        predictive_means,
        predictive_sigmas,
        target_values,
        prior_weights,
        batch_size=batch_size,
    )
    return {
        "reference_expected_risk": float(np.mean(risks)),
        "posterior_state_expected_risk": posterior_state_risk,
        "raw_quantile_expected_risk": float(np.dot(raw_probabilities, raw_risks)),
        "medoid_indices": medoids.tolist(),
        "representative_observations": representatives.tolist(),
        "branch_probabilities": probabilities.tolist(),
    }


def _method_summary(cases: Sequence[Mapping[str, Any]], method: str) -> dict[str, float]:
    correlations = np.asarray([case[method]["spearman"] for case in cases])
    regrets = np.asarray([case[method]["normalized_top_one_regret"] for case in cases])
    return {
        "median_spearman": float(np.median(correlations)),
        "fraction_spearman_at_least_080": float(np.mean(correlations >= 0.80)),
        "fraction_normalized_regret_at_most_003": float(np.mean(regrets <= 0.03)),
        "mean_normalized_top_one_regret": float(np.mean(regrets)),
    }


def evaluate_gates(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    posterior = _method_summary(cases, "posterior_state")
    raw = _method_summary(cases, "raw_quantile")
    component = {
        bank: {
            "fraction_normalized_regret_at_most_003": float(
                np.mean([case["component_bank_regret"][bank] <= 0.03 for case in cases])
            ),
            "mean_normalized_regret": float(
                np.mean([case["component_bank_regret"][bank] for case in cases])
            ),
        }
        for bank in ("bank_1", "bank_2")
    }
    conditions = {
        "all_cases_finite_and_reproducible": len(cases) == 36
        and all(case["finite_and_reproducible"] for case in cases),
        "median_spearman_at_least_090": posterior["median_spearman"] >= 0.90,
        "fraction_spearman_at_least_080_is_090": posterior[
            "fraction_spearman_at_least_080"
        ]
        >= 0.90,
        "fraction_regret_at_most_003_is_090": posterior[
            "fraction_normalized_regret_at_most_003"
        ]
        >= 0.90,
        "mean_normalized_regret_at_most_001": posterior[
            "mean_normalized_top_one_regret"
        ]
        <= 0.01,
        "component_bank_regret_passes": all(
            values["fraction_normalized_regret_at_most_003"] >= 0.90
            and values["mean_normalized_regret"] <= 0.01
            for values in component.values()
        ),
        "posterior_state_nonworse_than_raw_quantile": posterior["median_spearman"]
        >= raw["median_spearman"]
        and posterior["mean_normalized_top_one_regret"]
        <= raw["mean_normalized_top_one_regret"],
    }
    return {
        "posterior_state": posterior,
        "raw_quantile": raw,
        "component_banks": component,
        "conditions": conditions,
        "pass": all(conditions.values()),
    }


def _fit_root_bank(
    source: Any,
    domain: str,
    difficulty: str,
    prior: Any,
    root_designs: np.ndarray,
    root_observations: np.ndarray,
    seed_base: int,
) -> Any:
    return adaptive_tempered_smc(
        prior,
        _make_log_likelihood(source, domain, prior, root_designs, root_observations),
        num_particles=NUM_PARTICLES,
        seed=_stable_seed(seed_base, difficulty, domain),
        initialization="sobol",
        proposal_geometry="full",
    )


def _evaluate_case(
    source: Any,
    domain: str,
    difficulty: str,
    prior: Any,
    root_designs: np.ndarray,
    root_observations: np.ndarray,
    action_designs: np.ndarray,
    action_indices: Sequence[int],
    target_designs: np.ndarray,
    predecessor_cases: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    bank_1 = _fit_root_bank(
        source,
        domain,
        difficulty,
        prior,
        root_designs,
        root_observations,
        BANK_1_SEED_BASE,
    )
    bank_2 = _fit_root_bank(
        source,
        domain,
        difficulty,
        prior,
        root_designs,
        root_observations,
        BANK_2_SEED_BASE,
    )
    particles = np.concatenate((bank_1.particles, bank_2.particles), axis=0)
    weights = np.full(len(particles), 1.0 / len(particles))
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
        NOISE_LEVEL * action_rates / (1.0 + action_rates), ABSOLUTE_LOG_NOISE_FLOOR
    )
    target_values = np.log1p(
        _evaluate_rates(
            source,
            domain,
            parameters,
            target_designs,
            apply_secondary_effects=False,
        )
    )
    root_mean = weights @ target_values
    root_variance = weights @ np.square(target_values) - np.square(root_mean)
    root_risk = float(np.mean(np.maximum(0.0, root_variance)))
    if not math.isfinite(root_risk) or root_risk <= 0:
        raise ValueError("pooled root risk must be finite and positive")

    references = []
    posterior_values = []
    raw_values = []
    actions = []
    for local_index, assay_index in enumerate(action_indices):
        outcomes = _draw_reference_outcomes(
            action_means[:, local_index],
            action_sigmas[:, local_index],
            weights,
            seed=_stable_seed(
                OUTCOME_SEED_BASE, "pooled", difficulty, domain, str(assay_index)
            ),
            count=REFERENCE_OUTCOMES,
        )
        values = posterior_state_branch_expected_risk(
            outcomes=outcomes,
            predictive_means=action_means[:, local_index],
            predictive_sigmas=action_sigmas[:, local_index],
            target_values=target_values,
            prior_weights=weights,
            batch_size=batch_size,
        )
        references.append(values["reference_expected_risk"])
        posterior_values.append(values["posterior_state_expected_risk"])
        raw_values.append(values["raw_quantile_expected_risk"])
        actions.append(
            {
                "assay_index": assay_index,
                "reference_expected_risk": values["reference_expected_risk"],
                "posterior_state_expected_risk": values[
                    "posterior_state_expected_risk"
                ],
                "raw_quantile_expected_risk": values["raw_quantile_expected_risk"],
                "medoid_indices": values["medoid_indices"],
                "representative_observations": values[
                    "representative_observations"
                ],
                "branch_probabilities": values["branch_probabilities"],
            }
        )

    reference_array = np.asarray(references)
    methods = {}
    for name, approximate in (
        ("posterior_state", np.asarray(posterior_values)),
        ("raw_quantile", np.asarray(raw_values)),
    ):
        selected = int(np.argmin(approximate))
        regret = max(0.0, float(reference_array[selected] - np.min(reference_array)))
        methods[name] = {
            "spearman": _spearman(reference_array, approximate),
            "selected_action_index": int(action_indices[selected]),
            "reference_best_action_index": int(
                action_indices[int(np.argmin(reference_array))]
            ),
            "top_one_regret": regret,
            "normalized_top_one_regret": regret / root_risk,
        }
    selected_assay = methods["posterior_state"]["selected_action_index"]
    component_regret = {}
    for bank in ("bank_1", "bank_2"):
        component_case = predecessor_cases[(difficulty, domain, bank)]
        action_risks = {
            action["assay_index"]: action["reference_expected_risk"]
            for action in component_case["actions"]
        }
        component_regret[bank] = max(
            0.0,
            float(action_risks[selected_assay] - min(action_risks.values())),
        ) / float(component_case["root_risk"])
    finite = all(
        math.isfinite(value)
        for value in (
            root_risk,
            *references,
            *posterior_values,
            *raw_values,
            *component_regret.values(),
        )
    )
    return {
        "domain": domain,
        "difficulty": difficulty,
        "root_risk": root_risk,
        "pooled_particle_sha256": _array_sha256(particles, weights),
        "finite_and_reproducible": finite,
        "posterior_state": methods["posterior_state"],
        "raw_quantile": methods["raw_quantile"],
        "component_bank_regret": component_regret,
        "actions": actions,
    }


def run(
    source_root: Path,
    v1_result_path: Path,
    v3_result_path: Path,
    predecessor_result_path: Path,
    *,
    batch_size: int = 128,
    progress: bool = False,
) -> dict[str, Any]:
    if _sha256(v3_result_path) != V3_RESULT_SHA256:
        raise ValueError("V3 result hash does not match the frozen binding")
    if _sha256(predecessor_result_path) != PREDECESSOR_RESULT_SHA256:
        raise ValueError("predecessor result hash does not match the frozen binding")
    v3_result = json.loads(v3_result_path.read_text())
    if not v3_result["source_only"]["gates"]["pass"]:
        raise ValueError("V3 source-only posterior gate did not pass")
    predecessor = json.loads(predecessor_result_path.read_text())
    predecessor_cases = {
        (case["difficulty"], case["domain"], case["bank"]): case
        for case in predecessor["cases"]
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
            v1_case = v1_cases[("source_only", difficulty, domain)]
            compatibility = {
                "prior_sha256": _prior_sha256(prior),
                "truth_sha256": _parameter_sha256(truth),
                "history_sha256": _array_sha256(full_history_designs, observations),
                "query_sha256": _array_sha256(query_designs[difficulty]),
            }
            if any(compatibility[name] != v1_case[name] for name in compatibility):
                raise ValueError(f"V1 compatibility failed for {difficulty}/{domain}")
            case = _evaluate_case(
                source,
                domain,
                difficulty,
                prior,
                root_designs,
                observations[:4],
                action_designs,
                action_indices,
                query_designs[difficulty][:NUM_TARGETS],
                predecessor_cases,
                batch_size=batch_size,
            )
            case["v1_compatibility"] = compatibility
            cases.append(case)
            if progress:
                print(
                    f"{difficulty} {domain_index:02d}/12 {domain} "
                    f"rho={case['posterior_state']['spearman']:.3f} "
                    f"regret={case['posterior_state']['normalized_top_one_regret']:.4f} "
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
        "predecessor_result": {
            "path": PREDECESSOR_RESULT_PATH,
            "sha256": _sha256(predecessor_result_path),
        },
        "source": source_binding,
        "settings": {
            "panel_domains": PANEL_DOMAINS,
            "difficulties": DIFFICULTIES,
            "root_history_indices": ROOT_HISTORY_INDICES,
            "candidate_action_indices": action_indices,
            "particles_per_bank": NUM_PARTICLES,
            "pooled_particles": 2 * NUM_PARTICLES,
            "reference_outcomes": REFERENCE_OUTCOMES,
            "num_branches": NUM_BRANCHES,
            "num_targets": NUM_TARGETS,
            "outcome_seed_base": OUTCOME_SEED_BASE,
            "feature_scale_floor": FEATURE_SCALE_FLOOR,
            "max_cluster_iterations": MAX_CLUSTER_ITERATIONS,
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
        "--predecessor-result",
        type=Path,
        default=REPO_ROOT / PREDECESSOR_RESULT_PATH,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    result = run(
        args.source_root,
        args.v1_result,
        args.v3_result,
        args.predecessor_result,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
