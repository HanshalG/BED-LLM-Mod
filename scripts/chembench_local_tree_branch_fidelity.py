#!/usr/bin/env python3
"""Run the frozen ChemBench local-tree branch-fidelity gate."""

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
from scripts.chembench_mopen_nonmyopic_opportunity import (
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-local-tree-branch-fidelity-v1"
PROTOCOL_PATH = (
    "results/nonmyopic/CHEMBENCH_LOCAL_TREE_BRANCH_FIDELITY_PROTOCOL_20260815.md"
)
V3_RESULT_PATH = "results/nonmyopic/chembench_adaptive_smc_v3/result.json"
V3_RESULT_SHA256 = "dce0832190aa8b76c9b345220a9043edb6e622db3b8cf0f524b68b5c06c783c1"
PANEL_DOMAINS = (
    "c10_mm_competitive_arrhenius",
    "c23_pingpong_arrhenius",
    "c33_hill_competitive",
    "c37_hill_arrhenius",
    "c48_sinh_competitive",
    "c65_ordered_bi_bi",
    "c67_allosteric_act",
    "c70_mixed_inhibition",
    "c71_coop_inhibition",
    "c73_metal_activation",
    "c78_allosteric_act_arrhenius",
    "c93_fractal_competitive",
)
ROOT_HISTORY_INDICES = HISTORY_ASSAY_INDICES[:4]
REFERENCE_OUTCOMES = 2048
BRANCH_COUNTS = (3, 5, 9)
NUM_TARGETS = 128
OUTCOME_SEED_BASE = 2026083700


def _normalize_log_weight_rows(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("log weights must be a nonempty matrix")
    maxima = np.max(values, axis=1, keepdims=True)
    if not np.isfinite(maxima).all():
        raise ValueError("log weights have no finite row normalizer")
    weights = np.exp(values - maxima)
    totals = weights.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("log weights cannot be normalized")
    return weights / totals


def posterior_risks_for_observations(
    observations: np.ndarray,
    predictive_means: np.ndarray,
    predictive_sigmas: np.ndarray,
    target_values: np.ndarray,
    prior_weights: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Return task Bayes risk after each scalar transformed observation."""

    outcomes = np.asarray(observations, dtype=float)
    means = np.asarray(predictive_means, dtype=float)
    sigmas = np.asarray(predictive_sigmas, dtype=float)
    targets = np.asarray(target_values, dtype=float)
    weights = np.asarray(prior_weights, dtype=float)
    particle_count = len(means)
    if outcomes.ndim != 1 or not len(outcomes):
        raise ValueError("observations must be a nonempty vector")
    if sigmas.shape != (particle_count,) or targets.ndim != 2:
        raise ValueError("predictive or target arrays have incompatible shapes")
    if targets.shape[0] != particle_count or weights.shape != (particle_count,):
        raise ValueError("particle arrays have incompatible shapes")
    if batch_size <= 0 or np.any(sigmas <= 0) or np.any(weights < 0):
        raise ValueError("risk settings are invalid")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("prior weights must sum to one")
    if not all(np.isfinite(value).all() for value in (outcomes, means, sigmas, targets, weights)):
        raise ValueError("risk inputs must be finite")

    log_prior = np.full(particle_count, -math.inf, dtype=float)
    positive = weights > 0
    log_prior[positive] = np.log(weights[positive])
    target_squares = np.square(targets)
    risks = np.empty(len(outcomes), dtype=float)
    constant = math.log(2.0 * math.pi)
    for start in range(0, len(outcomes), batch_size):
        stop = min(start + batch_size, len(outcomes))
        residuals = (outcomes[start:stop, None] - means[None, :]) / sigmas[None, :]
        log_likelihoods = -0.5 * (np.square(residuals) + constant) - np.log(
            sigmas[None, :]
        )
        posterior = _normalize_log_weight_rows(log_likelihoods + log_prior[None, :])
        first_moment = posterior @ targets
        second_moment = posterior @ target_squares
        variances = np.maximum(0.0, second_moment - np.square(first_moment))
        risks[start:stop] = np.mean(variances, axis=1)
    if not np.isfinite(risks).all():
        raise ValueError("posterior risks are not finite")
    return risks


def quantile_branch_observations(
    reference_outcomes: np.ndarray, branch_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Represent empirical equal-mass bins by their median observations."""

    outcomes = np.asarray(reference_outcomes, dtype=float)
    if outcomes.ndim != 1 or not len(outcomes) or not np.isfinite(outcomes).all():
        raise ValueError("reference outcomes must be a finite nonempty vector")
    if branch_count <= 0 or branch_count > len(outcomes):
        raise ValueError("branch count is invalid")
    bins = np.array_split(np.sort(outcomes), branch_count)
    representatives = np.asarray([np.median(values) for values in bins], dtype=float)
    probabilities = np.asarray([len(values) / len(outcomes) for values in bins], dtype=float)
    return representatives, probabilities


def expected_action_risks(
    *,
    predictive_means: np.ndarray,
    predictive_sigmas: np.ndarray,
    target_values: np.ndarray,
    prior_weights: np.ndarray,
    reference_outcomes: np.ndarray,
    branch_counts: Sequence[int] = BRANCH_COUNTS,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Compare Monte Carlo and quantile expected risks for one action."""

    reference_risks = posterior_risks_for_observations(
        reference_outcomes,
        predictive_means,
        predictive_sigmas,
        target_values,
        prior_weights,
        batch_size=batch_size,
    )
    approximations = {}
    for count in branch_counts:
        representatives, probabilities = quantile_branch_observations(
            reference_outcomes, count
        )
        branch_risks = posterior_risks_for_observations(
            representatives,
            predictive_means,
            predictive_sigmas,
            target_values,
            prior_weights,
            batch_size=batch_size,
        )
        approximations[str(count)] = {
            "expected_risk": float(np.dot(probabilities, branch_risks)),
            "representatives": representatives.tolist(),
            "probabilities": probabilities.tolist(),
        }
    return {
        "reference_expected_risk": float(np.mean(reference_risks)),
        "approximations": approximations,
    }


def _root_risk(target_values: np.ndarray, weights: np.ndarray) -> float:
    means = np.dot(weights, target_values)
    variances = np.dot(weights, np.square(target_values)) - np.square(means)
    result = float(np.mean(np.maximum(0.0, variances)))
    if not math.isfinite(result) or result <= 0:
        raise ValueError("root terminal risk must be finite and positive")
    return result


def _draw_reference_outcomes(
    means: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    count: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(weights), size=count, p=weights)
    return means[indices] + sigmas[indices] * rng.normal(size=count)


def summarize_bank_cases(cases: Sequence[Mapping[str, Any]], branch_count: int) -> dict[str, Any]:
    key = str(branch_count)
    correlations = np.asarray([case["branches"][key]["spearman"] for case in cases])
    regrets = np.asarray(
        [case["branches"][key]["normalized_top_one_regret"] for case in cases]
    )
    return {
        "median_spearman": float(np.median(correlations)),
        "fraction_spearman_at_least_080": float(np.mean(correlations >= 0.80)),
        "fraction_normalized_regret_at_most_003": float(np.mean(regrets <= 0.03)),
        "mean_normalized_top_one_regret": float(np.mean(regrets)),
    }


def evaluate_gates(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_bank = {
        bank: [case for case in cases if case["bank"] == bank]
        for bank in ("bank_1", "bank_2")
    }
    summaries = {
        bank: {str(count): summarize_bank_cases(bank_cases, count) for count in BRANCH_COUNTS}
        for bank, bank_cases in by_bank.items()
    }
    paired = {
        (case["difficulty"], case["domain"]): case
        for case in cases
        if case["bank"] == "bank_1"
    }
    agreements = []
    for case in by_bank["bank_2"]:
        first = paired[(case["difficulty"], case["domain"])]
        agreements.append(
            first["branches"]["9"]["selected_action_index"]
            == case["branches"]["9"]["selected_action_index"]
        )
    conditions = {
        "all_cases_finite_and_reproducible": len(cases) == 72
        and all(case["finite_and_reproducible"] for case in cases),
        "median_spearman_at_least_090_both_banks": all(
            summaries[bank]["9"]["median_spearman"] >= 0.90 for bank in by_bank
        ),
        "fraction_spearman_at_least_080_is_090_both_banks": all(
            summaries[bank]["9"]["fraction_spearman_at_least_080"] >= 0.90
            for bank in by_bank
        ),
        "fraction_regret_at_most_003_is_090_both_banks": all(
            summaries[bank]["9"]["fraction_normalized_regret_at_most_003"] >= 0.90
            for bank in by_bank
        ),
        "mean_normalized_regret_at_most_001_both_banks": all(
            summaries[bank]["9"]["mean_normalized_top_one_regret"] <= 0.01
            for bank in by_bank
        ),
        "bank_selected_action_agreement_at_least_075": float(np.mean(agreements)) >= 0.75,
        "nine_branches_nonworse_than_five_both_banks": all(
            summaries[bank]["9"]["median_spearman"]
            >= summaries[bank]["5"]["median_spearman"]
            and summaries[bank]["9"]["mean_normalized_top_one_regret"]
            <= summaries[bank]["5"]["mean_normalized_top_one_regret"]
            for bank in by_bank
        ),
    }
    return {
        "summaries": summaries,
        "bank_selected_action_agreement": float(np.mean(agreements)),
        "conditions": conditions,
        "pass": all(conditions.values()),
    }


def _evaluate_bank(
    source: Any,
    domain: str,
    difficulty: str,
    bank_name: str,
    bank_seed_base: int,
    prior: Any,
    root_designs: np.ndarray,
    root_observations: np.ndarray,
    action_designs: np.ndarray,
    action_indices: Sequence[int],
    target_designs: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, Any]:
    log_likelihood = _make_log_likelihood(
        source, domain, prior, root_designs, root_observations
    )
    posterior = adaptive_tempered_smc(
        prior,
        log_likelihood,
        num_particles=NUM_PARTICLES,
        seed=_stable_seed(bank_seed_base, difficulty, domain),
        initialization="sobol",
        proposal_geometry="full",
    )
    parameters = prior.decode_many(posterior.particles)
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
    root_risk = _root_risk(target_values, posterior.weights)
    references = []
    approximations = {str(count): [] for count in BRANCH_COUNTS}
    actions = []
    outcome_digest = hashlib.sha256()
    for local_index, assay_index in enumerate(action_indices):
        outcomes = _draw_reference_outcomes(
            action_means[:, local_index],
            action_sigmas[:, local_index],
            posterior.weights,
            seed=_stable_seed(
                OUTCOME_SEED_BASE,
                bank_name,
                difficulty,
                domain,
                str(assay_index),
            ),
            count=REFERENCE_OUTCOMES,
        )
        outcome_digest.update(np.asarray(outcomes, dtype=np.float64).tobytes())
        evaluated = expected_action_risks(
            predictive_means=action_means[:, local_index],
            predictive_sigmas=action_sigmas[:, local_index],
            target_values=target_values,
            prior_weights=posterior.weights,
            reference_outcomes=outcomes,
            batch_size=batch_size,
        )
        references.append(evaluated["reference_expected_risk"])
        for count in BRANCH_COUNTS:
            approximations[str(count)].append(
                evaluated["approximations"][str(count)]["expected_risk"]
            )
        actions.append(
            {
                "assay_index": assay_index,
                "reference_expected_risk": evaluated["reference_expected_risk"],
                "branch_expected_risks": {
                    str(count): evaluated["approximations"][str(count)]["expected_risk"]
                    for count in BRANCH_COUNTS
                },
            }
        )

    reference_values = np.asarray(references)
    branch_payload = {}
    for count in BRANCH_COUNTS:
        approximate = np.asarray(approximations[str(count)])
        selected = int(np.argmin(approximate))
        regret = max(0.0, float(reference_values[selected] - np.min(reference_values)))
        branch_payload[str(count)] = {
            "spearman": _spearman(reference_values, approximate),
            "selected_action_index": int(action_indices[selected]),
            "reference_best_action_index": int(action_indices[int(np.argmin(reference_values))]),
            "top_one_regret": regret,
            "normalized_top_one_regret": regret / root_risk,
        }
    finite = all(
        math.isfinite(value)
        for value in (
            root_risk,
            posterior.log_evidence,
            *(item for values in approximations.values() for item in values),
            *references,
        )
    )
    return {
        "domain": domain,
        "difficulty": difficulty,
        "bank": bank_name,
        "root_risk": root_risk,
        "root_particle_sha256": _array_sha256(posterior.particles, posterior.weights),
        "outcome_sha256": outcome_digest.hexdigest(),
        "finite_and_reproducible": finite,
        "smc": {
            "log_evidence": posterior.log_evidence,
            "num_rungs": posterior.diagnostics.num_rungs,
            "acceptance_rate": posterior.diagnostics.aggregate_acceptance_rate,
        },
        "branches": branch_payload,
        "actions": actions,
    }


def run(
    source_root: Path,
    v1_result_path: Path,
    v3_result_path: Path,
    *,
    batch_size: int = 256,
    progress: bool = False,
) -> dict[str, Any]:
    if _sha256(v3_result_path) != V3_RESULT_SHA256:
        raise ValueError("V3 result hash does not match the branch-fidelity binding")
    v3_result = json.loads(v3_result_path.read_text())
    if not v3_result["source_only"]["gates"]["pass"]:
        raise ValueError("V3 source-only posterior gate did not pass")
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    _, v1_cases = _load_v1_cases(v1_result_path)
    assays = frozen_assays()
    action_indices = tuple(
        index for index in range(len(assays)) if index not in ROOT_HISTORY_INDICES
    )
    root_designs = np.asarray([assays[index].values for index in ROOT_HISTORY_INDICES])
    action_designs = np.asarray([assays[index].values for index in action_indices])
    query_designs = {
        difficulty: _query_assays(
            QUERY_SEEDS[difficulty],
            source.CHEM_INPUT_BOUNDS,
            source.CHEM_LOG_VARS,
            512,
        )[:NUM_TARGETS]
        for difficulty in DIFFICULTIES
    }
    full_history_designs = np.asarray(
        [assays[index].values for index in HISTORY_ASSAY_INDICES]
    )
    cases = []
    started = time.perf_counter()
    for difficulty in DIFFICULTIES:
        for domain_index, domain in enumerate(PANEL_DOMAINS, start=1):
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
            full_observations = np.maximum(
                0.0, truth_means * (1.0 + NOISE_LEVEL * rng.normal(size=8))
            )
            v1_case = v1_cases[("source_only", difficulty, domain)]
            compatibility = {
                "prior_sha256": _prior_sha256(prior),
                "truth_sha256": _parameter_sha256(truth),
                "history_sha256": _array_sha256(
                    full_history_designs, full_observations
                ),
                "query_sha256": _array_sha256(
                    _query_assays(
                        QUERY_SEEDS[difficulty],
                        source.CHEM_INPUT_BOUNDS,
                        source.CHEM_LOG_VARS,
                        512,
                    )
                ),
            }
            if any(compatibility[name] != v1_case[name] for name in compatibility):
                raise ValueError(f"V1 compatibility failed for {difficulty}/{domain}")
            for bank_name, seed_base in (
                ("bank_1", BANK_1_SEED_BASE),
                ("bank_2", BANK_2_SEED_BASE),
            ):
                case_started = time.perf_counter()
                case = _evaluate_bank(
                    source,
                    domain,
                    difficulty,
                    bank_name,
                    seed_base,
                    prior,
                    root_designs,
                    full_observations[:4],
                    action_designs,
                    action_indices,
                    query_designs[difficulty],
                    batch_size=batch_size,
                )
                case["v1_compatibility"] = compatibility
                cases.append(case)
                if progress:
                    print(
                        f"{difficulty} {domain_index:02d}/12 {domain} {bank_name} "
                        f"rho9={case['branches']['9']['spearman']:.3f} "
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
        "source": source_binding,
        "settings": {
            "panel_domains": PANEL_DOMAINS,
            "difficulties": DIFFICULTIES,
            "root_history_indices": ROOT_HISTORY_INDICES,
            "candidate_action_indices": action_indices,
            "num_particles": NUM_PARTICLES,
            "reference_outcomes": REFERENCE_OUTCOMES,
            "branch_counts": BRANCH_COUNTS,
            "num_targets": NUM_TARGETS,
            "outcome_seed_base": OUTCOME_SEED_BASE,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    result = run(
        args.source_root,
        args.v1_result,
        args.v3_result,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
