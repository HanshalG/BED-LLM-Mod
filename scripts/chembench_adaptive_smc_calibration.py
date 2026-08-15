#!/usr/bin/env python3
"""Run the frozen zero-call ChemBench adaptive-SMC calibration."""

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

from environments.chembench_mopen.smc import (
    ImportanceResult,
    SMCResult,
    TransformedParameterPrior,
    adaptive_tempered_smc,
    static_importance_sample,
)
from environments.chembench_mopen.source import _query_assays
from scripts.chembench_mopen_nonmyopic_opportunity import (
    active_domains,
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-adaptive-smc-calibration-v1"
PROTOCOL_PATH = (
    "results/nonmyopic/CHEMBENCH_ADAPTIVE_SMC_CALIBRATION_PROTOCOL_20260815.md"
)
CLARIFICATION_PATH = (
    "results/nonmyopic/"
    "CHEMBENCH_ADAPTIVE_SMC_CALIBRATION_POSITIVITY_CLARIFICATION_20260815.md"
)
DIFFICULTIES = ("easy", "medium", "hard")
QUERY_SEEDS = {"easy": 2026081701, "medium": 2026081702, "hard": 2026081703}
HISTORY_ASSAY_INDICES = (1, 4, 5, 8, 9, 11, 13, 15)
TRUTH_SEED_BASE = 2026083100
OBSERVATION_SEED_BASE = 2026083200
STATIC_16_SEED_BASE = 2026083300
STATIC_100_AND_SMC_1_SEED_BASE = 2026083301
SMC_2_SEED_BASE = 2026083302
NOISE_LEVEL = 0.01
ABSOLUTE_LOG_NOISE_FLOOR = 1e-4
NUM_QUERIES = 512
NUM_PARTICLES_SMC = 100
PRACTICAL_TOLERANCE = 1e-8


def _stable_seed(base: int, *parts: str) -> int:
    payload = ":".join((str(base), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array, dtype=np.float64)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _parameter_sha256(parameters: Mapping[str, Any]) -> str:
    payload = {name: float(value) for name, value in parameters.items()}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evaluate_rates(
    source: Any,
    domain: str,
    parameter_states: Sequence[Mapping[str, Any]],
    designs: np.ndarray,
    *,
    apply_secondary_effects: bool,
) -> np.ndarray:
    design_values = np.asarray(designs, dtype=float)
    if design_values.ndim != 2 or design_values.shape[1] != 7:
        raise ValueError("ChemBench designs must have shape (count, 7)")
    result = np.empty((len(parameter_states), len(design_values)), dtype=float)
    rate_fn = source._RATE_FNS[domain]
    secondary = np.asarray(
        [source._secondary_effects(row[5], row[6]) for row in design_values],
        dtype=float,
    )
    for parameter_index, parameters in enumerate(parameter_states):
        for design_index, design in enumerate(design_values):
            value = float(rate_fn(parameters, *design))
            if apply_secondary_effects:
                value *= float(secondary[design_index])
            result[parameter_index, design_index] = value
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"non-finite or negative rate for {domain}")
    return result


def _make_log_likelihood(
    source: Any,
    domain: str,
    prior: TransformedParameterPrior,
    history_designs: np.ndarray,
    observations: np.ndarray,
):
    transformed_observations = np.log1p(observations)

    def log_likelihood(coordinates: np.ndarray) -> np.ndarray:
        parameters = prior.decode_many(coordinates)
        means = _evaluate_rates(
            source,
            domain,
            parameters,
            history_designs,
            apply_secondary_effects=True,
        )
        log_means = np.log1p(means)
        sigmas = np.maximum(
            NOISE_LEVEL * means / (1.0 + means),
            ABSOLUTE_LOG_NOISE_FLOOR,
        )
        residuals = (transformed_observations[None, :] - log_means) / sigmas
        values = -0.5 * (residuals**2 + math.log(2.0 * math.pi)) - np.log(sigmas)
        return values.sum(axis=1)

    return log_likelihood


def _posterior_mse(
    source: Any,
    domain: str,
    prior: TransformedParameterPrior,
    result: ImportanceResult | SMCResult,
    query_designs: np.ndarray,
    truth_targets: np.ndarray,
) -> float:
    parameters = prior.decode_many(result.particles)
    rates = _evaluate_rates(
        source,
        domain,
        parameters,
        query_designs,
        apply_secondary_effects=False,
    )
    forecast = np.dot(result.weights, np.log1p(rates))
    return float(np.mean(np.square(forecast - truth_targets)))


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    first_ranks = _rankdata(first)
    second_ranks = _rankdata(second)
    if np.std(first_ranks) == 0 or np.std(second_ranks) == 0:
        return math.nan
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def _paired_counts(
    challenger: Sequence[float],
    baseline: Sequence[float],
    *,
    tolerance: float = PRACTICAL_TOLERANCE,
) -> dict[str, int]:
    differences = np.asarray(baseline) - np.asarray(challenger)
    return {
        "wins": int(np.count_nonzero(differences > tolerance)),
        "ties": int(np.count_nonzero(np.abs(differences) <= tolerance)),
        "losses": int(np.count_nonzero(differences < -tolerance)),
    }


def _prior_for(source: Any, domain: str, difficulty: str) -> TransformedParameterPrior:
    states = tuple(source._PARAMS[domain][difficulty][version] for version in ("v0", "v1", "v2", "v3"))
    return TransformedParameterPrior.from_parameter_states(states, expansion_factor=1.5)


def _prior_sha256(prior: TransformedParameterPrior) -> str:
    payload = {
        "names": prior.names,
        "transforms": prior.transforms,
        "lower": prior.lower.tolist(),
        "upper": prior.upper.tolist(),
        "ordered_pairs": prior.ordered_pairs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _truth_parameters(
    source: Any,
    prior: TransformedParameterPrior,
    domain: str,
    difficulty: str,
    cohort: str,
) -> dict[str, float]:
    if cohort == "source_only":
        seed = _stable_seed(TRUTH_SEED_BASE, difficulty, domain)
        return prior.decode(prior.sample(np.random.default_rng(seed), 1)[0])
    if cohort == "opened_v4":
        return {
            name: float(value)
            for name, value in source._PARAMS[domain][difficulty]["v4"].items()
        }
    raise ValueError(f"unknown cohort: {cohort}")


def evaluate_case(
    source: Any,
    domain: str,
    difficulty: str,
    cohort: str,
    history_designs: np.ndarray,
    query_designs: np.ndarray,
) -> dict[str, Any]:
    prior = _prior_for(source, domain, difficulty)
    truth_parameters = _truth_parameters(source, prior, domain, difficulty, cohort)
    truth_history_means = _evaluate_rates(
        source,
        domain,
        (truth_parameters,),
        history_designs,
        apply_secondary_effects=True,
    )[0]
    observation_rng = np.random.default_rng(
        _stable_seed(OBSERVATION_SEED_BASE, cohort, difficulty, domain)
    )
    observations = np.maximum(
        0.0,
        truth_history_means * (1.0 + NOISE_LEVEL * observation_rng.normal(size=8)),
    )
    truth_targets = np.log1p(
        _evaluate_rates(
            source,
            domain,
            (truth_parameters,),
            query_designs,
            apply_secondary_effects=False,
        )[0]
    )
    log_likelihood = _make_log_likelihood(
        source,
        domain,
        prior,
        history_designs,
        observations,
    )

    static_16_seed = _stable_seed(STATIC_16_SEED_BASE, difficulty, domain)
    static_100_seed = _stable_seed(STATIC_100_AND_SMC_1_SEED_BASE, difficulty, domain)
    smc_2_seed = _stable_seed(SMC_2_SEED_BASE, difficulty, domain)
    static_16 = static_importance_sample(
        prior,
        log_likelihood,
        num_particles=16,
        seed=static_16_seed,
    )
    static_100 = static_importance_sample(
        prior,
        log_likelihood,
        num_particles=100,
        seed=static_100_seed,
    )
    smc_1 = adaptive_tempered_smc(
        prior,
        log_likelihood,
        num_particles=NUM_PARTICLES_SMC,
        seed=static_100_seed,
        target_ess_fraction=0.6,
        rejuvenation_moves=3,
        max_tempering_rungs=80,
    )
    smc_2 = adaptive_tempered_smc(
        prior,
        log_likelihood,
        num_particles=NUM_PARTICLES_SMC,
        seed=smc_2_seed,
        target_ess_fraction=0.6,
        rejuvenation_moves=3,
        max_tempering_rungs=80,
    )
    mse = {
        "static_16": _posterior_mse(
            source, domain, prior, static_16, query_designs, truth_targets
        ),
        "static_100": _posterior_mse(
            source, domain, prior, static_100, query_designs, truth_targets
        ),
        "smc_1": _posterior_mse(source, domain, prior, smc_1, query_designs, truth_targets),
        "smc_2": _posterior_mse(source, domain, prior, smc_2, query_designs, truth_targets),
    }
    mse["smc_mean"] = 0.5 * (mse["smc_1"] + mse["smc_2"])
    outside_count = prior.outside_coordinate_count(truth_parameters)
    return {
        "domain": domain,
        "difficulty": difficulty,
        "cohort": cohort,
        "prior_sha256": _prior_sha256(prior),
        "truth_sha256": _parameter_sha256(truth_parameters),
        "history_sha256": _array_sha256(history_designs, observations),
        "query_sha256": _array_sha256(query_designs),
        "num_parameters": prior.dimension,
        "outside_prior_coordinates": outside_count,
        "truth_inside_prior": outside_count == 0,
        "mse": mse,
        "importance": {
            "static_16_ess": static_16.effective_sample_size,
            "static_100_ess": static_100.effective_sample_size,
        },
        "smc": {
            "bank_1": {
                "log_evidence": smc_1.log_evidence,
                "num_rungs": smc_1.diagnostics.num_rungs,
                "acceptance_rate": smc_1.diagnostics.aggregate_acceptance_rate,
                "invalid_proposal_rate": smc_1.diagnostics.aggregate_invalid_proposal_rate,
                "accepted": smc_1.diagnostics.total_accepted,
                "proposals": smc_1.diagnostics.total_proposals,
            },
            "bank_2": {
                "log_evidence": smc_2.log_evidence,
                "num_rungs": smc_2.diagnostics.num_rungs,
                "acceptance_rate": smc_2.diagnostics.aggregate_acceptance_rate,
                "invalid_proposal_rate": smc_2.diagnostics.aggregate_invalid_proposal_rate,
                "accepted": smc_2.diagnostics.total_accepted,
                "proposals": smc_2.diagnostics.total_proposals,
            },
        },
    }


def summarize_cases(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    methods = ("static_16", "static_100", "smc_1", "smc_2", "smc_mean")
    mse = {
        method: float(np.mean([case["mse"][method] for case in cases]))
        for method in methods
    }
    smc_values = [case["mse"]["smc_mean"] for case in cases]
    evidence_1 = [case["smc"]["bank_1"]["log_evidence"] for case in cases]
    evidence_2 = [case["smc"]["bank_2"]["log_evidence"] for case in cases]
    total_coordinates = sum(case["num_parameters"] for case in cases)
    outside_coordinates = sum(case["outside_prior_coordinates"] for case in cases)
    banks = {}
    for bank in ("bank_1", "bank_2"):
        accepted = sum(case["smc"][bank]["accepted"] for case in cases)
        proposals = sum(case["smc"][bank]["proposals"] for case in cases)
        banks[bank] = {
            "aggregate_acceptance_rate": accepted / proposals,
            "nonzero_acceptance_fraction": float(
                np.mean([case["smc"][bank]["accepted"] > 0 for case in cases])
            ),
            "mean_invalid_proposal_rate": float(
                np.mean([case["smc"][bank]["invalid_proposal_rate"] for case in cases])
            ),
            "mean_num_rungs": float(
                np.mean([case["smc"][bank]["num_rungs"] for case in cases])
            ),
            "max_num_rungs": max(case["smc"][bank]["num_rungs"] for case in cases),
        }
    return {
        "num_cases": len(cases),
        "mse": mse,
        "smc_relative_change_vs_static_16": (mse["smc_mean"] - mse["static_16"])
        / mse["static_16"],
        "smc_relative_change_vs_static_100": (mse["smc_mean"] - mse["static_100"])
        / mse["static_100"],
        "paired_vs_static_16": _paired_counts(
            smc_values, [case["mse"]["static_16"] for case in cases]
        ),
        "paired_vs_static_100": _paired_counts(
            smc_values, [case["mse"]["static_100"] for case in cases]
        ),
        "log_evidence_spearman": _spearman(evidence_1, evidence_2),
        "median_absolute_log_evidence_difference": float(
            np.median(np.abs(np.asarray(evidence_1) - np.asarray(evidence_2)))
        ),
        "outside_prior_coordinate_fraction": outside_coordinates / total_coordinates,
        "truth_inside_prior_fraction": float(
            np.mean([case["truth_inside_prior"] for case in cases])
        ),
        "smc_banks": banks,
    }


def _numeric_gates(
    cohort_summaries: Mapping[str, Any],
) -> dict[str, Any]:
    source_tiers = cohort_summaries["source_only"]["tiers"]
    source_aggregate = cohort_summaries["source_only"]["aggregate"]
    v4_aggregate = cohort_summaries["opened_v4"]["aggregate"]
    health_groups = [
        cohort_summaries[cohort]["tiers"][difficulty]
        for cohort in ("source_only", "opened_v4")
        for difficulty in DIFFICULTIES
    ]
    conditions = {
        "all_smc_runs_reach_temperature_one_within_80_rungs": all(
            summary["smc_banks"][bank]["max_num_rungs"] <= 80
            for summary in health_groups
            for bank in ("bank_1", "bank_2")
        ),
        "acceptance_between_005_and_090_every_tier": all(
            0.05 <= summary["smc_banks"][bank]["aggregate_acceptance_rate"] <= 0.90
            for summary in health_groups
            for bank in ("bank_1", "bank_2")
        ),
        "nonzero_acceptance_at_least_095": all(
            summary["smc_banks"][bank]["nonzero_acceptance_fraction"] >= 0.95
            for summary in health_groups
            for bank in ("bank_1", "bank_2")
        ),
        "source_smc_at_least_10pct_better_than_static16": source_aggregate[
            "smc_relative_change_vs_static_16"
        ]
        <= -0.10,
        "source_smc_no_more_than_2pct_worse_than_static100": source_aggregate[
            "smc_relative_change_vs_static_100"
        ]
        <= 0.02,
        "source_smc_better_than_static16_each_tier": all(
            source_tiers[difficulty]["mse"]["smc_mean"]
            < source_tiers[difficulty]["mse"]["static_16"]
            for difficulty in DIFFICULTIES
        ),
        "source_smc_no_more_than_5pct_worse_than_static100_each_tier": all(
            source_tiers[difficulty]["mse"]["smc_mean"]
            <= 1.05 * source_tiers[difficulty]["mse"]["static_100"]
            for difficulty in DIFFICULTIES
        ),
        "source_paired_wins_exceed_losses_vs_static16": source_aggregate[
            "paired_vs_static_16"
        ]["wins"]
        > source_aggregate["paired_vs_static_16"]["losses"],
        "source_evidence_spearman_at_least_095_each_tier": all(
            source_tiers[difficulty]["log_evidence_spearman"] >= 0.95
            for difficulty in DIFFICULTIES
        ),
        "source_evidence_median_absolute_difference_at_most_1_each_tier": all(
            source_tiers[difficulty]["median_absolute_log_evidence_difference"] <= 1.0
            for difficulty in DIFFICULTIES
        ),
        "opened_v4_smc_better_than_static16": v4_aggregate["mse"]["smc_mean"]
        < v4_aggregate["mse"]["static_16"],
        "opened_v4_support_fraction_disclosed": all(
            "outside_prior_coordinate_fraction"
            in cohort_summaries["opened_v4"]["tiers"][difficulty]
            for difficulty in DIFFICULTIES
        ),
    }
    return {
        "conditions": conditions,
        "numerical_gate_pass": all(conditions.values()),
        "requires_focused_test_pass_for_final_status": True,
    }


def run(source_root: Path, *, progress: bool = False) -> dict[str, Any]:
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    domains = active_domains(source)[9:]
    if len(domains) != 48:
        raise ValueError(f"expected 48 compound domains, got {len(domains)}")
    assays = frozen_assays()
    history_designs = np.asarray(
        [assays[index].values for index in HISTORY_ASSAY_INDICES], dtype=float
    )
    query_designs = {
        difficulty: _query_assays(
            QUERY_SEEDS[difficulty],
            source.CHEM_INPUT_BOUNDS,
            source.CHEM_LOG_VARS,
            NUM_QUERIES,
        )
        for difficulty in DIFFICULTIES
    }
    cases: list[dict[str, Any]] = []
    started = time.perf_counter()
    for cohort in ("source_only", "opened_v4"):
        for difficulty in DIFFICULTIES:
            for index, domain in enumerate(domains, start=1):
                case_started = time.perf_counter()
                case = evaluate_case(
                    source,
                    domain,
                    difficulty,
                    cohort,
                    history_designs,
                    query_designs[difficulty],
                )
                cases.append(case)
                if progress:
                    print(
                        f"{cohort} {difficulty} {index:02d}/48 {domain} "
                        f"smc={case['mse']['smc_mean']:.6g} "
                        f"elapsed={time.perf_counter() - case_started:.2f}s",
                        file=sys.stderr,
                        flush=True,
                    )
    cohort_summaries = {}
    for cohort in ("source_only", "opened_v4"):
        cohort_cases = [case for case in cases if case["cohort"] == cohort]
        cohort_summaries[cohort] = {
            "aggregate": summarize_cases(cohort_cases),
            "tiers": {
                difficulty: summarize_cases(
                    [case for case in cohort_cases if case["difficulty"] == difficulty]
                )
                for difficulty in DIFFICULTIES
            },
        }
    prior_consistency = all(
        next(
            case["prior_sha256"]
            for case in cases
            if case["cohort"] == "source_only"
            and case["difficulty"] == difficulty
            and case["domain"] == domain
        )
        == next(
            case["prior_sha256"]
            for case in cases
            if case["cohort"] == "opened_v4"
            and case["difficulty"] == difficulty
            and case["domain"] == domain
        )
        for difficulty in DIFFICULTIES
        for domain in domains
    )
    gates = _numeric_gates(cohort_summaries)
    gates["conditions"]["identical_prior_hashes_across_cohorts"] = prior_consistency
    gates["numerical_gate_pass"] = all(gates["conditions"].values())
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "path": PROTOCOL_PATH,
            "sha256": _sha256(REPO_ROOT / PROTOCOL_PATH),
            "clarification_path": CLARIFICATION_PATH,
            "clarification_sha256": _sha256(REPO_ROOT / CLARIFICATION_PATH),
        },
        "source": source_binding,
        "settings": {
            "difficulties": DIFFICULTIES,
            "cohorts": ("source_only", "opened_v4"),
            "compound_domains": len(domains),
            "history_assay_indices": HISTORY_ASSAY_INDICES,
            "history_assay_names": tuple(assays[index].name for index in HISTORY_ASSAY_INDICES),
            "num_queries": NUM_QUERIES,
            "query_seeds": QUERY_SEEDS,
            "truth_seed_base": TRUTH_SEED_BASE,
            "observation_seed_base": OBSERVATION_SEED_BASE,
            "static_16_seed_base": STATIC_16_SEED_BASE,
            "static_100_and_smc_1_seed_base": STATIC_100_AND_SMC_1_SEED_BASE,
            "smc_2_seed_base": SMC_2_SEED_BASE,
            "noise_level": NOISE_LEVEL,
            "absolute_log_noise_floor": ABSOLUTE_LOG_NOISE_FLOOR,
            "smc_num_particles": NUM_PARTICLES_SMC,
            "smc_target_ess_fraction": 0.6,
            "smc_rejuvenation_moves": 3,
            "smc_max_tempering_rungs": 80,
        },
        "summaries": cohort_summaries,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    result = run(args.source_root, progress=args.progress)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
