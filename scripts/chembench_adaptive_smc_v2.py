#!/usr/bin/env python3
"""Run the frozen 256-particle Sobol/full-covariance ChemBench SMC V2."""

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
    NUM_QUERIES,
    OBSERVATION_SEED_BASE,
    QUERY_SEEDS,
    _array_sha256,
    _evaluate_rates,
    _make_log_likelihood,
    _parameter_sha256,
    _posterior_mse,
    _prior_for,
    _prior_sha256,
    _sha256,
    _stable_seed,
    _truth_parameters,
    summarize_cases,
)
from scripts.chembench_mopen_nonmyopic_opportunity import (
    active_domains,
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-adaptive-smc-v2"
PROTOCOL_PATH = "results/nonmyopic/CHEMBENCH_ADAPTIVE_SMC_V2_PROTOCOL_20260815.md"
V1_RESULT_PATH = "results/nonmyopic/chembench_adaptive_smc_calibration/result.json"
V1_RESULT_SHA256 = "34b28639ce9b8625a69b8a5ba74e6e4f78573c0c1829f70ffb1c080895d1ae8f"
SMC_V2_BANK_1_SEED_BASE = 2026083401
SMC_V2_BANK_2_SEED_BASE = 2026083402
NUM_PARTICLES = 256
V1_SOURCE_MSE = 0.04656124523514992
V1_V4_MSE = 0.03461759019005903


def _load_v1_cases(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str, str], Any]]:
    if _sha256(path) != V1_RESULT_SHA256:
        raise ValueError("V1 result hash does not match the frozen binding")
    result = json.loads(path.read_text())
    cases = {
        (case["cohort"], case["difficulty"], case["domain"]): case
        for case in result["cases"]
    }
    if len(cases) != 288:
        raise ValueError("V1 result does not contain 288 unique cases")
    return result, cases


def evaluate_v2_case(
    source: Any,
    domain: str,
    difficulty: str,
    cohort: str,
    history_designs: np.ndarray,
    query_designs: np.ndarray,
    v1_case: Mapping[str, Any],
    *,
    num_particles: int = NUM_PARTICLES,
    bank_1_seed_base: int = SMC_V2_BANK_1_SEED_BASE,
    bank_2_seed_base: int = SMC_V2_BANK_2_SEED_BASE,
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
    hashes = {
        "prior_sha256": _prior_sha256(prior),
        "truth_sha256": _parameter_sha256(truth_parameters),
        "history_sha256": _array_sha256(history_designs, observations),
        "query_sha256": _array_sha256(query_designs),
    }
    for name, value in hashes.items():
        if value != v1_case[name]:
            raise ValueError(f"V2 {name} does not match V1 for {cohort}/{difficulty}/{domain}")

    log_likelihood = _make_log_likelihood(
        source,
        domain,
        prior,
        history_designs,
        observations,
    )
    bank_1 = adaptive_tempered_smc(
        prior,
        log_likelihood,
        num_particles=num_particles,
        seed=_stable_seed(bank_1_seed_base, difficulty, domain),
        initialization="sobol",
        proposal_geometry="full",
    )
    bank_2 = adaptive_tempered_smc(
        prior,
        log_likelihood,
        num_particles=num_particles,
        seed=_stable_seed(bank_2_seed_base, difficulty, domain),
        initialization="sobol",
        proposal_geometry="full",
    )
    mse_1 = _posterior_mse(source, domain, prior, bank_1, query_designs, truth_targets)
    mse_2 = _posterior_mse(source, domain, prior, bank_2, query_designs, truth_targets)

    def bank_payload(result: Any) -> dict[str, Any]:
        return {
            "log_evidence": result.log_evidence,
            "num_rungs": result.diagnostics.num_rungs,
            "acceptance_rate": result.diagnostics.aggregate_acceptance_rate,
            "invalid_proposal_rate": result.diagnostics.aggregate_invalid_proposal_rate,
            "accepted": result.diagnostics.total_accepted,
            "proposals": result.diagnostics.total_proposals,
        }

    return {
        "domain": domain,
        "difficulty": difficulty,
        "cohort": cohort,
        **hashes,
        "v1_hashes_match": True,
        "num_parameters": prior.dimension,
        "outside_prior_coordinates": v1_case["outside_prior_coordinates"],
        "truth_inside_prior": v1_case["truth_inside_prior"],
        "mse": {
            "static_16": v1_case["mse"]["static_16"],
            "static_100": v1_case["mse"]["static_100"],
            "smc_1": mse_1,
            "smc_2": mse_2,
            "smc_mean": 0.5 * (mse_1 + mse_2),
        },
        "smc": {"bank_1": bank_payload(bank_1), "bank_2": bank_payload(bank_2)},
    }


def _bank_ratio_pass(summary: Mapping[str, Any]) -> bool:
    first = float(summary["mse"]["smc_1"])
    second = float(summary["mse"]["smc_2"])
    if first < 1e-6 and second < 1e-6:
        return True
    smaller = min(first, second)
    return smaller > 0 and max(first, second) <= 1.5 * smaller


def v2_gates(summaries: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = summaries["source_only"]
    opened_v4 = summaries["opened_v4"]
    health = [
        summaries[cohort]["tiers"][difficulty]
        for cohort in ("source_only", "opened_v4")
        for difficulty in DIFFICULTIES
    ]
    conditions = {
        "all_v1_hashes_match": all(case["v1_hashes_match"] for case in cases),
        "all_runs_within_80_rungs": all(
            summary["smc_banks"][bank]["max_num_rungs"] <= 80
            for summary in health
            for bank in ("bank_1", "bank_2")
        ),
        "acceptance_between_005_and_090_every_tier": all(
            0.05 <= summary["smc_banks"][bank]["aggregate_acceptance_rate"] <= 0.90
            for summary in health
            for bank in ("bank_1", "bank_2")
        ),
        "nonzero_acceptance_at_least_095": all(
            summary["smc_banks"][bank]["nonzero_acceptance_fraction"] >= 0.95
            for summary in health
            for bank in ("bank_1", "bank_2")
        ),
        "source_evidence_spearman_at_least_095_each_tier": all(
            source["tiers"][difficulty]["log_evidence_spearman"] >= 0.95
            for difficulty in DIFFICULTIES
        ),
        "source_evidence_median_difference_at_most_1_each_tier": all(
            source["tiers"][difficulty]["median_absolute_log_evidence_difference"] <= 1.0
            for difficulty in DIFFICULTIES
        ),
        "source_mse_nonworse_than_v1": source["aggregate"]["mse"]["smc_mean"]
        <= V1_SOURCE_MSE,
        "opened_v4_mse_nonworse_than_v1": opened_v4["aggregate"]["mse"]["smc_mean"]
        <= V1_V4_MSE,
        "bank_mse_ratio_at_most_15_every_cohort_tier": all(
            _bank_ratio_pass(summary) for summary in health
        ),
        "source_better_than_both_static_baselines_each_tier": all(
            source["tiers"][difficulty]["mse"]["smc_mean"]
            < min(
                source["tiers"][difficulty]["mse"]["static_16"],
                source["tiers"][difficulty]["mse"]["static_100"],
            )
            for difficulty in DIFFICULTIES
        ),
        "opened_v4_better_than_both_static_baselines_aggregate": opened_v4[
            "aggregate"
        ]["mse"]["smc_mean"]
        < min(
            opened_v4["aggregate"]["mse"]["static_16"],
            opened_v4["aggregate"]["mse"]["static_100"],
        ),
    }
    return {"conditions": conditions, "gate_pass": all(conditions.values())}


def run(source_root: Path, v1_result_path: Path, *, progress: bool = False) -> dict[str, Any]:
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    v1_result, v1_cases = _load_v1_cases(v1_result_path)
    domains = active_domains(source)[9:]
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
                case = evaluate_v2_case(
                    source,
                    domain,
                    difficulty,
                    cohort,
                    history_designs,
                    query_designs[difficulty],
                    v1_cases[(cohort, difficulty, domain)],
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
    summaries = {}
    for cohort in ("source_only", "opened_v4"):
        cohort_cases = [case for case in cases if case["cohort"] == cohort]
        summaries[cohort] = {
            "aggregate": summarize_cases(cohort_cases),
            "tiers": {
                difficulty: summarize_cases(
                    [case for case in cohort_cases if case["difficulty"] == difficulty]
                )
                for difficulty in DIFFICULTIES
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {"path": PROTOCOL_PATH, "sha256": _sha256(REPO_ROOT / PROTOCOL_PATH)},
        "v1_result": {"path": V1_RESULT_PATH, "sha256": _sha256(v1_result_path)},
        "source": source_binding,
        "settings": {
            "num_particles": NUM_PARTICLES,
            "initialization": "scrambled_sobol",
            "proposal_geometry": "full_covariance",
            "proposal_covariance_scale": 0.25,
            "proposal_floor_fraction": 0.01,
            "target_ess_fraction": 0.6,
            "rejuvenation_moves": 3,
            "max_tempering_rungs": 80,
            "bank_seed_bases": (SMC_V2_BANK_1_SEED_BASE, SMC_V2_BANK_2_SEED_BASE),
            "v1_settings_sha256": _array_sha256(
                np.asarray(v1_result["settings"]["history_assay_indices"], dtype=float)
            ),
            "noise_level": NOISE_LEVEL,
            "absolute_log_noise_floor": ABSOLUTE_LOG_NOISE_FLOOR,
        },
        "summaries": summaries,
        "gates": v2_gates(summaries, cases),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    result = run(args.source_root, args.v1_result, progress=args.progress)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
