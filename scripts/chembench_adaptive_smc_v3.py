#!/usr/bin/env python3
"""Run the frozen source-first 512-particle ChemBench SMC V3."""

from __future__ import annotations

import argparse
import json
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
    DIFFICULTIES,
    HISTORY_ASSAY_INDICES,
    NUM_QUERIES,
    QUERY_SEEDS,
    _sha256,
    summarize_cases,
)
from scripts.chembench_adaptive_smc_v2 import (
    V1_RESULT_PATH,
    V1_RESULT_SHA256,
    _bank_ratio_pass,
    _load_v1_cases,
    evaluate_v2_case,
)
from scripts.chembench_mopen_nonmyopic_opportunity import (
    active_domains,
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-adaptive-smc-v3"
PROTOCOL_PATH = "results/nonmyopic/CHEMBENCH_ADAPTIVE_SMC_V3_PROTOCOL_20260815.md"
NUM_PARTICLES = 512
BANK_1_SEED_BASE = 2026083501
BANK_2_SEED_BASE = 2026083502
V2_SOURCE_MSE = 0.04252595039735338
V2_V4_MSE = 0.016629854723866702


def _health_pass(tier_summaries: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        summary["smc_banks"][bank]["max_num_rungs"] <= 80
        and 0.05 <= summary["smc_banks"][bank]["aggregate_acceptance_rate"] <= 0.90
        and summary["smc_banks"][bank]["nonzero_acceptance_fraction"] >= 0.95
        for summary in tier_summaries
        for bank in ("bank_1", "bank_2")
    )


def source_gates(summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tiers = summary["tiers"]
    conditions = {
        "all_v1_hashes_match": all(case["v1_hashes_match"] for case in cases),
        "health_passes_every_tier": _health_pass([tiers[difficulty] for difficulty in DIFFICULTIES]),
        "evidence_spearman_at_least_095_every_tier": all(
            tiers[difficulty]["log_evidence_spearman"] >= 0.95
            for difficulty in DIFFICULTIES
        ),
        "evidence_median_difference_at_most_1_every_tier": all(
            tiers[difficulty]["median_absolute_log_evidence_difference"] <= 1.0
            for difficulty in DIFFICULTIES
        ),
        "aggregate_mse_nonworse_than_v2": summary["aggregate"]["mse"]["smc_mean"]
        <= V2_SOURCE_MSE,
        "better_than_static16_and_static100_every_tier": all(
            tiers[difficulty]["mse"]["smc_mean"]
            < min(
                tiers[difficulty]["mse"]["static_16"],
                tiers[difficulty]["mse"]["static_100"],
            )
            for difficulty in DIFFICULTIES
        ),
        "bank_mse_ratio_at_most_15_every_tier": all(
            _bank_ratio_pass(tiers[difficulty]) for difficulty in DIFFICULTIES
        ),
    }
    return {"conditions": conditions, "pass": all(conditions.values())}


def opened_v4_gates(
    summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    tiers = summary["tiers"]
    conditions = {
        "all_v1_hashes_match": all(case["v1_hashes_match"] for case in cases),
        "health_passes_every_tier": _health_pass([tiers[difficulty] for difficulty in DIFFICULTIES]),
        "aggregate_mse_nonworse_than_v2": summary["aggregate"]["mse"]["smc_mean"]
        <= V2_V4_MSE,
        "aggregate_better_than_static16_and_static100": summary["aggregate"]["mse"][
            "smc_mean"
        ]
        < min(
            summary["aggregate"]["mse"]["static_16"],
            summary["aggregate"]["mse"]["static_100"],
        ),
        "bank_mse_ratio_at_most_15_every_tier": all(
            _bank_ratio_pass(tiers[difficulty]) for difficulty in DIFFICULTIES
        ),
    }
    return {"conditions": conditions, "pass": all(conditions.values())}


def _summarize(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "aggregate": summarize_cases(cases),
        "tiers": {
            difficulty: summarize_cases(
                [case for case in cases if case["difficulty"] == difficulty]
            )
            for difficulty in DIFFICULTIES
        },
    }


def _run_cohort(
    source: Any,
    cohort: str,
    domains: Sequence[str],
    history_designs: np.ndarray,
    query_designs: Mapping[str, np.ndarray],
    v1_cases: Mapping[tuple[str, str, str], Any],
    *,
    progress: bool,
) -> list[dict[str, Any]]:
    cases = []
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
                num_particles=NUM_PARTICLES,
                bank_1_seed_base=BANK_1_SEED_BASE,
                bank_2_seed_base=BANK_2_SEED_BASE,
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
    return cases


def run(source_root: Path, v1_result_path: Path, *, progress: bool = False) -> dict[str, Any]:
    if _sha256(v1_result_path) != V1_RESULT_SHA256:
        raise ValueError("V1 result hash does not match V3 binding")
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    _, v1_cases = _load_v1_cases(v1_result_path)
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
    started = time.perf_counter()
    source_cases = _run_cohort(
        source,
        "source_only",
        domains,
        history_designs,
        query_designs,
        v1_cases,
        progress=progress,
    )
    source_summary = _summarize(source_cases)
    source_gate = source_gates(source_summary, source_cases)
    opened_v4_cases: list[dict[str, Any]] = []
    opened_v4_summary = None
    opened_v4_gate = None
    if source_gate["pass"]:
        opened_v4_cases = _run_cohort(
            source,
            "opened_v4",
            domains,
            history_designs,
            query_designs,
            v1_cases,
            progress=progress,
        )
        opened_v4_summary = _summarize(opened_v4_cases)
        opened_v4_gate = opened_v4_gates(opened_v4_summary, opened_v4_cases)
    full_pass = source_gate["pass"] and bool(opened_v4_gate and opened_v4_gate["pass"])
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {"path": PROTOCOL_PATH, "sha256": _sha256(REPO_ROOT / PROTOCOL_PATH)},
        "v1_result": {"path": V1_RESULT_PATH, "sha256": _sha256(v1_result_path)},
        "source": source_binding,
        "settings": {
            "num_particles": NUM_PARTICLES,
            "initialization": "digitally_shifted_sobol",
            "proposal_geometry": "regularized_full_covariance",
            "bank_seed_bases": (BANK_1_SEED_BASE, BANK_2_SEED_BASE),
            "source_first": True,
        },
        "source_only": {"summary": source_summary, "gates": source_gate},
        "opened_v4": {
            "opened": source_gate["pass"],
            "summary": opened_v4_summary,
            "gates": opened_v4_gate,
        },
        "full_pass": full_pass,
        "cases": source_cases + opened_v4_cases,
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
