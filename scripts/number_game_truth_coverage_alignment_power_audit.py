#!/usr/bin/env python3
"""Estimate plug-in power for the prospective truth-coverage claim tier."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_support_capacity_selector_audit as capacity
from scripts import number_game_two_draw_diversity_bonus_confirmation64_staged as staged


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-truth-coverage-alignment-power-audit-1"
SIMULATION_SEED = 112_900
SIMULATION_DRAWS = 10_000
SAMPLE_SIZES = (32, 64, 96)
HISTORICAL_SOURCE = "later_fresh32"
PRIOR_BRIER_POWER_RESULT = REPO_ROOT / (
    "results/nonmyopic/number_game_two_draw_diversity_bonus_power_audit/"
    "number-game-two-draw-diversity-bonus-power-audit-20260806T013000Z/"
    "RESULT.json"
)
PRIOR_BRIER_POWER_RESULT_SHA256 = (
    "3dfd1605deba87641c791718a8b50d81e971062fee85612fa94638fc674f156e"
)


def _coverage_gate(differences: Sequence[float]) -> bool:
    if len(differences) < 2:
        return False
    mean = statistics.fmean(differences)
    lower = mean - 1.96 * statistics.stdev(differences) / math.sqrt(
        len(differences)
    )
    wins = sum(value > 1e-15 for value in differences)
    losses = sum(value < -1e-15 for value in differences)
    return mean > 0.0 and lower > 0.0 and wins > losses


def simulate_power(
    rows: Sequence[dict[str, Any]],
    *,
    sample_sizes: Sequence[int] = SAMPLE_SIZES,
    draws: int = SIMULATION_DRAWS,
    seed: int = SIMULATION_SEED,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("power audit requires historical rows")
    if draws <= 0 or any(size <= 1 for size in sample_sizes):
        raise ValueError("power audit sizes and draws must be positive")
    specs = staged.COVERAGE_COMPARISON_SPECS
    coverage_rows = {
        name: staged._coverage_comparison_rows(
            rows,
            candidate_key=candidate,
            baseline_key=baseline,
        )
        for name, (candidate, baseline) in specs.items()
    }
    alignment_rows = {
        name: staged._coverage_brier_alignment_rows(
            rows,
            candidate_key=candidate,
            baseline_key=baseline,
        )
        for name, (candidate, baseline) in specs.items()
    }
    rng = random.Random(seed)
    output = {}
    for sample_size in sample_sizes:
        simulated = []
        for _ in range(draws):
            indices = [rng.randrange(len(rows)) for _ in range(sample_size)]
            coverage_passes = {}
            rhos = {}
            for name in specs:
                coverage_passes[name] = _coverage_gate(
                    [
                        float(coverage_rows[name][index]["difference"])
                        for index in indices
                    ]
                )
                rhos[name] = staged._changed_root_spearman(
                    [alignment_rows[name][index] for index in indices]
                )
            simulated.append(
                {
                    "coverage": coverage_passes,
                    "rhos": rhos,
                    "family_mean_rho": statistics.fmean(rhos.values()),
                }
            )
        family_sd = statistics.stdev(
            float(item["family_mean_rho"]) for item in simulated
        )
        family_lower_positive_threshold = 1.96 * family_sd
        coverage_family = [
            all(item["coverage"].values()) for item in simulated
        ]
        alignment_family = [
            all(float(value) > 0.0 for value in item["rhos"].values())
            and float(item["family_mean_rho"])
            > family_lower_positive_threshold
            for item in simulated
        ]
        output[str(sample_size)] = {
            "draws": draws,
            "coverage_component_normal_interval_pass_probability": {
                name: statistics.fmean(
                    float(item["coverage"][name]) for item in simulated
                )
                for name in specs
            },
            "coverage_family_pass_probability": statistics.fmean(
                float(value) for value in coverage_family
            ),
            "alignment_component_positive_probability": {
                name: statistics.fmean(
                    float(float(item["rhos"][name]) > 0.0)
                    for item in simulated
                )
                for name in specs
            },
            "family_mean_rho_sampling_sd": family_sd,
            "family_lower_positive_normal_threshold": (
                family_lower_positive_threshold
            ),
            "alignment_family_approximate_pass_probability": statistics.fmean(
                float(value) for value in alignment_family
            ),
            "coverage_and_alignment_approximate_pass_probability": (
                statistics.fmean(
                    float(coverage and alignment)
                    for coverage, alignment in zip(
                        coverage_family, alignment_family, strict=True
                    )
                )
            ),
        }
    return output


def run_audit(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    source_spec = next(
        dict(spec)
        for spec in capacity.diversity.SOURCE_SPECS
        if spec["name"] == HISTORICAL_SOURCE
    )
    rows = capacity._source_rows(source_spec)
    observed_coverage = staged.truth_coverage_comparisons(rows)
    observed_alignment = staged.truth_coverage_brier_alignment(rows)
    power = simulate_power(rows)
    if (
        capacity.diversity.sha256_file(PRIOR_BRIER_POWER_RESULT)
        != PRIOR_BRIER_POWER_RESULT_SHA256
    ):
        raise ValueError("prior Brier power audit changed")
    prior = json.loads(PRIOR_BRIER_POWER_RESULT.read_text(encoding="utf-8"))
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "retrospective_power_boundary",
        "decision": (
            "aug8_9_justified_for_dynamic_brier_confirmation_not_expected_"
            "truth_coverage_family"
        ),
        "protocol": {
            "analysis_is_retrospective": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "historical_source": HISTORICAL_SOURCE,
            "historical_source_tree_count": len(rows),
            "future_responses_or_endpoints_opened": False,
            "simulation_seed": SIMULATION_SEED,
            "simulation_draws_per_size": SIMULATION_DRAWS,
            "sample_sizes": list(SAMPLE_SIZES),
            "sampling": "complete historical trees with replacement",
            "coverage_interval_approximation": (
                "mean minus 1.96 standard errors; final result retains the "
                "frozen 20000-sample paired tree bootstrap"
            ),
            "alignment_interval_approximation": (
                "family mean rho exceeds 1.96 times its empirical sampling SD; "
                "final result retains the frozen joint tree bootstrap"
            ),
            "plug_in_power_is_not_a_scientific_gate": True,
            "prior_brier_power_result_sha256": (
                PRIOR_BRIER_POWER_RESULT_SHA256
            ),
        },
        "observed_open32_truth_coverage": observed_coverage,
        "observed_open32_alignment": observed_alignment,
        "plug_in_power": power,
        "registered_brier_power_reference": {
            "pooled_64_primary_probability": prior["estimates"]["pooled128"][
                "64"
            ]["primary_only"],
            "pooled_64_primary_plus_nonworsening_probability": prior[
                "estimates"
            ]["pooled128"]["64"]["primary_plus_nonworsening"],
        },
        "interpretation": {
            "aug8_9_spend_remains_justified": True,
            "main_value": (
                "prospective full LLM-native dynamic-support Brier and depth "
                "confirmation"
            ),
            "truth_coverage_alignment_is_a_bonus_tier": True,
            "truth_coverage_family_expected_to_pass": False,
            "future_capacity_selector_requires_fresh_preregistration": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
