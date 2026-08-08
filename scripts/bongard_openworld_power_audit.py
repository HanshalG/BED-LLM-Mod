#!/usr/bin/env python3
"""Audit Bongard paired-effect and changed-path gate sensitivity at zero calls."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable


DEVELOPMENT_EVIDENCE_PROBABILITY = 0.80
CONFIRMATION_UPPER_QUANTILE = 0.975
TARGET_POWER = 0.80
MINIMUM_RELATIVE_GAIN = 0.03
MAX_ACCEPTED_RESPONSES_PER_TASK = 43
MAX_REQUEST_COST_USD = 0.004


@dataclass(frozen=True)
class Design:
    name: str
    tasks: int
    blocks: int
    minimum_changed_paths: int
    evidence_z: float

    @property
    def block_size(self) -> int:
        if self.tasks % self.blocks:
            raise ValueError("tasks must divide evenly across blocks")
        return self.tasks // self.blocks


DESIGNS = (
    Design(
        name="development32",
        tasks=32,
        blocks=4,
        minimum_changed_paths=12,
        evidence_z=NormalDist().inv_cdf(DEVELOPMENT_EVIDENCE_PROBABILITY),
    ),
    Design(
        name="development64_candidate",
        tasks=64,
        blocks=4,
        minimum_changed_paths=24,
        evidence_z=NormalDist().inv_cdf(DEVELOPMENT_EVIDENCE_PROBABILITY),
    ),
    Design(
        name="confirmation64",
        tasks=64,
        blocks=4,
        minimum_changed_paths=24,
        evidence_z=NormalDist().inv_cdf(CONFIRMATION_UPPER_QUANTILE),
    ),
    Design(
        name="confirmation96_candidate",
        tasks=96,
        blocks=4,
        minimum_changed_paths=36,
        evidence_z=NormalDist().inv_cdf(CONFIRMATION_UPPER_QUANTILE),
    ),
)


def normal_cdf(value: float) -> float:
    return NormalDist().cdf(value)


def observed_gain_threshold(
    design: Design,
    *,
    paired_difference_sd_ratio: float,
    minimum_relative_gain: float = MINIMUM_RELATIVE_GAIN,
) -> float:
    """Minimum observed relative gain satisfying effect and evidence gates."""
    if paired_difference_sd_ratio <= 0:
        raise ValueError("paired difference SD ratio must be positive")
    evidence_threshold = (
        design.evidence_z
        * paired_difference_sd_ratio
        / math.sqrt(design.tasks)
    )
    return max(minimum_relative_gain, evidence_threshold)


def paired_effect_gate_power(
    design: Design,
    *,
    true_relative_gain: float,
    paired_difference_sd_ratio: float,
) -> float:
    """Normal-approximation power for the joint relative/evidence gate.

    The paired task difference is scaled by the baseline policy's mean Brier,
    so ``paired_difference_sd_ratio`` is SD(diff) / baseline_mean_brier.
    """
    threshold = observed_gain_threshold(
        design,
        paired_difference_sd_ratio=paired_difference_sd_ratio,
    )
    standard_error = paired_difference_sd_ratio / math.sqrt(design.tasks)
    return normal_cdf((true_relative_gain - threshold) / standard_error)


def true_gain_for_target_power(
    design: Design,
    *,
    paired_difference_sd_ratio: float,
    target_power: float = TARGET_POWER,
) -> float:
    if not 0 < target_power < 1:
        raise ValueError("target power must lie strictly between zero and one")
    threshold = observed_gain_threshold(
        design,
        paired_difference_sd_ratio=paired_difference_sd_ratio,
    )
    standard_error = paired_difference_sd_ratio / math.sqrt(design.tasks)
    return threshold + NormalDist().inv_cdf(target_power) * standard_error


def changed_path_gate_probability(
    design: Design,
    *,
    per_task_change_probability: float,
) -> float:
    """Exact probability of count threshold plus at least one per block."""
    if not 0 <= per_task_change_probability <= 1:
        raise ValueError("change probability must lie in [0, 1]")
    block_distribution = {
        changed: math.comb(design.block_size, changed)
        * per_task_change_probability**changed
        * (1.0 - per_task_change_probability)
        ** (design.block_size - changed)
        for changed in range(1, design.block_size + 1)
    }
    totals: dict[int, float] = {0: 1.0}
    for _ in range(design.blocks):
        updated: defaultdict[int, float] = defaultdict(float)
        for prior, prior_probability in totals.items():
            for changed, probability in block_distribution.items():
                updated[prior + changed] += prior_probability * probability
        totals = dict(updated)
    return sum(
        probability
        for changed, probability in totals.items()
        if changed >= design.minimum_changed_paths
    )


def transport_budget(design: Design) -> dict[str, float | int]:
    accepted = design.block_size * MAX_ACCEPTED_RESPONSES_PER_TASK
    retries = max(4, math.ceil(0.02 * accepted))
    attempts = accepted + retries
    return {
        "tasks_per_block": design.block_size,
        "maximum_accepted_responses": accepted,
        "maximum_transport_retries": retries,
        "maximum_http_attempts": attempts,
        "maximum_precharged_exposure_usd": attempts * MAX_REQUEST_COST_USD,
    }


def _effect_rows(
    designs: Iterable[Design],
    *,
    sd_ratios: Iterable[float],
    true_gains: Iterable[float],
) -> list[dict[str, float | str | int]]:
    rows = []
    for design in designs:
        for sd_ratio in sd_ratios:
            row: dict[str, float | str | int] = {
                "design": design.name,
                "tasks": design.tasks,
                "paired_difference_sd_ratio": sd_ratio,
                "observed_gain_threshold": observed_gain_threshold(
                    design,
                    paired_difference_sd_ratio=sd_ratio,
                ),
                "true_gain_for_80pct_power": true_gain_for_target_power(
                    design,
                    paired_difference_sd_ratio=sd_ratio,
                ),
            }
            for gain in true_gains:
                row[f"power_at_true_gain_{gain:.3f}"] = paired_effect_gate_power(
                    design,
                    true_relative_gain=gain,
                    paired_difference_sd_ratio=sd_ratio,
                )
            rows.append(row)
    return rows


def build_audit() -> dict[str, object]:
    sd_ratios = (0.10, 0.20, 0.30, 0.50)
    true_gains = (0.03, 0.05, 0.075, 0.10)
    change_probabilities = (0.30, 0.375, 0.40, 0.50, 0.60)
    changed_path_rows = [
        {
            "design": design.name,
            "tasks": design.tasks,
            "minimum_changed_paths": design.minimum_changed_paths,
            "per_task_change_probability": probability,
            "gate_probability": changed_path_gate_probability(
                design,
                per_task_change_probability=probability,
            ),
        }
        for design in DESIGNS
        for probability in change_probabilities
    ]
    return {
        "schema_version": 1,
        "status": "zero_call_design_sensitivity_audit",
        "model_calls_made": 0,
        "endpoint_data_accessed": False,
        "assumptions": {
            "paired_task_differences_are_normally_approximated": True,
            "sd_ratio_definition": (
                "SD(dynamic_brier-control_brier) / control_mean_brier"
            ),
            "effect_and_evidence_gates_share_the_same_observed_mean": True,
            "changed_path_events_are_iid_for_sensitivity_only": True,
            "ranking_log_loss_and_multi_control_conjunction_not_modelled": True,
            "values_are_design_sensitivity_not_outcome_forecasts": True,
        },
        "constants": {
            "minimum_relative_gain": MINIMUM_RELATIVE_GAIN,
            "development_evidence_probability": DEVELOPMENT_EVIDENCE_PROBABILITY,
            "confirmation_upper_quantile": CONFIRMATION_UPPER_QUANTILE,
            "target_power": TARGET_POWER,
            "maximum_accepted_responses_per_task": (
                MAX_ACCEPTED_RESPONSES_PER_TASK
            ),
            "maximum_request_cost_usd": MAX_REQUEST_COST_USD,
        },
        "designs": [
            {
                **asdict(design),
                "block_size": design.block_size,
                "transport_budget": transport_budget(design),
            }
            for design in DESIGNS
        ],
        "paired_effect_sensitivity": _effect_rows(
            DESIGNS,
            sd_ratios=sd_ratios,
            true_gains=true_gains,
        ),
        "changed_path_sensitivity": changed_path_rows,
        "interpretation": {
            "three_percent_is_a_claim_floor_not_a_powered_target": True,
            "larger_samples_do_not_relax_any_scientific_threshold": True,
            "candidate_development64_fits_daily_cap": (
                transport_budget(DESIGNS[1])[
                    "maximum_precharged_exposure_usd"
                ]
                < 5.0
            ),
            "candidate_confirmation96_fits_daily_cap": (
                transport_budget(DESIGNS[3])[
                    "maximum_precharged_exposure_usd"
                ]
                < 5.0
            ),
        },
    }


def render_markdown(audit: dict[str, object]) -> str:
    effect_rows = audit["paired_effect_sensitivity"]
    selected = [
        row
        for row in effect_rows
        if row["paired_difference_sd_ratio"] in (0.20, 0.30)
    ]
    lines = [
        "# Bongard OpenWorld Sample-Size Power Audit",
        "",
        "Date: 2026-08-08. Model calls: `0`. Endpoint data accessed: `false`.",
        "",
        "This is a design-sensitivity calculation, not an outcome forecast. It",
        "uses the exact frozen relative-Brier and evidence thresholds while",
        "leaving ranking, log-loss, and multi-control conjunctions unmodelled.",
        "Actual full-tier power is therefore no greater than any listed marginal",
        "effect-gate power.",
        "",
        "## Paired-Effect Sensitivity",
        "",
        "`SD ratio` means paired task-difference SD divided by control mean Brier.",
        "The final column is the true relative gain needed for 80% marginal power",
        "on the joint 3% effect-size and evidence gate.",
        "",
        "| Design | N | SD ratio | Observed gate | True gain for 80% power |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| {design} | {tasks} | {paired_difference_sd_ratio:.0%} | "
            "{observed_gain_threshold:.2%} | "
            "{true_gain_for_80pct_power:.2%} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Changed-Path Sensitivity",
            "",
            "The exact calculation requires both the total changed-path threshold",
            "and at least one change in every fixed execution block.",
            "",
            "| Design | Per-task change rate | Gate probability |",
            "|---|---:|---:|",
        ]
    )
    for row in audit["changed_path_sensitivity"]:
        if row["per_task_change_probability"] in (0.375, 0.50):
            lines.append(
                "| {design} | {per_task_change_probability:.1%} | "
                "{gate_probability:.1%} |".format(**row)
            )
    lines.extend(
        [
            "",
            "## Daily-Cap Feasibility",
            "",
            "| Design | Tasks/block | Accepted | HTTP attempts | Exposure |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for design in audit["designs"]:
        budget = design["transport_budget"]
        lines.append(
            "| {name} | {tasks_per_block} | {maximum_accepted_responses} | "
            "{maximum_http_attempts} | ${maximum_precharged_exposure_usd:.3f} |".format(
                name=design["name"], **budget
            )
        )
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "The frozen 3% value is a minimum claim threshold, not a true-effect",
            "target with 80% power. At moderate 20--30% paired SD ratios, moving",
            "development from 32 to 64 tasks and confirmation from 64 to 96 tasks",
            "materially lowers the detectable true gain while fitting the existing",
            "hard `$5` daily cap. Any expansion must be frozen before planner or",
            "endpoint responses, retain disjoint opaque tasks, scale changed-path",
            "counts proportionally, and leave every semantic and efficacy gate",
            "otherwise unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit()
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(audit), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
