#!/usr/bin/env python3
"""Audit sensitivity of the frozen Bongard call-matched myopic gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_power_audit as base


AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_COMPUTE_MATCHED_MYOPIC_ENSEMBLE_AMENDMENT_20260809.md"
)
DEVELOPMENT_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_development64/"
    "PROTOCOL_MANIFEST_V18.json"
)
CONFIRMATION_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_confirmation64/"
    "PROTOCOL_MANIFEST_V15.json"
)
BOUND_FILES = {
    "base_sample_size_power_audit": (
        REPO_ROOT / "scripts/bongard_openworld_power_audit.py",
        "97b6fcd0266f074c3bd5d12efa20bbebf6f7e0e3c20cf8abfc44225c4d5fe1af",
    ),
    "call_matched_amendment": (
        AMENDMENT,
        "064fb13dd1fbbcea255e38b1b9eae41dfb9347bcf1b91fa5bc5711eba7985368",
    ),
    "development_manifest_v18": (
        DEVELOPMENT_MANIFEST,
        "df55546302190161ab2c4005f936e967e24dafe5f42483288c39100e0b03614f",
    ),
    "confirmation_manifest_v15": (
        CONFIRMATION_MANIFEST,
        "63cda4a2cd964eeae2227292483760fade0c7738e9c35fe3b4e0dddac8152c0e",
    ),
}
TARGET_POWER = 0.80
SD_RATIOS = (0.20, 0.30)
CALL_MATCHED_CONFIRMATION_GATES = {
    "at_least_36_dynamic_final_histories_differ_from_compute_matched_myopic",
    "at_least_36_dynamic_action_changes_from_compute_matched_myopic_clear_numerical_tie_margin",
    "dynamic_and_compute_matched_myopic_differ_in_every_execution_block",
    "dynamic_brier_relative_improvement_vs_compute_matched_myopic_at_least_3_percent",
    "dynamic_brier_vs_compute_matched_myopic_paired_tree_bootstrap_95pct_upper_below_zero",
    "dynamic_log_loss_is_not_worse_than_compute_matched_myopic",
    "dynamic_ranking_fidelity_is_not_worse_than_compute_matched_myopic",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bindings() -> dict[str, dict[str, str]]:
    observed = {
        name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
        for name, (path, _) in BOUND_FILES.items()
    }
    expected = {
        name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": digest}
        for name, (path, digest) in BOUND_FILES.items()
    }
    if observed != expected:
        raise ValueError("call-matched power-audit binding changed")
    confirmation = json.loads(CONFIRMATION_MANIFEST.read_text(encoding="utf-8"))
    policy_gates = set(confirmation["science_gates"]["policy"])
    if not CALL_MATCHED_CONFIRMATION_GATES.issubset(policy_gates):
        raise ValueError("call-matched confirmation gate family is incomplete")
    return observed


def change_probability_for_target_power(
    design: base.Design, *, target_power: float = TARGET_POWER
) -> float:
    if not 0 < target_power < 1:
        raise ValueError("target power must lie strictly between zero and one")
    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        probability = base.changed_path_gate_probability(
            design, per_task_change_probability=midpoint
        )
        if probability < target_power:
            low = midpoint
        else:
            high = midpoint
    return high


def standardized_mean_margin_for_target_power(
    tasks: int, *, target_power: float = TARGET_POWER
) -> float:
    """True favorable mean difference divided by its task-level SD."""
    if tasks <= 0 or not 0 < target_power < 1:
        raise ValueError("invalid standardized-margin inputs")
    return NormalDist().inv_cdf(target_power) / math.sqrt(tasks)


def _design_row(design: base.Design) -> dict[str, Any]:
    return {
        "design": design.name,
        "tasks": design.tasks,
        "blocks": design.blocks,
        "minimum_changed_actions": design.minimum_changed_paths,
        "minimum_changed_action_fraction": (
            design.minimum_changed_paths / design.tasks
        ),
        "true_change_probability_for_80pct_marginal_power": (
            change_probability_for_target_power(design)
        ),
        "ranking_or_log_loss_standardized_favorable_margin_for_80pct_marginal_power": (
            standardized_mean_margin_for_target_power(design.tasks)
        ),
        "brier_sensitivity": [
            {
                "paired_difference_sd_ratio": ratio,
                "observed_relative_gain_threshold": base.observed_gain_threshold(
                    design, paired_difference_sd_ratio=ratio
                ),
                "true_relative_gain_for_80pct_marginal_power": (
                    base.true_gain_for_target_power(
                        design, paired_difference_sd_ratio=ratio
                    )
                ),
            }
            for ratio in SD_RATIOS
        ],
    }


def build_audit() -> dict[str, Any]:
    bindings = verify_bindings()
    development = next(
        design for design in base.DESIGNS if design.name == "development64_candidate"
    )
    confirmation = next(
        design for design in base.DESIGNS if design.name == "confirmation96_candidate"
    )
    return {
        "schema_version": 1,
        "status": "call_matched_gate_sensitivity_complete",
        "frozen_before_first_bongard_response": True,
        "model_calls": 0,
        "cost_usd": 0.0,
        "endpoint_data_accessed": False,
        "changes_scientific_gates": False,
        "authorizes_paid_calls": False,
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "bindings": bindings,
        "assumptions": {
            "brier_uses_normal_approximation_to_paired_task_mean": True,
            "action_changes_are_iid_for_sensitivity_only": True,
            "ranking_and_log_loss_use_normal_approximation_to_mean_difference": True,
            "gate_dependencies_are_unknown": True,
            "values_are_marginal_sensitivity_not_joint_power_or_outcome_forecasts": True,
        },
        "designs": [_design_row(development), _design_row(confirmation)],
        "interpretation": {
            "three_percent_brier_floor_is_not_an_80pct_powered_target": True,
            "changed_action_gate_requires_more_than_37_5pct_true_change_rate_for_80pct_power": True,
            "full_conjunction_power_is_no_greater_than_its_weakest_marginal_gate": True,
            "a_null_can_reflect_small_effect_or_insufficient_action_separation": True,
            "no_post_outcome_threshold_relaxation_is_allowed": True,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Bongard Call-Matched Myopic Gate Sensitivity Audit",
        "",
        "Date: 2026-08-09. Model calls: `0`. Endpoint data accessed: `false`.",
        "",
        "This audit was frozen before the first Bongard response. It is a marginal",
        "design-sensitivity calculation, not joint power and not an outcome forecast.",
        "It does not alter a gate or authorize a paid call.",
        "",
        "| Stage | N | True action-change rate for 80% power | Rank/log standardized margin | 80% Brier gain at SD 20% | 80% Brier gain at SD 30% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit["designs"]:
        brier = row["brier_sensitivity"]
        lines.append(
            "| {design} | {tasks} | {change:.2%} | {margin:.3f} SD | "
            "{gain20:.2%} | {gain30:.2%} |".format(
                design=row["design"],
                tasks=row["tasks"],
                change=row["true_change_probability_for_80pct_marginal_power"],
                margin=row[
                    "ranking_or_log_loss_standardized_favorable_margin_for_80pct_marginal_power"
                ],
                gain20=brier[0]["true_relative_gain_for_80pct_marginal_power"],
                gain30=brier[1]["true_relative_gain_for_80pct_marginal_power"],
            )
        )
    lines.extend(
        [
            "",
            "The 37.5% changed-action floor has only about 50% marginal power when",
            "the true rate sits at the floor. The Brier gate likewise needs a true",
            "gain above 3% at realistic paired variance. Because dependencies among",
            "Brier, action-change, ranking, and log-loss gates are unknown, full-tier",
            "power cannot be inferred by multiplying these marginals and is bounded",
            "above by the weakest one. A null must not trigger threshold relaxation.",
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
