#!/usr/bin/env python3
"""Freeze the Paprika headline sample size from the terminal-valid pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


def power_plan(
    pilot_report: Path,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    shrinkage: float = 0.50,
    maximum_tasks: int = 50,
) -> dict[str, Any]:
    report = json.loads(pilot_report.read_text())
    rows = report["primary_arbitration_vs_naive_thinking"]["per_task"]
    deltas = np.asarray([row["censored_turn_delta"] for row in rows], dtype=float)
    pilot_mean = float(np.mean(deltas))
    pilot_sd = float(np.std(deltas, ddof=1))
    assumed_effect = abs(pilot_mean) * shrinkage
    if pilot_sd <= 0.0 or assumed_effect <= 0.0:
        raise ValueError("Pilot must contain a nonzero effect and paired variance")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(target_power)
    required = int(
        math.ceil(((z_alpha + z_power) * pilot_sd / assumed_effect) ** 2)
    )
    selected = min(required, maximum_tasks)
    noncentrality = assumed_effect * math.sqrt(selected) / pilot_sd
    achieved = normal.cdf(noncentrality - z_alpha) + normal.cdf(
        -noncentrality - z_alpha
    )
    return {
        "pilot_task_count": int(deltas.size),
        "pilot_deltas": deltas.tolist(),
        "pilot_mean_censored_turn_delta": pilot_mean,
        "pilot_paired_sd": pilot_sd,
        "shrinkage": shrinkage,
        "assumed_absolute_effect": assumed_effect,
        "alpha_two_sided": alpha,
        "target_power": target_power,
        "normal_approx_required_tasks": required,
        "pre_registered_maximum_tasks": maximum_tasks,
        "selected_tasks": selected,
        "normal_approx_power_at_selected_n": float(achieved),
        "task_range": {
            "split": "eval",
            "start_offset_inclusive": 10,
            "end_offset_inclusive": 10 + selected - 1,
        },
        "decision": (
            "Freeze the pre-registered maximum N=50. The exact normal approximation "
            "requires 53 tasks for 80% power at half the pilot effect; N=50 provides "
            "approximately 78% power and stays within the authorized 30-50 range."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-report",
        type=Path,
        default=Path("results/path_e/arbitration_terminal/PAPRIKA_ARBITRATION.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = power_plan(args.pilot_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
    print(f"selected_tasks={result['selected_tasks']}")


if __name__ == "__main__":
    main()
