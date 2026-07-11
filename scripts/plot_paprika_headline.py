#!/usr/bin/env python3
"""Render the preregistered Paprika headline result figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARM_STYLE = {
    "arbitration": ("Arbitration", "#147D64", "o"),
    "candidate0": ("Candidate 0", "#D97706", "s"),
    "naive_thinking": ("Thinking naive", "#2563A6", "^"),
    "naive_nonthinking": ("Non-thinking naive", "#6B7280", "D"),
    "best_n_eig": ("Best-N EIG", "#A23B72", "P"),
}


def _comparison_rows(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = [
        ("Arbitration - thinking naive", result["primary_arbitration_vs_naive_thinking"]),
        ("Arbitration - candidate 0", result["coprimary_arbitration_vs_candidate0"]),
    ]
    if "context_best_n_vs_naive_thinking" in result:
        rows.append(("Best-N EIG - thinking naive", result["context_best_n_vs_naive_thinking"]))
    if "context_best_n_vs_arbitration" in result:
        rows.append(("Best-N EIG - arbitration", result["context_best_n_vs_arbitration"]))
    return rows


def plot_headline(result: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    round_budget = int(result["round_budget"])
    rounds = np.arange(1, round_budget + 1)
    fig, (curve_ax, delta_ax) = plt.subplots(1, 2, figsize=(10.2, 3.9))

    for arm_name, arm in result["arms"].items():
        label, color, marker = ARM_STYLE[arm_name]
        curve_ax.plot(
            rounds,
            arm["resolution_curve"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
        )
    curve_ax.set_xlabel("Interaction round")
    curve_ax.set_ylabel("Cumulative resolution rate")
    curve_ax.set_xticks(rounds)
    curve_ax.set_ylim(-0.02, 1.02)
    curve_ax.grid(axis="y", alpha=0.25)
    curve_ax.legend(frameon=False, fontsize=8, loc="upper left")
    curve_ax.set_title("A. Held-out resolution curves", loc="left", fontweight="bold")

    comparisons = _comparison_rows(result)
    y = np.arange(len(comparisons))[::-1]
    means = np.asarray([row[1]["mean_censored_turn_delta"] for row in comparisons])
    intervals = np.asarray([row[1]["bootstrap_ci95"] for row in comparisons], dtype=float)
    errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
    colors = ["#147D64", "#D97706", "#A23B72", "#6B7280"][: len(comparisons)]
    for index, (position, mean) in enumerate(zip(y, means)):
        delta_ax.errorbar(
            mean,
            position,
            xerr=errors[:, index : index + 1],
            fmt="o",
            color=colors[index],
            capsize=3,
            linewidth=1.8,
        )
    delta_ax.axvline(0.0, color="#111827", linewidth=1, linestyle="--")
    delta_ax.set_yticks(y, [row[0] for row in comparisons], fontsize=8)
    delta_ax.set_xlabel("Paired censored-turn difference (95% bootstrap CI)")
    delta_ax.grid(axis="x", alpha=0.25)
    delta_ax.set_title("B. Paired endpoint effects", loc="left", fontweight="bold")

    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.analysis.read_text())
    for path in plot_headline(result, args.output_prefix):
        print(path)


if __name__ == "__main__":
    main()
