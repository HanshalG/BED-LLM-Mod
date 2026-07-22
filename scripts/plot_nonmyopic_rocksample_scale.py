"""Plot the five preregistered Rock Diagnosis StrategyEIG confirmations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_nonmyopic_rock_branch_result import (
    ARMS,
    ARM_LABELS,
    ARM_STYLES,
)


def _round_means(payload: dict[str, Any], map_name: str) -> dict[str, list[float]]:
    summary = payload["maps"][map_name]["summary"]
    return {arm: summary[arm]["round_entropy_mean"] for arm in ARMS}


def plot_scale_results(
    base_payload: dict[str, Any],
    eight_rock_payload: dict[str, Any],
    eleven_rock_payload: dict[str, Any],
    fifteen_rock_payload: dict[str, Any],
    output_path: Path,
) -> None:
    panels = [
        ("3-6", _round_means(base_payload, "3-6")),
        ("5-7", _round_means(base_payload, "5-7")),
        ("7-8", _round_means(eight_rock_payload, "7-8")),
        ("11-11", _round_means(eleven_rock_payload, "11-11")),
        ("15-15", _round_means(fifteen_rock_payload, "15-15")),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18.8, 4.6))
    for axis, (map_name, means) in zip(axes, panels):
        rounds = range(1, len(next(iter(means.values()))) + 1)
        for arm in ARMS:
            axis.plot(
                rounds,
                means[arm],
                label=ARM_LABELS[arm],
                markersize=3.0,
                **ARM_STYLES[arm],
            )
        axis.set_title(f"RockSample[{map_name}]")
        axis.set_xlabel("Round")
        round_values = list(rounds)
        tick_step = 2 if len(round_values) > 10 else 1
        ticks = round_values[::tick_step]
        if ticks[-1] != round_values[-1]:
            ticks.append(round_values[-1])
        axis.set_xticks(ticks)
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean entropy (nats)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.3)
    fig.suptitle("Preregistered Rock Diagnosis confirmations")
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_result", type=Path)
    parser.add_argument("eight_rock_result", type=Path)
    parser.add_argument("eleven_rock_result", type=Path)
    parser.add_argument("fifteen_rock_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_scale_results(
        json.loads(args.base_result.read_text(encoding="utf-8")),
        json.loads(args.eight_rock_result.read_text(encoding="utf-8")),
        json.loads(args.eleven_rock_result.read_text(encoding="utf-8")),
        json.loads(args.fifteen_rock_result.read_text(encoding="utf-8")),
        args.output,
    )


if __name__ == "__main__":
    main()
