#!/usr/bin/env python3
"""Estimate fresh-tree power for the frozen two-draw diversity selector."""

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
from scripts.number_game_two_draw_diversity_bonus_audit import sha256_file


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-two-draw-diversity-bonus-power-audit-1"
SOURCE_RESULT = REPO_ROOT / (
    "results/nonmyopic/number_game_two_draw_diversity_bonus_audit/"
    "number-game-two-draw-diversity-bonus-audit-20260806T003000Z/"
    "RESULT.json"
)
SOURCE_RESULT_SHA256 = (
    "49ef14d369a7afe1e530b4cfcb377ee1b2cc280ef4af758fe8f59fd0ecb4caea"
)
SIMULATION_SEED = 108_700
SIMULATION_DRAWS = 50_000
SAMPLE_SIZES = (32, 64, 96)


def _actual(row: dict[str, Any], root: int) -> float:
    return next(
        float(item["realized_brier"])
        for item in row["root_rows"]
        if int(item["root"]) == root
    )


def paired_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in source["rows"]:
        bonus = _actual(row, int(row["bonus_root"]))
        depth_two = _actual(row, int(row["depth_two_root"]))
        original = _actual(row, int(row["original_root"]))
        rows.append(
            {
                "source": str(row["source"]),
                "bonus_brier": bonus,
                "depth_two_brier": depth_two,
                "original_brier": original,
                "bonus_minus_depth_two": bonus - depth_two,
                "bonus_minus_original": bonus - original,
                "bonus_changes_original": (
                    int(row["bonus_root"]) != int(row["original_root"])
                ),
            }
        )
    return rows


def sample_statistics(sample: Sequence[dict[str, Any]]) -> dict[str, Any]:
    depth_two = [float(row["bonus_minus_depth_two"]) for row in sample]
    original = [float(row["bonus_minus_original"]) for row in sample]
    mean_depth_two = statistics.fmean(depth_two)
    standard_error = statistics.stdev(depth_two) / math.sqrt(len(sample))
    changed = [row for row in sample if row["bonus_changes_original"]]
    return {
        "relative_reduction_vs_depth_two": (
            -mean_depth_two
            / statistics.fmean(float(row["depth_two_brier"]) for row in sample)
        ),
        "mean_bonus_minus_depth_two": mean_depth_two,
        "normal_approximation_upper_95pct": (
            mean_depth_two + 1.96 * standard_error
        ),
        "depth_two_wins": sum(value < -1e-15 for value in depth_two),
        "depth_two_losses": sum(value > 1e-15 for value in depth_two),
        "changed_original_roots": len(changed),
        "mean_bonus_minus_original": statistics.fmean(original),
        "original_changed_root_wins": sum(
            float(row["bonus_minus_original"]) < -1e-15 for row in changed
        ),
        "original_changed_root_losses": sum(
            float(row["bonus_minus_original"]) > 1e-15 for row in changed
        ),
    }


def gate_values(stats: dict[str, Any], *, sample_size: int) -> dict[str, bool]:
    primary = {
        "relative_reduction_at_least_three_percent": (
            stats["relative_reduction_vs_depth_two"] >= 0.03
        ),
        "normal_approximation_interval_below_zero": (
            stats["normal_approximation_upper_95pct"] < 0.0
        ),
        "depth_two_wins_exceed_losses": (
            stats["depth_two_wins"] > stats["depth_two_losses"]
        ),
    }
    mechanism = {
        "at_least_one_quarter_roots_change": (
            stats["changed_original_roots"] >= sample_size // 4
        ),
        "bonus_mean_not_worse_than_original": (
            stats["mean_bonus_minus_original"] <= 0.0
        ),
    }
    original_redundant_counts = {
        "scaled_depth_two_win_floor": (
            stats["depth_two_wins"] >= 14 * sample_size // 32
        ),
        "original_changed_root_wins_exceed_losses": (
            stats["original_changed_root_wins"]
            > stats["original_changed_root_losses"]
        ),
    }
    return {
        "primary_only": all(primary.values()),
        "primary_plus_nonworsening": all(
            (primary | mechanism).values()
        ),
        "original_redundant_gate": all(
            (primary | mechanism | original_redundant_counts).values()
        ),
        **{f"component_{name}": value for name, value in primary.items()},
        **{f"component_{name}": value for name, value in mechanism.items()},
        **{
            f"component_{name}": value
            for name, value in original_redundant_counts.items()
        },
    }


def simulate(
    pool: Sequence[dict[str, Any]],
    *,
    sample_size: int,
    rng: random.Random,
) -> dict[str, float]:
    counts: dict[str, int] = {}
    for _ in range(SIMULATION_DRAWS):
        sample = [rng.choice(pool) for _ in range(sample_size)]
        gates = gate_values(
            sample_statistics(sample),
            sample_size=sample_size,
        )
        for name, passed in gates.items():
            counts[name] = counts.get(name, 0) + int(passed)
    return {
        name: count / SIMULATION_DRAWS for name, count in counts.items()
    }


def run_analysis(output_dir: Path) -> dict[str, Any]:
    if sha256_file(SOURCE_RESULT) != SOURCE_RESULT_SHA256:
        raise ValueError("diversity-bonus audit source hash changed")
    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    rows = paired_rows(source)
    pools = {
        "development96": [
            row for row in rows if row["source"] == "development96"
        ],
        "later_fresh32": [
            row for row in rows if row["source"] == "later_fresh32"
        ],
        "pooled128": rows,
    }
    rng = random.Random(SIMULATION_SEED)
    estimates = {
        name: {
            str(sample_size): simulate(
                pool,
                sample_size=sample_size,
                rng=rng,
            )
            for sample_size in SAMPLE_SIZES
        }
        for name, pool in pools.items()
    }
    pooled = estimates["pooled128"]
    gates = {
        "thirty_two_primary_plus_nonworsening_below_half": (
            pooled["32"]["primary_plus_nonworsening"] < 0.5
        ),
        "sixty_four_primary_probability_at_least_eighty_percent": (
            pooled["64"]["primary_only"] >= 0.8
        ),
        "sixty_four_joint_probability_at_least_sixty_five_percent": (
            pooled["64"]["primary_plus_nonworsening"] >= 0.65
        ),
        "ninety_six_joint_gain_over_sixty_four_below_fifteen_points": (
            pooled["96"]["primary_plus_nonworsening"]
            - pooled["64"]["primary_plus_nonworsening"]
            < 0.15
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "retrospective_power_recommends_64"
            if all(gates.values())
            else "retrospective_power_inconclusive"
        ),
        "protocol": {
            "analysis_is_retrospective": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "simulation_seed": SIMULATION_SEED,
            "simulation_draws_per_pool_and_size": SIMULATION_DRAWS,
            "sample_sizes": list(SAMPLE_SIZES),
            "sampling": "tree rows with replacement within each empirical pool",
            "interval_approximation": (
                "mean plus 1.96 standard errors; final experiment retains "
                "the frozen 20000-sample paired tree bootstrap"
            ),
            "coefficient_remains_frozen": -0.5,
            "no_future_seed_or_response_used": True,
        },
        "decision_gates": gates,
        "estimates": estimates,
        "decision": {
            "tree_count": 64,
            "execution": "two mandatory fresh 32-tree blocks on separate budget days",
            "why_not_32": "pooled joint pass probability is below one half",
            "why_not_96": (
                "third Qwen day adds less than fifteen percentage points "
                "of pooled joint pass probability"
            ),
            "original_redundant_win_counts_removed": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = run_analysis(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
