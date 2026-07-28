#!/usr/bin/env python3
"""Replicate Number Game predictive-risk BED with swapped model roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.number_game_predictive_risk_powered_replication import (
    run_powered_replication,
)


INTERFACE_VERSION = "number-game-cross-planner-replication-1"
PLANNING_MODEL_ID = "openai/gpt-5.4-mini"
TARGET_MODEL_ID = "google/gemini-2.5-flash"
TREE_SEEDS = tuple(range(27000, 27032))
TARGET_SEEDS = tuple(range(27100, 27132))
RUN_BUDGET_USD = 3.00
PROJECTED_PLANNING_COST_USD = 0.10
PROJECTED_TARGET_COST_USD = 0.02


def run_cross_planner_replication(
    *,
    output_dir: Path,
    run_id: str,
) -> dict:
    return run_powered_replication(
        output_dir=output_dir,
        run_id=run_id,
        tree_seeds=TREE_SEEDS,
        target_seeds=TARGET_SEEDS,
        planning_model=PLANNING_MODEL_ID,
        target_model=TARGET_MODEL_ID,
        planning_concurrency=16,
        target_concurrency=1,
        projected_planning_cost=PROJECTED_PLANNING_COST_USD,
        projected_target_cost=PROJECTED_TARGET_COST_USD,
        interface_version=INTERFACE_VERSION,
        run_budget_usd=RUN_BUDGET_USD,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_cross_planner_replication(
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
