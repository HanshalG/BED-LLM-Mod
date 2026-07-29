#!/usr/bin/env python3
"""Replicate cross-fitted Number Game depth three with a GLM planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts import number_game_qwen_planner_depth_three as engine
from scripts.number_game_glm_planner_serving_smoke import (
    INTERFACE_VERSION as SMOKE_INTERFACE_VERSION,
    MODEL_ID as PLANNING_MODEL_ID,
)
from scripts.number_game_grok_planner_depth_three import formal_gates


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-glm-planner-depth-three-1"
TARGET_MODEL_ID = "openai/gpt-5.4-mini"
FORMAL_TREE_SEEDS = tuple(range(47_000, 47_032))
FORMAL_TARGET_SEEDS = tuple(range(47_100, 47_132))
FORMAL_VALIDATION_SEED_START = 47_200
FORMAL_EXTRA_ENDPOINT_SEED_START = 47_500
FORMAL_EXPECTED_REQUESTS = (
    len(FORMAL_TREE_SEEDS) * engine.REQUESTS_PER_TREE
)
FORMAL_BUDGET_USD = 7.50
MIN_FORMAL_STARTING_BALANCE_USD = 7.50


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_FORMAL_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_FORMAL_STARTING_BALANCE_USD:.2f} formal projection"
        )


def validate_smoke_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    protocol = result.get("protocol") or {}
    gates = result.get("gates") or {}
    if result.get("status") != "passed" or gates.get("all_pass") is not True:
        raise ValueError("GLM planner exact-10 serving smoke did not pass")
    if protocol.get("interface_version") != SMOKE_INTERFACE_VERSION:
        raise ValueError("GLM planner smoke interface changed")
    if protocol.get("model") != PLANNING_MODEL_ID:
        raise ValueError("GLM planner smoke model changed")
    if protocol.get("expected_requests") != 10:
        raise ValueError("GLM planner smoke was not exact-10")
    if protocol.get("efficacy_used_for_authorization") is not False:
        raise ValueError("GLM planner smoke used efficacy for authorization")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    if not args.skip_balance_check:
        require_starting_balance(openrouter_remaining_credit())
    result = engine.run_study(
        stage="formal",
        tree_seeds=FORMAL_TREE_SEEDS,
        target_seeds=FORMAL_TARGET_SEEDS,
        validation_seed_start=FORMAL_VALIDATION_SEED_START,
        extra_endpoint_seed_start=FORMAL_EXTRA_ENDPOINT_SEED_START,
        output_dir=args.output_dir,
        run_id=args.run_id,
        run_budget_usd=FORMAL_BUDGET_USD,
        smoke_result_path=args.smoke_result,
        planning_model=PLANNING_MODEL_ID,
        target_model=TARGET_MODEL_ID,
        interface_version=INTERFACE_VERSION,
        smoke_validator=validate_smoke_result,
        formal_gate_fn=formal_gates,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "usage": result["usage"],
                "gates": result["gates"],
                "primary": result["aggregate"]["comparisons"][
                    "crossfit_depth_two"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
