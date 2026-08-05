#!/usr/bin/env python3
"""Run the frozen GPT-5.6 Luna paired Number Game efficacy gate."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_deepseek_v4_flash_paired_efficacy32 as base
from scripts.openrouter_daily_budget import read_live_credits, require_budget


INTERFACE_VERSION = "number-game-luna-paired-efficacy32-1"
MODEL_ID = "openai/gpt-5.6-luna"
RUN_BUDGET_USD = 0.85
MIN_STARTING_BALANCE_USD = 1.00
BOOTSTRAP_SEED = 1_080_600
SMOKE_RESULT = REPO_ROOT / (
    "results/nonmyopic/number_game_planner_frontier_smoke/"
    "number-game-planner-frontier-luna-20260805T120000Z/RESULT.json"
)
SMOKE_RESULT_SHA256 = (
    "9affa0b48ebc7ae2adfc6867e4b5bf56a47e0dc28c92c8894f94ae1896211496"
)
DAILY_LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/2026-08-05.json"
)


@contextmanager
def configured_base() -> Iterator[None]:
    overrides = {
        "INTERFACE_VERSION": INTERFACE_VERSION,
        "MODEL_ID": MODEL_ID,
        "RUN_BUDGET_USD": RUN_BUDGET_USD,
        "MIN_STARTING_BALANCE_USD": MIN_STARTING_BALANCE_USD,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "SMOKE_RESULT": SMOKE_RESULT,
        "SMOKE_RESULT_SHA256": SMOKE_RESULT_SHA256,
    }
    originals = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def run_efficacy(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path = SMOKE_RESULT,
    tree_runner: Callable[..., tuple[dict[str, Any], dict[str, Any]]]
    | None = None,
    max_tree_workers: int = 4,
) -> dict[str, Any]:
    with configured_base():
        result = base.run_efficacy(
            output_dir=output_dir,
            run_id=run_id,
            smoke_result_path=smoke_result_path,
            tree_runner=tree_runner,
            max_tree_workers=max_tree_workers,
        )
    if result["status"] in {"gated_null", "mechanics_failed"}:
        result["decision"] = "close_luna_route"
        base.checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--daily-ledger", type=Path, default=DAILY_LEDGER)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    ledger = json.loads(args.daily_ledger.read_text(encoding="utf-8"))
    live = read_live_credits()
    budget = require_budget(
        ledger,
        projected_cost_usd=RUN_BUDGET_USD,
        total_usage_usd=live["total_usage_usd"],
    )
    if live["balance_usd"] + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError("OpenRouter balance is below the frozen start gate")
    result = run_efficacy(output_dir=args.output_dir, run_id=args.run_id)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "daily_budget_at_start": budget,
                "usage": result["usage"],
                "mechanics_gates": result["mechanics_gates"],
                "intelligence_gates": result["intelligence_gates"],
                "qwen_noninferiority": {
                    key: value
                    for key, value in result["qwen_noninferiority"].items()
                    if key != "per_tree_differences"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
