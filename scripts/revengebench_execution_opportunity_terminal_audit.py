#!/usr/bin/env python3
"""Independently enforce the terminal RevengeBench opportunity decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BETAS = ("0.5", "1.0", "2.0")


def audit(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    battle = result["arenas"]["battlesnake"]
    halite = result["arenas"]["halite"]
    husky = result["arenas"]["huskybench"]
    battle_nonqualifying = (
        battle["complete_cells"] == 27
        and battle["all_fresh_arms_exact"] is True
        and battle["all_self_distances_zero"] is True
        and battle["qualifies"] is False
        and any(not battle["beta_results"][beta]["changed_first_action"] for beta in BETAS)
        and all(
            math.isfinite(battle["beta_results"][beta]["depth_two_margin_nats"])
            and battle["beta_results"][beta]["depth_two_margin_nats"] < 0.01
            for beta in BETAS
        )
    )
    failed = halite["failed_cell"]
    halite_failed = (
        halite["status"] == "failed_native_probe_gate"
        and halite["qualifies"] is False
        and failed["fresh_arms_exact"] is True
        and failed["self_distance"] == 0.0
        and failed["decision_count_per_arm"] == [1, 1]
        and min(failed["decision_count_per_arm"]) < 3
    )
    remaining_max = 1 if husky["status"] == "not_executed_gate_preserving_stop" else 0
    impossible = battle_nonqualifying and halite_failed and remaining_max < result["minimum_qualifying_arenas"]
    accounting_zero = result["accounting"] == {
        "cluster_use": 0,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
    }
    passed = impossible and accounting_zero and result["status"] == "no_horizon_opportunity"
    return {
        "status": "pass" if passed else "fail",
        "decision": "close_revengebench_route" if passed else "reject_terminal_result",
        "gates": {
            "battlesnake_nonqualifying": battle_nonqualifying,
            "halite_native_probe_gate_failed": halite_failed,
            "minimum_two_arenas_now_impossible": impossible,
            "gate_preserving_stop_valid": husky["status"] == "not_executed_gate_preserving_stop",
            "accounting_zero": accounting_zero,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.result)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
