#!/usr/bin/env python3
"""Apply the frozen final gate to naive-primary Paprika arbitration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_paprika_step1 import (
        _arm_summary,
        _comparison,
        _gate_outcome,
        _task_values,
        load_arm,
    )
except ModuleNotFoundError:
    from analyze_paprika_step1 import (
        _arm_summary,
        _comparison,
        _gate_outcome,
        _task_values,
        load_arm,
    )


def analyze(
    arbitration_run: Path,
    naive_nonthinking_run: Path,
    naive_thinking_run: Path,
    *,
    round_budget: int = 5,
) -> dict[str, Any]:
    arms = {
        "arbitration": load_arm(arbitration_run, "NaivePrimaryArbitration"),
        "naive_nonthinking": load_arm(naive_nonthinking_run, "naive"),
        "naive_thinking": load_arm(naive_thinking_run, "naive"),
    }
    values = {name: _task_values(arm, round_budget) for name, arm in arms.items()}
    task_sets = {tuple(sorted(rows)) for rows in values.values()}
    if len(task_sets) != 1 or len(next(iter(task_sets))) != 10:
        raise ValueError("Paprika arbitration requires ten identical paired task IDs")
    summaries = {
        name: _arm_summary(arm, values[name], round_budget) for name, arm in arms.items()
    }
    primary = _gate_outcome(
        _comparison("arbitration", "naive_thinking", values),
        summaries,
        better="arbitration",
        baseline="naive_thinking",
        require_six_wins=False,
    )
    context = _comparison("arbitration", "naive_nonthinking", values)
    endpoint_valid = all(summary["endpoint_valid"] for summary in summaries.values())
    if not endpoint_valid:
        status = "invalid_endpoint_stop_and_discuss"
    else:
        status = {
            "pass": "arbitration_pass_stop_and_discuss",
            "insufficient_signal": "arbitration_insufficient_stop_and_discuss",
            "fail": "arbitration_fail_stop_and_discuss",
        }[primary["gate_status"]]
    return {
        "status": status,
        "round_budget": round_budget,
        "censoring_rule": "unresolved tasks score round_budget + 1 censored turns",
        "gate_rule": "arbitration must directionally beat thinking naive unless >=6 ties; clear edge is >=0.2 resolution or <=-0.2 mean-turn delta",
        "endpoint_valid": endpoint_valid,
        "arms": summaries,
        "primary_arbitration_vs_naive_thinking": primary,
        "context_arbitration_vs_naive_nonthinking": context,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Paprika Naive-Primary Arbitration",
        "",
        f"Status: **{result['status']}**",
        "",
        f"| arm | resolution@{result['round_budget']} | mean censored turns | coverage | final inconsistency | cost (USD) | requests |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("naive_nonthinking", "naive_thinking", "arbitration"):
        arm = result["arms"][name]
        lines.append(
            f"| {name} | {arm['resolution_at_budget']:.3f} | {arm['mean_censored_turns']:.3f} | "
            f"{arm['answer_set_coverage']:.3f} | "
            f"{arm['simulator_faithfulness_final_inconsistency_rate']:.3f} | "
            f"{arm['backend_cost_usd']:.4f} | {arm['backend_requests']} |"
        )
    for title, key in (
        ("Primary: arbitration vs thinking naive", "primary_arbitration_vs_naive_thinking"),
        ("Context: arbitration vs non-thinking naive", "context_arbitration_vs_naive_nonthinking"),
    ):
        value = result[key]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"Wins/losses/ties: {value['wins']}/{value['losses']}/{value['ties']}.",
                f"Mean paired censored-turn delta: {value['mean_censored_turn_delta']:.3f}; "
                f"95% bootstrap CI {value['bootstrap_ci95']}; Wilcoxon p={value['wilcoxon_signed_rank_p']}.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arbitration-run", type=Path, required=True)
    parser.add_argument("--naive-nonthinking-run", type=Path, required=True)
    parser.add_argument("--naive-thinking-run", type=Path, required=True)
    parser.add_argument("--round-budget", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.arbitration_run,
        args.naive_nonthinking_run,
        args.naive_thinking_run,
        round_budget=args.round_budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(_markdown(result))
    print(args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
