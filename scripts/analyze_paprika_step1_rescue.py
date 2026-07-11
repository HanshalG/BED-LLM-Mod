#!/usr/bin/env python3
"""Apply the frozen Claim 1 gate to the sole generation-thinking rescue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_paprika_step1 import (
    _arm_summary,
    _comparison,
    _gate_outcome,
    _task_values,
    load_arm,
)


def analyze(
    rescue_run: Path,
    naive_nonthinking_run: Path,
    naive_thinking_run: Path,
    *,
    round_budget: int = 5,
) -> dict[str, Any]:
    arms = {
        "rescue_eig": load_arm(rescue_run, "EIG"),
        "naive_nonthinking": load_arm(naive_nonthinking_run, "naive"),
        "naive_thinking": load_arm(naive_thinking_run, "naive"),
    }
    values = {name: _task_values(arm, round_budget) for name, arm in arms.items()}
    task_sets = {tuple(sorted(rows)) for rows in values.values()}
    if len(task_sets) != 1 or len(next(iter(task_sets))) != 10:
        raise ValueError("Step 1 rescue requires exactly ten identical paired task IDs")
    summaries = {
        name: _arm_summary(arm, values[name], round_budget) for name, arm in arms.items()
    }
    matched = _gate_outcome(
        _comparison("rescue_eig", "naive_nonthinking", values),
        summaries,
        better="rescue_eig",
        baseline="naive_nonthinking",
        require_six_wins=False,
    )
    adversarial = _comparison("rescue_eig", "naive_thinking", values)
    status = {
        "pass": "rescue_pass_continue_claim1_transfer",
        "insufficient_signal": "rescue_insufficient_stop_and_discuss",
        "fail": "rescue_fail_stop_and_discuss",
    }[matched["gate_status"]]
    return {
        "status": status,
        "round_budget": round_budget,
        "censoring_rule": "unresolved tasks score round_budget + 1 censored turns",
        "gate_rule": "directional wins unless >=6 ties; or >=0.2 resolution / -0.2 mean-turn clear edge",
        "arms": summaries,
        "claim1_matched_rescue_vs_naive_nonthinking": matched,
        "claim1_adversarial_rescue_vs_naive_thinking": adversarial,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Paprika Step 1 Generation-Thinking Rescue",
        "",
        f"Status: **{result['status']}**",
        "",
        f"| arm | resolution@{result['round_budget']} | mean censored turns | coverage | cost (USD) | requests |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("naive_nonthinking", "naive_thinking", "rescue_eig"):
        arm = result["arms"][name]
        lines.append(
            f"| {name} | {arm['resolution_at_budget']:.3f} | {arm['mean_censored_turns']:.3f} | "
            f"{arm['answer_set_coverage']:.3f} | {arm['backend_cost_usd']:.4f} | {arm['backend_requests']} |"
        )
    for title, key in (
        ("Matched rescue vs naive non-thinking", "claim1_matched_rescue_vs_naive_nonthinking"),
        ("Adversarial rescue vs naive thinking", "claim1_adversarial_rescue_vs_naive_thinking"),
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
    parser.add_argument("--rescue-run", type=Path, required=True)
    parser.add_argument("--naive-nonthinking-run", type=Path, required=True)
    parser.add_argument("--naive-thinking-run", type=Path, required=True)
    parser.add_argument("--round-budget", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.rescue_run,
        args.naive_nonthinking_run,
        args.naive_thinking_run,
        round_budget=args.round_budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(_markdown(result))
    print(args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
