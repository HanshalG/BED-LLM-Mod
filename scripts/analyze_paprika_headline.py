#!/usr/bin/env python3
"""Analyze the pre-registered held-out Paprika arbitration headline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.analyze_paprika_step1 import (
        _arm_summary,
        _comparison,
        _task_values,
        load_arm,
    )
except ModuleNotFoundError:
    from analyze_paprika_step1 import (
        _arm_summary,
        _comparison,
        _task_values,
        load_arm,
    )


def _exact_paired_binary_p(
    better_values: dict[str, dict[str, Any]],
    baseline_values: dict[str, dict[str, Any]],
) -> tuple[int, int, float | None]:
    wins = sum(
        better_values[task_id]["resolved"]
        and not baseline_values[task_id]["resolved"]
        for task_id in better_values
    )
    losses = sum(
        baseline_values[task_id]["resolved"]
        and not better_values[task_id]["resolved"]
        for task_id in better_values
    )
    discordant = wins + losses
    if discordant == 0:
        return wins, losses, None
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    p_value = min(1.0, 2.0 * tail / (2**discordant))
    return wins, losses, float(p_value)


def _headline_comparison(
    better: str,
    baseline: str,
    values: dict[str, dict[str, dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = _comparison(better, baseline, values)
    resolution_wins, resolution_losses, p_value = _exact_paired_binary_p(
        values[better], values[baseline]
    )
    result.update(
        {
            "resolution_rate_delta": summaries[better]["resolution_at_budget"]
            - summaries[baseline]["resolution_at_budget"],
            "resolution_discordant_wins": resolution_wins,
            "resolution_discordant_losses": resolution_losses,
            "paired_binary_exact_p": p_value,
        }
    )
    return result


def _proposal_pairing(
    arbitration_records: tuple[dict[str, Any], ...],
    candidate0_records: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    arbitration = {str(record["task_id"]): record for record in arbitration_records}
    candidate0 = {str(record["task_id"]): record for record in candidate0_records}
    eligible = 0
    matched = 0
    mismatches: list[dict[str, Any]] = []
    for task_id in sorted(arbitration):
        arbitration_turns = list(arbitration[task_id].get("turns", []))
        candidate0_turns = list(candidate0[task_id].get("turns", []))
        arbitration_history: list[tuple[str, str]] = []
        candidate0_history: list[tuple[str, str]] = []
        for turn_index, (arbitration_turn, candidate0_turn) in enumerate(
            zip(arbitration_turns, candidate0_turns),
            start=1,
        ):
            if arbitration_history == candidate0_history:
                eligible += 1
                arbitration_queries = (
                    arbitration_turn.get("selection_extras", {}).get("candidate_queries")
                )
                candidate0_queries = (
                    candidate0_turn.get("selection_extras", {}).get("candidate_queries")
                )
                if arbitration_queries == candidate0_queries and arbitration_queries:
                    matched += 1
                else:
                    mismatches.append(
                        {
                            "task_id": task_id,
                            "turn": turn_index,
                            "arbitration_candidates": arbitration_queries,
                            "candidate0_candidates": candidate0_queries,
                        }
                    )
            arbitration_history.append(
                (str(arbitration_turn.get("query")), str(arbitration_turn.get("reply")))
            )
            candidate0_history.append(
                (str(candidate0_turn.get("query")), str(candidate0_turn.get("reply")))
            )
    return {
        "eligible_identical_history_turns": eligible,
        "matched_proposal_turns": matched,
        "pairing_rate": matched / eligible if eligible else 0.0,
        "mismatches": mismatches,
        "valid": bool(eligible and not mismatches),
    }


def _mechanism_summary(
    arbitration_records: tuple[dict[str, Any], ...],
    causal_comparison: dict[str, Any],
) -> dict[str, Any]:
    task_outcomes = {
        str(row["task_id"]): str(row["outcome"])
        for row in causal_comparison["per_task"]
    }
    overrides: list[dict[str, Any]] = []
    total_turns = 0
    for record in arbitration_records:
        task_id = str(record["task_id"])
        for turn_index, turn in enumerate(record.get("turns", []), start=1):
            total_turns += 1
            extras = turn.get("selection_extras") or {}
            if extras.get("native_overridden") is not True:
                continue
            gap = float(extras.get("score_gap_vs_native", 0.0))
            threshold = float(extras.get("one_se_threshold", 0.0))
            overrides.append(
                {
                    "task_id": task_id,
                    "turn": turn_index,
                    "selected_index": int(extras.get("selected_index", 0)),
                    "score_gap": gap,
                    "one_se_threshold": threshold,
                    "margin_excess": gap - threshold,
                    "immediate_resolution": bool(turn.get("goal_reached")),
                    "task_outcome_vs_candidate0": task_outcomes[task_id],
                }
            )
    grouped: dict[str, dict[str, Any]] = {}
    for outcome in ("win", "loss", "tie"):
        rows = [row for row in overrides if row["task_outcome_vs_candidate0"] == outcome]
        grouped[outcome] = {
            "count": len(rows),
            "immediate_resolutions": sum(row["immediate_resolution"] for row in rows),
            "mean_score_gap": (
                float(np.mean([row["score_gap"] for row in rows])) if rows else None
            ),
            "mean_margin_excess": (
                float(np.mean([row["margin_excess"] for row in rows])) if rows else None
            ),
        }
    return {
        "total_turns": total_turns,
        "override_count": len(overrides),
        "override_rate": len(overrides) / total_turns if total_turns else 0.0,
        "immediate_resolution_count": sum(
            row["immediate_resolution"] for row in overrides
        ),
        "by_task_outcome_vs_candidate0": grouped,
        "overrides": overrides,
    }


def _manual_review_plan(
    values: dict[str, dict[str, dict[str, Any]]],
    *,
    seed: int = 1304,
) -> dict[str, Any]:
    task_ids = sorted(next(iter(values.values())))
    disagreements = [
        task_id
        for task_id in task_ids
        if len({values[arm][task_id]["resolved"] for arm in values}) > 1
    ]
    remaining = [task_id for task_id in task_ids if task_id not in disagreements]
    rng = np.random.default_rng(seed)
    spot_count = min(10, len(remaining))
    spot_checks = sorted(
        rng.choice(remaining, size=spot_count, replace=False).tolist()
        if spot_count
        else []
    )
    return {
        "rule": "review every success-disagreement task and ten seeded random remaining tasks",
        "seed": seed,
        "success_disagreement_task_ids": disagreements,
        "random_spot_check_task_ids": spot_checks,
        "all_review_task_ids": sorted(set(disagreements + spot_checks)),
    }


def analyze(
    headline_run: Path,
    naive_nonthinking_run: Path,
    *,
    round_budget: int = 5,
    expected_start: int = 10,
    expected_count: int = 50,
) -> dict[str, Any]:
    arms = {
        "arbitration": load_arm(headline_run, "NaivePrimaryArbitration"),
        "candidate0": load_arm(headline_run, "NaivePrimaryCandidate0"),
        "naive_thinking": load_arm(headline_run, "naive"),
        "naive_nonthinking": load_arm(naive_nonthinking_run, "naive"),
    }
    values = {name: _task_values(arm, round_budget) for name, arm in arms.items()}
    expected_ids = {
        f"customer_service:eval:{index:04d}"
        for index in range(expected_start, expected_start + expected_count)
    }
    if any(set(rows) != expected_ids for rows in values.values()):
        raise ValueError("Headline arms do not cover the exact pre-registered task IDs")
    summaries = {
        name: _arm_summary(arm, values[name], round_budget) for name, arm in arms.items()
    }
    for summary in summaries.values():
        resolved = summary["resolution_at_budget"] * summary["num_tasks"]
        summary["cost_per_resolution_usd"] = (
            summary["backend_cost_usd"] / resolved if resolved else None
        )
    primary = _headline_comparison(
        "arbitration", "naive_thinking", values, summaries
    )
    causal = _headline_comparison("arbitration", "candidate0", values, summaries)
    context = _headline_comparison(
        "arbitration", "naive_nonthinking", values, summaries
    )
    pairing = _proposal_pairing(arms["arbitration"].records, arms["candidate0"].records)
    endpoint_valid = all(summary["endpoint_valid"] for summary in summaries.values())
    primary_supported = primary["bootstrap_ci95"][1] < 0.0
    causal_supported = causal["bootstrap_ci95"][1] < 0.0
    primary_directional = primary["mean_censored_turn_delta"] < 0.0
    causal_directional = causal["mean_censored_turn_delta"] < 0.0
    if not endpoint_valid:
        claim_read = "invalid_endpoint"
    elif not pairing["valid"]:
        claim_read = "invalid_candidate_pairing"
    elif primary_supported and causal_supported:
        claim_read = "claim_b_confirmed"
    elif primary_supported and causal_directional:
        claim_read = "native_gain_supported_causal_gain_uncertain"
    elif primary_directional and causal_directional:
        claim_read = "directional_but_uncertain"
    elif primary_directional:
        claim_read = "native_gain_without_eig_override_gain"
    else:
        claim_read = "claim_b_not_confirmed"
    mechanism = _mechanism_summary(arms["arbitration"].records, causal)
    return {
        "status": f"{claim_read}_requires_manual_review",
        "claim_read_before_manual_review": claim_read,
        "manual_review_required": True,
        "round_budget": round_budget,
        "expected_start": expected_start,
        "expected_count": expected_count,
        "censoring_rule": "unresolved tasks score round_budget + 1 censored turns",
        "claim_rule": (
            "Claim B confirmed only when endpoint and candidate pairing are valid and "
            "both arbitration-vs-thinking-naive and arbitration-vs-candidate0 paired "
            "censored-turn bootstrap CIs exclude zero in arbitration's favor"
        ),
        "endpoint_valid_automated": endpoint_valid,
        "candidate_pairing": pairing,
        "arms": summaries,
        "primary_arbitration_vs_naive_thinking": primary,
        "coprimary_arbitration_vs_candidate0": causal,
        "context_arbitration_vs_naive_nonthinking": context,
        "mechanism": mechanism,
        "manual_review_plan": _manual_review_plan(values),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Paprika Arbitration Headline",
        "",
        f"Automated status: **{result['status']}**",
        "",
        "Manual endpoint review is required before interpreting the claim.",
        "",
        f"| arm | resolution@{result['round_budget']} | mean censored turns | coverage | cost (USD) | cost/resolution | requests |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("arbitration", "candidate0", "naive_thinking", "naive_nonthinking"):
        arm = result["arms"][name]
        cost_per_resolution = arm["cost_per_resolution_usd"]
        cost_text = "n/a" if cost_per_resolution is None else f"{cost_per_resolution:.4f}"
        lines.append(
            f"| {name} | {arm['resolution_at_budget']:.3f} | "
            f"{arm['mean_censored_turns']:.3f} | {arm['answer_set_coverage']:.3f} | "
            f"{arm['backend_cost_usd']:.4f} | {cost_text} | {arm['backend_requests']} |"
        )
    for title, key in (
        ("Primary: arbitration vs thinking naive", "primary_arbitration_vs_naive_thinking"),
        ("Co-primary: arbitration vs candidate 0", "coprimary_arbitration_vs_candidate0"),
        ("Context: arbitration vs non-thinking naive", "context_arbitration_vs_naive_nonthinking"),
    ):
        value = result[key]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"Turn wins/losses/ties: {value['wins']}/{value['losses']}/{value['ties']}.",
                f"Mean paired censored-turn delta: {value['mean_censored_turn_delta']:.3f}; "
                f"95% bootstrap CI {value['bootstrap_ci95']}; Wilcoxon p={value['wilcoxon_signed_rank_p']}.",
                f"Resolution delta: {value['resolution_rate_delta']:.3f}; discordant "
                f"wins/losses {value['resolution_discordant_wins']}/"
                f"{value['resolution_discordant_losses']}; exact p={value['paired_binary_exact_p']}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Mechanism",
            "",
            f"Override rate: {result['mechanism']['override_count']}/"
            f"{result['mechanism']['total_turns']} = {result['mechanism']['override_rate']:.3f}.",
            f"Immediate resolutions after overrides: "
            f"{result['mechanism']['immediate_resolution_count']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline-run", type=Path, required=True)
    parser.add_argument("--naive-nonthinking-run", type=Path, required=True)
    parser.add_argument("--round-budget", type=int, default=5)
    parser.add_argument("--expected-start", type=int, default=10)
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.headline_run,
        args.naive_nonthinking_run,
        round_budget=args.round_budget,
        expected_start=args.expected_start,
        expected_count=args.expected_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(_markdown(result))
    print(args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
