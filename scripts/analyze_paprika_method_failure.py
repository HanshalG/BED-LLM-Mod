#!/usr/bin/env python3
"""Reproduce diagnostics for the invalidated Paprika BED headline.

This script does not re-run the frozen headline analyzer and does not make model calls.
It reads the accepted combined artifacts only and emits descriptive diagnostics that
are explicitly unsuitable as endpoint-valid policy evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_arm_records(run_dir: Path, method: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = _load_json(run_dir / "metrics.json")
    matches = [item for item in metrics.get("items", []) if item.get("method") == method]
    if len(matches) != 1:
        raise ValueError(f"Expected one {method!r} item in {run_dir}, found {len(matches)}")
    item = matches[0]
    artifact = item.get("artifacts", {}).get("paprika_smoke")
    if not isinstance(artifact, str):
        raise ValueError(f"{method!r} in {run_dir} has no paprika_smoke artifact")
    records = _load_json(run_dir / artifact)
    if not isinstance(records, list):
        raise ValueError(f"Paprika artifact for {method!r} must be a list")
    return records, item


def _resolution_turn(record: dict[str, Any], round_budget: int) -> int | None:
    for turn_index, turn in enumerate(record.get("turns", []), start=1):
        if turn.get("goal_reached") is True:
            return turn_index
    return None


def _censored_turns(record: dict[str, Any], round_budget: int) -> int:
    return _resolution_turn(record, round_budget) or round_budget + 1


def _selected_score(turn: dict[str, Any]) -> float | None:
    extras = turn.get("selection_extras") or {}
    scores = extras.get("candidate_scores")
    if not isinstance(scores, list) or not scores:
        return None
    selected_index = extras.get("selected_index")
    if isinstance(selected_index, int) and 0 <= selected_index < len(scores):
        return float(scores[selected_index])
    return float(max(scores))


def action_summary(
    records: Sequence[dict[str, Any]], round_budget: int
) -> dict[str, Any]:
    turns = [turn for record in records for turn in record.get("turns", [])]
    resolution_histogram = Counter(
        turn
        for record in records
        if (turn := _resolution_turn(record, round_budget)) is not None
    )
    cumulative = 0
    resolution_curve: list[float] = []
    for turn in range(1, round_budget + 1):
        cumulative += resolution_histogram[turn]
        resolution_curve.append(cumulative / len(records) if records else 0.0)

    exact_masses = [
        float(record.get("final_metrics", {}).get("true_solution_exact_mass", 0.0))
        for record in records
        if "true_solution_exact_mass" in record.get("final_metrics", {})
    ]
    scored_turns: list[tuple[float, bool, bool]] = []
    for record in records:
        eventual = _resolution_turn(record, round_budget) is not None
        for turn in record.get("turns", []):
            score = _selected_score(turn)
            if score is not None:
                scored_turns.append((score, bool(turn.get("goal_reached")), eventual))

    def score_group(index: int, value: bool) -> dict[str, float | int | None]:
        selected = [row[0] for row in scored_turns if row[index] is value]
        return {
            "count": len(selected),
            "mean": fmean(selected) if selected else None,
        }

    return {
        "num_tasks": len(records),
        "resolved": sum(_resolution_turn(record, round_budget) is not None for record in records),
        "resolution_curve": resolution_curve,
        "num_turns": len(turns),
        "action_kinds": dict(sorted(Counter(str(turn.get("kind")) for turn in turns).items())),
        "solution_action_fraction": (
            sum(turn.get("kind") == "solution" for turn in turns) / len(turns)
            if turns
            else 0.0
        ),
        "unclean_mappings": sum(turn.get("mapped_cleanly") is not True for turn in turns),
        "mapping_coverage": (
            sum(turn.get("mapped_cleanly") is True for turn in turns) / len(turns)
            if turns
            else 0.0
        ),
        "outcome_cardinality": {
            str(size): sum(len(turn.get("outcomes", [])) == size for turn in turns)
            for size in (3, 4, 5)
        },
        "true_solution_exact_mass": {
            "logged_tasks": len(exact_masses),
            "nonzero_tasks": sum(value > 0.0 for value in exact_masses),
        },
        "selected_score_descriptives": {
            "immediate_resolution": score_group(1, True),
            "no_immediate_resolution": score_group(1, False),
            "eventually_resolved_task": score_group(2, True),
            "unresolved_task": score_group(2, False),
        },
    }


def _paired_label(selected: bool, default: bool) -> str:
    if selected and default:
        return "both"
    if selected:
        return "selected_only"
    if default:
        return "default_only"
    return "neither"


def arbitration_diagnostics(
    arbitration_records: Sequence[dict[str, Any]],
    candidate0_records: Sequence[dict[str, Any]],
    round_budget: int,
) -> dict[str, Any]:
    arbitration = {str(record["task_id"]): record for record in arbitration_records}
    candidate0 = {str(record["task_id"]): record for record in candidate0_records}
    if set(arbitration) != set(candidate0):
        raise ValueError("Arbitration and candidate-0 task IDs differ")

    default_ranks: Counter[int] = Counter()
    selected_indices: Counter[int] = Counter()
    overrides: list[dict[str, Any]] = []
    override_by_turn: Counter[int] = Counter()
    overrides_after_unclean = 0
    for record in arbitration_records:
        task_id = str(record["task_id"])
        previous_unclean = False
        selected_turns = _censored_turns(record, round_budget)
        default_turns = _censored_turns(candidate0[task_id], round_budget)
        task_outcome = (
            "win" if selected_turns < default_turns else "loss" if selected_turns > default_turns else "tie"
        )
        for turn_index, turn in enumerate(record.get("turns", []), start=1):
            extras = turn.get("selection_extras") or {}
            scores = [float(value) for value in extras.get("candidate_scores", [])]
            if len(scores) == 3:
                default_ranks[1 + sum(score > scores[0] for score in scores[1:])] += 1
            selected_index = extras.get("selected_index")
            if isinstance(selected_index, int):
                selected_indices[selected_index] += 1
            if extras.get("native_overridden") is True:
                gap = float(extras.get("score_gap_vs_native", 0.0))
                threshold = float(extras.get("one_se_threshold", 0.0))
                overrides.append(
                    {
                        "task_id": task_id,
                        "turn": turn_index,
                        "task_outcome": task_outcome,
                        "immediate_resolution": bool(turn.get("goal_reached")),
                        "score_gap": gap,
                        "threshold": threshold,
                        "margin_excess": gap - threshold,
                    }
                )
                override_by_turn[turn_index] += 1
                overrides_after_unclean += int(previous_unclean)
            previous_unclean = turn.get("mapped_cleanly") is not True

    grouped: dict[str, Any] = {}
    for outcome in ("win", "loss", "tie"):
        rows = [row for row in overrides if row["task_outcome"] == outcome]
        grouped[outcome] = {
            "count": len(rows),
            "immediate_resolutions": sum(row["immediate_resolution"] for row in rows),
            "mean_score_gap": fmean(row["score_gap"] for row in rows) if rows else None,
            "mean_margin_excess": fmean(row["margin_excess"] for row in rows) if rows else None,
        }

    first_divergences: list[dict[str, Any]] = []
    immediate_pairs: Counter[str] = Counter()
    task_outcomes: Counter[str] = Counter()
    for task_id in sorted(arbitration):
        selected_record = arbitration[task_id]
        default_record = candidate0[task_id]
        selected_history: list[tuple[str, str]] = []
        default_history: list[tuple[str, str]] = []
        for turn_index, (selected_turn, default_turn) in enumerate(
            zip(selected_record.get("turns", []), default_record.get("turns", [])),
            start=1,
        ):
            if (
                selected_history == default_history
                and selected_turn.get("query") != default_turn.get("query")
            ):
                selected_censored = _censored_turns(selected_record, round_budget)
                default_censored = _censored_turns(default_record, round_budget)
                outcome = (
                    "win"
                    if selected_censored < default_censored
                    else "loss"
                    if selected_censored > default_censored
                    else "tie"
                )
                immediate = _paired_label(
                    bool(selected_turn.get("goal_reached")),
                    bool(default_turn.get("goal_reached")),
                )
                immediate_pairs[immediate] += 1
                task_outcomes[outcome] += 1
                first_divergences.append(
                    {
                        "task_id": task_id,
                        "turn": turn_index,
                        "outcome": outcome,
                        "selected_censored_turns": selected_censored,
                        "default_censored_turns": default_censored,
                        "selected_query": selected_turn.get("query"),
                        "default_query": default_turn.get("query"),
                        "selected_reply": selected_turn.get("reply"),
                        "default_reply": default_turn.get("reply"),
                        "immediate_pair": immediate,
                    }
                )
                break
            selected_history.append(
                (str(selected_turn.get("query")), str(selected_turn.get("reply")))
            )
            default_history.append(
                (str(default_turn.get("query")), str(default_turn.get("reply")))
            )

    total_turns = sum(len(record.get("turns", [])) for record in arbitration_records)
    return {
        "total_turns": total_turns,
        "override_count": len(overrides),
        "override_rate": len(overrides) / total_turns if total_turns else 0.0,
        "overrides_by_turn": {str(key): value for key, value in sorted(override_by_turn.items())},
        "overrides_immediately_after_unclean_mapping": overrides_after_unclean,
        "native_default_eig_rank": {str(key): value for key, value in sorted(default_ranks.items())},
        "selected_candidate_index": {str(key): value for key, value in sorted(selected_indices.items())},
        "override_margin_by_task_outcome": grouped,
        "first_divergence": {
            "count": len(first_divergences),
            "immediate_resolution_pair": dict(sorted(immediate_pairs.items())),
            "eventual_censored_turn_outcome": dict(sorted(task_outcomes.items())),
            "rows": first_divergences,
        },
    }


def _metric(item: dict[str, Any], name: str) -> float:
    values = item.get("metrics", {}).get(name)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"Missing scalar metric {name!r}")
    return float(values[0])


def _compact_comparison(value: dict[str, Any]) -> dict[str, Any]:
    """Drop duplicated per-task payloads while retaining the frozen comparison read."""
    return {key: item for key, item in value.items() if key != "per_task"}


def build_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    arbitration, arbitration_item = load_arm_records(args.headline_run, "NaivePrimaryArbitration")
    candidate0, candidate0_item = load_arm_records(args.headline_run, "NaivePrimaryCandidate0")
    naive_thinking, naive_thinking_item = load_arm_records(args.headline_run, "naive")
    naive_nonthinking, naive_nonthinking_item = load_arm_records(
        args.naive_nonthinking_run, "naive"
    )
    best_n, best_n_item = load_arm_records(args.best_n_run, "EIG")
    one_step, one_step_item = load_arm_records(args.step1_run, "EIG")
    full_two_step, full_two_step_item = load_arm_records(args.step1_run, "Full2StepEIG")

    headline = _load_json(args.headline_analysis)
    step1 = _load_json(args.step1_analysis)
    arbitration_pilot = _load_json(args.arbitration_pilot_analysis)
    best_n_pilot = _load_json(args.best_n_pilot_analysis)

    arm_records = {
        "arbitration": arbitration,
        "candidate0": candidate0,
        "naive_thinking": naive_thinking,
        "naive_nonthinking": naive_nonthinking,
        "best_n_eig": best_n,
        "step1_eig": one_step,
        "step1_full2": full_two_step,
    }
    item_by_arm = {
        "arbitration": arbitration_item,
        "candidate0": candidate0_item,
        "naive_thinking": naive_thinking_item,
        "naive_nonthinking": naive_nonthinking_item,
        "best_n_eig": best_n_item,
    }
    action_summaries = {
        name: action_summary(records, args.round_budget)
        for name, records in arm_records.items()
    }
    for name, item in item_by_arm.items():
        action_summaries[name]["backend_requests"] = int(_metric(item, "backend_requests"))
        action_summaries[name]["backend_cost_usd"] = _metric(item, "backend_cost_usd")

    one_step_requests = _metric(one_step_item, "backend_requests")
    full_two_step_requests = _metric(full_two_step_item, "backend_requests")
    one_step_cost = _metric(one_step_item, "backend_cost_usd")
    full_two_step_cost = _metric(full_two_step_item, "backend_cost_usd")
    return {
        "validity": "INVALID-ENDPOINT_DIAGNOSTIC_ONLY",
        "source_policy": (
            "Accepted combined artifacts only; no model calls; frozen headline analyzer was not rerun"
        ),
        "round_budget": args.round_budget,
        "arms": action_summaries,
        "arbitration": arbitration_diagnostics(arbitration, candidate0, args.round_budget),
        "heldout_endpoint": {
            "arbitration_vs_thinking": _compact_comparison(
                headline["primary_arbitration_vs_naive_thinking"]
            ),
            "arbitration_vs_candidate0": _compact_comparison(
                headline["coprimary_arbitration_vs_candidate0"]
            ),
            "best_n_vs_thinking": _compact_comparison(
                headline["context_best_n_vs_naive_thinking"]
            ),
        },
        "development_to_heldout": {
            "arbitration_pilot_vs_thinking": _compact_comparison(
                arbitration_pilot["primary_arbitration_vs_naive_thinking"]
            ),
            "best_n_pilot_vs_thinking": _compact_comparison(
                best_n_pilot["comparisons"]["best_n_eig_vs_naive_thinking"]
            ),
            "heldout_arbitration_vs_thinking": _compact_comparison(
                headline["primary_arbitration_vs_naive_thinking"]
            ),
            "heldout_best_n_vs_thinking": _compact_comparison(
                headline["context_best_n_vs_naive_thinking"]
            ),
        },
        "two_step": {
            "endpoint_comparison": _compact_comparison(step1["claim2_full2_vs_eig"]),
            "one_step_requests": int(one_step_requests),
            "full_two_step_requests": int(full_two_step_requests),
            "request_ratio": full_two_step_requests / one_step_requests,
            "one_step_cost_usd": one_step_cost,
            "full_two_step_cost_usd": full_two_step_cost,
            "cost_ratio": full_two_step_cost / one_step_cost,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline-run", type=Path, required=True)
    parser.add_argument("--naive-nonthinking-run", type=Path, required=True)
    parser.add_argument("--best-n-run", type=Path, required=True)
    parser.add_argument("--step1-run", type=Path, required=True)
    parser.add_argument("--headline-analysis", type=Path, required=True)
    parser.add_argument("--step1-analysis", type=Path, required=True)
    parser.add_argument("--arbitration-pilot-analysis", type=Path, required=True)
    parser.add_argument("--best-n-pilot-analysis", type=Path, required=True)
    parser.add_argument("--round-budget", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_diagnostics(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
