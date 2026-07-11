#!/usr/bin/env python3
"""Analyze the preregistered paired Paprika Step 1 gap pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class Arm:
    method: str
    records: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]


def _item_for_method(run_dir: Path, method: str) -> dict[str, Any]:
    payload = json.loads((run_dir / "metrics.json").read_text())
    matches = [item for item in payload.get("items", []) if item.get("method") == method]
    if len(matches) != 1:
        raise ValueError(f"Expected one {method!r} item in {run_dir}, found {len(matches)}")
    return matches[0]


def load_arm(run_dir: Path, method: str) -> Arm:
    item = _item_for_method(run_dir, method)
    artifact = run_dir / item["artifacts"]["paprika_smoke"]
    records = json.loads(artifact.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"No Paprika records in {artifact}")
    return Arm(method=method, records=tuple(records), metrics=dict(item.get("metrics", {})))


def _task_values(arm: Arm, round_budget: int) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for record in arm.records:
        task_id = str(record["task_id"])
        turns = list(record.get("turns", []))
        resolution_turn = next(
            (index + 1 for index, turn in enumerate(turns) if turn.get("goal_reached") is True),
            None,
        )
        clean = sum(turn.get("mapped_cleanly") is True for turn in turns)
        values[task_id] = {
            "resolved": resolution_turn is not None,
            "resolution_turn": resolution_turn,
            "censored_turns": resolution_turn if resolution_turn is not None else round_budget + 1,
            "num_turns": len(turns),
            "clean_turns": clean,
        }
    return values


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 1304) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(5000, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _wilcoxon_p(values: np.ndarray) -> float | None:
    nonzero = values[np.abs(values) > 1e-12]
    if nonzero.size == 0:
        return None
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(nonzero, alternative="two-sided").pvalue)
    except (ImportError, ValueError):
        return None


def _arm_summary(arm: Arm, values: dict[str, dict[str, Any]], round_budget: int) -> dict[str, Any]:
    rows = list(values.values())
    total_turns = sum(row["num_turns"] for row in rows)
    resolution_curve = [
        sum(bool(row["resolved"] and row["resolution_turn"] <= turn) for row in rows) / len(rows)
        for turn in range(1, round_budget + 1)
    ]
    metric = lambda name: (arm.metrics.get(name) or [0])[-1]
    faithfulness_metrics = (
        "simulator_faithfulness_observations",
        "simulator_faithfulness_checks",
        "simulator_faithfulness_raw_contradictions",
        "simulator_faithfulness_repairs",
        "simulator_faithfulness_failures",
        "simulator_faithfulness_final_inconsistency_rate",
        "simulator_terminal_claims",
        "simulator_terminal_checks",
        "simulator_terminal_rejections",
    )
    faithfulness_present = all(name in arm.metrics for name in faithfulness_metrics)
    structured_parse_failures = float(metric("structured_parse_failures"))
    faithfulness_failures = float(metric("simulator_faithfulness_failures"))
    final_inconsistency_rate = float(
        metric("simulator_faithfulness_final_inconsistency_rate")
    )
    return {
        "num_tasks": len(rows),
        "resolution_curve": resolution_curve,
        "resolution_at_budget": resolution_curve[-1],
        "mean_censored_turns": float(np.mean([row["censored_turns"] for row in rows])),
        "answer_set_coverage": (
            sum(row["clean_turns"] for row in rows) / total_turns if total_turns else 0.0
        ),
        "structured_parse_failures": structured_parse_failures,
        "simulator_faithfulness_metrics_present": faithfulness_present,
        "simulator_faithfulness_observations": float(
            metric("simulator_faithfulness_observations")
        ),
        "simulator_faithfulness_checks": float(
            metric("simulator_faithfulness_checks")
        ),
        "simulator_faithfulness_raw_contradictions": float(
            metric("simulator_faithfulness_raw_contradictions")
        ),
        "simulator_faithfulness_repairs": float(
            metric("simulator_faithfulness_repairs")
        ),
        "simulator_faithfulness_failures": faithfulness_failures,
        "simulator_faithfulness_final_inconsistency_rate": final_inconsistency_rate,
        "simulator_terminal_claims": float(metric("simulator_terminal_claims")),
        "simulator_terminal_checks": float(metric("simulator_terminal_checks")),
        "simulator_terminal_rejections": float(metric("simulator_terminal_rejections")),
        "endpoint_valid": bool(
            faithfulness_present
            and structured_parse_failures == 0.0
            and faithfulness_failures == 0.0
            and final_inconsistency_rate == 0.0
        ),
        "backend_cost_usd": float(metric("backend_cost_usd")),
        "backend_requests": int(metric("backend_requests")),
        "backend_prompt_tokens": int(metric("backend_prompt_tokens")),
        "backend_completion_tokens": int(metric("backend_completion_tokens")),
        "backend_reasoning_tokens": int(metric("backend_reasoning_tokens")),
        "backend_forced_exits": int(metric("backend_forced_exits")),
    }


def _comparison(
    better: str,
    baseline: str,
    arm_values: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    better_values = arm_values[better]
    baseline_values = arm_values[baseline]
    if set(better_values) != set(baseline_values):
        raise ValueError(f"Task IDs differ between {better} and {baseline}")
    task_ids = sorted(better_values)
    deltas = np.asarray(
        [better_values[task]["censored_turns"] - baseline_values[task]["censored_turns"] for task in task_ids],
        dtype=float,
    )
    per_task = []
    for task_id, delta in zip(task_ids, deltas):
        outcome = "win" if delta < 0 else "loss" if delta > 0 else "tie"
        per_task.append(
            {
                "task_id": task_id,
                "outcome": outcome,
                "censored_turn_delta": float(delta),
                better: better_values[task_id],
                baseline: baseline_values[task_id],
            }
        )
    return {
        "better_arm": better,
        "baseline_arm": baseline,
        "wins": int(np.sum(deltas < 0)),
        "losses": int(np.sum(deltas > 0)),
        "ties": int(np.sum(deltas == 0)),
        "mean_censored_turn_delta": float(np.mean(deltas)),
        "bootstrap_ci95": _bootstrap_mean_ci(deltas),
        "wilcoxon_signed_rank_p": _wilcoxon_p(deltas),
        "per_task": per_task,
    }


def _gate_outcome(
    comparison: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    *,
    better: str,
    baseline: str,
    require_six_wins: bool,
) -> dict[str, Any]:
    resolution_advantage = (
        summaries[better]["resolution_at_budget"]
        - summaries[baseline]["resolution_at_budget"]
    )
    clear_edge = bool(
        resolution_advantage >= 0.2 - 1e-12
        or comparison["mean_censored_turn_delta"] <= -0.2 + 1e-12
    )
    mostly_ties = comparison["ties"] >= 6
    directional = comparison["wins"] > comparison["losses"]
    passed = clear_edge or (
        comparison["wins"] >= 6 if require_six_wins else directional and not mostly_ties
    )
    status = "pass" if passed else "insufficient_signal" if mostly_ties else "fail"
    comparison.update(
        {
            "gate_status": status,
            "gate_pass": passed,
            "mostly_ties": mostly_ties,
            "resolution_rate_advantage": resolution_advantage,
            "clear_edge": clear_edge,
        }
    )
    return comparison


def analyze(
    scaffolded_run: Path,
    naive_nonthinking_run: Path,
    naive_thinking_run: Path,
    *,
    round_budget: int = 5,
) -> dict[str, Any]:
    arms = {
        "EIG": load_arm(scaffolded_run, "EIG"),
        "Full2StepEIG": load_arm(scaffolded_run, "Full2StepEIG"),
        "naive_nonthinking": load_arm(naive_nonthinking_run, "naive"),
        "naive_thinking": load_arm(naive_thinking_run, "naive"),
    }
    arm_values = {name: _task_values(arm, round_budget) for name, arm in arms.items()}
    task_sets = {tuple(sorted(values)) for values in arm_values.values()}
    if len(task_sets) != 1 or len(next(iter(task_sets))) != 10:
        raise ValueError("Step 1 requires exactly ten identical paired task IDs across arms")
    summaries = {
        name: _arm_summary(arm, arm_values[name], round_budget) for name, arm in arms.items()
    }
    claim1 = _gate_outcome(
        _comparison("EIG", "naive_nonthinking", arm_values),
        summaries,
        better="EIG",
        baseline="naive_nonthinking",
        require_six_wins=False,
    )
    adversarial = _comparison("EIG", "naive_thinking", arm_values)
    claim2 = _gate_outcome(
        _comparison("Full2StepEIG", "EIG", arm_values),
        summaries,
        better="Full2StepEIG",
        baseline="EIG",
        require_six_wins=True,
    )
    endpoint_valid = all(summary["endpoint_valid"] for summary in summaries.values())
    if not endpoint_valid:
        status = "invalid_endpoint_stop"
    elif claim1["gate_status"] == "fail":
        status = "claim1_matched_fail_rescue_or_stop"
    elif claim1["gate_status"] == "insufficient_signal":
        status = "claim1_matched_insufficient_signal"
    elif claim2["gate_status"] == "pass":
        status = "claims1_and_2_pass"
    elif claim2["gate_status"] == "insufficient_signal":
        status = "claim1_pass_claim2_insufficient_signal"
    else:
        status = "claim1_only_descope_lookahead"
    return {
        "status": status,
        "round_budget": round_budget,
        "censoring_rule": "unresolved tasks score round_budget + 1 censored turns",
        "claim2_clear_edge_rule": "at least 2/10 additional resolutions or mean censored-turn delta <= -0.2",
        "arms": summaries,
        "endpoint_valid": endpoint_valid,
        "claim1_matched_eig_vs_naive_nonthinking": claim1,
        "claim1_adversarial_eig_vs_naive_thinking": adversarial,
        "claim2_full2_vs_eig": claim2,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Paprika Step 1 Gap Pilot",
        "",
        f"Status: **{result['status']}**",
        "",
        f"| arm | resolution@{result['round_budget']} | mean censored turns | coverage | cost (USD) | requests |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("naive_nonthinking", "naive_thinking", "EIG", "Full2StepEIG"):
        arm = result["arms"][name]
        lines.append(
            f"| {name} | {arm['resolution_at_budget']:.3f} | {arm['mean_censored_turns']:.3f} | "
            f"{arm['answer_set_coverage']:.3f} | {arm['backend_cost_usd']:.4f} | {arm['backend_requests']} |"
        )
    for title, key in (
        ("Claim 1 matched: EIG vs naive non-thinking", "claim1_matched_eig_vs_naive_nonthinking"),
        ("Claim 1 adversarial: EIG vs naive thinking", "claim1_adversarial_eig_vs_naive_thinking"),
        ("Claim 2: full2 vs EIG", "claim2_full2_vs_eig"),
    ):
        value = result[key]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"Wins/losses/ties: {value['wins']}/{value['losses']}/{value['ties']}. ",
                f"Mean paired censored-turn delta: {value['mean_censored_turn_delta']:.3f}; "
                f"95% bootstrap CI {value['bootstrap_ci95']}; Wilcoxon p={value['wilcoxon_signed_rank_p']}.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffolded-run", type=Path, required=True)
    parser.add_argument("--naive-nonthinking-run", type=Path, required=True)
    parser.add_argument("--naive-thinking-run", type=Path, required=True)
    parser.add_argument("--round-budget", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.scaffolded_run,
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
