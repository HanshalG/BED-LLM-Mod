#!/usr/bin/env python3
"""Probe whether 20-Questions candidate choices alter future truth coverage.

For each ordinary one-round live trajectory, this script scores the next
candidate pool with immediate EIG and then branches each candidate through the
same belief regeneration and filtering path used by the environment.  The
target is used only after each counterfactual update has returned, to measure
whether that support contains it; it is never included in questioner scoring
or regeneration prompts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from environments.animals.env import AnimalsBEDEnvironment
from environments.animals.questions import (
    CandidateCoverageDynamics,
    evaluate_candidate_coverage_dynamics,
)
from helpers import Config, load_config
from model_factory import build_model_adapter


class CoverageProbeError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _history_messages(history: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        message
        for question, answer in history
        for message in (
            {"role": "assistant", "content": question},
            {"role": "user", "content": answer},
        )
    ]


def _serialize_dynamics(entry: CandidateCoverageDynamics) -> dict[str, Any]:
    return {
        **asdict(entry),
        "expected_truth_coverage": entry.expected_truth_coverage,
    }


def _average_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    left_centered = left_array - left_array.mean()
    right_centered = right_array - right_array.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def summarize_probe(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize candidate-level coverage variation without using it as a policy."""
    immediate_eigs: list[float] = []
    expected_coverages: list[float] = []
    state_spreads: list[float] = []
    immediate_eig_coverage_regrets: list[float] = []

    for record in records:
        dynamics = record["candidate_dynamics"]
        if not dynamics:
            continue
        coverage_values = [float(entry["expected_truth_coverage"]) for entry in dynamics]
        eig_values = [float(entry["immediate_eig"]) for entry in dynamics]
        state_spreads.append(max(coverage_values) - min(coverage_values))
        selected_index = max(range(len(dynamics)), key=lambda index: eig_values[index])
        immediate_eig_coverage_regrets.append(max(coverage_values) - coverage_values[selected_index])
        immediate_eigs.extend(eig_values)
        expected_coverages.extend(coverage_values)

    spearman = _pearson_correlation(_average_ranks(immediate_eigs), _average_ranks(expected_coverages))
    return {
        "num_states": len(records),
        "num_candidate_rows": len(immediate_eigs),
        "mean_within_state_coverage_spread": float(np.mean(state_spreads)) if state_spreads else None,
        "median_within_state_coverage_spread": float(np.median(state_spreads)) if state_spreads else None,
        "max_within_state_coverage_spread": max(state_spreads) if state_spreads else None,
        "states_with_coverage_spread_at_least_0_20": sum(spread >= 0.20 for spread in state_spreads),
        "mean_immediate_eig_coverage_regret": (
            float(np.mean(immediate_eig_coverage_regrets)) if immediate_eig_coverage_regrets else None
        ),
        "states_with_immediate_eig_coverage_regret_at_least_0_20": sum(
            regret >= 0.20 for regret in immediate_eig_coverage_regrets
        ),
        "spearman_immediate_eig_vs_expected_truth_coverage": spearman,
    }


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("num_states", "candidate_width", "max_attempts"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_attempts < args.num_states:
        raise ValueError("--max-attempts must be at least --num-states")


def run_probe(
    runtime_config: Config,
    *,
    num_states: int,
    candidate_width: int,
    max_attempts: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not runtime_config.model_pairs:
        raise ValueError("coverage probe config requires one questioner/answerer model pair")
    if not runtime_config.animals or not runtime_config.animals[runtime_config.version]:
        raise ValueError("coverage probe config requires a non-empty animals[version] target pool")

    np.random.seed(seed)
    pair = runtime_config.model_pairs[0]
    questioner = build_model_adapter(pair.questioner, config=runtime_config)
    answerer = build_model_adapter(pair.answerer, config=runtime_config)
    targets = list(runtime_config.animals[runtime_config.version])
    env = AnimalsBEDEnvironment(runtime_config, answerer=answerer, target_animals=targets)
    records: list[dict[str, Any]] = []

    try:
        for attempt_index in range(max_attempts):
            if len(records) == num_states:
                break
            target = targets[attempt_index % len(targets)]
            beliefs = env.initial_belief_state(questioner, runtime_config)
            bootstrap_candidates = env.generate_candidate_actions(beliefs, [], questioner, runtime_config)
            bootstrap_candidates = list(dict.fromkeys(bootstrap_candidates))
            if not bootstrap_candidates:
                continue
            bootstrap_question = bootstrap_candidates[attempt_index % len(bootstrap_candidates)]
            bootstrap_answer = env.observe(bootstrap_question, target, np.random.default_rng(seed + attempt_index))
            if bootstrap_answer not in {"Yes", "No"}:
                continue

            history = [(bootstrap_question, bootstrap_answer)]
            beliefs = env.update_belief_state(beliefs, history, questioner, runtime_config)
            candidate_questions = env.generate_candidate_actions(beliefs, history, questioner, runtime_config)
            candidate_questions = list(dict.fromkeys(candidate_questions))[:candidate_width]
            if len(candidate_questions) < 2:
                continue

            dynamics = evaluate_candidate_coverage_dynamics(
                beliefs,
                _history_messages(history),
                candidate_questions,
                target,
                deterministic=False,
                questioner=questioner,
                config=runtime_config,
            )
            records.append(
                {
                    "state_index": len(records),
                    "attempt_index": attempt_index,
                    "target_measurement_only": target,
                    "bootstrap_question": bootstrap_question,
                    "bootstrap_answer": bootstrap_answer,
                    "history": [{"question": question, "answer": answer} for question, answer in history],
                    "belief_support_size": beliefs.support_size,
                    "candidate_dynamics": [_serialize_dynamics(entry) for entry in dynamics],
                }
            )
    except Exception as exc:
        usage = {
            "questioner": questioner.usage_snapshot(),
            "answerer": answerer.usage_snapshot(),
        }
        raise CoverageProbeError(str(exc), usage) from exc

    usage = {
        "questioner": questioner.usage_snapshot(),
        "answerer": answerer.usage_snapshot(),
    }

    if len(records) != num_states:
        raise CoverageProbeError(
            f"Collected {len(records)}/{num_states} usable states after {max_attempts} attempts",
            usage,
        )
    return records, summarize_probe(records), usage


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = [
        "# 20 Questions Coverage-Dynamics Probe",
        "",
        "Exploratory diagnostic only. The target is measurement-only after each counterfactual belief update; it is not supplied to questioner scoring or regeneration.",
        "",
        f"- States: `{summary['num_states']}`",
        f"- Candidate rows: `{summary['num_candidate_rows']}`",
        f"- Mean within-state coverage spread: `{summary['mean_within_state_coverage_spread']}`",
        f"- Median within-state coverage spread: `{summary['median_within_state_coverage_spread']}`",
        f"- Max within-state coverage spread: `{summary['max_within_state_coverage_spread']}`",
        f"- Mean immediate-EIG coverage regret: `{summary['mean_immediate_eig_coverage_regret']}`",
        f"- Spearman(immediate EIG, expected truth coverage): `{summary['spearman_immediate_eig_vs_expected_truth_coverage']}`",
        "",
        "Per-state candidate/branch values are in `COVERAGE_PROBE.json`.",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_animals_coverage_dynamics_openrouter.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/nonmyopic/animals_coverage_dynamics"))
    parser.add_argument("--run-id", default="animals-coverage-dynamics-20260718")
    parser.add_argument("--num-states", type=int, default=10)
    parser.add_argument("--candidate-width", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1304)
    args = parser.parse_args()
    _validate_args(args)

    runtime_config = load_config(str(args.config))
    runtime_config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        records, summary, usage = run_probe(
            runtime_config,
            num_states=args.num_states,
            candidate_width=args.candidate_width,
            max_attempts=args.max_attempts,
            seed=args.seed,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "run_id": args.run_id,
            "error": f"{type(exc).__name__}: {exc}",
            "usage": getattr(exc, "usage", None),
        }
        (args.output_dir / "COVERAGE_PROBE_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    payload = {
        "schema_version": 1,
        "exploratory_only": True,
        "target_measurement_only": True,
        "run_id": args.run_id,
        "config_path": str(args.config),
        "probe": {
            "num_states": args.num_states,
            "candidate_width": args.candidate_width,
            "max_attempts": args.max_attempts,
            "seed": args.seed,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }
    (args.output_dir / "COVERAGE_PROBE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "COVERAGE_PROBE.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
