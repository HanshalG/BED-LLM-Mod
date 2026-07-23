#!/usr/bin/env python3
"""Run a sealed fresh-target gate for the Animals belief-recall ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts.animals_belief_recall_ranker import run_ranker, summarize_rankings
from scripts.animals_coverage_dynamics import run_probe


BOOTSTRAP_SEED = 24272
BOOTSTRAP_REPLICATES = 10_000
MIN_ACTIVE_STATES = 15


def paired_coverage_gains(records: list[dict[str, Any]]) -> list[float]:
    gains = []
    for record in records:
        dynamics = record["candidate_dynamics"]
        ranker_scores = [float(value) for value in record["belief_recall_scores"]]
        immediate_scores = [float(entry["immediate_eig"]) for entry in dynamics]
        coverages = [
            float(entry["expected_truth_coverage"]) for entry in dynamics
        ]
        ranker_index = max(
            range(len(ranker_scores)),
            key=ranker_scores.__getitem__,
        )
        immediate_index = max(
            range(len(immediate_scores)),
            key=immediate_scores.__getitem__,
        )
        gains.append(coverages[ranker_index] - coverages[immediate_index])
    return gains


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_gain(
    gains: list[float],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    if not gains:
        raise ValueError("paired coverage gains must be non-empty")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = random.Random(seed)
    n = len(gains)
    samples = [
        sum(gains[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(replicates)
    ]
    return {
        "seed": seed,
        "replicates": replicates,
        "mean": sum(gains) / n,
        "ci95": [
            _percentile(samples, 0.025),
            _percentile(samples, 0.975),
        ],
    }


def evaluate_gates(
    summary: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, bool]:
    wins, _ties, losses = summary["ranker_immediate_wins_ties_losses"]
    gates = {
        "sixty_states_completed": summary["num_states"] == 60,
        "at_least_fifteen_active_states": summary["num_active_states"]
        >= MIN_ACTIVE_STATES,
        "paired_bootstrap_lower_bound_positive": bootstrap["ci95"][0] > 0.0,
        "more_ranker_wins_than_losses": wins > losses,
        "active_state_regret_better_than_immediate_eig": (
            summary["mean_active_state_regret_belief_recall"]
            < summary["mean_active_state_regret_immediate_eig"]
        ),
    }
    gates["all_pass"] = all(gates.values())
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-config",
        type=Path,
        default=Path(
            "configs/config_animals_belief_recall_holdout_openrouter.yaml"
        ),
    )
    parser.add_argument(
        "--ranker-config",
        type=Path,
        default=Path(
            "configs/config_animals_belief_recall_ranker_openrouter.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=24271)
    parser.add_argument("--audit-seed", type=int, default=24273)
    args = parser.parse_args()

    coverage_config = load_config(str(args.coverage_config))
    ranker_config = load_config(str(args.ranker_config))
    coverage_config.run_id = (
        f"animals-belief-recall-holdout-coverage-{args.seed}"
    )
    ranker_config.run_id = f"animals-belief-recall-holdout-ranker-{args.seed}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        records, coverage_summary, coverage_usage = run_probe(
            coverage_config,
            num_states=60,
            candidate_width=3,
            max_attempts=60,
            seed=args.seed,
        )
        ranked_records, ranker_usage = run_ranker(records, ranker_config)
        ranking_summary = summarize_rankings(ranked_records)
        bootstrap = bootstrap_mean_gain(
            paired_coverage_gains(ranked_records),
            seed=BOOTSTRAP_SEED,
        )
        gates = evaluate_gates(ranking_summary, bootstrap)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "error": f"{type(exc).__name__}: {exc}",
            "coverage_seed": args.seed,
            "producer_bootstrap_seed": BOOTSTRAP_SEED,
            "audit_bootstrap_seed": args.audit_seed,
            "coverage_usage": getattr(exc, "usage", None),
            "partial_records": getattr(exc, "records", None),
        }
        (args.output_dir / "HOLDOUT_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    payload = {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "scientific_gate_failed",
        "target_measurement_only": True,
        "no_intermediate_endpoint_written": True,
        "coverage_config_path": str(args.coverage_config),
        "ranker_config_path": str(args.ranker_config),
        "coverage_seed": args.seed,
        "producer_bootstrap_seed": BOOTSTRAP_SEED,
        "audit_bootstrap_seed": args.audit_seed,
        "coverage_summary": coverage_summary,
        "ranking_summary": ranking_summary,
        "paired_coverage_gain_bootstrap": bootstrap,
        "gates": gates,
        "records": ranked_records,
        "usage": {
            "coverage": coverage_usage,
            "ranker": ranker_usage,
        },
    }
    (args.output_dir / "HOLDOUT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ranking_summary": ranking_summary,
                "paired_coverage_gain_bootstrap": bootstrap,
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
