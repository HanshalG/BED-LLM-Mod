#!/usr/bin/env python3
"""Run the sealed fresh Animals multi-sample branch-ranker holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts.animals_belief_recall_holdout import (
    bootstrap_mean_gain,
    paired_coverage_gains,
)
from scripts.animals_belief_recall_ranker import summarize_rankings
from scripts.animals_branch_support_ranker import run_ranker
from scripts.animals_coverage_dynamics import run_probe


PRODUCER_BOOTSTRAP_SEED = 24282


def union_covered_states(records: list[dict[str, Any]]) -> int:
    return sum(
        any(
            bool(candidate["truth_covered_if_yes"])
            or bool(candidate["truth_covered_if_no"])
            for candidate in record["candidate_dynamics"]
        )
        for record in records
    )


def evaluate_gates(
    summary: dict[str, Any],
    bootstrap: dict[str, Any],
    *,
    union_covered: int,
) -> dict[str, bool]:
    wins, _ties, losses = summary["ranker_immediate_wins_ties_losses"]
    ranker_rho = summary[
        "spearman_belief_recall_score_vs_expected_truth_coverage"
    ]
    eig_rho = summary[
        "spearman_immediate_eig_vs_expected_truth_coverage"
    ]
    gates = {
        "sixty_states_completed": summary["num_states"] == 60,
        "at_least_twenty_union_covered_states": union_covered >= 20,
        "at_least_twenty_active_states": summary["num_active_states"] >= 20,
        "ranker_spearman_positive": (
            ranker_rho is not None and ranker_rho > 0.0
        ),
        "ranker_spearman_exceeds_immediate_eig": (
            ranker_rho is not None
            and eig_rho is not None
            and ranker_rho > eig_rho
        ),
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
            "configs/config_animals_multisample_branch_holdout_openrouter.yaml"
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
    parser.add_argument("--seed", type=int, default=24281)
    parser.add_argument("--audit-seed", type=int, default=24283)
    args = parser.parse_args()

    coverage_config = load_config(str(args.coverage_config))
    ranker_config = load_config(str(args.ranker_config))
    coverage_config.run_id = f"animals-multisample-holdout-coverage-{args.seed}"
    ranker_config.run_id = f"animals-multisample-holdout-ranker-{args.seed}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, coverage_summary, coverage_usage = run_probe(
        coverage_config,
        num_states=60,
        candidate_width=3,
        max_attempts=60,
        seed=args.seed,
    )
    ranked, ranker_usage = run_ranker(records, ranker_config)
    summary = summarize_rankings(ranked)
    bootstrap = bootstrap_mean_gain(
        paired_coverage_gains(ranked),
        seed=PRODUCER_BOOTSTRAP_SEED,
    )
    union_covered = union_covered_states(ranked)
    gates = evaluate_gates(
        summary,
        bootstrap,
        union_covered=union_covered,
    )
    payload = {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "scientific_gate_failed",
        "target_measurement_only": True,
        "no_intermediate_endpoint_written": True,
        "coverage_seed": args.seed,
        "producer_bootstrap_seed": PRODUCER_BOOTSTRAP_SEED,
        "audit_bootstrap_seed": args.audit_seed,
        "belief_generation_num_calls": coverage_config.belief_generation_num_calls,
        "coverage_summary": coverage_summary,
        "ranking_summary": summary,
        "states_with_truth_in_branch_union": union_covered,
        "paired_coverage_gain_bootstrap": bootstrap,
        "gates": gates,
        "records": ranked,
        "usage": {"coverage": coverage_usage, "ranker": ranker_usage},
    }
    (args.output_dir / "HOLDOUT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ranking_summary": summary,
                "states_with_truth_in_branch_union": union_covered,
                "paired_coverage_gain_bootstrap": bootstrap,
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
