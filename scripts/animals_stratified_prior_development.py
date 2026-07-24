#!/usr/bin/env python3
"""Run the fixed-target Animals stratified-prior development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts.animals_belief_recall_ranker import summarize_rankings
from scripts.animals_branch_support_ranker import run_ranker
from scripts.animals_coverage_dynamics import run_probe
from scripts.animals_multisample_branch_holdout import union_covered_states


def recovered_after_initial_omission(records: list[dict[str, Any]]) -> int:
    return sum(
        not record["truth_covered_before_counterfactuals"]
        and any(
            candidate["truth_covered_if_yes"]
            or candidate["truth_covered_if_no"]
            for candidate in record["candidate_dynamics"]
        )
        for record in records
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ranker-config", type=Path, required=True)
    parser.add_argument("--target-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=24286)
    args = parser.parse_args()

    config = load_config(str(args.config))
    ranker_config = load_config(str(args.ranker_config))
    pool = json.loads(args.target_pool.read_text())
    targets = list(pool["development_targets"])
    config.animals = [targets]
    config.run_id = f"animals-stratified-development-coverage-{args.seed}"
    ranker_config.run_id = f"animals-stratified-development-ranker-{args.seed}"
    records, coverage_summary, coverage_usage = run_probe(
        config,
        num_states=20,
        candidate_width=3,
        max_attempts=20,
        seed=args.seed,
    )
    ranked, ranker_usage = run_ranker(records, ranker_config)
    summary = summarize_rankings(ranked)
    union_covered = union_covered_states(ranked)
    recovered = recovered_after_initial_omission(ranked)
    ranker_rho = summary[
        "spearman_belief_recall_score_vs_expected_truth_coverage"
    ]
    eig_rho = summary[
        "spearman_immediate_eig_vs_expected_truth_coverage"
    ]
    wins, _ties, losses = summary["ranker_immediate_wins_ties_losses"]
    gates = {
        "twenty_states_completed": summary["num_states"] == 20,
        "at_least_twelve_union_covered": union_covered >= 12,
        "at_least_six_recovered_after_initial_omission": recovered >= 6,
        "ranker_spearman_positive_and_above_eig": (
            ranker_rho is not None
            and eig_rho is not None
            and ranker_rho > 0.0
            and ranker_rho > eig_rho
        ),
        "ranker_selected_coverage_above_eig": (
            summary["mean_paired_selected_coverage_gain"] > 0.0
        ),
        "ranker_wins_exceed_losses": wins > losses,
    }
    gates["all_pass"] = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "development_gate_failed",
        "target_pool_path": str(args.target_pool),
        "target_pool_split": "development_targets",
        "coverage_seed": args.seed,
        "coverage_summary": coverage_summary,
        "ranking_summary": summary,
        "states_with_truth_in_branch_union": union_covered,
        "states_recovered_after_initial_omission": recovered,
        "gates": gates,
        "records": ranked,
        "usage": {"coverage": coverage_usage, "ranker": ranker_usage},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "DEVELOPMENT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "ranking_summary",
                    "states_with_truth_in_branch_union",
                    "states_recovered_after_initial_omission",
                    "gates",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
