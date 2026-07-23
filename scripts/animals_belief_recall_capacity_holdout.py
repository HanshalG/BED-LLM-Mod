#!/usr/bin/env python3
"""Run the fresh Animals support-capacity-gated belief-recall holdout."""

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
    evaluate_gates,
    paired_coverage_gains,
)
from scripts.animals_belief_recall_ranker import run_ranker, summarize_rankings
from scripts.animals_coverage_dynamics import run_probe


PRODUCER_BOOTSTRAP_SEED = 24276


def capacity_gated_records(
    records: list[dict[str, Any]],
    *,
    support_capacity: int,
) -> list[dict[str, Any]]:
    """Represent EIG fallback by replacing inactive ranker scores with EIG."""
    if support_capacity <= 0:
        raise ValueError("support capacity must be positive")
    gated = []
    for record in records:
        active = int(record["belief_support_size"]) <= support_capacity
        scores = (
            [float(value) for value in record["belief_recall_scores"]]
            if active
            else [
                float(entry["immediate_eig"])
                for entry in record["candidate_dynamics"]
            ]
        )
        gated.append(
            {
                **record,
                "capacity_gate_active": active,
                "capacity_gate_threshold": support_capacity,
                "capacity_gated_scores": scores,
                "belief_recall_scores": scores,
            }
        )
    return gated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-config",
        type=Path,
        default=Path(
            "configs/config_animals_belief_recall_capacity_holdout_openrouter.yaml"
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
    parser.add_argument("--seed", type=int, default=24275)
    parser.add_argument("--audit-seed", type=int, default=24277)
    args = parser.parse_args()

    coverage_config = load_config(str(args.coverage_config))
    ranker_config = load_config(str(args.ranker_config))
    coverage_config.run_id = (
        f"animals-belief-recall-capacity-holdout-coverage-{args.seed}"
    )
    ranker_config.run_id = (
        f"animals-belief-recall-capacity-holdout-ranker-{args.seed}"
    )
    support_capacity = int(coverage_config.max_num_samples)
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
        ungated_summary = summarize_rankings(ranked_records)
        gated_records = capacity_gated_records(
            ranked_records,
            support_capacity=support_capacity,
        )
        gated_summary = summarize_rankings(gated_records)
        bootstrap = bootstrap_mean_gain(
            paired_coverage_gains(gated_records),
            seed=PRODUCER_BOOTSTRAP_SEED,
        )
        gates = evaluate_gates(gated_summary, bootstrap)
        gates["support_capacity_is_exactly_max_num_samples"] = (
            support_capacity == coverage_config.max_num_samples == 16
        )
        gates["all_pass"] = all(
            value for key, value in gates.items() if key != "all_pass"
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "error": f"{type(exc).__name__}: {exc}",
            "coverage_seed": args.seed,
            "producer_bootstrap_seed": PRODUCER_BOOTSTRAP_SEED,
            "audit_bootstrap_seed": args.audit_seed,
            "coverage_usage": getattr(exc, "usage", None),
            "partial_records": getattr(exc, "records", None),
        }
        (args.output_dir / "CAPACITY_HOLDOUT_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    payload = {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "scientific_gate_failed",
        "target_measurement_only": True,
        "no_intermediate_endpoint_written": True,
        "selector": "belief_recall_if_support_at_or_below_generation_capacity",
        "support_capacity": support_capacity,
        "coverage_config_path": str(args.coverage_config),
        "ranker_config_path": str(args.ranker_config),
        "coverage_seed": args.seed,
        "producer_bootstrap_seed": PRODUCER_BOOTSTRAP_SEED,
        "audit_bootstrap_seed": args.audit_seed,
        "coverage_summary": coverage_summary,
        "ungated_ranking_summary": ungated_summary,
        "capacity_gated_summary": gated_summary,
        "paired_coverage_gain_bootstrap": bootstrap,
        "gates": gates,
        "records": ranked_records,
        "usage": {
            "coverage": coverage_usage,
            "ranker": ranker_usage,
        },
    }
    (args.output_dir / "CAPACITY_HOLDOUT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ungated_ranking_summary": ungated_summary,
                "capacity_gated_summary": gated_summary,
                "paired_coverage_gain_bootstrap": bootstrap,
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
