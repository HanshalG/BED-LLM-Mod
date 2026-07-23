#!/usr/bin/env python3
"""Independently replay the Animals multi-sample branch-ranker holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.animals_belief_recall_holdout import (
    bootstrap_mean_gain,
    paired_coverage_gains,
)
from scripts.animals_belief_recall_ranker import parse_scores, summarize_rankings
from scripts.animals_branch_support_ranker import prompt_payload
from scripts.animals_multisample_branch_holdout import (
    evaluate_gates,
    union_covered_states,
)


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload["records"]
    payload_checks = []
    response_checks = []
    for record in records:
        visible = record["model_visible_payload"]
        encoded = json.dumps(visible)
        payload_checks.extend(
            [
                visible == prompt_payload(record),
                "target_measurement_only" not in encoded,
                "expected_truth_coverage" not in encoded,
                "truth_covered" not in encoded,
                "immediate_eig" not in encoded,
            ]
        )
        response_checks.append(
            parse_scores(
                record["raw_ranker_response"],
                len(record["candidate_dynamics"]),
            )
            == record["belief_recall_scores"]
        )
    summary = summarize_rankings(records)
    union_covered = union_covered_states(records)
    producer_gates = evaluate_gates(
        summary,
        payload["paired_coverage_gain_bootstrap"],
        union_covered=union_covered,
    )
    bootstrap = bootstrap_mean_gain(
        paired_coverage_gains(records),
        seed=int(payload["audit_bootstrap_seed"]),
    )
    gates = evaluate_gates(
        summary,
        bootstrap,
        union_covered=union_covered,
    )
    checks = {
        "all_model_visible_payloads_rebuilt_exactly": all(payload_checks),
        "all_raw_responses_reparse_exactly": all(response_checks),
        "ranking_summary_replays": summary == payload["ranking_summary"],
        "union_coverage_replays": (
            union_covered == payload["states_with_truth_in_branch_union"]
        ),
        "producer_gates_replay": producer_gates == payload["gates"],
        "four_generation_calls_registered": (
            payload["belief_generation_num_calls"] == 4
        ),
        "sixty_distinct_targets": (
            len(records) == 60
            and len(
                {
                    str(record["target_measurement_only"]).casefold()
                    for record in records
                }
            )
            == 60
        ),
        "audit_scientific_gates_pass": gates["all_pass"],
    }
    integrity_keys = [
        key for key in checks if key != "audit_scientific_gates_pass"
    ]
    checks["integrity_pass"] = all(checks[key] for key in integrity_keys)
    checks["all_pass"] = all(checks.values())
    return {
        "schema_version": 1,
        "status": "passed" if checks["all_pass"] else "failed",
        "checks": checks,
        "audit_ranking_summary": summary,
        "audit_union_covered_states": union_covered,
        "audit_paired_coverage_gain_bootstrap": bootstrap,
        "audit_gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.input.read_text()))
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["checks"]["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
