#!/usr/bin/env python3
"""Independently replay the support-capacity-gated Animals holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.animals_belief_recall_capacity_holdout import capacity_gated_records
from scripts.animals_belief_recall_holdout import (
    bootstrap_mean_gain,
    evaluate_gates,
    paired_coverage_gains,
)
from scripts.animals_belief_recall_ranker import (
    parse_scores,
    prompt_payload,
    summarize_rankings,
)


def _equal(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _equal(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equal(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload["records"]
    support_capacity = int(payload["support_capacity"])
    payload_checks = []
    response_checks = []
    for record in records:
        visible = record["model_visible_payload"]
        payload_checks.extend(
            [
                visible == prompt_payload(record),
                "target_measurement_only" not in json.dumps(visible),
                "expected_truth_coverage" not in json.dumps(visible),
            ]
        )
        response_checks.append(
            parse_scores(
                record["raw_ranker_response"],
                len(record["candidate_dynamics"]),
            )
            == record["belief_recall_scores"]
        )

    gated_records = capacity_gated_records(
        records,
        support_capacity=support_capacity,
    )
    summary = summarize_rankings(gated_records)
    bootstrap = bootstrap_mean_gain(
        paired_coverage_gains(gated_records),
        seed=int(payload["audit_bootstrap_seed"]),
    )
    producer_gates = evaluate_gates(
        summary,
        payload["paired_coverage_gain_bootstrap"],
    )
    producer_gates["support_capacity_is_exactly_max_num_samples"] = (
        support_capacity == 16
    )
    producer_gates["all_pass"] = all(
        value for key, value in producer_gates.items() if key != "all_pass"
    )
    gates = evaluate_gates(summary, bootstrap)
    gates["support_capacity_is_exactly_max_num_samples"] = support_capacity == 16
    gates["all_pass"] = all(
        value for key, value in gates.items() if key != "all_pass"
    )
    checks = {
        "all_model_visible_payloads_rebuilt_exactly": all(payload_checks),
        "all_raw_responses_reparse_exactly": all(response_checks),
        "capacity_gated_summary_replays": _equal(
            summary,
            payload["capacity_gated_summary"],
        ),
        "producer_gate_status_replays": producer_gates == payload["gates"],
        "audit_bootstrap_gates_pass": gates["all_pass"],
        "sixty_distinct_measurement_targets": (
            len(records) == 60
            and len(
                {
                    str(record["target_measurement_only"]).casefold()
                    for record in records
                }
            )
            == 60
        ),
    }
    checks["all_pass"] = all(checks.values())
    return {
        "schema_version": 1,
        "status": "passed" if checks["all_pass"] else "failed",
        "checks": checks,
        "audit_capacity_gated_summary": summary,
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
