#!/usr/bin/env python3
"""Post-hoc audit of Animals branch-transition selector scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SELECTORS = {
    "support_expansion": "expected_support_size",
    "support_retention": "expected_current_support_retention",
    "immediate_eig": "immediate_eig",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_index(
    candidates: Sequence[Mapping[str, Any]],
    score_key: str,
) -> int:
    if not candidates:
        raise ValueError("cannot select from an empty candidate list")
    return max(
        range(len(candidates)),
        key=lambda index: (float(candidates[index][score_key]), -index),
    )


def within_state_pairwise_accuracy(
    records: Sequence[Mapping[str, Any]],
    score_key: str,
    endpoint_key: str,
) -> dict[str, float | int]:
    points = 0.0
    pairs = 0
    informative_states = 0
    for record in records:
        candidates = record["candidate_dynamics"]
        state_pairs = 0
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                endpoint_delta = (
                    float(candidates[left][endpoint_key])
                    - float(candidates[right][endpoint_key])
                )
                if endpoint_delta == 0.0:
                    continue
                score_delta = (
                    float(candidates[left][score_key])
                    - float(candidates[right][score_key])
                )
                pairs += 1
                state_pairs += 1
                if score_delta == 0.0:
                    points += 0.5
                elif (score_delta > 0.0) == (endpoint_delta > 0.0):
                    points += 1.0
        informative_states += state_pairs > 0
    return {
        "accuracy": points / pairs if pairs else 0.5,
        "pairs": pairs,
        "informative_states": informative_states,
    }


def _realized_coverage(candidate: Mapping[str, Any]) -> float:
    return float(candidate["realized_endpoint"]["truth_covered"])


def _realized_truth_probability(candidate: Mapping[str, Any]) -> float:
    return float(candidate["realized_endpoint"]["uniform_truth_probability"])


def summarize_aligned_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eig_indices = [
        select_index(record["candidate_dynamics"], "immediate_eig")
        for record in records
    ]
    selectors: dict[str, Any] = {}
    for name, score_key in SELECTORS.items():
        indices = [
            select_index(record["candidate_dynamics"], score_key)
            for record in records
        ]
        coverages = [
            _realized_coverage(record["candidate_dynamics"][index])
            for record, index in zip(records, indices, strict=True)
        ]
        truth_probabilities = [
            _realized_truth_probability(
                record["candidate_dynamics"][index]
            )
            for record, index in zip(records, indices, strict=True)
        ]
        expected_coverages = [
            float(
                record["candidate_dynamics"][index][
                    "expected_truth_coverage"
                ]
            )
            for record, index in zip(records, indices, strict=True)
        ]
        eig_coverages = [
            _realized_coverage(record["candidate_dynamics"][index])
            for record, index in zip(records, eig_indices, strict=True)
        ]
        differences = [
            selected - baseline
            for selected, baseline in zip(
                coverages,
                eig_coverages,
                strict=True,
            )
        ]
        recoveries = sum(
            not bool(record["truth_covered_before_counterfactuals"])
            and bool(coverage)
            for record, coverage in zip(records, coverages, strict=True)
        )
        selectors[name] = {
            "mean_realized_truth_coverage": float(np.mean(coverages)),
            "mean_expected_truth_coverage": float(
                np.mean(expected_coverages)
            ),
            "mean_uniform_truth_probability": float(
                np.mean(truth_probabilities)
            ),
            "selector_changes_vs_eig": sum(
                left != right
                for left, right in zip(indices, eig_indices, strict=True)
            ),
            "realized_coverage_vs_eig_wins_ties_losses": [
                sum(value > 0.0 for value in differences),
                sum(value == 0.0 for value in differences),
                sum(value < 0.0 for value in differences),
            ],
            "recoveries_after_initial_omission": recoveries,
        }

    augmented_records = []
    for record in records:
        candidates = []
        for candidate in record["candidate_dynamics"]:
            candidates.append(
                {
                    **candidate,
                    "realized_truth_coverage": _realized_coverage(
                        candidate
                    ),
                }
            )
        augmented_records.append(
            {**record, "candidate_dynamics": candidates}
        )
    return {
        "selectors": selectors,
        "within_state_realized_coverage_pairwise": {
            name: within_state_pairwise_accuracy(
                augmented_records,
                score_key,
                "realized_truth_coverage",
            )
            for name, score_key in SELECTORS.items()
        },
    }


def summarize_expected_coverage_artifact(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    usable = [
        record
        for record in records
        if record.get("candidate_dynamics")
        and all(
            "expected_current_support_retention" in candidate
            and "immediate_eig" in candidate
            and "expected_truth_coverage" in candidate
            for candidate in record["candidate_dynamics"]
        )
    ]
    if not usable:
        return None
    retention_values = []
    eig_values = []
    changes = 0
    for record in usable:
        candidates = record["candidate_dynamics"]
        retention_index = select_index(
            candidates,
            "expected_current_support_retention",
        )
        eig_index = select_index(candidates, "immediate_eig")
        retention_values.append(
            float(candidates[retention_index]["expected_truth_coverage"])
        )
        eig_values.append(
            float(candidates[eig_index]["expected_truth_coverage"])
        )
        changes += retention_index != eig_index
    differences = [
        retention - eig
        for retention, eig in zip(
            retention_values,
            eig_values,
            strict=True,
        )
    ]
    return {
        "states": len(usable),
        "retention_mean_expected_truth_coverage": float(
            np.mean(retention_values)
        ),
        "eig_mean_expected_truth_coverage": float(np.mean(eig_values)),
        "retention_minus_eig": float(np.mean(differences)),
        "wins_ties_losses": [
            sum(value > 0.0 for value in differences),
            sum(value == 0.0 for value in differences),
            sum(value < 0.0 for value in differences),
        ],
        "selector_changes": changes,
    }


def _records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("artifact has no records list")
    return records


def analyze(
    development_path: Path,
    comparison_paths: Sequence[Path],
) -> dict[str, Any]:
    development = json.loads(development_path.read_text(encoding="utf-8"))
    comparisons = []
    for path in comparison_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = summarize_expected_coverage_artifact(_records(payload))
        if summary is not None:
            comparisons.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    **summary,
                }
            )
    return {
        "status": "posthoc_selector_audit",
        "disclosure": (
            "Support retention was selected after the aligned development "
            "endpoint was opened; these statistics are not confirmatory."
        ),
        "development_path": str(development_path),
        "development_sha256": sha256_file(development_path),
        "aligned_development": summarize_aligned_records(
            _records(development)
        ),
        "cached_expected_coverage_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument(
        "--comparison",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.development, args.comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
