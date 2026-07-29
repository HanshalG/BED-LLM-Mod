#!/usr/bin/env python3
"""Pool two independent Qwen canonical cohorts retrospectively."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_crossfit_depth_three_pooled_audit import audit_pooled


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-external-canonical-pooled64-1"
TREE_COUNT_PER_STUDY = 32
TOTAL_TREE_COUNT = 64
TARGET_COUNT = 33
STUDIES = (
    {
        "name": "confirmation_v1",
        "path": (
            REPO_ROOT
            / "results/nonmyopic/number_game_qwen_external_canonical_confirmation"
            / "number-game-qwen-external-canonical-confirmation-20260729"
            / "RESULT.json"
        ),
        "sha256": (
            "370e1c2923e56fb6a8344558db0a69bd5f86a8b013da7d9452675df380f7f12b"
        ),
    },
    {
        "name": "replication_v2",
        "path": (
            REPO_ROOT
            / "results/nonmyopic"
            / "number_game_qwen_external_canonical_replication_v2"
            / "number-game-qwen-external-canonical-replication-v2-20260729T063429Z"
            / "RESULT.json"
        ),
        "sha256": (
            "a03c5a6f6e01af403ce27ef7984e29176bf0aa5f40caeae2bb6d213ce8c5dc83"
        ),
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(study: dict[str, Any]) -> dict[str, Any]:
    path = Path(study["path"])
    if sha256_file(path) != study["sha256"]:
        raise ValueError(f"{study['name']} result hash changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] != "gated_null":
        raise ValueError("source composite status changed")
    if result["protocol"]["tree_count"] != TREE_COUNT_PER_STUDY:
        raise ValueError("source must contain exactly 32 trees")
    if result["protocol"]["target_count"] != TARGET_COUNT:
        raise ValueError("source target count changed")
    if result["protocol"]["planning_model"] != "qwen/qwen3.7-plus":
        raise ValueError("source planning model changed")
    if result["usage"]["adapter_requests"] != 1856:
        raise ValueError("source accepted-call count changed")
    if len(result["trees"]) != TREE_COUNT_PER_STUDY:
        raise ValueError("source scored-tree count changed")
    seeds = [int(tree["tree_seed"]) for tree in result["trees"]]
    if len(set(seeds)) != TREE_COUNT_PER_STUDY:
        raise ValueError("source tree seeds are not unique")
    return result


def robustness_checks(
    *,
    sources: list[dict[str, Any]],
    pooled: dict[str, Any],
) -> dict[str, bool]:
    source_myopic = [
        source["aggregate"]["comparisons"]["myopic_eig"]
        for source in sources
    ]
    pooled_myopic = pooled["comparisons"]["myopic_eig"]
    return {
        "exactly_64_hash_bound_disjoint_trees": (
            sum(len(source["trees"]) for source in sources)
            == TOTAL_TREE_COUNT
        ),
        "both_cohorts_myopic_gain_at_least_eight_percent": all(
            comparison["relative_brier_reduction"] >= 0.08
            for comparison in source_myopic
        ),
        "both_cohorts_myopic_ci_below_zero": all(
            comparison[
                "tree_cluster_brier_difference_95pct_bootstrap"
            ][1]
            < 0.0
            for comparison in source_myopic
        ),
        "both_cohorts_myopic_wins_at_least_twenty": all(
            comparison["brier_tree_wins"] >= 20
            for comparison in source_myopic
        ),
        "pooled_myopic_gain_at_least_ten_percent": (
            pooled_myopic["relative_brier_reduction"] >= 0.10
        ),
        "pooled_myopic_ci_below_zero": (
            pooled_myopic[
                "stratified_tree_bootstrap_brier_difference_95pct"
            ][1]
            < 0.0
        ),
        "pooled_myopic_wins_at_least_forty": (
            pooled_myopic["wins"] >= 40
        ),
    }


def run_synthesis(output_dir: Path) -> dict[str, Any]:
    sources = [validate_source(study) for study in STUDIES]
    seed_sets = [
        {int(tree["tree_seed"]) for tree in source["trees"]}
        for source in sources
    ]
    if seed_sets[0] & seed_sets[1]:
        raise ValueError("source cohorts share tree seeds")
    scored = [
        {
            "aggregate": source["aggregate"],
            "trees": source["trees"],
        }
        for source in sources
    ]
    pooled = audit_pooled(scored)
    checks = robustness_checks(sources=sources, pooled=pooled)
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "retrospective_robustness_positive"
            if all(checks.values())
            else "retrospective_robustness_mixed"
        ),
        "protocol": {
            "analysis_is_retrospective": True,
            "cannot_rescue_source_composite_statuses": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "study_count": len(STUDIES),
            "tree_count": TOTAL_TREE_COUNT,
            "target_count": TARGET_COUNT,
            "bootstrap": (
                "resample 32 trees independently within each source cohort"
            ),
            "source_hashes": {
                study["name"]: study["sha256"] for study in STUDIES
            },
        },
        "sources": [
            {
                "name": study["name"],
                "status": source["status"],
                "usage": source["usage"],
                "myopic_comparison": source["aggregate"]["comparisons"][
                    "myopic_eig"
                ],
                "depth_two_comparison": source["aggregate"]["comparisons"][
                    "crossfit_depth_two"
                ],
            }
            for study, source in zip(STUDIES, sources, strict=True)
        ],
        "pooled": pooled,
        "robustness_checks": checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = run_synthesis(args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "robustness_checks": result["robustness_checks"],
                "myopic": result["pooled"]["comparisons"]["myopic_eig"],
                "depth_two": result["pooled"]["comparisons"][
                    "crossfit_depth_two"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
