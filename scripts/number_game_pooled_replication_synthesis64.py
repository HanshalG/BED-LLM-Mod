#!/usr/bin/env python3
"""Synthesize two independent pooled-Qwen Number Game cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_crossfit_depth_three_confirmation import _mean
from scripts.number_game_qwen_first_link_mechanism64 import _interval


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-pooled-replication-synthesis64-1"
TREE_COUNT_PER_COHORT = 32
TOTAL_TREE_COUNT = 64
BOOTSTRAP_SEED = 64_600
BOOTSTRAP_SAMPLES = 20_000

FIRST_POLICY_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_pooled_first_link_confirmation32"
    / "number-game-qwen-pooled-first-link-confirmation32-20260729T094455Z"
    / "RESULT.json"
)
FIRST_POLICY_SHA256 = (
    "cef6ded08050e6df07de62bbbf548635f229a8bf73fac149b4d0bd960f7b0253"
)
FIRST_ABLATION_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_pooled_second_refresh_ablation"
    / "number-game-pooled-second-refresh-ablation-20260729T104014Z"
    / "RESULT.json"
)
FIRST_ABLATION_SHA256 = (
    "77fa26cc9d4599804521081bcb201c6d9cf1dfdc0d56d5f51487314d8e70f95c"
)
SECOND_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_pooled_second_refresh_confirmation32"
    / "number-game-qwen-pooled-second-refresh-confirmation32-20260729T104705Z"
    / "RESULT.json"
)
SECOND_SHA256 = (
    "71281c483297e7ee4cc011d0ba795dabd28ec2d8598366771e11afee2af2a459"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_hash_bound(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"source hash changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    first_policy = _load_hash_bound(FIRST_POLICY_RESULT, FIRST_POLICY_SHA256)
    first_ablation = _load_hash_bound(
        FIRST_ABLATION_RESULT,
        FIRST_ABLATION_SHA256,
    )
    second = _load_hash_bound(SECOND_RESULT, SECOND_SHA256)

    if first_policy["status"] != "gated_null":
        raise ValueError("first policy source status changed")
    if first_ablation["status"] != "retrospective_second_refresh_positive":
        raise ValueError("first ablation source status changed")
    if second["status"] != "gated_null":
        raise ValueError("second source status changed")
    for source in (first_policy, second):
        if source["protocol"]["tree_count"] != TREE_COUNT_PER_COHORT:
            raise ValueError("source tree count changed")
        if len(source["trees"]) != TREE_COUNT_PER_COHORT:
            raise ValueError("source scored-tree count changed")
        if source["protocol"]["planning_model"] != "qwen/qwen3.7-plus":
            raise ValueError("source planning model changed")
        if source["protocol"]["target_count"] != 33:
            raise ValueError("source target count changed")
    if len(first_ablation["rows"]) != TREE_COUNT_PER_COHORT:
        raise ValueError("first ablation row count changed")
    return first_policy, first_ablation, second


def _policy_values(tree: dict[str, Any]) -> dict[str, float]:
    candidate = tree["endpoint"]["crossfit_depth_three"]
    baseline = tree["endpoint"]["myopic_eig"]
    return {
        "candidate_brier": float(candidate["mean_posterior_predictive_brier"]),
        "baseline_brier": float(baseline["mean_posterior_predictive_brier"]),
        "candidate_hamming": float(candidate["mean_best_hamming_error"]),
        "baseline_hamming": float(baseline["mean_best_hamming_error"]),
        "candidate_coverage": float(candidate["truth_extension_coverage_rate"]),
        "baseline_coverage": float(baseline["truth_extension_coverage_rate"]),
    }


def cohort_rows(
    first_policy: dict[str, Any],
    first_ablation: dict[str, Any],
    second: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    first_ablation_by_seed = {
        int(row["tree_seed"]): row for row in first_ablation["rows"]
    }
    first_rows = []
    for tree in first_policy["trees"]:
        seed = int(tree["tree_seed"])
        ablation = first_ablation_by_seed[seed]
        policy = _policy_values(tree)
        if abs(
            policy["candidate_brier"]
            - float(ablation["endpoint_brier"]["merged_retained_generated"])
        ) > 1e-12:
            raise ValueError(f"first cohort endpoint mismatch for tree {seed}")
        first_rows.append(
            {
                "tree_seed": seed,
                "policy": policy,
                "support": {
                    "candidate_brier": float(
                        ablation["endpoint_brier"][
                            "merged_retained_generated"
                        ]
                    ),
                    "parent_only_brier": float(
                        ablation["endpoint_brier"]["parent_only"]
                    ),
                    "generated_only_brier": float(
                        ablation["endpoint_brier"]["generated_only"]
                    ),
                    "candidate_root": int(
                        ablation["selected_roots"][
                            "merged_retained_generated"
                        ]
                    ),
                    "parent_only_root": int(
                        ablation["selected_roots"]["parent_only"]
                    ),
                    "generated_only_root": int(
                        ablation["selected_roots"]["generated_only"]
                    ),
                },
            }
        )

    second_ablation_by_seed = {
        int(row["tree_seed"]): row
        for row in second["second_refresh"]["rows"]
    }
    second_rows = []
    for tree in second["trees"]:
        seed = int(tree["tree_seed"])
        ablation = second_ablation_by_seed[seed]
        policy = _policy_values(tree)
        if abs(
            policy["candidate_brier"]
            - float(ablation["endpoint_brier"]["merged_retained_generated"])
        ) > 1e-12:
            raise ValueError(
                f"second cohort endpoint mismatch for tree {seed}"
            )
        second_rows.append(
            {
                "tree_seed": seed,
                "policy": policy,
                "support": {
                    "candidate_brier": float(
                        ablation["endpoint_brier"][
                            "merged_retained_generated"
                        ]
                    ),
                    "parent_only_brier": float(
                        ablation["endpoint_brier"]["parent_only"]
                    ),
                    "generated_only_brier": float(
                        ablation["endpoint_brier"]["generated_only"]
                    ),
                    "candidate_root": int(
                        ablation["selected_roots"][
                            "merged_retained_generated"
                        ]
                    ),
                    "parent_only_root": int(
                        ablation["selected_roots"]["parent_only"]
                    ),
                    "generated_only_root": int(
                        ablation["selected_roots"]["generated_only"]
                    ),
                },
            }
        )

    first_seeds = {row["tree_seed"] for row in first_rows}
    second_seeds = {row["tree_seed"] for row in second_rows}
    if len(first_seeds) != 32 or len(second_seeds) != 32:
        raise ValueError("source tree seeds are not unique")
    if first_seeds & second_seeds:
        raise ValueError("source cohorts share tree seeds")
    return [first_rows, second_rows]


def stratified_bootstrap_indices(
    *,
    cohort_sizes: Sequence[int],
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> list[list[list[int]]]:
    rng = random.Random(seed)
    return [
        [
            [rng.randrange(size) for _ in range(size)]
            for size in cohort_sizes
        ]
        for _ in range(samples)
    ]


def _wtl(differences: Sequence[float]) -> tuple[int, int, int]:
    wins = sum(value < -1e-15 for value in differences)
    losses = sum(value > 1e-15 for value in differences)
    return wins, len(differences) - wins - losses, losses


def summarize_difference(
    cohorts: Sequence[Sequence[dict[str, Any]]],
    *,
    candidate_path: tuple[str, str],
    baseline_path: tuple[str, str],
    bootstrap_indices: Sequence[Sequence[Sequence[int]]],
) -> dict[str, Any]:
    candidate_by_cohort = [
        [float(row[candidate_path[0]][candidate_path[1]]) for row in cohort]
        for cohort in cohorts
    ]
    baseline_by_cohort = [
        [float(row[baseline_path[0]][baseline_path[1]]) for row in cohort]
        for cohort in cohorts
    ]
    differences_by_cohort = [
        [
            candidate - baseline
            for candidate, baseline in zip(
                candidate_values,
                baseline_values,
                strict=True,
            )
        ]
        for candidate_values, baseline_values in zip(
            candidate_by_cohort,
            baseline_by_cohort,
            strict=True,
        )
    ]
    candidate = [value for cohort in candidate_by_cohort for value in cohort]
    baseline = [value for cohort in baseline_by_cohort for value in cohort]
    differences = [
        value for cohort in differences_by_cohort for value in cohort
    ]
    bootstrap_means = []
    bootstrap_contrasts = []
    for replicate in bootstrap_indices:
        replicate_differences = [
            [
                differences_by_cohort[cohort_index][index]
                for index in indices
            ]
            for cohort_index, indices in enumerate(replicate)
        ]
        bootstrap_means.append(
            _mean(
                [
                    value
                    for cohort in replicate_differences
                    for value in cohort
                ]
            )
        )
        bootstrap_contrasts.append(
            _mean(replicate_differences[0])
            - _mean(replicate_differences[1])
        )
    wins, ties, losses = _wtl(differences)
    candidate_mean = _mean(candidate)
    baseline_mean = _mean(baseline)
    source_mean_differences = [
        _mean(cohort) for cohort in differences_by_cohort
    ]
    return {
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "mean_candidate_minus_baseline": _mean(differences),
        "relative_reduction": (
            (baseline_mean - candidate_mean) / baseline_mean
        ),
        "stratified_bootstrap_difference_95pct": _interval(bootstrap_means),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "source_mean_differences": source_mean_differences,
        "cohort_one_minus_two_mean_difference": (
            source_mean_differences[0] - source_mean_differences[1]
        ),
        "cohort_one_minus_two_95pct_bootstrap": _interval(
            bootstrap_contrasts
        ),
    }


def summarize_root_differences(
    cohorts: Sequence[Sequence[dict[str, Any]]],
    baseline: str,
) -> dict[str, Any]:
    per_cohort = [
        sum(
            row["support"]["candidate_root"]
            != row["support"][f"{baseline}_root"]
            for row in cohort
        )
        for cohort in cohorts
    ]
    return {
        "pooled": sum(per_cohort),
        "per_cohort": per_cohort,
    }


def run_synthesis(output_dir: Path) -> dict[str, Any]:
    first_policy, first_ablation, second = load_sources()
    cohorts = cohort_rows(first_policy, first_ablation, second)
    bootstrap = stratified_bootstrap_indices(
        cohort_sizes=[len(cohort) for cohort in cohorts]
    )
    brier = summarize_difference(
        cohorts,
        candidate_path=("policy", "candidate_brier"),
        baseline_path=("policy", "baseline_brier"),
        bootstrap_indices=bootstrap,
    )
    hamming = summarize_difference(
        cohorts,
        candidate_path=("policy", "candidate_hamming"),
        baseline_path=("policy", "baseline_hamming"),
        bootstrap_indices=bootstrap,
    )
    coverage = summarize_difference(
        cohorts,
        candidate_path=("policy", "candidate_coverage"),
        baseline_path=("policy", "baseline_coverage"),
        bootstrap_indices=bootstrap,
    )
    parent = summarize_difference(
        cohorts,
        candidate_path=("support", "candidate_brier"),
        baseline_path=("support", "parent_only_brier"),
        bootstrap_indices=bootstrap,
    )
    generated = summarize_difference(
        cohorts,
        candidate_path=("support", "candidate_brier"),
        baseline_path=("support", "generated_only_brier"),
        bootstrap_indices=bootstrap,
    )
    policy_source_gates = [
        first_policy["efficacy_gates"],
        second["myopic_policy_gates"],
    ]
    checks = {
        "exactly_two_disjoint_32_tree_cohorts": (
            len(cohorts) == 2
            and all(len(cohort) == TREE_COUNT_PER_COHORT for cohort in cohorts)
        ),
        "both_source_policy_efficacy_gate_sets_pass": all(
            all(gates.values()) for gates in policy_source_gates
        ),
        "pooled_policy_brier_interval_below_zero": (
            brier["stratified_bootstrap_difference_95pct"][1] < 0.0
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "retrospective_policy_robustness_positive"
            if all(checks.values())
            else "retrospective_policy_robustness_mixed"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "analysis_is_retrospective": True,
            "prospective_mechanism_status_remains_binding": True,
            "cannot_rescue_or_reclassify_sources": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "cohort_count": 2,
            "tree_count": TOTAL_TREE_COUNT,
            "trees_per_cohort": TREE_COUNT_PER_COHORT,
            "target_count": 33,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap": "resample 32 trees independently within each cohort",
            "source_hashes": {
                "first_policy": FIRST_POLICY_SHA256,
                "first_second_refresh_ablation": FIRST_ABLATION_SHA256,
                "prospective_confirmation": SECOND_SHA256,
            },
        },
        "sources": [
            {
                "name": "retrospective_ablation_cohort",
                "policy_status": first_policy["status"],
                "mechanism_status": first_ablation["status"],
                "policy_myopic": first_policy["aggregate"]["comparisons"][
                    "myopic_eig"
                ],
                "second_refresh_parent_only": first_ablation["comparisons"][
                    "parent_only"
                ],
                "second_refresh_generated_only": first_ablation[
                    "comparisons"
                ]["generated_only"],
            },
            {
                "name": "prospective_confirmation_cohort",
                "policy_status": second["status"],
                "mechanism_status": second["status"],
                "policy_myopic": second["aggregate"]["comparisons"][
                    "myopic_eig"
                ],
                "second_refresh_parent_only": second["second_refresh"][
                    "comparisons"
                ]["parent_only"],
                "second_refresh_generated_only": second["second_refresh"][
                    "comparisons"
                ]["generated_only"],
            },
        ],
        "pooled": {
            "policy_vs_myopic": {
                "brier": brier,
                "hamming": hamming,
                "coverage": coverage,
            },
            "second_refresh": {
                "parent_only": {
                    **parent,
                    "root_differences": summarize_root_differences(
                        cohorts,
                        "parent_only",
                    ),
                },
                "generated_only": {
                    **generated,
                    "root_differences": summarize_root_differences(
                        cohorts,
                        "generated_only",
                    ),
                },
            },
        },
        "robustness_checks": checks,
        "interpretation": {
            "policy_effect_replicates_across_both_fresh_cohorts": all(
                checks.values()
            ),
            "prospective_regeneration_increment_confirmed": False,
            "retention_contribution_is_directionally_replicated": (
                first_ablation["comparisons"]["generated_only"][
                    "mean_candidate_minus_baseline_brier"
                ]
                < 0.0
                and second["second_refresh"]["comparisons"]["generated_only"][
                    "mean_candidate_minus_baseline_brier"
                ]
                < 0.0
            ),
        },
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
                "pooled": result["pooled"],
                "interpretation": result["interpretation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
