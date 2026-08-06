#!/usr/bin/env python3
"""Pool two disjoint matched history-conditioning controls without model calls."""

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
from scripts.number_game_qwen_history_blind_matched32 import (
    spearman_correlation,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-history-conditioning-pooled64-1"
BOOTSTRAP_SEED = 10_700_000
BOOTSTRAP_SAMPLES = 20_000

DEVELOPMENT_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_history_blind_matched32_v3"
    / "number-game-qwen-history-blind-matched32-v3-20260730T173000Z"
    / "RESULT.json"
)
CONFIRMATION_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_history_blind_first_link_confirmation32"
    / "number-game-qwen-history-blind-first-link-confirmation32-20260730T044135Z"
    / "RESULT.json"
)
DEVELOPMENT_RESULT_SHA256 = (
    "29bb76074dfb53a6efef352fde88fbca6c3a7dc591d0936d132c82050f1dc71d"
)
CONFIRMATION_RESULT_SHA256 = (
    "c47a6aba6c1ac5d9028f234c4670d62418c4ce8d5e09162145c188e6a2d2b725"
)
EXPECTED_TREE_INDICES = (tuple(range(32)), tuple(range(32, 64)))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(values: Sequence[float]) -> list[float]:
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def _changed(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["selected_roots"]["roots_differ"]]


def _tree_metric(
    row: dict[str, Any],
    *,
    stage: str,
    arm: str,
    metric: str,
) -> float:
    return float(row["tree_mean"][stage][arm][metric])


def _difference(
    row: dict[str, Any],
    *,
    stage: str,
    metric: str,
) -> float:
    return float(row["tree_mean"][stage]["differences"][metric])


def _selected_values(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    contrasts = [
        float(
            row["selected_roots"][
                "dynamic_minus_fixed_root_prompt_conditioning_benefit"
            ]
        )
        for row in rows
    ]
    advantages = [
        float(row["selected_roots"]["realized_advantage"])
        for row in rows
    ]
    return contrasts, advantages


def load_and_validate_sources(
    paths: Sequence[Path] = (DEVELOPMENT_RESULT, CONFIRMATION_RESULT),
    hashes: Sequence[str] = (
        DEVELOPMENT_RESULT_SHA256,
        CONFIRMATION_RESULT_SHA256,
    ),
) -> list[list[dict[str, Any]]]:
    if len(paths) != 2 or len(hashes) != 2:
        raise ValueError("pooled audit requires exactly two source cohorts")
    payloads = []
    expected_statuses = ("gated_null", "passed")
    seen_indices: set[int] = set()
    for cohort_index, (path, expected_hash) in enumerate(zip(paths, hashes)):
        if sha256_file(path) != expected_hash:
            raise ValueError(f"source cohort {cohort_index} hash changed")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != expected_statuses[cohort_index]:
            raise ValueError(f"source cohort {cohort_index} status changed")
        rows = payload.get("trees") or []
        observed = tuple(int(row["tree_index"]) for row in rows)
        if observed != EXPECTED_TREE_INDICES[cohort_index]:
            raise ValueError(f"source cohort {cohort_index} tree block changed")
        if seen_indices.intersection(observed):
            raise ValueError("source cohorts overlap")
        seen_indices.update(observed)
        if payload.get("protocol", {}).get("analysis_was_preregistered") is not True:
            raise ValueError(f"source cohort {cohort_index} was not preregistered")
        payloads.append(rows)
    return payloads


def bootstrap_analysis(
    blocks: Sequence[Sequence[dict[str, Any]]],
    *,
    seed: int,
    samples: int,
) -> dict[str, list[float]]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    changed_blocks = [_changed(block) for block in blocks]
    if any(not block for block in changed_blocks):
        raise ValueError("every source cohort needs changed-root trees")
    rng = random.Random(seed)
    replicates = {
        "second_conditional_minus_history_blind_predictive_mse_95pct": [],
        "second_conditional_minus_history_blind_truth_coverage_95pct": [],
        "changed_root_prompt_benefit_contrast_95pct": [],
        "prompt_benefit_contrast_to_realized_spearman_95pct": [],
    }
    for _ in range(samples):
        sampled = [
            rng.choice(block)
            for block in blocks
            for _ in range(len(block))
        ]
        sampled_changed = [
            rng.choice(block)
            for block in changed_blocks
            for _ in range(len(block))
        ]
        replicates[
            "second_conditional_minus_history_blind_predictive_mse_95pct"
        ].append(
            _mean(
                [
                    _difference(
                        row,
                        stage="second",
                        metric=(
                            "conditional_minus_history_blind_predictive_mse"
                        ),
                    )
                    for row in sampled
                ]
            )
        )
        replicates[
            "second_conditional_minus_history_blind_truth_coverage_95pct"
        ].append(
            _mean(
                [
                    _difference(
                        row,
                        stage="second",
                        metric=(
                            "conditional_minus_history_blind_truth_coverage"
                        ),
                    )
                    for row in sampled
                ]
            )
        )
        contrasts, advantages = _selected_values(sampled_changed)
        replicates["changed_root_prompt_benefit_contrast_95pct"].append(
            _mean(contrasts)
        )
        replicates[
            "prompt_benefit_contrast_to_realized_spearman_95pct"
        ].append(spearman_correlation(contrasts, advantages))
    return {key: _interval(values) for key, values in replicates.items()}


def summarize_blocks(
    blocks: Sequence[Sequence[dict[str, Any]]],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if len(blocks) != 2 or any(len(block) != 32 for block in blocks):
        raise ValueError("expected two complete 32-tree cohorts")
    rows = [row for block in blocks for row in block]
    changed_blocks = [_changed(block) for block in blocks]
    changed = [row for block in changed_blocks for row in block]
    stages = {}
    for stage in ("first", "second"):
        stages[stage] = {
            arm: {
                metric: _mean(
                    [
                        _tree_metric(
                            row,
                            stage=stage,
                            arm=arm,
                            metric=metric,
                        )
                        for row in rows
                    ]
                )
                for metric in (
                    "posterior_predictive_mse",
                    "truth_extension_coverage",
                    "support_size",
                )
            }
            for arm in ("conditional", "history_blind")
        }
        stages[stage]["differences"] = {
            metric: _mean(
                [
                    _difference(row, stage=stage, metric=metric)
                    for row in rows
                ]
            )
            for metric in rows[0]["tree_mean"][stage]["differences"]
        }

    contrasts, advantages = _selected_values(changed)
    selected = {
        "changed_root_tree_count": len(changed),
        "changed_root_tree_count_by_cohort": [
            len(block) for block in changed_blocks
        ],
        "mean_dynamic_root_prompt_conditioning_benefit": _mean(
            [
                float(
                    row["selected_roots"][
                        "dynamic_root_prompt_conditioning_benefit"
                    ]
                )
                for row in changed
            ]
        ),
        "mean_fixed_root_prompt_conditioning_benefit": _mean(
            [
                float(
                    row["selected_roots"][
                        "fixed_root_prompt_conditioning_benefit"
                    ]
                )
                for row in changed
            ]
        ),
        "mean_dynamic_minus_fixed_root_prompt_conditioning_benefit": (
            _mean(contrasts)
        ),
        "prompt_benefit_contrast_to_realized_advantage_spearman": (
            spearman_correlation(contrasts, advantages)
        ),
    }
    bootstrap = bootstrap_analysis(
        blocks,
        seed=BOOTSTRAP_SEED,
        samples=bootstrap_samples,
    )
    findings = {
        "second_stage_conditioning_mse_ci_below_zero": (
            bootstrap[
                "second_conditional_minus_history_blind_predictive_mse_95pct"
            ][1]
            < 0.0
        ),
        "second_stage_conditioning_coverage_ci_above_zero": (
            bootstrap[
                "second_conditional_minus_history_blind_truth_coverage_95pct"
            ][0]
            > 0.0
        ),
        "root_specific_benefit_spearman_ci_above_zero": (
            bootstrap[
                "prompt_benefit_contrast_to_realized_spearman_95pct"
            ][0]
            > 0.0
        ),
        "selected_root_mean_benefit_ci_above_zero": (
            bootstrap["changed_root_prompt_benefit_contrast_95pct"][0]
            > 0.0
        ),
    }
    return {
        "tree_count": len(rows),
        "stages": stages,
        "selected_root_prompt_conditioning": selected,
        "bootstrap": bootstrap,
        "retrospective_findings": findings,
    }


def run_audit(
    *,
    output_dir: Path,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    paths: Sequence[Path] = (DEVELOPMENT_RESULT, CONFIRMATION_RESULT),
    hashes: Sequence[str] = (
        DEVELOPMENT_RESULT_SHA256,
        CONFIRMATION_RESULT_SHA256,
    ),
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    blocks = load_and_validate_sources(paths=paths, hashes=hashes)
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "retrospective_replication_synthesis",
        "protocol": {
            "analysis_is_retrospective": True,
            "source_statuses_unchanged": True,
            "cannot_rescue_or_relabel_sources": True,
            "cohort_count": 2,
            "trees_per_cohort": 32,
            "tree_count": 64,
            "cohort_tree_indices": [list(indices) for indices in EXPECTED_TREE_INDICES],
            "bootstrap_design": "cohort_stratified_tree_resampling",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": bootstrap_samples,
            "model_calls": 0,
            "cost_usd": 0.0,
            "source_result_paths": [str(path.relative_to(REPO_ROOT)) for path in paths],
            "source_result_sha256": list(hashes),
        },
        "analysis": summarize_blocks(
            blocks,
            bootstrap_samples=bootstrap_samples,
        ),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(output_dir=args.output_dir)
    print(json.dumps(result["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
