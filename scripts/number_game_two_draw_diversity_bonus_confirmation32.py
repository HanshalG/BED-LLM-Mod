#!/usr/bin/env python3
"""Reusable source and scoring helpers for staged diversity confirmation."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import statistics
import sys
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_qwen_fully_fresh_source_control32 as base
from scripts import number_game_two_draw_diversity_bonus_audit as audit
from scripts.number_game_ranking_fidelity_audit import spearman_correlation


SOURCE_INTERFACE_VERSION = (
    "number-game-two-draw-diversity-bonus-confirmation32-source-1"
)
TREE_SEEDS = tuple(range(110_000, 110_032))
TARGET_SEEDS = tuple(range(110_100, 110_132))
VALIDATION_SEED_START = 110_200
BOOTSTRAP_SEED = 110_800
TREE_COUNT = 32
DAILY_CAP_USD = 5.0
MIN_STARTING_BALANCE_USD = 5.0
PREREGISTRATION = REPO_ROOT / (
    "results/nonmyopic/"
    "NUMBER_GAME_TWO_DRAW_DIVERSITY_BONUS_PROSPECTIVE_PREREGISTRATION.md"
)


@contextmanager
def configured_fresh_source(
    *,
    tree_seeds: Sequence[int] = TREE_SEEDS,
    target_seeds: Sequence[int] = TARGET_SEEDS,
    validation_seed_start: int = VALIDATION_SEED_START,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    source_interface_version: str = SOURCE_INTERFACE_VERSION,
) -> Iterator[None]:
    overrides = {
        "SOURCE_INTERFACE_VERSION": source_interface_version,
        "TREE_SEEDS": tuple(tree_seeds),
        "TARGET_SEEDS": tuple(target_seeds),
        "VALIDATION_SEED_START": validation_seed_start,
        "SOURCE_BOOTSTRAP_SEED": bootstrap_seed,
        "SOURCE_RUN_BUDGET_USD": DAILY_CAP_USD,
        "MIN_STARTING_BALANCE_USD": MIN_STARTING_BALANCE_USD,
    }
    originals = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def _mean_rank_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    original_correlations = []
    adjusted_correlations = []
    original_regrets = []
    adjusted_regrets = []
    for row in rows:
        root_rows = row["root_rows"]
        roots = [int(item["root"]) for item in root_rows]
        realized = {
            int(item["root"]): float(item["realized_brier"])
            for item in root_rows
        }
        predicted = {
            int(item["root"]): float(item["predicted_brier"])
            for item in root_rows
        }
        adjusted = {
            int(root): float(value)
            for root, value in row["adjusted_scores"].items()
        }
        original_correlations.append(
            spearman_correlation(
                [predicted[root] for root in roots],
                [realized[root] for root in roots],
            )
        )
        adjusted_correlations.append(
            spearman_correlation(
                [adjusted[root] for root in roots],
                [realized[root] for root in roots],
            )
        )
        oracle_brier = min(realized.values())
        original_regrets.append(
            realized[int(row["original_root"])] - oracle_brier
        )
        adjusted_regrets.append(
            realized[int(row["bonus_root"])] - oracle_brier
        )
    return {
        "original_mean_candidate_root_spearman": statistics.fmean(
            original_correlations
        ),
        "bonus_mean_candidate_root_spearman": statistics.fmean(
            adjusted_correlations
        ),
        "original_mean_candidate_set_oracle_regret": statistics.fmean(
            original_regrets
        ),
        "bonus_mean_candidate_set_oracle_regret": statistics.fmean(
            adjusted_regrets
        ),
    }


def score_source_directory(
    source_dir: Path,
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    spec = {
        "name": "prospective_confirmation_block",
        "role": "prospective_confirmation",
        "directory": source_dir,
        "tree_count": TREE_COUNT,
        "result_sha256": audit.sha256_file(source_dir / "RESULT.json"),
        "trees_sha256": audit.sha256_file(source_dir / "TREES.json"),
        "raw_sha256": audit.sha256_file(
            source_dir / "private" / "RAW_RESPONSES.json"
        ),
        "targets_sha256": audit.sha256_file(source_dir / "TARGETS.json"),
    }
    rows = audit.load_source(
        spec,
        coefficients=(0.0, audit.DIVERSITY_COEFFICIENT),
    )
    summary = audit.source_summary(
        rows,
        seed=bootstrap_seed,
        include_coefficient_grid=False,
    )
    public_rows = []
    for row in rows:
        public_row = dict(row)
        public_row.pop("coefficient_grid_roots")
        public_row["adjusted_scores"] = {
            str(root): float(value)
            for root, value in public_row["adjusted_scores"].items()
        }
        public_rows.append(public_row)
    return {
        "source_artifacts": {
            key: spec[key]
            for key in (
                "result_sha256",
                "trees_sha256",
                "targets_sha256",
                "raw_sha256",
            )
        },
        "summary": summary,
        "rank_metrics": _mean_rank_metrics(rows),
        "rows": public_rows,
    }
