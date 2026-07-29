#!/usr/bin/env python3
"""Replay and diagnose the 36 complete trees from a failed Qwen run."""

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

from scripts import number_game_depth_three_development as depth
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_crossfit_depth_three_confirmation import (
    aggregate_scored_trees,
    score_public_tree as score_crossfit_public_tree,
)
from scripts.number_game_crossfit_endpoint_precision import score_fixed_tree
from scripts.number_game_external_canonical_replay import canonical_targets
from scripts.number_game_qwen_first_link_confirmation64 import first_link_rows
from scripts.number_game_qwen_first_link_mechanism64 import _interval
from scripts.number_game_ranking_fidelity_audit import spearman_correlation


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-first-link-failed-prefix36-1"
SOURCE_DIR = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_first_link_confirmation64"
    / "number-game-qwen-first-link-confirmation64-20260729T083156Z"
)
SOURCE_RAW = SOURCE_DIR / "private" / "RAW_RESPONSES.json"
SOURCE_RAW_SHA256 = (
    "6c2106ae7a676d0dbd4a7eb8e9b34cea20775e1f8a763132f43476452a1dc2a8"
)
SOURCE_FAILURE_SHA256 = (
    "3606024a5cff2a30eee96343c385f6cc142c300239d8be8f177470dc5c359893"
)
TREE_SEEDS = tuple(range(60_100, 60_136))
TARGET_SEEDS = tuple(range(60_200, 60_236))
VALIDATION_SEED_START = 61_100
VALIDATION_DRAWS_PER_TREE = 8
BOOTSTRAP_SEED = 61_800
BOOTSTRAP_SAMPLES = 20_000
PLANNING_MODEL = "qwen/qwen3.7-plus"
TARGET_MODEL = "google/gemini-2.5-flash"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplayAdapter:
    def __init__(self, batches: Sequence[Sequence[str]]) -> None:
        self._batches = [list(batch) for batch in batches]
        self._position = 0

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: Sequence[Sequence[dict[str, Any]]],
        **_: Any,
    ) -> list[str]:
        if self._position >= len(self._batches):
            raise ValueError("replay adapter received an unexpected batch")
        response = self._batches[self._position]
        self._position += 1
        if len(response) != len(batch_messages):
            raise ValueError("replay response batch length changed")
        return response

    def assert_exhausted(self) -> None:
        if self._position != len(self._batches):
            raise ValueError("replay adapter has unconsumed batches")

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": 0,
            "http_attempts": 0,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
        }


def validate_source_checkpoint() -> dict[str, Any]:
    if sha256_file(SOURCE_RAW) != SOURCE_RAW_SHA256:
        raise ValueError("failed-run raw checkpoint hash changed")
    if sha256_file(SOURCE_DIR / "FAILURE.json") != SOURCE_FAILURE_SHA256:
        raise ValueError("failed-run failure artifact hash changed")
    source = json.loads(SOURCE_RAW.read_text(encoding="utf-8"))
    if len(source.get("trees", [])) != len(TREE_SEEDS):
        raise ValueError("failed-run complete-tree count changed")
    for tree in source["trees"]:
        if (
            len(tree["first_responses"]) != 16
            or len(tree["second_responses"]) != 32
            or len(tree["validation_responses"])
            != VALIDATION_DRAWS_PER_TREE
        ):
            raise ValueError("failed-run response batch shape changed")
    return source


def validation_seeds_for_tree(tree_index: int) -> tuple[int, ...]:
    start = VALIDATION_SEED_START + tree_index * VALIDATION_DRAWS_PER_TREE
    return tuple(range(start, start + VALIDATION_DRAWS_PER_TREE))


def reconstruct_public_tree(
    *,
    raw_tree: dict[str, Any],
    tree_index: int,
) -> dict[str, Any]:
    planning = ReplayAdapter(
        (
            (raw_tree["initial_response"],),
            tuple(row["response"] for row in raw_tree["first_responses"]),
            tuple(row["response"] for row in raw_tree["second_responses"]),
        )
    )
    target = ReplayAdapter(((raw_tree["target_response"],),))
    original_adapter = depth._adapter

    def replay_factory(*, model: str, **_: Any) -> ReplayAdapter:
        if model == PLANNING_MODEL:
            return planning
        if model == TARGET_MODEL:
            return target
        raise ValueError(f"unexpected replay model {model}")

    depth._adapter = replay_factory
    try:
        _, artifacts = depth.run_tree_depth_three(
            tree_index=tree_index,
            tree_seed=TREE_SEEDS[tree_index],
            target_seed=TARGET_SEEDS[tree_index],
            output_dir=SOURCE_DIR,
            run_id="failed-prefix36-zero-call-replay",
            planning_model=PLANNING_MODEL,
            target_model=TARGET_MODEL,
            planning_concurrency=32,
            target_concurrency=1,
            projected_planning_cost=0.0,
            projected_target_cost=0.0,
            run_budget_usd=0.0,
            shared_budget_run_id="failed-prefix36-zero-call-replay",
            first_support_mode=depth.FIRST_SUPPORT_RETAINED_REJUVENATION,
            second_support_mode=depth.SECOND_SUPPORT_RETAINED_REJUVENATION,
            brier_tolerance=0.0,
        )
    finally:
        depth._adapter = original_adapter
    planning.assert_exhausted()
    target.assert_exhausted()

    expected_validation_seeds = validation_seeds_for_tree(tree_index)
    saved_validation_seeds = tuple(
        int(row["seed"]) for row in raw_tree["validation_responses"]
    )
    if saved_validation_seeds != expected_validation_seeds:
        raise ValueError("validation seed sequence changed")
    supports = []
    diagnostics = []
    for row in raw_tree["validation_responses"]:
        support, diagnostic = depth.parse_proposals(row["response"])
        supports.append(support)
        diagnostics.append(diagnostic)
    public = artifacts["public"]
    public["validation_seeds"] = list(expected_validation_seeds)
    public["validation_supports"] = [
        [hypothesis.public_dict() for hypothesis in support]
        for support in supports
    ]
    public["validation_diagnostics"] = diagnostics
    return public


def summarize_first_link(
    scored_trees: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = first_link_rows(scored_trees)
    changed = [row for row in rows if row["roots_differ"]]
    predicted = [row["predicted_advantage"] for row in changed]
    realized = [row["realized_advantage"] for row in changed]
    rng = random.Random(BOOTSTRAP_SEED)
    mean_bootstrap = []
    rho_bootstrap = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [rng.choice(changed) for _ in changed]
        sample_predicted = [row["predicted_advantage"] for row in sample]
        sample_realized = [row["realized_advantage"] for row in sample]
        mean_bootstrap.append(sum(sample_realized) / len(sample_realized))
        rho_bootstrap.append(
            spearman_correlation(sample_predicted, sample_realized)
        )
    wins = sum(value > 1e-15 for value in realized)
    losses = sum(value < -1e-15 for value in realized)
    return {
        "tree_count": len(rows),
        "root_differences": len(changed),
        "mean_predicted_advantage": sum(predicted) / len(predicted),
        "mean_realized_advantage": sum(realized) / len(realized),
        "mean_realized_advantage_95pct_bootstrap": _interval(
            mean_bootstrap
        ),
        "score_to_realized_advantage_spearman": spearman_correlation(
            predicted,
            realized,
        ),
        "score_to_realized_spearman_95pct_bootstrap": _interval(
            rho_bootstrap
        ),
        "wins": wins,
        "ties": len(realized) - wins - losses,
        "losses": losses,
        "rows": rows,
    }


def run_analysis(output_dir: Path) -> dict[str, Any]:
    source = validate_source_checkpoint()
    public_trees = [
        reconstruct_public_tree(
            raw_tree=raw_tree,
            tree_index=tree_index,
        )
        for tree_index, raw_tree in enumerate(source["trees"])
    ]
    targets = canonical_targets()
    public_targets = [target.public_dict() for target in targets]
    source_metrics = [
        score_crossfit_public_tree(tree) for tree in public_trees
    ]
    scored_trees = [
        score_fixed_tree(tree, metrics, [public_targets])
        for tree, metrics in zip(
            public_trees,
            source_metrics,
            strict=True,
        )
    ]
    aggregate = aggregate_scored_trees(scored_trees)
    first_link = summarize_first_link(scored_trees)
    protocol = {
        "analysis_was_frozen_before_endpoint_scoring": True,
        "analysis_is_underpowered_failed_prefix_diagnostic": True,
        "cannot_rescue_confirmation": True,
        "model_calls": 0,
        "cost_usd": 0.0,
        "tree_count": len(scored_trees),
        "planned_confirmation_tree_count": 64,
        "target_count": len(targets),
        "source_raw_sha256": SOURCE_RAW_SHA256,
        "source_failure_sha256": SOURCE_FAILURE_SHA256,
        "tree_seeds": list(TREE_SEEDS),
        "target_seeds": list(TARGET_SEEDS),
        "validation_seeds": [
            list(validation_seeds_for_tree(index))
            for index in range(len(TREE_SEEDS))
        ],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    trees_document = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "protocol": protocol,
        "trees": public_trees,
    }
    checkpoint(output_dir / "TREES.json", trees_document)
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "failed_run_prefix_diagnostic",
        "protocol": protocol,
        "aggregate": aggregate,
        "first_link": first_link,
        "trees": scored_trees,
        "trees_sha256": sha256_file(output_dir / "TREES.json"),
        "interpretation": {
            "confirmation_status_unchanged": "failed_closed",
            "no_original_gate_evaluated": True,
            "no_continuation_or_seed_substitution": True,
        },
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    source = validate_source_checkpoint()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "source_valid",
                    "tree_count": len(source["trees"]),
                    "source_raw_sha256": SOURCE_RAW_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = run_analysis(args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "protocol": result["protocol"],
                "first_link": {
                    key: value
                    for key, value in result["first_link"].items()
                    if key != "rows"
                },
                "comparisons": result["aggregate"]["comparisons"],
                "interpretation": result["interpretation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
