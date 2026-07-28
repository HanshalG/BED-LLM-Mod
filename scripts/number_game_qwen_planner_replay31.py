#!/usr/bin/env python3
"""Replay the 31 complete trees from the failed Qwen planner formal run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
from scripts.number_game_qwen_planner_depth_three import (
    ENDPOINT_DRAWS_PER_TREE,
    EXTRA_ENDPOINT_DRAWS_PER_TREE,
    FORMAL_TARGET_SEEDS,
    FORMAL_TREE_SEEDS,
    INTERFACE_VERSION as SOURCE_INTERFACE_VERSION,
    PLANNING_MODEL_ID,
    REQUESTS_PER_TREE,
    TARGET_MODEL_ID,
    VALIDATION_DRAWS_PER_TREE,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-planner-replay31-1"
TREE_COUNT = 31
RAW_PATH = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_planner_depth_three"
    / "number-game-qwen-planner-depth-three-20260728"
    / "private/RAW_RESPONSES.json"
)
RUN_LOG_PATH = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_planner_depth_three"
    / "number-game-qwen-planner-depth-three-20260728"
    / "run.log"
)
RAW_SHA256 = (
    "bed953cb8163f98fa046c42a713e6556571047a73bff0a9293696636c03d7fc1"
)
RUN_LOG_SHA256 = (
    "48523bcf77b5ebaf653cefa781e90ec93a502d65eaa814af9216a94356c3bd4e"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplayAdapter:
    def __init__(self, batches: Sequence[Sequence[str]]) -> None:
        self.batches = [list(batch) for batch in batches]

    def chat_complete_messages_batched_structured(
        self,
        messages: Sequence[Sequence[dict[str, Any]]],
        **kwargs: Any,
    ) -> list[str]:
        del kwargs
        if not self.batches:
            raise ValueError("replay adapter has no response batch")
        responses = self.batches.pop(0)
        if len(responses) != len(messages):
            raise ValueError("replay response count does not match messages")
        return responses

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


def development_gates(aggregate: dict[str, Any]) -> dict[str, bool]:
    primary = aggregate["comparisons"]["crossfit_depth_two"]
    myopic = aggregate["comparisons"]["myopic_eig"]
    fixed = aggregate["comparisons"]["fixed_support_depth_three"]
    pts = aggregate["comparisons"]["positive_test_strategy"]
    novel = aggregate["novel_target_mean_differences"]
    ranking = aggregate["ranking"]
    return {
        "crossfit_depth_roots_differ_on_at_least_twelve_trees": (
            aggregate["root_differences"]["crossfit_depth_two"] >= 12
        ),
        "depth_three_brier_gain_at_least_one_percent": (
            primary["relative_brier_reduction"] >= 0.01
        ),
        "depth_three_brier_ci_below_zero": (
            primary["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_wins_at_least_twelve_trees": (
            primary["brier_tree_wins"] >= 12
        ),
        "no_hamming_regression_vs_depth_two": (
            primary["mean_candidate_minus_baseline_hamming"] <= 0.0
        ),
        "no_coverage_regression_vs_depth_two": (
            primary["mean_coverage_difference"] >= 0.0
        ),
        "novel_targets_do_not_regress": (
            novel["candidate_minus_baseline_brier"] <= 0.0
            and novel["candidate_minus_baseline_hamming"] <= 0.0
            and novel["coverage_difference"] >= 0.0
        ),
        "depth_three_beats_myopic_by_five_percent_with_ci": (
            myopic["relative_brier_reduction"] >= 0.05
            and myopic["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_beats_fixed_by_five_percent_with_ci": (
            fixed["relative_brier_reduction"] >= 0.05
            and fixed["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_beats_pts_by_three_percent_with_ci": (
            pts["relative_brier_reduction"] >= 0.03
            and pts["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_rank_rho_at_least_point_seven": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"]
            >= 0.7
        ),
        "depth_three_rho_exceeds_depth_two_by_point_one_five": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"]
            - ranking["crossfit_depth_two_spearman_brier"]["mean"]
            >= 0.15
        ),
    }


def _usage_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "llm_token_usage":
            rows.append(row)
    return rows


def _parse_support_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    supports = []
    diagnostics = []
    for record in records:
        support, diagnostic = depth.parse_proposals(record["response"])
        supports.append(support)
        diagnostics.append(diagnostic)
    return supports, diagnostics


def replay31(*, output_dir: Path, run_id: str) -> dict[str, Any]:
    if sha256_file(RAW_PATH) != RAW_SHA256:
        raise ValueError("Qwen planner raw checkpoint hash changed")
    if sha256_file(RUN_LOG_PATH) != RUN_LOG_SHA256:
        raise ValueError("Qwen planner run log hash changed")
    raw_document = json.loads(RAW_PATH.read_text())
    if len(raw_document["trees"]) != TREE_COUNT:
        raise ValueError("raw checkpoint does not contain exactly 31 trees")
    usage_rows = _usage_rows(RUN_LOG_PATH)
    if len(usage_rows) != 32 * REQUESTS_PER_TREE:
        raise ValueError("formal run log request count changed")
    used_rows = usage_rows[: TREE_COUNT * REQUESTS_PER_TREE]
    if any(row.get("finish_reasons") != ["stop"] for row in used_rows):
        raise ValueError("a replayed tree contains a non-stop response")

    output_dir.mkdir(parents=True, exist_ok=True)
    public_trees = []
    public_endpoint_trees = []
    original_adapter = depth._adapter
    try:
        for index, raw_tree in enumerate(raw_document["trees"]):
            if (
                len(raw_tree["first_responses"]) != 16
                or len(raw_tree["second_responses"]) != 32
                or len(raw_tree["validation_responses"])
                != VALIDATION_DRAWS_PER_TREE
                or len(raw_tree["extra_endpoint_responses"])
                != EXTRA_ENDPOINT_DRAWS_PER_TREE
            ):
                raise ValueError(f"tree {index} raw batch counts changed")
            planning = ReplayAdapter(
                (
                    (raw_tree["initial_response"],),
                    tuple(
                        item["response"]
                        for item in raw_tree["first_responses"]
                    ),
                    tuple(
                        item["response"]
                        for item in raw_tree["second_responses"]
                    ),
                )
            )
            target = ReplayAdapter(((raw_tree["target_response"],),))
            adapters = iter((planning, target))
            depth._adapter = lambda **kwargs: next(adapters)
            live, artifacts = depth.run_tree_depth_three(
                tree_index=index,
                tree_seed=FORMAL_TREE_SEEDS[index],
                target_seed=FORMAL_TARGET_SEEDS[index],
                output_dir=output_dir,
                run_id=run_id,
                planning_model=PLANNING_MODEL_ID,
                target_model=TARGET_MODEL_ID,
                first_support_mode=(
                    depth.FIRST_SUPPORT_RETAINED_REJUVENATION
                ),
                second_support_mode=(
                    depth.SECOND_SUPPORT_RETAINED_REJUVENATION
                ),
                brier_tolerance=0.0,
            )
            if planning.batches or target.batches:
                raise ValueError(f"tree {index} left replay batches unused")
            validation_supports, validation_diagnostics = (
                _parse_support_records(raw_tree["validation_responses"])
            )
            endpoint_supports, endpoint_diagnostics = (
                _parse_support_records(raw_tree["extra_endpoint_responses"])
            )
            artifacts["public"]["validation_seeds"] = [
                int(item["seed"])
                for item in raw_tree["validation_responses"]
            ]
            artifacts["public"]["validation_supports"] = [
                [hypothesis.public_dict() for hypothesis in support]
                for support in validation_supports
            ]
            artifacts["public"]["validation_diagnostics"] = (
                validation_diagnostics
            )
            public_trees.append(artifacts["public"])
            public_endpoint_trees.append(
                {
                    "tree_index": index,
                    "seeds": [
                        FORMAL_TARGET_SEEDS[index],
                        *[
                            int(item["seed"])
                            for item in raw_tree[
                                "extra_endpoint_responses"
                            ]
                        ],
                    ],
                    "supports": [
                        artifacts["public"]["targets"],
                        *[
                            [
                                hypothesis.public_dict()
                                for hypothesis in support
                            ]
                            for support in endpoint_supports
                        ],
                    ],
                    "diagnostics": [
                        live["target_diagnostics"],
                        *endpoint_diagnostics,
                    ],
                }
            )
    finally:
        depth._adapter = original_adapter

    source_metrics = [
        score_crossfit_public_tree(tree) for tree in public_trees
    ]
    scored_trees = [
        score_fixed_tree(tree, metrics, endpoints["supports"])
        for tree, metrics, endpoints in zip(
            public_trees,
            source_metrics,
            public_endpoint_trees,
            strict=True,
        )
    ]
    aggregate = aggregate_scored_trees(scored_trees)
    gates = development_gates(aggregate)
    protocol = {
        "interface_version": INTERFACE_VERSION,
        "source_interface_version": SOURCE_INTERFACE_VERSION,
        "status_scope": "posthoc development only",
        "model_calls": 0,
        "tree_count": TREE_COUNT,
        "planning_model": PLANNING_MODEL_ID,
        "target_and_validation_model": TARGET_MODEL_ID,
        "tree_seeds": list(FORMAL_TREE_SEEDS[:TREE_COUNT]),
        "target_seeds": list(FORMAL_TARGET_SEEDS[:TREE_COUNT]),
        "validation_draws_per_tree": VALIDATION_DRAWS_PER_TREE,
        "endpoint_draws_per_tree": ENDPOINT_DRAWS_PER_TREE,
        "first_support_mode": (
            depth.FIRST_SUPPORT_RETAINED_REJUVENATION
        ),
        "second_support_mode": (
            depth.SECOND_SUPPORT_RETAINED_REJUVENATION
        ),
        "raw_checkpoint_sha256": RAW_SHA256,
        "formal_run_log_sha256": RUN_LOG_SHA256,
        "original_accepted_responses_used": len(used_rows),
        "original_response_cost_usd": sum(
            float(row["cost_usd"]) for row in used_rows
        ),
        "formal_failure_status_unchanged": True,
    }
    trees_document = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "trees": public_trees,
    }
    endpoints_document = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "trees": public_endpoint_trees,
    }
    checkpoint(output_dir / "TREES.json", trees_document)
    checkpoint(output_dir / "ENDPOINTS.json", endpoints_document)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "posthoc_development_gate_passed"
            if all(gates.values())
            else "posthoc_development_gate_failed"
        ),
        "protocol": protocol,
        "aggregate": aggregate,
        "gates": gates,
        "trees": scored_trees,
        "trees_sha256": sha256_file(output_dir / "TREES.json"),
        "endpoints_sha256": sha256_file(
            output_dir / "ENDPOINTS.json"
        ),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = replay31(output_dir=args.output_dir, run_id=args.run_id)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": result["gates"],
                "primary": result["aggregate"]["comparisons"][
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
