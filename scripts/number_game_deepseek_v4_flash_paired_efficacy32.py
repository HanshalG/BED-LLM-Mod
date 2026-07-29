#!/usr/bin/env python3
"""Test DeepSeek V4 Flash on the frozen Qwen Number Game evaluation bank."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

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
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts.number_game_predictive_risk_replication import (
    aggregate_tree_comparisons,
)
from scripts.number_game_ranking_fidelity_audit import (
    bootstrap_mean_interval,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-deepseek-v4-flash-paired-efficacy32-1"
MODEL_ID = "deepseek/deepseek-v4-flash"
TREE_SEEDS = tuple(range(49_000, 49_032))
TREE_COUNT = len(TREE_SEEDS)
PLANNING_HISTORIES_PER_TREE = 49
EXPECTED_REQUESTS = TREE_COUNT * PLANNING_HISTORIES_PER_TREE
RUN_BUDGET_USD = 0.50
MIN_STARTING_BALANCE_USD = 0.50
MAX_RETRIES = 16
BOOTSTRAP_SEED = 66_900
BOOTSTRAP_SAMPLES = 20_000
NONINFERIORITY_MARGIN = 0.005
MIN_INITIAL_VALID = 16
MIN_FIRST_BRANCH_VALID = 8
MIN_SECOND_BRANCH_VALID = 4

SOURCE_DIR = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_external_canonical_confirmation"
    / "number-game-qwen-external-canonical-confirmation-20260729"
)
SOURCE_TREES = SOURCE_DIR / "TREES.json"
SOURCE_TARGETS = SOURCE_DIR / "TARGETS.json"
SOURCE_RESULT = SOURCE_DIR / "RESULT.json"
SOURCE_TREES_SHA256 = (
    "39b79f391ae3b613d15794c9dd6c86ef02eb96907fbaa33591b157b2ac19cc63"
)
SOURCE_TARGETS_SHA256 = (
    "9e788da25b8431f457d044e9f7724bcea77312ca989aaf94b92001a21bf01a44"
)
SOURCE_RESULT_SHA256 = (
    "370e1c2923e56fb6a8344558db0a69bd5f86a8b013da7d9452675df380f7f12b"
)
SMOKE_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_planner_frontier_smoke"
    / "number-game-planner-frontier-deepseekv4flash-20260729T224000Z"
    / "RESULT.json"
)
SMOKE_RESULT_SHA256 = (
    "5342c72b3eb02e9038f7447dded89e9d726785169e9834335f09da91a03b6b71"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} projection"
        )


def validate_smoke_result(path: Path = SMOKE_RESULT) -> dict[str, Any]:
    if sha256_file(path) != SMOKE_RESULT_SHA256:
        raise ValueError("DeepSeek V4 Flash exact-10 RESULT hash changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    gates = result.get("gates") or {}
    if result.get("status") != "passed" or gates.get("all_pass") is not True:
        raise ValueError("DeepSeek V4 Flash exact-10 gate did not pass")
    if protocol.get("model") != MODEL_ID:
        raise ValueError("DeepSeek V4 Flash exact-10 model changed")
    if protocol.get("expected_requests") != 10:
        raise ValueError("DeepSeek V4 Flash serving gate was not exact-10")
    if protocol.get("reasoning") is not False:
        raise ValueError("DeepSeek V4 Flash serving gate used reasoning")
    return result


def load_frozen_bank(
    *,
    trees_path: Path = SOURCE_TREES,
    targets_path: Path = SOURCE_TARGETS,
    result_path: Path = SOURCE_RESULT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected = (
        (trees_path, SOURCE_TREES_SHA256, "TREES"),
        (targets_path, SOURCE_TARGETS_SHA256, "TARGETS"),
        (result_path, SOURCE_RESULT_SHA256, "RESULT"),
    )
    for path, expected_hash, label in expected:
        if sha256_file(path) != expected_hash:
            raise ValueError(f"frozen Qwen {label} hash changed")
    trees = json.loads(trees_path.read_text(encoding="utf-8"))
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    source_trees = trees.get("trees") or []
    source_targets = targets.get("targets") or []
    source_scored = result.get("trees") or []
    if len(source_trees) != TREE_COUNT or len(source_scored) != TREE_COUNT:
        raise ValueError("frozen Qwen bank does not contain exactly 32 trees")
    if [int(tree["tree_seed"]) for tree in source_trees] != list(TREE_SEEDS):
        raise ValueError("frozen Qwen tree seeds changed")
    if [int(tree["tree_seed"]) for tree in source_scored] != list(TREE_SEEDS):
        raise ValueError("frozen Qwen scored-tree seeds changed")
    if (
        len(source_targets) != 33
        or len({target["extension_sha256"] for target in source_targets})
        != 33
    ):
        raise ValueError("frozen canonical endpoint is not 33 unique targets")
    if any(len(tree.get("validation_supports") or []) != 8 for tree in source_trees):
        raise ValueError("frozen Qwen validation bank is incomplete")
    return trees, targets, result


class LocalTargetAdapter:
    """Supply a parser-valid local target support without a provider call."""

    def __init__(self, targets: Sequence[dict[str, Any]]) -> None:
        if len(targets) < 24:
            raise ValueError("local target support requires at least 24 targets")
        self.calls = 0
        self.response = json.dumps(
            {
                "hypotheses": [
                    {
                        "name": str(target["name"]),
                        "expression": str(target["expression"]),
                    }
                    for target in targets[:24]
                ]
            },
            sort_keys=True,
        )
        parsed, _ = depth.parse_proposals(self.response)
        if len(parsed) != 24:
            raise ValueError("local target support is not 24 unique valid rules")

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        **_: Any,
    ) -> list[str]:
        if len(batch_messages) != 1:
            raise ValueError("local target adapter expects one request")
        self.calls += 1
        return [self.response]

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": 0,
            "http_attempts": 0,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "forced_final_requests": 0,
            "forced_final_successes": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def _strict_parse_events_complete(tree: dict[str, Any]) -> bool:
    initial = tree.get("initial_diagnostics") or {}
    first = tree.get("first_branch_diagnostics") or {}
    second = tree.get("second_branch_diagnostics") or {}
    return (
        initial.get("raw_count") == 24
        and len(first) == 16
        and all(item.get("raw_count") == 24 for item in first.values())
        and len(second) == 32
        and all(item.get("raw_count") == 24 for item in second.values())
    )


def aggregate_usage(trees: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "adapter_requests",
        "http_attempts",
        "retry_count",
        "provider_error_retries",
        "adapter_reasoning_tokens",
        "forced_exits",
        "run_cost_usd",
    )
    return {
        field: sum(tree["usage"].get(field, 0) for tree in trees)
        for field in fields
    }


def mechanics_gates(
    *,
    live_trees: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    local_target_calls: int,
) -> dict[str, bool]:
    return {
        "exactly_32_complete_trees": len(live_trees) == TREE_COUNT,
        "exactly_1568_candidate_planner_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "retries_within_cap": usage["retry_count"] <= MAX_RETRIES,
        "zero_provider_error_retries": (
            usage["provider_error_retries"] == 0
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_49_response_shapes_strict_per_tree": all(
            _strict_parse_events_complete(tree) for tree in live_trees
        ),
        "all_initial_supports_valid": all(
            tree["mechanics"]["initial_valid"] >= MIN_INITIAL_VALID
            for tree in live_trees
        ),
        "all_merged_first_branches_valid": all(
            tree["mechanics"]["minimum_first_branch_valid"]
            >= MIN_FIRST_BRANCH_VALID
            for tree in live_trees
        ),
        "all_merged_second_branches_valid": all(
            tree["mechanics"]["minimum_second_branch_valid"]
            >= MIN_SECOND_BRANCH_VALID
            for tree in live_trees
        ),
        "exactly_32_local_target_stub_invocations": (
            local_target_calls == TREE_COUNT
        ),
        "zero_provider_target_or_validation_calls": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }


def comparison_with_frozen_bootstrap(
    trees: Sequence[dict[str, Any]],
    *,
    baseline: str,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    comparison = aggregate_tree_comparisons(
        trees,
        baseline=baseline,
        candidate="crossfit_depth_three",
    )
    differences = [
        float(
            tree["comparisons"][baseline][
                "candidate_minus_baseline_brier"
            ]
        )
        for tree in trees
    ]
    comparison[
        "tree_cluster_brier_difference_95pct_bootstrap"
    ] = bootstrap_mean_interval(
        differences,
        seed=seed,
        samples=BOOTSTRAP_SAMPLES,
    )
    return comparison


def intelligence_gates(comparison: dict[str, Any]) -> dict[str, bool]:
    return {
        "depth_three_beats_own_myopic_by_eight_percent": (
            comparison["relative_brier_reduction"] >= 0.08
        ),
        "depth_three_vs_own_myopic_ci_below_zero": (
            comparison["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_wins_at_least_twenty_trees": (
            comparison["brier_tree_wins"] >= 20
        ),
    }


def qwen_noninferiority(
    candidate_trees: Sequence[dict[str, Any]],
    qwen_trees: Sequence[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if len(candidate_trees) != len(qwen_trees):
        raise ValueError("candidate and Qwen tree counts differ")
    differences = []
    for candidate, qwen in zip(candidate_trees, qwen_trees, strict=True):
        if int(candidate["tree_seed"]) != int(qwen["tree_seed"]):
            raise ValueError("candidate and Qwen tree seeds are misaligned")
        differences.append(
            float(
                candidate["endpoint"]["crossfit_depth_three"][
                    "mean_posterior_predictive_brier"
                ]
            )
            - float(
                qwen["endpoint"]["crossfit_depth_three"][
                    "mean_posterior_predictive_brier"
                ]
            )
        )
    interval = bootstrap_mean_interval(
        differences,
        seed=seed,
        samples=BOOTSTRAP_SAMPLES,
    )
    tolerance = 1e-12
    return {
        "candidate_mean_brier": sum(
            float(
                tree["endpoint"]["crossfit_depth_three"][
                    "mean_posterior_predictive_brier"
                ]
            )
            for tree in candidate_trees
        )
        / len(candidate_trees),
        "qwen_mean_brier": sum(
            float(
                tree["endpoint"]["crossfit_depth_three"][
                    "mean_posterior_predictive_brier"
                ]
            )
            for tree in qwen_trees
        )
        / len(qwen_trees),
        "mean_candidate_minus_qwen_brier": sum(differences)
        / len(differences),
        "paired_tree_bootstrap_95pct": interval,
        "candidate_wins": sum(value < -tolerance for value in differences),
        "ties": sum(abs(value) <= tolerance for value in differences),
        "candidate_losses": sum(value > tolerance for value in differences),
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "upper_ci_below_margin": interval[1] < NONINFERIORITY_MARGIN,
        "per_tree_differences": differences,
    }


def run_efficacy(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path = SMOKE_RESULT,
    tree_runner: Callable[..., tuple[dict[str, Any], dict[str, Any]]]
    | None = None,
    max_tree_workers: int = 4,
) -> dict[str, Any]:
    if max_tree_workers <= 0:
        raise ValueError("max_tree_workers must be positive")
    smoke = validate_smoke_result(smoke_result_path)
    source_trees_doc, targets_doc, source_result = load_frozen_bank()
    source_trees = source_trees_doc["trees"]
    canonical_targets = targets_doc["targets"]
    source_scored = source_result["trees"]
    runner = tree_runner or depth.run_tree_depth_three

    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"trees": []}
    live_by_index: list[dict[str, Any] | None] = [None] * TREE_COUNT
    public_by_index: list[dict[str, Any] | None] = [None] * TREE_COUNT
    raw_by_index: list[dict[str, Any] | None] = [None] * TREE_COUNT
    local_target_calls = 0
    try:
        def run_one_tree(
            tree_index: int,
            tree_seed: int,
            source_tree: dict[str, Any],
        ) -> tuple[
            int,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            int,
        ]:
            local_target = LocalTargetAdapter(canonical_targets)
            live, artifacts = runner(
                tree_index=tree_index,
                tree_seed=tree_seed,
                target_seed=int(source_tree["target_seed"]),
                output_dir=output_dir,
                run_id=run_id,
                planning_model=MODEL_ID,
                target_model="local/frozen-canonical-stub",
                planning_concurrency=32,
                target_concurrency=1,
                projected_planning_cost=0.02,
                projected_target_cost=0.0,
                run_budget_usd=RUN_BUDGET_USD,
                shared_budget_run_id=run_id,
                first_support_mode=depth.FIRST_SUPPORT_RETAINED_REJUVENATION,
                second_support_mode=depth.SECOND_SUPPORT_RETAINED_REJUVENATION,
                brier_tolerance=0.0,
                target_adapter=local_target,
            )
            if local_target.calls != 1:
                raise ValueError("local target stub was not invoked exactly once")
            public = artifacts["public"]
            public["validation_seeds"] = copy.deepcopy(
                source_tree["validation_seeds"]
            )
            public["validation_supports"] = copy.deepcopy(
                source_tree["validation_supports"]
            )
            public["validation_diagnostics"] = copy.deepcopy(
                source_tree["validation_diagnostics"]
            )
            if (
                public["validation_supports"]
                != source_tree["validation_supports"]
            ):
                raise AssertionError("validation supports changed during copy")
            raw_tree = {
                "tree_index": tree_index,
                **artifacts["raw"],
            }
            return (
                tree_index,
                live,
                public,
                raw_tree,
                local_target.calls,
            )

        with ThreadPoolExecutor(max_workers=max_tree_workers) as executor:
            futures = [
                executor.submit(
                    run_one_tree,
                    tree_index,
                    tree_seed,
                    source_tree,
                )
                for tree_index, (tree_seed, source_tree) in enumerate(
                    zip(TREE_SEEDS, source_trees, strict=True)
                )
            ]
            for future in as_completed(futures):
                (
                    tree_index,
                    live,
                    public,
                    raw_tree,
                    target_calls,
                ) = future.result()
                live_by_index[tree_index] = live
                public_by_index[tree_index] = public
                raw_by_index[tree_index] = raw_tree
                local_target_calls += target_calls
                raw["trees"] = [
                    item for item in raw_by_index if item is not None
                ]
                checkpoint(raw_path, raw)

        if (
            any(tree is None for tree in live_by_index)
            or any(tree is None for tree in public_by_index)
        ):
            raise AssertionError("tree executor returned an incomplete bank")
        live_trees = [tree for tree in live_by_index if tree is not None]
        public_trees = [tree for tree in public_by_index if tree is not None]

        source_metrics = [
            score_crossfit_public_tree(tree) for tree in public_trees
        ]
        scored_trees = [
            score_fixed_tree(
                tree,
                metrics,
                [canonical_targets],
            )
            for tree, metrics in zip(
                public_trees,
                source_metrics,
                strict=True,
            )
        ]
        aggregate = aggregate_scored_trees(scored_trees)
        myopic = comparison_with_frozen_bootstrap(
            scored_trees,
            baseline="myopic_eig",
        )
        aggregate["comparisons"]["myopic_eig"] = myopic
        usage = aggregate_usage(live_trees)
        mechanics = mechanics_gates(
            live_trees=live_trees,
            usage=usage,
            local_target_calls=local_target_calls,
        )
        intelligence = intelligence_gates(myopic)
        noninferiority = qwen_noninferiority(
            scored_trees,
            source_scored,
        )
        all_mechanics = all(mechanics.values())
        all_intelligence = all(intelligence.values())
        noninferior = noninferiority["upper_ci_below_margin"]
        if all_mechanics and all_intelligence and noninferior:
            status = "passed"
            decision = "replace_qwen_for_future_scaled_runs"
        elif all_mechanics and all_intelligence:
            status = "efficacy_only"
            decision = "retain_as_lower_cost_exploratory_planner"
        elif all_mechanics:
            status = "gated_null"
            decision = "close_deepseek_route"
        else:
            status = "mechanics_failed"
            decision = "close_deepseek_route"

        protocol = {
            "interface_version": INTERFACE_VERSION,
            "candidate_planning_model": MODEL_ID,
            "reasoning": False,
            "temperature": depth.TEMPERATURE,
            "tree_seeds": list(TREE_SEEDS),
            "tree_count": TREE_COUNT,
            "concurrent_tree_workers": max_tree_workers,
            "maximum_planner_request_concurrency": (
                max_tree_workers * 32
            ),
            "planning_histories_per_tree": PLANNING_HISTORIES_PER_TREE,
            "expected_candidate_provider_calls": EXPECTED_REQUESTS,
            "provider_target_generation_calls": 0,
            "provider_validation_generation_calls": 0,
            "target_and_validation_bank": (
                "frozen Qwen external-canonical confirmation"
            ),
            "support_update": "retained_rejuvenation_at_both_steps",
            "candidate_roots_per_tree": 8,
            "canonical_target_count": 33,
            "tree_weighting": "equal",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "run_budget_usd": RUN_BUDGET_USD,
            "source_trees_sha256": SOURCE_TREES_SHA256,
            "source_targets_sha256": SOURCE_TARGETS_SHA256,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "smoke_result_sha256": SMOKE_RESULT_SHA256,
            "smoke_raw_responses_sha256": smoke[
                "raw_responses_sha256"
            ],
        }
        trees_document = {
            "schema_version": SCHEMA_VERSION,
            "protocol": protocol,
            "raw_responses_sha256": sha256_file(raw_path),
            "trees": public_trees,
        }
        checkpoint(output_dir / "TREES.json", trees_document)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "decision": decision,
            "protocol": protocol,
            "usage": usage,
            "mechanics_gates": mechanics,
            "intelligence_gates": intelligence,
            "qwen_noninferiority": noninferiority,
            "aggregate": aggregate,
            "trees": scored_trees,
            "trees_sha256": sha256_file(output_dir / "TREES.json"),
            "raw_responses_sha256": trees_document[
                "raw_responses_sha256"
            ],
        }
        checkpoint(output_dir / "RESULT.json", result)
        return result
    except Exception as exc:
        checkpoint(
            output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_trees": sum(
                    tree is not None for tree in live_by_index
                ),
                "raw_responses_sha256": (
                    sha256_file(raw_path) if raw_path.exists() else None
                ),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-result", type=Path, default=SMOKE_RESULT)
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if not args.skip_balance_check:
        require_starting_balance(openrouter_remaining_credit())
    result = run_efficacy(
        output_dir=args.output_dir,
        run_id=args.run_id,
        smoke_result_path=args.smoke_result,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "usage": result["usage"],
                "mechanics_gates": result["mechanics_gates"],
                "intelligence_gates": result["intelligence_gates"],
                "qwen_noninferiority": {
                    key: value
                    for key, value in result["qwen_noninferiority"].items()
                    if key != "per_tree_differences"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
