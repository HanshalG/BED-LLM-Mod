#!/usr/bin/env python3
"""Confirm path-dependent versus fixed-support Number Game planning."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_qwen_external_canonical_confirmation as engine
from scripts import number_game_qwen_planner_depth_three as qwen
from scripts import number_game_qwen_pooled_first_link_confirmation32 as pooled
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts.number_game_qwen_dynamic_vs_fixed_serving_smoke import (
    INTERFACE_VERSION as SMOKE_INTERFACE_VERSION,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-dynamic-vs-fixed-confirmation32-1"
TREE_SEEDS = tuple(range(66_000, 66_032))
TARGET_SEEDS = tuple(range(66_100, 66_132))
VALIDATION_SEED_START = 66_200
TREE_COUNT = 32
TARGET_COUNT = 33
VALIDATION_DRAWS_PER_TREE = 16
PLANNING_HISTORIES_PER_TREE = 49
PARSE_EVENTS_PER_TREE = (
    PLANNING_HISTORIES_PER_TREE + 1 + VALIDATION_DRAWS_PER_TREE
)
REQUESTS_PER_TREE = (
    2 * PLANNING_HISTORIES_PER_TREE + 1 + VALIDATION_DRAWS_PER_TREE
)
EXPECTED_REQUESTS = TREE_COUNT * REQUESTS_PER_TREE
EXPECTED_POOLED_PARSE_EVENTS = TREE_COUNT * PLANNING_HISTORIES_PER_TREE
EXPECTED_PARSE_EVENTS = TREE_COUNT * PARSE_EVENTS_PER_TREE
RUN_BUDGET_USD = 5.25
MIN_STARTING_BALANCE_USD = 5.50
MAX_RETRIES = 96
MAX_ITEM_SALVAGED_DRAWS = 16
BOOTSTRAP_SEED = 66_800
BOOTSTRAP_SAMPLES = 20_000
SMOKE_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_dynamic_vs_fixed_serving_smoke"
    / "number-game-qwen-dynamic-vs-fixed-serving-smoke-20260729T114808Z"
    / "RESULT.json"
)
SMOKE_RESULT_SHA256 = (
    "aa89d0443427d165e4d37d90dd85915eebad9b3637f3304eac30382309a31470"
)


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} projection"
        )


def validate_smoke_result(path: Path) -> dict[str, Any]:
    if engine.sha256_file(path) != SMOKE_RESULT_SHA256:
        raise ValueError("dynamic-vs-fixed serving smoke hash changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise ValueError("dynamic-vs-fixed serving smoke did not pass")
    protocol = result.get("protocol") or {}
    if protocol.get("interface_version") != SMOKE_INTERFACE_VERSION:
        raise ValueError("dynamic-vs-fixed serving interface changed")
    if protocol.get("model") != pooled.MODEL_ID:
        raise ValueError("dynamic-vs-fixed serving model changed")
    if protocol.get("expected_requests") != 10:
        raise ValueError("dynamic-vs-fixed serving request count changed")
    if protocol.get("efficacy_used_for_authorization") is not False:
        raise ValueError("dynamic-vs-fixed smoke used efficacy")
    if not all((result.get("gates") or {}).values()):
        raise ValueError("dynamic-vs-fixed smoke has a failed gate")
    return result


def mechanics_gates(
    *,
    scored_trees: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    targets: Sequence[Any],
    parse_summary: dict[str, Any],
) -> dict[str, bool]:
    return {
        "exactly_32_fresh_trees": len(scored_trees) == TREE_COUNT,
        "exactly_33_unique_canonical_targets": (
            len(targets) == TARGET_COUNT
            and len({target.extension for target in targets}) == TARGET_COUNT
        ),
        "accepted_request_count_exact": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "retries_within_cap": usage["retry_count"] <= MAX_RETRIES,
        "provider_error_retries_within_cap": (
            usage["provider_error_retries"] <= MAX_RETRIES
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "parse_event_count_exact": (
            parse_summary["parse_events"] == EXPECTED_PARSE_EVENTS
        ),
        "pooled_parse_event_count_exact": (
            parse_summary["pooled_parse_events"]
            == EXPECTED_POOLED_PARSE_EVENTS
        ),
        "item_salvaged_draws_within_cap": (
            parse_summary["item_salvaged_draws"]
            <= MAX_ITEM_SALVAGED_DRAWS
        ),
        "all_pooled_initial_supports_valid": all(
            tree["mechanics"]["initial_valid"] >= 24
            for tree in scored_trees
        ),
        "all_deployed_retained_branches_non_degenerate": all(
            tree["mechanics"]["minimum_first_branch_valid"] >= 12
            and tree["mechanics"]["minimum_retained_second_branch_valid"]
            >= 8
            for tree in scored_trees
        ),
        "all_sixteen_validation_supports_valid": all(
            tree["mechanics"]["validation_support_count"]
            == VALIDATION_DRAWS_PER_TREE
            and tree["mechanics"]["minimum_validation_support_valid"] >= 16
            for tree in scored_trees
        ),
    }


def myopic_policy_gates(aggregate: dict[str, Any]) -> dict[str, bool]:
    myopic = aggregate["comparisons"]["myopic_eig"]
    return {
        "depth_three_beats_myopic_by_eight_percent": (
            myopic["relative_brier_reduction"] >= 0.08
        ),
        "depth_three_vs_myopic_ci_below_zero": (
            myopic["tree_cluster_brier_difference_95pct_bootstrap"][1] < 0.0
        ),
        "depth_three_wins_at_least_twenty_trees": (
            myopic["brier_tree_wins"] >= 20
        ),
    }


def dynamic_support_gates(
    *,
    comparison: dict[str, Any],
    root_differences: int,
) -> dict[str, bool]:
    return {
        "dynamic_and_fixed_roots_differ_on_at_least_twenty_four_trees": (
            root_differences >= 24
        ),
        "dynamic_beats_fixed_by_three_percent": (
            comparison["relative_brier_reduction"] >= 0.03
        ),
        "dynamic_vs_fixed_ci_below_zero": (
            comparison["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "dynamic_wins_at_least_sixteen_trees": (
            comparison["brier_tree_wins"] >= 16
        ),
    }


def _interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    return [
        ordered[int(0.025 * len(ordered))],
        ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))],
    ]


def comparison_with_frozen_bootstrap(
    trees: Sequence[dict[str, Any]],
    *,
    baseline: str,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    registered = dict(
        engine.aggregate_scored_trees(trees)["comparisons"][baseline]
    )
    brier = [
        float(tree["comparisons"][baseline]["candidate_minus_baseline_brier"])
        for tree in trees
    ]
    hamming = [
        float(
            tree["comparisons"][baseline][
                "candidate_minus_baseline_hamming"
            ]
        )
        for tree in trees
    ]
    coverage = [
        float(tree["comparisons"][baseline]["coverage_difference"])
        for tree in trees
    ]
    rng = random.Random(seed)
    bootstrap_indices = [
        [rng.randrange(len(trees)) for _ in trees]
        for _ in range(samples)
    ]

    def bootstrap(values: Sequence[float]) -> list[float]:
        return _interval(
            [
                sum(values[index] for index in indices) / len(indices)
                for indices in bootstrap_indices
            ]
        )

    registered[
        "tree_cluster_brier_difference_95pct_bootstrap"
    ] = bootstrap(brier)
    registered[
        "tree_cluster_hamming_difference_95pct_bootstrap"
    ] = bootstrap(hamming)
    registered[
        "tree_cluster_coverage_difference_95pct_bootstrap"
    ] = bootstrap(coverage)
    return registered


@contextmanager
def configured_engine(
    smoke_result_path: Path,
    parse_events: list[dict[str, Any]],
) -> Iterator[None]:
    overrides = {
        "INTERFACE_VERSION": INTERFACE_VERSION,
        "TREE_SEEDS": TREE_SEEDS,
        "TARGET_SEEDS": TARGET_SEEDS,
        "VALIDATION_SEED_START": VALIDATION_SEED_START,
        "TREE_COUNT": TREE_COUNT,
        "TARGET_COUNT": TARGET_COUNT,
        "VALIDATION_DRAWS_PER_TREE": VALIDATION_DRAWS_PER_TREE,
        "PLANNING_HISTORIES_PER_TREE": PLANNING_HISTORIES_PER_TREE,
        "PARSE_EVENTS_PER_TREE": PARSE_EVENTS_PER_TREE,
        "REQUESTS_PER_TREE": REQUESTS_PER_TREE,
        "EXPECTED_REQUESTS": EXPECTED_REQUESTS,
        "EXPECTED_POOLED_PARSE_EVENTS": EXPECTED_POOLED_PARSE_EVENTS,
        "EXPECTED_PARSE_EVENTS": EXPECTED_PARSE_EVENTS,
        "RUN_BUDGET_USD": RUN_BUDGET_USD,
        "MIN_STARTING_BALANCE_USD": MIN_STARTING_BALANCE_USD,
        "MAX_RETRIES": MAX_RETRIES,
        "MAX_ITEM_SALVAGED_DRAWS": MAX_ITEM_SALVAGED_DRAWS,
        "SMOKE_RESULT": smoke_result_path,
        "SMOKE_RESULT_SHA256": SMOKE_RESULT_SHA256,
        "validate_smoke_result": validate_smoke_result,
        "mechanics_gates": mechanics_gates,
        "efficacy_gates": myopic_policy_gates,
    }
    originals = {name: getattr(pooled, name) for name in overrides}
    original_validation_draws = qwen.VALIDATION_DRAWS_PER_TREE
    try:
        for name, value in overrides.items():
            setattr(pooled, name, value)
        qwen.VALIDATION_DRAWS_PER_TREE = VALIDATION_DRAWS_PER_TREE
        with pooled.configured_engine(smoke_result_path, parse_events):
            yield
    finally:
        qwen.VALIDATION_DRAWS_PER_TREE = original_validation_draws
        for name, value in originals.items():
            setattr(pooled, name, value)


def run_confirmation(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path = SMOKE_RESULT,
) -> dict[str, Any]:
    validate_smoke_result(smoke_result_path)
    parse_events: list[dict[str, Any]] = []
    with configured_engine(smoke_result_path, parse_events):
        result = engine.run_confirmation(
            output_dir=output_dir,
            run_id=run_id,
            smoke_result_path=smoke_result_path,
        )

    fixed = comparison_with_frozen_bootstrap(
        result["trees"],
        baseline="fixed_support_depth_three",
    )
    myopic = comparison_with_frozen_bootstrap(
        result["trees"],
        baseline="myopic_eig",
    )
    result["aggregate"]["comparisons"]["fixed_support_depth_three"] = fixed
    result["aggregate"]["comparisons"]["myopic_eig"] = myopic
    root_differences = sum(
        tree["selection"]["crossfit_depth_three_root"]
        != tree["selection"]["fixed_support_depth_three_root"]
        for tree in result["trees"]
    )
    parse_summary = pooled.parser_accounting(parse_events)
    result["protocol"].update(
        {
            "analysis_was_preregistered": True,
            "planning_support_pool_size": 2,
            "second_draw_seed_offset": pooled.SECOND_DRAW_SEED_OFFSET,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "parse_accounting": parse_summary,
            "sole_estimator_change": (
                "sixteen instead of eight independent validation supports"
            ),
        }
    )
    result["myopic_policy_gates"] = result.pop("primary_gates")
    result["dynamic_support"] = {
        "comparison": fixed,
        "root_differences": root_differences,
        "gates": dynamic_support_gates(
            comparison=fixed,
            root_differences=root_differences,
        ),
    }
    result["status"] = (
        "passed"
        if all(result["mechanics_gates"].values())
        and all(result["myopic_policy_gates"].values())
        and all(result["dynamic_support"]["gates"].values())
        else "gated_null"
    )
    checkpoint(output_dir / "RESULT.json", result)
    return result


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
    result = run_confirmation(
        output_dir=args.output_dir,
        run_id=args.run_id,
        smoke_result_path=args.smoke_result,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "usage": result["usage"],
                "mechanics_gates": result["mechanics_gates"],
                "myopic_policy_gates": result["myopic_policy_gates"],
                "dynamic_support": result["dynamic_support"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
