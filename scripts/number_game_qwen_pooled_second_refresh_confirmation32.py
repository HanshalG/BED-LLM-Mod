#!/usr/bin/env python3
"""Prospectively confirm pooled Qwen second-step regeneration value."""

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
from scripts import number_game_qwen_pooled_first_link_confirmation32 as pooled
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts.number_game_pooled_second_refresh_ablation import (
    comparison_summary,
    variant_root_selections,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-pooled-second-refresh-confirmation32-1"
TREE_SEEDS = tuple(range(64_000, 64_032))
TARGET_SEEDS = tuple(range(64_100, 64_132))
VALIDATION_SEED_START = 64_200
TREE_COUNT = 32
TARGET_COUNT = 33
VALIDATION_DRAWS_PER_TREE = 8
PLANNING_HISTORIES_PER_TREE = 49
PARSE_EVENTS_PER_TREE = 58
REQUESTS_PER_TREE = 2 * PLANNING_HISTORIES_PER_TREE + 1 + 8
EXPECTED_REQUESTS = TREE_COUNT * REQUESTS_PER_TREE
EXPECTED_POOLED_PARSE_EVENTS = TREE_COUNT * PLANNING_HISTORIES_PER_TREE
EXPECTED_PARSE_EVENTS = TREE_COUNT * PARSE_EVENTS_PER_TREE
RUN_BUDGET_USD = 5.25
MIN_STARTING_BALANCE_USD = 5.50
MAX_RETRIES = 96
MAX_ITEM_SALVAGED_DRAWS = 16
SECOND_REFRESH_BOOTSTRAP_SEED = 64_500
FIRST_LINK_BOOTSTRAP_SEED = 64_501
BOOTSTRAP_SAMPLES = 20_000
SMOKE_RESULT = pooled.SMOKE_RESULT
SMOKE_RESULT_SHA256 = pooled.SMOKE_RESULT_SHA256


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} projection"
        )


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
        "all_eight_validation_supports_valid": all(
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


def second_refresh_gates(comparison: dict[str, Any]) -> dict[str, bool]:
    return {
        "merged_and_parent_only_roots_differ_on_at_least_twelve_trees": (
            comparison["root_differences"] >= 12
        ),
        "merged_beats_parent_only_by_two_percent": (
            comparison["relative_brier_reduction"] >= 0.02
        ),
        "merged_vs_parent_only_interval_below_zero": (
            comparison["tree_bootstrap_brier_difference_95pct"][1] < 0.0
        ),
        "merged_wins_minus_losses_at_least_eight": (
            comparison["wins_minus_losses"] >= 8
        ),
    }


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
        "BOOTSTRAP_SEED": FIRST_LINK_BOOTSTRAP_SEED,
        "BOOTSTRAP_SAMPLES": BOOTSTRAP_SAMPLES,
        "SMOKE_RESULT": SMOKE_RESULT,
        "SMOKE_RESULT_SHA256": SMOKE_RESULT_SHA256,
        "mechanics_gates": mechanics_gates,
        "efficacy_gates": myopic_policy_gates,
    }
    originals = {name: getattr(pooled, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(pooled, name, value)
        with pooled.configured_engine(smoke_result_path, parse_events):
            yield
    finally:
        for name, value in originals.items():
            setattr(pooled, name, value)


def second_refresh_analysis(
    *,
    scored_trees: Sequence[dict[str, Any]],
    public_trees: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    scored_by_seed = {
        int(tree["tree_seed"]): tree for tree in scored_trees
    }
    rows = []
    for tree in public_trees:
        tree_seed = int(tree["tree_seed"])
        scored = scored_by_seed[tree_seed]
        selection = variant_root_selections(tree)
        roots = selection["roots"]
        registered = int(
            scored["selection"]["crossfit_depth_three_root"]
        )
        if roots["merged_retained_generated"] != registered:
            raise ValueError(
                f"tree {tree_seed} does not reproduce merged root"
            )
        per_root = {
            int(root): float(value)
            for root, value in scored["per_root_endpoint_brier"].items()
        }
        rows.append(
            {
                "tree_seed": tree_seed,
                "selected_roots": roots,
                "crossfit_risks": selection["risks"],
                "endpoint_brier": {
                    variant: per_root[root]
                    for variant, root in roots.items()
                },
            }
        )
    rng = random.Random(SECOND_REFRESH_BOOTSTRAP_SEED)
    bootstrap_indices = [
        [rng.randrange(TREE_COUNT) for _ in range(TREE_COUNT)]
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    comparisons = {
        baseline: comparison_summary(
            rows,
            baseline=baseline,
            bootstrap_indices=bootstrap_indices,
        )
        for baseline in ("parent_only", "generated_only")
    }
    return {
        "comparisons": comparisons,
        "gates": second_refresh_gates(comparisons["parent_only"]),
        "rows": rows,
    }


def summarize_first_link(scored_trees):
    original_seed = pooled.BOOTSTRAP_SEED
    original_samples = pooled.BOOTSTRAP_SAMPLES
    pooled.BOOTSTRAP_SEED = FIRST_LINK_BOOTSTRAP_SEED
    pooled.BOOTSTRAP_SAMPLES = BOOTSTRAP_SAMPLES
    try:
        return pooled.summarize_first_link(scored_trees)
    finally:
        pooled.BOOTSTRAP_SEED = original_seed
        pooled.BOOTSTRAP_SAMPLES = original_samples


def run_confirmation(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path = SMOKE_RESULT,
) -> dict[str, Any]:
    pooled.validate_smoke_result(smoke_result_path)
    parse_events: list[dict[str, Any]] = []
    with configured_engine(smoke_result_path, parse_events):
        result = engine.run_confirmation(
            output_dir=output_dir,
            run_id=run_id,
            smoke_result_path=smoke_result_path,
        )
    trees_document = json.loads(
        (output_dir / "TREES.json").read_text(encoding="utf-8")
    )
    refresh = second_refresh_analysis(
        scored_trees=result["trees"],
        public_trees=trees_document["trees"],
    )
    first_link_summary = summarize_first_link(result["trees"])
    result["protocol"].update(
        {
            "analysis_was_preregistered": True,
            "planning_support_pool_size": 2,
            "second_draw_seed_offset": pooled.SECOND_DRAW_SEED_OFFSET,
            "second_refresh_bootstrap_seed": SECOND_REFRESH_BOOTSTRAP_SEED,
            "first_link_bootstrap_seed": FIRST_LINK_BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "parse_accounting": pooled.parser_accounting(parse_events),
            "generated_only_support_minima_are_diagnostic": True,
        }
    )
    result["myopic_policy_gates"] = result.pop("primary_gates")
    result["second_refresh"] = refresh
    result["first_link"] = first_link_summary
    result["status"] = (
        "passed"
        if all(result["mechanics_gates"].values())
        and all(result["myopic_policy_gates"].values())
        and all(refresh["gates"].values())
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
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
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
                "second_refresh": {
                    key: value
                    for key, value in result["second_refresh"].items()
                    if key != "rows"
                },
                "first_link": {
                    key: value
                    for key, value in result["first_link"].items()
                    if key != "rows"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
