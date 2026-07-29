#!/usr/bin/env python3
"""Confirm Number Game first-link fidelity with pooled Qwen supports."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_depth_three_development as depth
from scripts import number_game_qwen_external_canonical_confirmation as engine
from scripts import number_game_qwen_first_link_confirmation64 as first_link
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts.number_game_pooled_support import (
    PooledStructuredAdapter,
    parse_pooled_proposals,
)
from scripts.number_game_qwen_pooled_support_serving_smoke import (
    INTERFACE_VERSION as SMOKE_INTERFACE_VERSION,
    MODEL_ID,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-pooled-first-link-confirmation32-1"
TREE_SEEDS = tuple(range(63_100, 63_132))
TARGET_SEEDS = tuple(range(63_200, 63_232))
VALIDATION_SEED_START = 63_300
TREE_COUNT = 32
TARGET_COUNT = 33
VALIDATION_DRAWS_PER_TREE = 8
PLANNING_HISTORIES_PER_TREE = 49
PARSE_EVENTS_PER_TREE = 58
REQUESTS_PER_TREE = 2 * PLANNING_HISTORIES_PER_TREE + 1 + 8
EXPECTED_REQUESTS = TREE_COUNT * REQUESTS_PER_TREE
EXPECTED_POOLED_PARSE_EVENTS = TREE_COUNT * PLANNING_HISTORIES_PER_TREE
EXPECTED_PARSE_EVENTS = TREE_COUNT * PARSE_EVENTS_PER_TREE
SECOND_DRAW_SEED_OFFSET = 1_000_000
RUN_BUDGET_USD = 5.25
MIN_STARTING_BALANCE_USD = 5.50
MAX_RETRIES = 24
MAX_ITEM_SALVAGED_DRAWS = 16
BOOTSTRAP_SEED = 63_600
BOOTSTRAP_SAMPLES = 20_000
SMOKE_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_pooled_support_serving_smoke"
    / "number-game-qwen-pooled-support-serving-smoke-20260729T093842Z"
    / "RESULT.json"
)
SMOKE_RESULT_SHA256 = (
    "f4c5371e9cbe80e4a344fd7cb649e7e02d883c00768456c472ee2474f60b880a"
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
        raise ValueError("pooled-support serving smoke hash changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise ValueError("pooled-support serving smoke did not pass")
    protocol = result.get("protocol") or {}
    if protocol.get("interface_version") != SMOKE_INTERFACE_VERSION:
        raise ValueError("pooled-support serving interface changed")
    if protocol.get("model") != MODEL_ID:
        raise ValueError("pooled-support serving model changed")
    if protocol.get("expected_requests") != 10:
        raise ValueError("pooled-support serving request count changed")
    if protocol.get("efficacy_used_for_authorization") is not False:
        raise ValueError("pooled-support smoke used efficacy")
    if not all((result.get("gates") or {}).values()):
        raise ValueError("pooled-support smoke has a failed gate")
    return result


def parser_accounting(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pooled = [event for event in events if event.get("pool_size") == 2]
    single = [event for event in events if event.get("pool_size") == 1]
    draw_diagnostics = []
    second_draw_contributions = []
    for event in pooled:
        draw_diagnostics.extend(event["draw_diagnostics"])
        second_draw_contributions.append(
            event["draw_novel_contributions"][1]
        )
    draw_diagnostics.extend(single)
    salvaged = sum(
        diagnostic.get("codec_mode") == "complete_item_salvage"
        for diagnostic in draw_diagnostics
    )
    return {
        "parse_events": len(events),
        "pooled_parse_events": len(pooled),
        "single_parse_events": len(single),
        "provider_draws_parsed": len(draw_diagnostics),
        "item_salvaged_draws": salvaged,
        "strict_json_draws": len(draw_diagnostics) - salvaged,
        "minimum_second_draw_novel_contribution": (
            min(second_draw_contributions)
            if second_draw_contributions
            else None
        ),
        "mean_second_draw_novel_contribution": (
            sum(second_draw_contributions)
            / len(second_draw_contributions)
            if second_draw_contributions
            else None
        ),
    }


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
        "all_pooled_generated_branches_valid": all(
            tree["mechanics"]["minimum_generated_first_branch_valid"] >= 8
            and tree["mechanics"]["minimum_generated_second_branch_valid"]
            >= 8
            for tree in scored_trees
        ),
        "all_retained_branches_non_degenerate": all(
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


def efficacy_gates(aggregate: dict[str, Any]) -> dict[str, bool]:
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


def first_link_gates(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "roots_differ_on_at_least_twenty_eight_trees": (
            summary["root_differences"] >= 28
        ),
        "mean_realized_advantage_at_least_point_zero_zero_eight": (
            summary["mean_realized_advantage"] >= 0.008
        ),
        "realized_advantage_interval_above_zero": (
            summary["mean_realized_advantage_95pct_bootstrap"][0] > 0.0
        ),
        "score_to_realized_spearman_at_least_point_two_five": (
            summary["score_to_realized_advantage_spearman"] >= 0.25
        ),
        "score_to_realized_spearman_interval_above_zero": (
            summary["score_to_realized_spearman_95pct_bootstrap"][0] > 0.0
        ),
        "wins_minus_losses_at_least_eight": (
            summary["wins"] - summary["losses"] >= 8
        ),
    }


@contextmanager
def configured_engine(
    smoke_result_path: Path,
    parse_events: list[dict[str, Any]],
) -> Iterator[None]:
    original_adapter = depth._adapter
    original_parser = depth.parse_proposals

    def pooled_adapter_factory(**kwargs):
        if kwargs["model"] != MODEL_ID:
            return original_adapter(**kwargs)
        first_adapter = original_adapter(**kwargs)
        second_kwargs = {
            **kwargs,
            "request_seed": (
                int(kwargs["request_seed"]) + SECOND_DRAW_SEED_OFFSET
            ),
        }
        second_adapter = original_adapter(**second_kwargs)
        return PooledStructuredAdapter([first_adapter, second_adapter])

    def recording_parser(response, *, observations=()):
        support, diagnostic = parse_pooled_proposals(
            response,
            observations=observations,
        )
        parse_events.append(diagnostic)
        return support, diagnostic

    def bound_mechanics(*, scored_trees, usage, targets):
        return mechanics_gates(
            scored_trees=scored_trees,
            usage=usage,
            targets=targets,
            parse_summary=parser_accounting(parse_events),
        )

    overrides = {
        "INTERFACE_VERSION": INTERFACE_VERSION,
        "TREE_SEEDS": TREE_SEEDS,
        "TARGET_SEEDS": TARGET_SEEDS,
        "VALIDATION_SEED_START": VALIDATION_SEED_START,
        "TREE_COUNT": TREE_COUNT,
        "TARGET_COUNT": TARGET_COUNT,
        "VALIDATION_DRAWS_PER_TREE": VALIDATION_DRAWS_PER_TREE,
        "REQUESTS_PER_TREE": REQUESTS_PER_TREE,
        "EXPECTED_REQUESTS": EXPECTED_REQUESTS,
        "RUN_BUDGET_USD": RUN_BUDGET_USD,
        "MIN_STARTING_BALANCE_USD": MIN_STARTING_BALANCE_USD,
        "MAX_RETRIES": MAX_RETRIES,
        "SMOKE_RESULT_SHA256": engine.sha256_file(smoke_result_path),
        "validate_smoke_result": validate_smoke_result,
        "mechanics_gates": bound_mechanics,
        "primary_gates": efficacy_gates,
        "diagnostic_gates": lambda aggregate: {},
    }
    originals = {name: getattr(engine, name) for name in overrides}
    depth._adapter = pooled_adapter_factory
    depth.parse_proposals = recording_parser
    try:
        for name, value in overrides.items():
            setattr(engine, name, value)
        yield
    finally:
        depth._adapter = original_adapter
        depth.parse_proposals = original_parser
        for name, value in originals.items():
            setattr(engine, name, value)


def summarize_first_link(scored_trees):
    original_seed = first_link.BOOTSTRAP_SEED
    original_samples = first_link.BOOTSTRAP_SAMPLES
    first_link.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    first_link.BOOTSTRAP_SAMPLES = BOOTSTRAP_SAMPLES
    try:
        return first_link.summarize_first_link(scored_trees)
    finally:
        first_link.BOOTSTRAP_SEED = original_seed
        first_link.BOOTSTRAP_SAMPLES = original_samples


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
    parse_summary = parser_accounting(parse_events)
    first_link_summary = summarize_first_link(result["trees"])
    result["protocol"].update(
        {
            "analysis_was_preregistered": True,
            "planning_support_pool_size": 2,
            "second_draw_seed_offset": SECOND_DRAW_SEED_OFFSET,
            "first_link_bootstrap_seed": BOOTSTRAP_SEED,
            "first_link_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "parse_accounting": parse_summary,
        }
    )
    result["first_link"] = first_link_summary
    result["efficacy_gates"] = result.pop("primary_gates")
    result["first_link_gates"] = first_link_gates(first_link_summary)
    result["status"] = (
        "passed"
        if all(result["mechanics_gates"].values())
        and all(result["efficacy_gates"].values())
        and all(result["first_link_gates"].values())
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
                "efficacy_gates": result["efficacy_gates"],
                "first_link_gates": result["first_link_gates"],
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
