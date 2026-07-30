#!/usr/bin/env python3
"""Run the fresh transport-resilient Qwen dynamic-support replication."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from threading import Lock
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_qwen_dynamic_vs_fixed_powered96 as p96
from scripts import number_game_qwen_external_canonical_confirmation as engine
from scripts import number_game_qwen_planner_depth_three as qwen
from scripts import number_game_qwen_pooled_first_link_confirmation32 as pooled
from scripts import number_game_validator_fallback_serving_smoke_v2 as smoke
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts.number_game_provider_seed_fallback import (
    DEFAULT_FALLBACK_OFFSETS,
    fallback_adapter_factory,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-dynamic-vs-fixed-resilient96-v2-1"
TREE_SEEDS = tuple(range(80_000, 80_096))
TARGET_SEEDS = tuple(range(81_000, 81_096))
VALIDATION_SEED_START = 82_000
TREE_COUNT = 96
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
RUN_BUDGET_USD = 15.75
MIN_STARTING_BALANCE_USD = 16.50
MAX_RETRIES = 480
MAX_ITEM_SALVAGED_DRAWS = 48
MAX_FALLBACK_EVENTS = 96
BOOTSTRAP_SEED = 84_000
BOOTSTRAP_SAMPLES = 20_000
SMOKE_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_validator_fallback_serving_smoke_v2"
    / "number-game-validator-fallback-serving-smoke-v2-20260730T021500Z"
    / "RESULT.json"
)
SMOKE_RESULT_SHA256 = (
    "26b9bcf95b72e89d081163d844bbbb06f368e7c1b8b3119c875c2b31940f1dcc"
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
        raise ValueError("validator fallback smoke hash changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    if result.get("status") != "passed":
        raise ValueError("validator fallback smoke did not pass")
    if protocol.get("interface_version") != smoke.INTERFACE_VERSION:
        raise ValueError("validator fallback smoke interface changed")
    if protocol.get("model") != qwen.TARGET_MODEL_ID:
        raise ValueError("validator fallback smoke model changed")
    if protocol.get("expected_requests") != 10:
        raise ValueError("validator fallback smoke request count changed")
    if protocol.get("efficacy_used_for_authorization") is not False:
        raise ValueError("validator fallback smoke used efficacy")
    if protocol.get("fallback_offsets") != list(DEFAULT_FALLBACK_OFFSETS):
        raise ValueError("validator fallback schedule changed")
    if not all((result.get("gates") or {}).values()):
        raise ValueError("validator fallback smoke has a failed gate")
    return result


def mechanics_gates(
    *,
    scored_trees: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    targets: Sequence[Any],
    parse_summary: dict[str, Any],
    fallback_events: Sequence[dict[str, Any]],
) -> dict[str, bool]:
    gates = p96.mechanics_gates(
        scored_trees=scored_trees,
        usage=usage,
        targets=targets,
        parse_summary=parse_summary,
    )
    gates["provider_seed_fallbacks_within_cap"] = (
        len(fallback_events) <= MAX_FALLBACK_EVENTS
    )
    return gates


@contextmanager
def configured_engine(
    smoke_result_path: Path,
    parse_events: list[dict[str, Any]],
    fallback_events: list[dict[str, Any]],
) -> Iterator[None]:
    original_adapter = pooled.depth._adapter
    pooled.depth._adapter = fallback_adapter_factory(
        original_adapter,
        events=fallback_events,
        event_lock=Lock(),
    )

    def bound_mechanics(
        *,
        scored_trees,
        usage,
        targets,
        parse_summary,
    ):
        return mechanics_gates(
            scored_trees=scored_trees,
            usage=usage,
            targets=targets,
            parse_summary=parse_summary,
            fallback_events=fallback_events,
        )

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
        "mechanics_gates": bound_mechanics,
        "efficacy_gates": p96.myopic_policy_gates,
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
        pooled.depth._adapter = original_adapter


def finalize_result(
    result: dict[str, Any],
    *,
    parse_events: Sequence[dict[str, Any]],
    fallback_events: Sequence[dict[str, Any]],
) -> None:
    fixed = p96.comparison_with_frozen_bootstrap(
        result["trees"],
        baseline="fixed_support_depth_three",
        seed=BOOTSTRAP_SEED,
        samples=BOOTSTRAP_SAMPLES,
    )
    myopic = p96.comparison_with_frozen_bootstrap(
        result["trees"],
        baseline="myopic_eig",
        seed=BOOTSTRAP_SEED,
        samples=BOOTSTRAP_SAMPLES,
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
            "provider_error_fallback_offsets": list(
                DEFAULT_FALLBACK_OFFSETS
            ),
            "provider_seed_fallback_events": list(fallback_events),
            "provider_seed_fallback_count": len(fallback_events),
        }
    )
    result.pop("primary_gates")
    result["myopic_policy_gates"] = p96.myopic_policy_gates(
        result["aggregate"]
    )
    result["dynamic_support"] = {
        "comparison": fixed,
        "root_differences": root_differences,
        "gates": p96.dynamic_support_gates(
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


def run_replication(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path = SMOKE_RESULT,
) -> dict[str, Any]:
    validate_smoke_result(smoke_result_path)
    parse_events: list[dict[str, Any]] = []
    fallback_events: list[dict[str, Any]] = []
    with configured_engine(
        smoke_result_path,
        parse_events,
        fallback_events,
    ):
        result = engine.run_confirmation(
            output_dir=output_dir,
            run_id=run_id,
            smoke_result_path=smoke_result_path,
        )
    finalize_result(
        result,
        parse_events=parse_events,
        fallback_events=fallback_events,
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
    result = run_replication(
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
