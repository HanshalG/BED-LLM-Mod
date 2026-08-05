#!/usr/bin/env python3
"""Run a fully fresh Number Game source and history-blind control study."""

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

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts import number_game_dynamic_support_quality96 as quality
from scripts import number_game_qwen_dynamic_vs_fixed_powered96 as p96
from scripts import number_game_qwen_dynamic_vs_fixed_resilient96_v2 as source
from scripts import number_game_qwen_history_blind_first_link_confirmation32 as first_link
from scripts import number_game_qwen_history_blind_matched32 as control
from scripts import number_game_qwen_history_blind_matched32_v3 as v3
from scripts.number_game_qwen_history_blind_serving_smoke import sha256_file
from scripts.number_game_provider_seed_fallback import DEFAULT_FALLBACK_OFFSETS


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-fully-fresh-source-control32-1"
SOURCE_INTERFACE_VERSION = (
    "number-game-qwen-fully-fresh-source-control32-source-1"
)
CONTROL_INTERFACE_VERSION = (
    "number-game-qwen-fully-fresh-source-control32-control-1"
)

TREE_COUNT = 32
TREE_SEEDS = tuple(range(100_000, 100_032))
TARGET_SEEDS = tuple(range(100_100, 100_132))
VALIDATION_SEED_START = 100_200
VALIDATION_DRAWS_PER_TREE = 16
PLANNING_HISTORIES_PER_TREE = 49
PARSE_EVENTS_PER_TREE = (
    PLANNING_HISTORIES_PER_TREE + 1 + VALIDATION_DRAWS_PER_TREE
)
SOURCE_REQUESTS_PER_TREE = (
    2 * PLANNING_HISTORIES_PER_TREE + 1 + VALIDATION_DRAWS_PER_TREE
)
SOURCE_EXPECTED_REQUESTS = TREE_COUNT * SOURCE_REQUESTS_PER_TREE
SOURCE_EXPECTED_POOLED_PARSE_EVENTS = (
    TREE_COUNT * PLANNING_HISTORIES_PER_TREE
)
SOURCE_EXPECTED_PARSE_EVENTS = TREE_COUNT * PARSE_EVENTS_PER_TREE
SOURCE_RUN_BUDGET_USD = 5.25
SOURCE_MAX_RETRIES = 96
SOURCE_MAX_ITEM_SALVAGED_DRAWS = 16
SOURCE_MAX_FALLBACK_EVENTS = 32
SOURCE_BOOTSTRAP_SEED = 100_800

CONTROL_SEED_START = 10_000_000
CONTROL_BOOTSTRAP_SEED = 10_100_000
CONTROL_RUN_BUDGET_USD = 4.25
CONTROL_EXPECTED_REQUESTS = control.EXPECTED_REQUESTS
COMPOSITE_EXPECTED_REQUESTS = (
    SOURCE_EXPECTED_REQUESTS + CONTROL_EXPECTED_REQUESTS
)
COMPOSITE_RUN_BUDGET_USD = (
    SOURCE_RUN_BUDGET_USD + CONTROL_RUN_BUDGET_USD
)
MIN_STARTING_BALANCE_USD = 10.25
MIN_CHANGED_ROOT_TREES = 20
BOOTSTRAP_SAMPLES = 20_000

SOURCE_PREDECESSOR_RESULT = source.REPO_ROOT / (
    "results/nonmyopic/number_game_qwen_dynamic_vs_fixed_resilient96_v2/"
    "number-game-qwen-dynamic-vs-fixed-resilient96-v2-20260730T023000Z/"
    "RESULT.json"
)
SOURCE_PREDECESSOR_RESULT_SHA256 = (
    "04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e"
)
CONTROL_PREDECESSOR_RESULT = first_link.DEVELOPMENT_DIR.parent.parent / (
    "number_game_qwen_history_blind_first_link_confirmation32/"
    "number-game-qwen-history-blind-first-link-confirmation32-"
    "20260730T044135Z/RESULT.json"
)
CONTROL_PREDECESSOR_RESULT_SHA256 = (
    "c47a6aba6c1ac5d9028f234c4670d62418c4ce8d5e09162145c188e6a2d2b725"
)

def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} requirement"
        )


def validate_predecessors(
    *,
    source_result_path: Path = SOURCE_PREDECESSOR_RESULT,
    control_result_path: Path = CONTROL_PREDECESSOR_RESULT,
) -> dict[str, Any]:
    if sha256_file(source_result_path) != SOURCE_PREDECESSOR_RESULT_SHA256:
        raise ValueError("source predecessor result hash changed")
    if sha256_file(control_result_path) != CONTROL_PREDECESSOR_RESULT_SHA256:
        raise ValueError("control predecessor result hash changed")
    source_result = json.loads(
        source_result_path.read_text(encoding="utf-8")
    )
    control_result = json.loads(
        control_result_path.read_text(encoding="utf-8")
    )
    if source_result.get("protocol", {}).get("tree_count") != 96:
        raise ValueError("source predecessor tree count changed")
    if source_result.get("usage", {}).get("adapter_requests") != 11_040:
        raise ValueError("source predecessor request count changed")
    source_mechanics = source_result.get("mechanics_gates") or {}
    failed_source_mechanics = {
        name for name, passed in source_mechanics.items() if not passed
    }
    if failed_source_mechanics != {
        "all_deployed_retained_branches_non_degenerate"
    }:
        raise ValueError("source predecessor mechanics pattern changed")
    if control_result.get("status") != "passed":
        raise ValueError("control predecessor did not pass")
    if (
        control_result.get("interface_version")
        != first_link.INTERFACE_VERSION
    ):
        raise ValueError("control predecessor interface changed")
    if not all(
        (control_result.get("mechanics_gates") or {}).values()
    ):
        raise ValueError("control predecessor mechanics did not all pass")
    return {
        "source": source_result,
        "control": control_result,
    }


def source_mechanics_gates(
    *,
    scored_trees: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    targets: Sequence[Any],
    parse_summary: dict[str, Any],
    fallback_events: Sequence[dict[str, Any]],
) -> dict[str, bool]:
    return {
        "exactly_32_fresh_trees": len(scored_trees) == TREE_COUNT,
        "exactly_33_unique_canonical_targets": (
            len(targets) == 33
            and len({target.extension for target in targets}) == 33
        ),
        "accepted_request_count_exact": (
            usage["adapter_requests"] == SOURCE_EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "retries_within_cap": (
            usage["retry_count"] <= SOURCE_MAX_RETRIES
        ),
        "provider_error_retries_within_cap": (
            usage["provider_error_retries"] <= SOURCE_MAX_RETRIES
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": (
            usage["run_cost_usd"] <= SOURCE_RUN_BUDGET_USD
        ),
        "parse_event_count_exact": (
            parse_summary["parse_events"]
            == SOURCE_EXPECTED_PARSE_EVENTS
        ),
        "pooled_parse_event_count_exact": (
            parse_summary["pooled_parse_events"]
            == SOURCE_EXPECTED_POOLED_PARSE_EVENTS
        ),
        "provider_draw_count_exact": (
            parse_summary["provider_draws_parsed"]
            == SOURCE_EXPECTED_REQUESTS
        ),
        "item_salvaged_draws_within_cap": (
            parse_summary["item_salvaged_draws"]
            <= SOURCE_MAX_ITEM_SALVAGED_DRAWS
        ),
        "provider_seed_fallbacks_within_cap": (
            len(fallback_events) <= SOURCE_MAX_FALLBACK_EVENTS
        ),
        "all_pooled_initial_supports_valid": all(
            tree["mechanics"]["initial_valid"] >= 24
            for tree in scored_trees
        ),
        "all_deployed_retained_branches_non_degenerate": all(
            tree["mechanics"]["minimum_first_branch_valid"] >= 12
            and tree["mechanics"][
                "minimum_retained_second_branch_valid"
            ]
            >= 8
            for tree in scored_trees
        ),
        "all_sixteen_validation_supports_valid": all(
            tree["mechanics"]["validation_support_count"]
            == VALIDATION_DRAWS_PER_TREE
            and tree["mechanics"]["minimum_validation_support_valid"]
            >= 16
            for tree in scored_trees
        ),
    }


def source_myopic_gates(aggregate: dict[str, Any]) -> dict[str, bool]:
    comparison = aggregate["comparisons"]["myopic_eig"]
    return {
        "depth_three_beats_myopic_by_eight_percent": (
            comparison["relative_brier_reduction"] >= 0.08
        ),
        "depth_three_vs_myopic_ci_below_zero": (
            comparison[
                "tree_cluster_brier_difference_95pct_bootstrap"
            ][1]
            < 0.0
        ),
        "depth_three_wins_at_least_twenty_trees": (
            comparison["brier_tree_wins"] >= 20
        ),
    }


def source_dynamic_gates(
    *,
    comparison: dict[str, Any],
    root_differences: int,
) -> dict[str, bool]:
    return {
        "dynamic_and_fixed_roots_differ_on_at_least_twenty_trees": (
            root_differences >= MIN_CHANGED_ROOT_TREES
        ),
        "dynamic_beats_fixed_by_three_percent": (
            comparison["relative_brier_reduction"] >= 0.03
        ),
        "dynamic_vs_fixed_ci_below_zero": (
            comparison[
                "tree_cluster_brier_difference_95pct_bootstrap"
            ][1]
            < 0.0
        ),
        "dynamic_wins_at_least_sixteen_trees": (
            comparison["brier_tree_wins"] >= 16
        ),
    }


def finalize_source_result(
    result: dict[str, Any],
    *,
    parse_events: Sequence[dict[str, Any]],
    fallback_events: Sequence[dict[str, Any]],
) -> None:
    fixed = p96.comparison_with_frozen_bootstrap(
        result["trees"],
        baseline="fixed_support_depth_three",
        seed=SOURCE_BOOTSTRAP_SEED,
        samples=BOOTSTRAP_SAMPLES,
    )
    myopic = p96.comparison_with_frozen_bootstrap(
        result["trees"],
        baseline="myopic_eig",
        seed=SOURCE_BOOTSTRAP_SEED,
        samples=BOOTSTRAP_SAMPLES,
    )
    result["aggregate"]["comparisons"]["fixed_support_depth_three"] = fixed
    result["aggregate"]["comparisons"]["myopic_eig"] = myopic
    root_differences = sum(
        tree["selection"]["crossfit_depth_three_root"]
        != tree["selection"]["fixed_support_depth_three_root"]
        for tree in result["trees"]
    )
    parse_summary = source.pooled.parser_accounting(parse_events)
    result["protocol"].update(
        {
            "analysis_was_preregistered": True,
            "planning_support_pool_size": 2,
            "second_draw_seed_offset": (
                source.pooled.SECOND_DRAW_SEED_OFFSET
            ),
            "bootstrap_seed": SOURCE_BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "parse_accounting": parse_summary,
            "provider_error_fallback_offsets": list(
                DEFAULT_FALLBACK_OFFSETS
            ),
            "provider_seed_fallback_events": list(fallback_events),
            "provider_seed_fallback_count": len(fallback_events),
            "fully_fresh_composite_source_stage": True,
        }
    )
    result.pop("primary_gates")
    result["myopic_policy_gates"] = source_myopic_gates(
        result["aggregate"]
    )
    result["dynamic_support"] = {
        "comparison": fixed,
        "root_differences": root_differences,
        "gates": source_dynamic_gates(
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


@contextmanager
def configured_source_module() -> Iterator[None]:
    overrides = {
        "INTERFACE_VERSION": SOURCE_INTERFACE_VERSION,
        "TREE_SEEDS": TREE_SEEDS,
        "TARGET_SEEDS": TARGET_SEEDS,
        "VALIDATION_SEED_START": VALIDATION_SEED_START,
        "TREE_COUNT": TREE_COUNT,
        "TARGET_COUNT": 33,
        "VALIDATION_DRAWS_PER_TREE": VALIDATION_DRAWS_PER_TREE,
        "PLANNING_HISTORIES_PER_TREE": PLANNING_HISTORIES_PER_TREE,
        "PARSE_EVENTS_PER_TREE": PARSE_EVENTS_PER_TREE,
        "REQUESTS_PER_TREE": SOURCE_REQUESTS_PER_TREE,
        "EXPECTED_REQUESTS": SOURCE_EXPECTED_REQUESTS,
        "EXPECTED_POOLED_PARSE_EVENTS": (
            SOURCE_EXPECTED_POOLED_PARSE_EVENTS
        ),
        "EXPECTED_PARSE_EVENTS": SOURCE_EXPECTED_PARSE_EVENTS,
        "RUN_BUDGET_USD": SOURCE_RUN_BUDGET_USD,
        "MIN_STARTING_BALANCE_USD": MIN_STARTING_BALANCE_USD,
        "MAX_RETRIES": SOURCE_MAX_RETRIES,
        "MAX_ITEM_SALVAGED_DRAWS": SOURCE_MAX_ITEM_SALVAGED_DRAWS,
        "MAX_FALLBACK_EVENTS": SOURCE_MAX_FALLBACK_EVENTS,
        "BOOTSTRAP_SEED": SOURCE_BOOTSTRAP_SEED,
        "BOOTSTRAP_SAMPLES": BOOTSTRAP_SAMPLES,
        "mechanics_gates": source_mechanics_gates,
        "finalize_result": finalize_source_result,
    }
    originals = {name: getattr(source, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(source, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(source, name, value)


def run_fresh_source(
    *,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    with configured_source_module():
        return source.run_replication(
            output_dir=output_dir,
            run_id=run_id,
            smoke_result_path=source.SMOKE_RESULT,
        )


@contextmanager
def configured_control_source(
    *,
    source_result_path: Path,
    source_trees_path: Path,
    source_targets_path: Path,
    source_result_sha256: str,
    source_trees_sha256: str,
    source_targets_sha256: str,
    tree_seeds: Sequence[int] = TREE_SEEDS,
) -> Iterator[None]:
    control_overrides = {
        "INTERFACE_VERSION": CONTROL_INTERFACE_VERSION,
        "SOURCE_RESULT": source_result_path,
        "SOURCE_TREES": source_trees_path,
        "SOURCE_TARGETS": source_targets_path,
        "SOURCE_RESULT_SHA256": source_result_sha256,
        "SOURCE_TREES_SHA256": source_trees_sha256,
        "SOURCE_TARGETS_SHA256": source_targets_sha256,
        "SOURCE_TREE_START": 0,
        "SOURCE_TREE_SEEDS": tuple(tree_seeds),
        "CONTROL_SEED_START": CONTROL_SEED_START,
        "BOOTSTRAP_SEED": CONTROL_BOOTSTRAP_SEED,
        "RUN_BUDGET_USD": CONTROL_RUN_BUDGET_USD,
        "mechanics_gates": v3.mechanics_gates,
        "summarize_scores": first_link.summarize_scores,
    }
    quality_overrides = {
        "SOURCE_RESULT_SHA256": source_result_sha256,
        "SOURCE_TREES_SHA256": source_trees_sha256,
        "SOURCE_TARGETS_SHA256": source_targets_sha256,
        "TREE_COUNT": TREE_COUNT,
    }
    control_originals = {
        name: getattr(control, name) for name in control_overrides
    }
    quality_originals = {
        name: getattr(quality, name) for name in quality_overrides
    }
    try:
        for name, value in control_overrides.items():
            setattr(control, name, value)
        for name, value in quality_overrides.items():
            setattr(quality, name, value)
        yield
    finally:
        for name, value in quality_originals.items():
            setattr(quality, name, value)
        for name, value in control_originals.items():
            setattr(control, name, value)


def run_fresh_control(
    *,
    source_dir: Path,
    output_dir: Path,
    run_id: str,
    adapter=None,
    remaining_credit: float | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    source_result_path = source_dir / "RESULT.json"
    source_trees_path = source_dir / "TREES.json"
    source_targets_path = source_dir / "TARGETS.json"
    source_hashes = {
        "result": sha256_file(source_result_path),
        "trees": sha256_file(source_trees_path),
        "targets": sha256_file(source_targets_path),
    }
    with configured_control_source(
        source_result_path=source_result_path,
        source_trees_path=source_trees_path,
        source_targets_path=source_targets_path,
        source_result_sha256=source_hashes["result"],
        source_trees_sha256=source_hashes["trees"],
        source_targets_sha256=source_hashes["targets"],
    ):
        result = control.run_formal(
            output_dir=output_dir,
            run_id=run_id,
            smoke_result_path=control.SMOKE_RESULT,
            adapter=adapter,
            remaining_credit=remaining_credit,
            bootstrap_samples=bootstrap_samples,
        )
    controls = json.loads(
        (output_dir / "CONTROLS.json").read_text(encoding="utf-8")
    )
    result["protocol"].update(
        {
            "fully_fresh_composite_control_stage": True,
            "source_result_sha256": source_hashes["result"],
            "source_trees_sha256": source_hashes["trees"],
            "source_targets_sha256": source_hashes["targets"],
            "source_tree_seeds": list(TREE_SEEDS),
            "control_seed_start": CONTROL_SEED_START,
            "bootstrap_seed": CONTROL_BOOTSTRAP_SEED,
            "development_responses_reused": False,
        }
    )
    result["second_draw_novelty"] = v3.second_draw_novelty(controls)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def _source_artifact_hashes(source_dir: Path) -> dict[str, str]:
    return {
        name: sha256_file(source_dir / filename)
        for name, filename in (
            ("result", "RESULT.json"),
            ("trees", "TREES.json"),
            ("targets", "TARGETS.json"),
        )
    }


def _control_artifact_hashes(control_dir: Path) -> dict[str, str]:
    return {
        name: sha256_file(control_dir / filename)
        for name, filename in (
            ("result", "RESULT.json"),
            ("controls", "CONTROLS.json"),
        )
    }


def _source_root_differences(result: dict[str, Any]) -> int:
    return int(result["dynamic_support"]["root_differences"])


def source_control_authorization(
    source_result: dict[str, Any],
) -> dict[str, Any]:
    mechanics_passed = all(source_result["mechanics_gates"].values())
    root_differences = _source_root_differences(source_result)
    opportunity_passed = root_differences >= MIN_CHANGED_ROOT_TREES
    authorized = mechanics_passed and opportunity_passed
    if not mechanics_passed:
        decision = "stop_before_control_source_mechanics"
    elif not opportunity_passed:
        decision = "stop_before_control_changed_root_floor"
    else:
        decision = "run_control_regardless_of_source_science"
    return {
        "source_mechanics_passed": mechanics_passed,
        "root_differences": root_differences,
        "minimum_root_differences": MIN_CHANGED_ROOT_TREES,
        "opportunity_passed": opportunity_passed,
        "control_authorized": authorized,
        "decision": decision,
    }


def _base_composite(
    *,
    run_id: str,
    starting_balance_usd: float,
    source_result: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "protocol": {
            "analysis_was_preregistered": True,
            "all_source_and_control_seeds_were_unopened": True,
            "source_science_did_not_authorize_control_stage": True,
            "control_authorization_used_only_mechanics_and_opportunity": True,
            "cannot_rescue_or_relabel_prior_results": True,
            "run_id": run_id,
            "tree_seeds": list(TREE_SEEDS),
            "target_seeds": list(TARGET_SEEDS),
            "validation_seed_start": VALIDATION_SEED_START,
            "source_expected_requests": SOURCE_EXPECTED_REQUESTS,
            "control_expected_requests": CONTROL_EXPECTED_REQUESTS,
            "composite_expected_requests": COMPOSITE_EXPECTED_REQUESTS,
            "source_run_budget_usd": SOURCE_RUN_BUDGET_USD,
            "control_run_budget_usd": CONTROL_RUN_BUDGET_USD,
            "composite_run_budget_usd": COMPOSITE_RUN_BUDGET_USD,
            "starting_balance_usd": starting_balance_usd,
            "source_predecessor_result_sha256": (
                SOURCE_PREDECESSOR_RESULT_SHA256
            ),
            "control_predecessor_result_sha256": (
                CONTROL_PREDECESSOR_RESULT_SHA256
            ),
            "source_artifacts": source_hashes,
        },
        "source": {
            "status": source_result["status"],
            "usage": source_result["usage"],
            "mechanics_gates": source_result["mechanics_gates"],
            "myopic_policy_gates": source_result[
                "myopic_policy_gates"
            ],
            "dynamic_support": source_result["dynamic_support"],
        },
    }


def finalize_composite_with_control(
    *,
    output_dir: Path,
    composite: dict[str, Any],
    source_result: dict[str, Any],
    control_result: dict[str, Any],
    control_dir: Path,
) -> dict[str, Any]:
    control_hashes = _control_artifact_hashes(control_dir)
    composite["protocol"]["control_artifacts"] = control_hashes
    composite["control"] = {
        "status": control_result["status"],
        "usage": control_result["usage"],
        "mechanics_gates": control_result["mechanics_gates"],
        "analysis": control_result["analysis"],
        "second_draw_novelty": control_result["second_draw_novelty"],
    }
    source_science_passed = (
        all(source_result["myopic_policy_gates"].values())
        and all(source_result["dynamic_support"]["gates"].values())
    )
    control_science_passed = all(
        control_result["analysis"]["scientific_gates"].values()
    )
    control_mechanics_passed = all(
        control_result["mechanics_gates"].values()
    )
    total_requests = (
        source_result["usage"]["adapter_requests"]
        + control_result["usage"]["adapter_requests"]
    )
    total_cost = (
        source_result["usage"]["run_cost_usd"]
        + control_result["usage"]["run_cost_usd"]
    )
    composite["composite_gates"] = {
        "source_mechanics_passed": True,
        "source_myopic_policy_passed": all(
            source_result["myopic_policy_gates"].values()
        ),
        "source_dynamic_support_passed": all(
            source_result["dynamic_support"]["gates"].values()
        ),
        "control_mechanics_passed": control_mechanics_passed,
        "control_science_passed": control_science_passed,
        "accepted_request_count_exact": (
            total_requests == COMPOSITE_EXPECTED_REQUESTS
        ),
        "within_composite_budget": (
            total_cost <= COMPOSITE_RUN_BUDGET_USD
        ),
    }
    composite["usage"] = {
        "source_requests": source_result["usage"]["adapter_requests"],
        "control_requests": control_result["usage"]["adapter_requests"],
        "total_requests": total_requests,
        "source_cost_usd": source_result["usage"]["run_cost_usd"],
        "control_cost_usd": control_result["usage"]["run_cost_usd"],
        "total_cost_usd": total_cost,
    }
    composite["status"] = (
        "passed"
        if source_science_passed
        and control_science_passed
        and control_mechanics_passed
        and total_requests == COMPOSITE_EXPECTED_REQUESTS
        and total_cost <= COMPOSITE_RUN_BUDGET_USD
        else "gated_null"
    )
    if not control_mechanics_passed:
        composite["status"] = "mechanics_failed"
        composite["decision"] = "stop_at_control_mechanics"
    else:
        composite["decision"] = "complete_composite_endpoint"
    checkpoint(output_dir / "RESULT.json", composite)
    return composite


def run_combined(
    *,
    output_dir: Path,
    run_id: str,
    remaining_credit: float | None = None,
    control_adapter=None,
    control_remaining_credit: float | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    validate_predecessors()
    available = (
        openrouter_remaining_credit()
        if remaining_credit is None
        else remaining_credit
    )
    require_starting_balance(available)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source"
    control_dir = output_dir / "control"
    source_result = run_fresh_source(
        output_dir=source_dir,
        run_id=f"{run_id}-source",
    )
    source_hashes = _source_artifact_hashes(source_dir)
    composite = _base_composite(
        run_id=run_id,
        starting_balance_usd=available,
        source_result=source_result,
        source_hashes=source_hashes,
    )
    authorization = source_control_authorization(source_result)
    composite["protocol"]["control_authorization"] = authorization
    if not authorization["source_mechanics_passed"]:
        composite["status"] = "mechanics_failed"
        composite["decision"] = authorization["decision"]
        checkpoint(output_dir / "RESULT.json", composite)
        return composite

    if not authorization["opportunity_passed"]:
        composite["status"] = "opportunity_failed"
        composite["decision"] = authorization["decision"]
        checkpoint(output_dir / "RESULT.json", composite)
        return composite

    try:
        control_result = run_fresh_control(
            source_dir=source_dir,
            output_dir=control_dir,
            run_id=f"{run_id}-control",
            adapter=control_adapter,
            remaining_credit=control_remaining_credit,
            bootstrap_samples=bootstrap_samples,
        )
    except RuntimeError as exc:
        failure_path = control_dir / "FAILURE.json"
        if (
            str(exc) != "formal history-blind mechanics gates failed"
            or not failure_path.exists()
        ):
            raise
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        composite["protocol"]["control_failure_sha256"] = sha256_file(
            failure_path
        )
        composite["control"] = {
            "status": "mechanics_failed",
            "usage": failure["usage"],
            "mechanics_gates": failure["mechanics_gates"],
        }
        composite["status"] = "mechanics_failed"
        composite["decision"] = "stop_at_control_mechanics"
        checkpoint(output_dir / "RESULT.json", composite)
        return composite
    return finalize_composite_with_control(
        output_dir=output_dir,
        composite=composite,
        source_result=source_result,
        control_result=control_result,
        control_dir=control_dir,
    )


def replay_fresh_control(
    *,
    source_dir: Path,
    control_dir: Path,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    source_hashes = _source_artifact_hashes(source_dir)
    control_result_path = control_dir / "RESULT.json"
    control_result = json.loads(
        control_result_path.read_text(encoding="utf-8")
    )
    controls_path = control_dir / "CONTROLS.json"
    if (
        sha256_file(controls_path)
        != control_result["protocol"]["controls_sha256"]
    ):
        raise ValueError("fully fresh control hash changed")
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    with configured_control_source(
        source_result_path=source_dir / "RESULT.json",
        source_trees_path=source_dir / "TREES.json",
        source_targets_path=source_dir / "TARGETS.json",
        source_result_sha256=source_hashes["result"],
        source_trees_sha256=source_hashes["trees"],
        source_targets_sha256=source_hashes["targets"],
    ):
        source_result, source_trees, targets = (
            quality.load_and_validate_sources(
                result_path=source_dir / "RESULT.json",
                trees_path=source_dir / "TREES.json",
                targets_path=source_dir / "TARGETS.json",
            )
        )
        rows = [
            control.score_tree(
                public_tree=public_tree,
                scored_tree=scored_tree,
                control_tree=control_tree,
                canonical_targets=targets,
            )
            for public_tree, scored_tree, control_tree in zip(
                source_trees["trees"],
                source_result["trees"],
                controls["trees"],
                strict=True,
            )
        ]
        analysis = control.summarize_scores(
            rows,
            bootstrap_samples=bootstrap_samples,
        )
    if rows != control_result["trees"]:
        raise ValueError("fully fresh replay tree rows differ")
    if analysis != control_result["analysis"]:
        raise ValueError("fully fresh replay analysis differs")
    return {"trees": rows, "analysis": analysis}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    try:
        result = run_combined(
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / "RUNNER_FAILURE.json"
        if not failure_path.exists():
            checkpoint(
                failure_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "usage": result.get("usage"),
                "composite_gates": result.get("composite_gates"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
