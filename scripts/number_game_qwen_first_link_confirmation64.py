#!/usr/bin/env python3
"""Prospectively confirm the Qwen Number Game first-link mechanism."""

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

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_qwen_external_canonical_confirmation as engine
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts.number_game_qwen_first_link_serving_smoke_v2 import (
    INTERFACE_VERSION as SMOKE_INTERFACE_VERSION,
    MODEL_ID,
)
from scripts.number_game_qwen_first_link_mechanism64 import _interval
from scripts.number_game_ranking_fidelity_audit import spearman_correlation


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-first-link-confirmation64-1"
TREE_SEEDS = tuple(range(60_100, 60_164))
TARGET_SEEDS = tuple(range(60_200, 60_264))
VALIDATION_SEED_START = 61_100
TREE_COUNT = 64
TARGET_COUNT = 33
VALIDATION_DRAWS_PER_TREE = 8
REQUESTS_PER_TREE = 58
EXPECTED_REQUESTS = TREE_COUNT * REQUESTS_PER_TREE
RUN_BUDGET_USD = 5.75
MIN_STARTING_BALANCE_USD = 5.25
MAX_RETRIES = 24
BOOTSTRAP_SEED = 61_700
BOOTSTRAP_SAMPLES = 20_000


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} projection"
        )


def validate_smoke_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    gates = result.get("gates") or {}
    if result.get("status") != "passed" or not gates.get("all_pass"):
        raise ValueError("Qwen first-link serving smoke did not pass")
    if protocol.get("interface_version") != SMOKE_INTERFACE_VERSION:
        raise ValueError("Qwen first-link smoke interface changed")
    if protocol.get("model") != MODEL_ID:
        raise ValueError("Qwen first-link smoke model changed")
    if protocol.get("expected_requests") != 10:
        raise ValueError("Qwen first-link smoke request count changed")
    if protocol.get("efficacy_used_for_authorization") is not False:
        raise ValueError("Qwen first-link smoke used efficacy")
    return result


def mechanics_gates(
    *,
    scored_trees: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    targets: Sequence[Any],
) -> dict[str, bool]:
    return {
        "exactly_64_fresh_trees": len(scored_trees) == TREE_COUNT,
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
        "all_initial_supports_valid": all(
            tree["mechanics"]["initial_valid"] >= 16
            for tree in scored_trees
        ),
        "all_eight_validation_supports_valid": all(
            tree["mechanics"]["validation_support_count"]
            == VALIDATION_DRAWS_PER_TREE
            and tree["mechanics"]["minimum_validation_support_valid"] >= 16
            for tree in scored_trees
        ),
        "all_retained_branches_non_degenerate": all(
            tree["mechanics"]["minimum_first_branch_valid"] >= 8
            and tree["mechanics"]["minimum_retained_second_branch_valid"] >= 4
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
            myopic["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_wins_at_least_forty_trees": (
            myopic["brier_tree_wins"] >= 40
        ),
    }


def diagnostic_gates(aggregate: dict[str, Any]) -> dict[str, bool]:
    del aggregate
    return {}


def first_link_rows(
    scored_trees: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    def root_value(mapping: dict[Any, Any], root: int) -> float:
        return float(mapping[root] if root in mapping else mapping[str(root)])

    rows = []
    for tree in scored_trees:
        selection = tree["selection"]
        depth_three_root = int(selection["crossfit_depth_three_root"])
        myopic_root = int(selection["myopic_root"])
        rows.append(
            {
                "tree_seed": int(tree["tree_seed"]),
                "roots_differ": depth_three_root != myopic_root,
                "predicted_advantage": (
                    root_value(
                        selection["crossfit_depth_three_brier"],
                        myopic_root,
                    )
                    - root_value(
                        selection["crossfit_depth_three_brier"],
                        depth_three_root,
                    )
                ),
                "realized_advantage": (
                    root_value(tree["per_root_endpoint_brier"], myopic_root)
                    - root_value(
                        tree["per_root_endpoint_brier"],
                        depth_three_root,
                    )
                ),
            }
        )
    return rows


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


def first_link_gates(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "roots_differ_on_at_least_fifty_six_trees": (
            summary["root_differences"] >= 56
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
        "wins_minus_losses_at_least_fifteen": (
            summary["wins"] - summary["losses"] >= 15
        ),
    }


@contextmanager
def configured_engine(smoke_result_path: Path) -> Iterator[None]:
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
        "mechanics_gates": mechanics_gates,
        "primary_gates": efficacy_gates,
        "diagnostic_gates": diagnostic_gates,
    }
    originals = {name: getattr(engine, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(engine, name, value)


def run_confirmation(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path,
) -> dict[str, Any]:
    validate_smoke_result(smoke_result_path)
    with configured_engine(smoke_result_path):
        result = engine.run_confirmation(
            output_dir=output_dir,
            run_id=run_id,
            smoke_result_path=smoke_result_path,
        )
    first_link = summarize_first_link(result["trees"])
    result["protocol"]["first_link_bootstrap_seed"] = BOOTSTRAP_SEED
    result["protocol"]["first_link_bootstrap_samples"] = BOOTSTRAP_SAMPLES
    result["protocol"]["analysis_was_preregistered"] = True
    result["first_link"] = first_link
    result["efficacy_gates"] = result.pop("primary_gates")
    result["first_link_gates"] = first_link_gates(first_link)
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
    parser.add_argument("--smoke-result", type=Path, required=True)
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
