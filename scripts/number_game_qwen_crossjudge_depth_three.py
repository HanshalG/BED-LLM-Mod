#!/usr/bin/env python3
"""Evaluate fixed Number Game depth-three policies on Qwen target supports."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
)
from scripts.number_game_crossfit_endpoint_precision import score_fixed_tree
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-crossjudge-depth-three-1"
MODEL_ID = "qwen/qwen3.7-plus"
SMOKE_SEEDS = tuple(range(31610, 31620))
FORMAL_SEED_START = 31700
ENDPOINT_DRAWS_PER_TREE = 16
SOURCE_TREE_COUNT = 32
TOTAL_TREE_COUNT = 64
EXPECTED_FORMAL_REQUESTS = TOTAL_TREE_COUNT * ENDPOINT_DRAWS_PER_TREE
SMOKE_BUDGET_USD = 0.10
FORMAL_BUDGET_USD = 3.50
MIN_FORMAL_STARTING_BALANCE_USD = 3.00
MIN_VALID_SUPPORT = 16
MIN_NOVEL_PER_TREE = 128
MAX_RETRIES = 8
CONCURRENCY = 128

SOURCE_STUDIES = (
    {
        "name": "fixed_policy_fresh_endpoints",
        "result": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_endpoint_precision"
            / "number-game-crossfit-endpoint-precision-20260728"
            / "RESULT.json"
        ),
        "result_sha256": (
            "47e684b11ac5c6340ff4f063ff7a14db8b8d0bd18b184d61f91e315903e4809e"
        ),
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_confirmation"
            / "number-game-crossfit-depth-three-confirmation-20260728"
            / "TREES.json"
        ),
        "trees_sha256": (
            "cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9"
        ),
    },
    {
        "name": "fresh_tree_replication",
        "result": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_fresh_replication"
            / "number-game-crossfit-depth-three-fresh-replication-20260728"
            / "RESULT.json"
        ),
        "result_sha256": (
            "25e0939164f6806b30481e48a88372f22639f6065096cb59978600509d43a3d8"
        ),
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_fresh_replication"
            / "number-game-crossfit-depth-three-fresh-replication-20260728"
            / "TREES.json"
        ),
        "trees_sha256": (
            "197dcfe3cb48eeb9a4b656d0f9b20ef2e0b45ac8d1df0ec4bd9358e07af18802"
        ),
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_seeds_for_tree(global_tree_index: int) -> tuple[int, ...]:
    start = FORMAL_SEED_START + (
        global_tree_index * ENDPOINT_DRAWS_PER_TREE
    )
    return tuple(range(start, start + ENDPOINT_DRAWS_PER_TREE))


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_FORMAL_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_FORMAL_STARTING_BALANCE_USD:.2f} formal projection"
        )


def _generate_supports(
    *,
    seeds: Sequence[int],
    output_dir: Path,
    run_id: str,
    run_budget_usd: float,
    concurrency: int,
) -> tuple[
    list[list[Any]],
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
]:
    adapters = [
        depth._adapter(
            model=MODEL_ID,
            run_id=run_id,
            output_dir=output_dir,
            request_seed=int(seed),
            concurrency=1,
            projected_cost=0.005,
            run_budget_usd=run_budget_usd,
        )
        for seed in seeds
    ]

    def request(adapter: Any) -> str:
        return adapter.chat_complete_messages_batched_structured(
            [depth.initial_messages()],
            temperature=depth.TEMPERATURE,
            block_size=1,
            response_format=depth.proposal_response_format(),
            max_new_tokens=depth.MAX_TOKENS,
        )[0]

    with ThreadPoolExecutor(
        max_workers=min(concurrency, len(adapters))
    ) as executor:
        responses = list(executor.map(request, adapters))
    supports = []
    diagnostics = []
    for response in responses:
        support, diagnostic = depth.parse_proposals(response)
        supports.append(support)
        diagnostics.append(diagnostic)
    return (
        supports,
        diagnostics,
        responses,
        [adapter.usage_snapshot() for adapter in adapters],
    )


def _usage(snapshots: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "adapter_requests": "adapter_requests",
        "http_attempts": "http_attempts",
        "retry_count": "retry_count",
        "provider_error_retries": "provider_error_retries",
        "adapter_reasoning_tokens": "adapter_reasoning_tokens",
        "forced_exits": "forced_exits",
        "run_cost_usd": "adapter_cost_usd",
    }
    return {
        output_key: sum(
            snapshot.get(snapshot_key, 0) for snapshot in snapshots
        )
        for output_key, snapshot_key in fields.items()
    }


def smoke_gates(
    *,
    supports: Sequence[Sequence[Any]],
    usage: dict[str, Any],
) -> dict[str, bool]:
    return {
        "exact_ten_accepted_requests": (
            usage["adapter_requests"] == len(SMOKE_SEEDS)
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "retries_within_cap": usage["retry_count"] <= MAX_RETRIES,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_smoke_budget": usage["run_cost_usd"] <= SMOKE_BUDGET_USD,
        "all_supports_have_at_least_sixteen_valid_rules": (
            len(supports) == len(SMOKE_SEEDS)
            and all(len(support) >= MIN_VALID_SUPPORT for support in supports)
        ),
        "all_support_extensions_are_unique_within_response": all(
            len({hypothesis.extension for hypothesis in support})
            == len(support)
            for support in supports
        ),
    }


def run_smoke(*, output_dir: Path, run_id: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    try:
        supports, diagnostics, responses, snapshots = _generate_supports(
            seeds=SMOKE_SEEDS,
            output_dir=output_dir,
            run_id=run_id,
            run_budget_usd=SMOKE_BUDGET_USD,
            concurrency=len(SMOKE_SEEDS),
        )
        usage = _usage(snapshots)
        gates = smoke_gates(supports=supports, usage=usage)
        raw_path = private_dir / "RAW_RESPONSES.json"
        checkpoint(
            raw_path,
            {
                "responses": [
                    {"seed": seed, "response": response}
                    for seed, response in zip(
                        SMOKE_SEEDS, responses, strict=True
                    )
                ]
            },
        )
        public_supports = {
            "schema_version": SCHEMA_VERSION,
            "model": MODEL_ID,
            "supports": [
                {
                    "seed": seed,
                    "hypotheses": [
                        hypothesis.public_dict() for hypothesis in support
                    ],
                    "diagnostics": diagnostic,
                }
                for seed, support, diagnostic in zip(
                    SMOKE_SEEDS,
                    supports,
                    diagnostics,
                    strict=True,
                )
            ],
            "raw_responses_sha256": sha256_file(raw_path),
        }
        checkpoint(output_dir / "SUPPORTS.json", public_supports)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(gates.values()) else "gated_null",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "stage": "support_serving_smoke",
                "model": MODEL_ID,
                "reasoning": False,
                "temperature": depth.TEMPERATURE,
                "seeds": list(SMOKE_SEEDS),
                "expected_requests": len(SMOKE_SEEDS),
                "run_budget_usd": SMOKE_BUDGET_USD,
                "scientific_endpoint_scored": False,
            },
            "usage": usage,
            "valid_support_counts": [len(support) for support in supports],
            "gates": gates,
            "supports_sha256": sha256_file(output_dir / "SUPPORTS.json"),
            "raw_responses_sha256": sha256_file(raw_path),
        }
        checkpoint(output_dir / "SMOKE.json", result)
        return result
    except Exception as exc:
        checkpoint(
            output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def validate_smoke_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    protocol = result.get("protocol") or {}
    if result.get("status") != "passed":
        raise ValueError("Qwen support smoke did not pass")
    if protocol.get("interface_version") != INTERFACE_VERSION:
        raise ValueError("Qwen support smoke interface changed")
    if protocol.get("model") != MODEL_ID:
        raise ValueError("Qwen support smoke model changed")
    if protocol.get("scientific_endpoint_scored") is not False:
        raise ValueError("Qwen support smoke opened a scientific endpoint")
    return result


def formal_gates(
    *,
    scored_trees: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    aggregate: dict[str, Any],
    study_aggregates: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    primary = aggregate["comparisons"]["crossfit_depth_two"]
    novel = aggregate["novel_target_mean_differences"]
    ranking = aggregate["ranking"]
    myopic = aggregate["comparisons"]["myopic_eig"]
    fixed = aggregate["comparisons"]["fixed_support_depth_three"]
    pts = aggregate["comparisons"]["positive_test_strategy"]
    return {
        "exact_1024_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_FORMAL_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "retries_within_cap": usage["retry_count"] <= MAX_RETRIES,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_formal_budget": (
            usage["run_cost_usd"] <= FORMAL_BUDGET_USD
        ),
        "all_sixteen_endpoint_supports_valid": all(
            tree["mechanics"]["endpoint_draw_count"]
            == ENDPOINT_DRAWS_PER_TREE
            and tree["mechanics"]["minimum_endpoint_support_valid"]
            >= MIN_VALID_SUPPORT
            for tree in scored_trees
        ),
        "every_tree_has_at_least_128_novel_endpoint_hypotheses": all(
            tree["mechanics"]["total_novel_endpoint_hypotheses"]
            >= MIN_NOVEL_PER_TREE
            for tree in scored_trees
        ),
        "fixed_depth_roots_differ_on_exactly_thirty_five_trees": (
            aggregate["root_differences"]["crossfit_depth_two"] == 35
        ),
        "depth_three_brier_gain_at_least_one_percent": (
            primary["relative_brier_reduction"] >= 0.01
        ),
        "depth_three_brier_ci_below_zero": (
            primary["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_wins_at_least_twenty_three_trees": (
            primary["brier_tree_wins"] >= 23
        ),
        "both_source_studies_directionally_positive": all(
            study["comparisons"]["crossfit_depth_two"][
                "mean_candidate_minus_baseline_brier"
            ]
            < 0.0
            for study in study_aggregates.values()
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
        "depth_three_beats_myopic_by_three_percent_with_ci": (
            myopic["relative_brier_reduction"] >= 0.03
            and myopic["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_beats_fixed_by_three_percent_with_ci": (
            fixed["relative_brier_reduction"] >= 0.03
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


def _load_sources() -> list[dict[str, Any]]:
    loaded = []
    for source in SOURCE_STUDIES:
        if sha256_file(source["result"]) != source["result_sha256"]:
            raise ValueError(f"{source['name']} RESULT.json hash changed")
        if sha256_file(source["trees"]) != source["trees_sha256"]:
            raise ValueError(f"{source['name']} TREES.json hash changed")
        result = json.loads(source["result"].read_text())
        trees = json.loads(source["trees"].read_text())
        if len(result["trees"]) != SOURCE_TREE_COUNT:
            raise ValueError(f"{source['name']} result tree count changed")
        if len(trees["trees"]) != SOURCE_TREE_COUNT:
            raise ValueError(f"{source['name']} public tree count changed")
        if [
            tree["tree_seed"] for tree in result["trees"]
        ] != [
            tree["tree_seed"] for tree in trees["trees"]
        ]:
            raise ValueError(f"{source['name']} tree order changed")
        loaded.append(
            {
                "name": source["name"],
                "result": result,
                "trees": trees,
            }
        )
    return loaded


def run_formal(
    *,
    smoke_result_path: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    smoke = validate_smoke_result(smoke_result_path)
    sources = _load_sources()
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    try:
        seeds = [
            seed
            for tree_index in range(TOTAL_TREE_COUNT)
            for seed in endpoint_seeds_for_tree(tree_index)
        ]
        supports, diagnostics, responses, snapshots = _generate_supports(
            seeds=seeds,
            output_dir=output_dir,
            run_id=run_id,
            run_budget_usd=FORMAL_BUDGET_USD,
            concurrency=CONCURRENCY,
        )
        raw_path = private_dir / "RAW_RESPONSES.json"
        checkpoint(
            raw_path,
            {
                "responses": [
                    {"seed": seed, "response": response}
                    for seed, response in zip(seeds, responses, strict=True)
                ]
            },
        )

        scored_trees = []
        public_endpoint_trees = []
        offset = 0
        for source in sources:
            for local_index in range(SOURCE_TREE_COUNT):
                tree_supports = supports[
                    offset : offset + ENDPOINT_DRAWS_PER_TREE
                ]
                tree_diagnostics = diagnostics[
                    offset : offset + ENDPOINT_DRAWS_PER_TREE
                ]
                tree_seeds = seeds[
                    offset : offset + ENDPOINT_DRAWS_PER_TREE
                ]
                scored = score_fixed_tree(
                    source["trees"]["trees"][local_index],
                    source["result"]["trees"][local_index],
                    [
                        [
                            hypothesis.public_dict()
                            for hypothesis in support
                        ]
                        for support in tree_supports
                    ],
                )
                scored["source_study"] = source["name"]
                scored["global_tree_index"] = len(scored_trees)
                scored_trees.append(scored)
                public_endpoint_trees.append(
                    {
                        "source_study": source["name"],
                        "tree_index": local_index,
                        "tree_seed": scored["tree_seed"],
                        "seeds": tree_seeds,
                        "supports": [
                            [
                                hypothesis.public_dict()
                                for hypothesis in support
                            ]
                            for support in tree_supports
                        ],
                        "diagnostics": tree_diagnostics,
                    }
                )
                offset += ENDPOINT_DRAWS_PER_TREE
        if offset != EXPECTED_FORMAL_REQUESTS:
            raise AssertionError("formal support grouping is incomplete")

        aggregate = aggregate_scored_trees(scored_trees)
        study_aggregates = {
            source["name"]: aggregate_scored_trees(
                [
                    tree
                    for tree in scored_trees
                    if tree["source_study"] == source["name"]
                ]
            )
            for source in sources
        }
        usage = _usage(snapshots)
        gates = formal_gates(
            scored_trees=scored_trees,
            usage=usage,
            aggregate=aggregate,
            study_aggregates=study_aggregates,
        )
        protocol = {
            "interface_version": INTERFACE_VERSION,
            "stage": "formal_crossjudge",
            "model": MODEL_ID,
            "reasoning": False,
            "temperature": depth.TEMPERATURE,
            "tree_count": TOTAL_TREE_COUNT,
            "endpoint_draws_per_tree": ENDPOINT_DRAWS_PER_TREE,
            "seed_start": FORMAL_SEED_START,
            "expected_requests": EXPECTED_FORMAL_REQUESTS,
            "concurrency": CONCURRENCY,
            "run_budget_usd": FORMAL_BUDGET_USD,
            "fixed_policy_roots": True,
            "root_reselection": False,
            "draw_weighting": "equal draw weight within tree",
            "tree_weighting": "equal tree weight",
            "smoke_result_sha256": sha256_file(smoke_result_path),
            "smoke_raw_responses_sha256": smoke[
                "raw_responses_sha256"
            ],
            "sources": [
                {
                    "name": source["name"],
                    "result_sha256": definition["result_sha256"],
                    "trees_sha256": definition["trees_sha256"],
                }
                for source, definition in zip(
                    sources, SOURCE_STUDIES, strict=True
                )
            ],
        }
        endpoints_document = {
            "schema_version": SCHEMA_VERSION,
            "protocol": protocol,
            "raw_responses_sha256": sha256_file(raw_path),
            "trees": public_endpoint_trees,
        }
        checkpoint(output_dir / "ENDPOINTS.json", endpoints_document)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(gates.values()) else "gated_null",
            "protocol": protocol,
            "usage": usage,
            "aggregate": aggregate,
            "study_aggregates": study_aggregates,
            "gates": gates,
            "trees": scored_trees,
            "endpoints_sha256": sha256_file(
                output_dir / "ENDPOINTS.json"
            ),
            "raw_responses_sha256": sha256_file(raw_path),
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
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-result", type=Path)
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.stage == "smoke":
        result = run_smoke(output_dir=args.output_dir, run_id=args.run_id)
    else:
        if args.smoke_result is None:
            parser.error("--smoke-result is required for formal stage")
        if not args.skip_balance_check:
            require_starting_balance(openrouter_remaining_credit())
        result = run_formal(
            smoke_result_path=args.smoke_result,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "usage": result["usage"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
