#!/usr/bin/env python3
"""Develop a multi-draw proposal prior on the open Number Game trees."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    MAX_TOKENS,
    RuleHypothesis,
    choose_predictive_bayes_risk_root,
    compile_expression,
    evaluate_policy_root,
    initial_messages,
    parse_proposals,
    predictive_bayes_risk_scores,
    proposal_response_format,
)
from scripts.number_game_predictive_risk_holdout import (
    aggregate_random_root_control,
    policy_comparison,
)
from scripts.number_game_predictive_risk_replication import (
    PLANNING_MODEL_ID,
    TEMPERATURE,
    _adapter,
    cluster_bootstrap_interval,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-multidraw-prior-development-1"
SOURCE_RESULT_SHA256 = (
    "0796310076a9cafd8c8aa0cba6e6c1adef7461d7b88e6c9dac203409e38c8855"
)
SOURCE_TREES_SHA256 = (
    "8817a10f1c121261524508c3b95699e489d072ceb9485bedf54915eaada9473f"
)
EXTRA_PRIOR_SEEDS = tuple(range(26300, 26316))
EXTRA_DRAWS_PER_TREE = 2
EXPECTED_REQUESTS = len(EXTRA_PRIOR_SEEDS)
MIN_DRAW_VALID = 16
MIN_ENSEMBLE_UNIQUE = 32
RUN_BUDGET_USD = 0.50


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rule(item: dict[str, Any]) -> RuleHypothesis:
    return RuleHypothesis(
        item["name"],
        item["expression"],
        compile_expression(item["expression"]),
    )


def dedupe_support(
    supports: list[list[RuleHypothesis]],
) -> list[RuleHypothesis]:
    merged = []
    seen = set()
    for support in supports:
        for hypothesis in support:
            if hypothesis.extension not in seen:
                merged.append(hypothesis)
                seen.add(hypothesis.extension)
    return merged


def aggregate_comparisons(
    trees: list[dict[str, Any]],
    baseline: str,
) -> dict[str, Any]:
    differences = [
        tree["comparisons"][baseline][
            "candidate_minus_baseline_brier"
        ]
        for tree in trees
    ]
    candidate = sum(
        tree["endpoint"]["multidraw_predictive_risk"][
            "mean_posterior_predictive_brier"
        ]
        for tree in trees
    ) / len(trees)
    control = sum(
        tree["endpoint"][baseline]["mean_posterior_predictive_brier"]
        for tree in trees
    ) / len(trees)
    return {
        "candidate_mean_brier": candidate,
        "baseline_mean_brier": control,
        "relative_brier_reduction": (control - candidate) / control,
        "mean_brier_difference": sum(differences) / len(differences),
        "tree_cluster_brier_difference_95pct_bootstrap": (
            cluster_bootstrap_interval(differences)
        ),
        "strict_tree_wins": sum(value < 0.0 for value in differences),
        "tree_ties": sum(abs(value) <= 1e-15 for value in differences),
        "mean_coverage_difference": sum(
            tree["comparisons"][baseline]["coverage_difference"]
            for tree in trees
        )
        / len(trees),
    }


def run_development(
    *,
    source_result: Path,
    source_trees: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    if sha256_file(source_result) != SOURCE_RESULT_SHA256:
        raise ValueError("source RESULT.json hash changed")
    if sha256_file(source_trees) != SOURCE_TREES_SHA256:
        raise ValueError("source TREES.json hash changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    source_result_payload = json.loads(source_result.read_text())
    source_trees_payload = json.loads(source_trees.read_text())

    adapters = [
        _adapter(
            model=PLANNING_MODEL_ID,
            run_id=f"{run_id}-prior-{seed}",
            output_dir=output_dir,
            request_seed=seed,
            concurrency=1,
            projected_cost=0.02,
        )
        for seed in EXTRA_PRIOR_SEEDS
    ]

    def request(index: int) -> str:
        return adapters[index].chat_complete_messages_batched_structured(
            [initial_messages()],
            temperature=TEMPERATURE,
            block_size=1,
            response_format=proposal_response_format(),
            max_new_tokens=MAX_TOKENS,
        )[0]

    with ThreadPoolExecutor(max_workers=16) as executor:
        responses = list(executor.map(request, range(len(adapters))))
    raw = {
        "seeds": list(EXTRA_PRIOR_SEEDS),
        "responses": responses,
    }
    checkpoint(raw_path, raw)
    parsed_draws = []
    draw_diagnostics = []
    for response in responses:
        support, diagnostics = parse_proposals(response)
        parsed_draws.append(support)
        draw_diagnostics.append(diagnostics)

    tree_results = []
    public_draws = []
    for tree_index, (source_tree, source_metrics) in enumerate(
        zip(
            source_trees_payload["trees"],
            source_result_payload["trees"],
            strict=True,
        )
    ):
        original = [_rule(item) for item in source_tree["initial"]]
        extras = parsed_draws[
            tree_index
            * EXTRA_DRAWS_PER_TREE : (tree_index + 1)
            * EXTRA_DRAWS_PER_TREE
        ]
        ensemble = dedupe_support([original, *extras])
        branches = {}
        for key, items in source_tree["branches"].items():
            root, label = key.split(":")
            branches[(int(root), bool(int(label)))] = [
                _rule(item) for item in items
            ]
        targets = {
            f"target_{index:02d}_{item['name']}": _rule(item)
            for index, item in enumerate(source_tree["targets"])
        }
        roots = [int(root) for root in source_tree["roots"]]
        scores = predictive_bayes_risk_scores(
            support=ensemble,
            roots=roots,
            branches=branches,
        )
        selected = choose_predictive_bayes_risk_root(scores)
        policy_roots = {
            "multidraw_predictive_risk": selected,
            "original_predictive_risk": int(
                source_metrics["selection"]["predictive_bayes_risk_root"]
            ),
            "myopic_eig": int(source_metrics["selection"]["myopic_root"]),
        }
        endpoint = {
            policy: evaluate_policy_root(
                policy=policy,
                root=root,
                targets=targets,
                branches=branches,
            )
            for policy, root in policy_roots.items()
        }
        pts_roots = [
            int(root) for root in source_metrics["selection"]["pts_roots"]
        ]
        endpoint["positive_test_strategy"] = (
            aggregate_random_root_control(
                {
                    root: evaluate_policy_root(
                        policy=f"pts_{root}",
                        root=root,
                        targets=targets,
                        branches=branches,
                    )
                    for root in pts_roots
                }
            )
            | {"policy": "positive_test_strategy"}
        )
        comparisons = {
            baseline: policy_comparison(
                endpoint["multidraw_predictive_risk"],
                endpoint[baseline],
            )
            for baseline in (
                "original_predictive_risk",
                "myopic_eig",
                "positive_test_strategy",
            )
        }
        tree_results.append(
            {
                "tree_index": tree_index,
                "ensemble_unique_count": len(ensemble),
                "selection": {
                    "multidraw_predictive_risk_root": selected,
                    "original_predictive_risk_root": policy_roots[
                        "original_predictive_risk"
                    ],
                    "myopic_root": policy_roots["myopic_eig"],
                    "pts_roots": pts_roots,
                },
                "endpoint": {
                    policy: {
                        key: value
                        for key, value in result.items()
                        if key != "targets"
                    }
                    for policy, result in endpoint.items()
                },
                "comparisons": comparisons,
            }
        )
        public_draws.append(
            {
                "tree_index": tree_index,
                "draws": [
                    [hypothesis.public_dict() for hypothesis in support]
                    for support in extras
                ],
                "ensemble_unique_count": len(ensemble),
            }
        )
    aggregate = {
        baseline: aggregate_comparisons(tree_results, baseline)
        for baseline in (
            "original_predictive_risk",
            "myopic_eig",
            "positive_test_strategy",
        )
    }
    snapshots = [adapter.usage_snapshot() for adapter in adapters]
    usage = {
        key: sum(snapshot.get(key, 0) for snapshot in snapshots)
        for key in (
            "adapter_requests",
            "http_attempts",
            "retry_count",
            "provider_error_retries",
            "adapter_reasoning_tokens",
            "forced_exits",
            "adapter_cost_usd",
        )
    }
    usage["run_cost_usd"] = usage.pop("adapter_cost_usd")
    original = aggregate["original_predictive_risk"]
    myopic = aggregate["myopic_eig"]
    pts = aggregate["positive_test_strategy"]
    gates = {
        "exact_16_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_draws_have_at_least_16_valid_unique_rules": all(
            diagnostics["valid_unique_count"] >= MIN_DRAW_VALID
            for diagnostics in draw_diagnostics
        ),
        "all_ensembles_have_at_least_32_unique_rules": all(
            tree["ensemble_unique_count"] >= MIN_ENSEMBLE_UNIQUE
            for tree in tree_results
        ),
        "brier_gain_vs_pts_at_least_5_percent": (
            pts["relative_brier_reduction"] >= 0.05
        ),
        "brier_cluster_ci_vs_pts_below_zero": (
            pts["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "brier_wins_vs_pts_on_at_least_6_trees": (
            pts["strict_tree_wins"] >= 6
        ),
        "brier_gain_vs_original_at_least_2_percent": (
            original["relative_brier_reduction"] >= 0.02
        ),
        "brier_cluster_ci_vs_original_not_above_zero": (
            original["tree_cluster_brier_difference_95pct_bootstrap"][1]
            <= 0.0
        ),
        "brier_wins_vs_original_on_at_least_3_trees": (
            original["strict_tree_wins"] >= 3
        ),
        "brier_gain_vs_myopic_at_least_10_percent": (
            myopic["relative_brier_reduction"] >= 0.10
        ),
        "brier_cluster_ci_vs_myopic_below_zero": (
            myopic["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "no_mean_coverage_loss_vs_myopic": (
            myopic["mean_coverage_difference"] >= 0.0
        ),
    }
    public = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_trees_sha256": SOURCE_TREES_SHA256,
            "model": PLANNING_MODEL_ID,
            "temperature": TEMPERATURE,
            "reasoning": "disabled",
            "extra_prior_seeds": list(EXTRA_PRIOR_SEEDS),
            "extra_draws_per_tree": EXTRA_DRAWS_PER_TREE,
        },
        "raw_responses_sha256": sha256_file(raw_path),
        "draws": public_draws,
    }
    checkpoint(output_dir / "EXTRA_PRIORS.json", public)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "development_failed",
        "protocol": public["protocol"],
        "gates": gates,
        "draw_diagnostics": draw_diagnostics,
        "aggregate": aggregate,
        "trees": tree_results,
        "usage": usage,
        "extra_priors_sha256": sha256_file(
            output_dir / "EXTRA_PRIORS.json"
        ),
        "raw_responses_sha256": public["raw_responses_sha256"],
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-trees", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_development(
        source_result=args.source_result.resolve(),
        source_trees=args.source_trees.resolve(),
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
