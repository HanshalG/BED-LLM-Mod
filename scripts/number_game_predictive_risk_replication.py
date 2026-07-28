#!/usr/bin/env python3
"""Replicate predictive-risk Number Game BED on independent proposal trees."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_dark_matter_structured_replication_v3 import (
    DefaultRoutingStructuredAdapter,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    MAX_TOKENS,
    NUM_ROOTS,
    RuleHypothesis,
    branch_messages,
    candidate_roots,
    choose_predictive_bayes_risk_root,
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


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-predictive-risk-replication-2"
PLANNING_MODEL_ID = "google/gemini-2.5-flash"
TARGET_MODEL_ID = "openai/gpt-5.4"
TREE_SEEDS = tuple(range(26080, 26088))
TARGET_SEEDS = tuple(range(26180, 26188))
TEMPERATURE = 0.7
EXPECTED_REQUESTS_PER_TREE = 18
EXPECTED_REQUESTS = len(TREE_SEEDS) * EXPECTED_REQUESTS_PER_TREE
MIN_INITIAL_VALID = 16
MIN_BRANCH_VALID = 8
MIN_TARGET_VALID = 16
MIN_NOVEL_TARGETS = 8
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 26270
RUN_BUDGET_USD = 2.00
PROJECTED_PLANNING_COST_USD = 0.10
PROJECTED_TARGET_COST_USD = 0.08


class SeededStructuredAdapter(DefaultRoutingStructuredAdapter):
    def __init__(
        self,
        spec: ModelSpec,
        config: Config,
        *,
        request_seed: int,
    ) -> None:
        super().__init__(spec, config)
        self.request_seed = request_seed
        self.provider_error_retries = 0

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=disable_reasoning,
            response_format=response_format,
        )
        payload["seed"] = self.request_seed
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            data = super()._post(payload)
            choices = data.get("choices")
            provider_error = (
                isinstance(choices, list)
                and choices
                and any(
                    choice.get("finish_reason") == "error"
                    for choice in choices
                    if isinstance(choice, dict)
                )
            )
            if not provider_error:
                return data
            cost = float((data.get("usage") or {}).get("cost", 0.0) or 0.0)
            if cost > 1e-12:
                raise RuntimeError(
                    "provider-error response reported nonzero cost"
                )
            if attempt >= self.max_retries:
                raise RuntimeError(
                    "provider-error response persisted after retries"
                )
            with self._usage_lock:
                self.retry_count += 1
                self.provider_error_retries += 1
            time.sleep(self.backoff_seconds * (2**attempt))
        raise AssertionError("unreachable")

    def usage_snapshot(self) -> dict[str, Any]:
        snapshot = super().usage_snapshot()
        snapshot["provider_error_retries"] = self.provider_error_retries
        return snapshot


def _adapter(
    *,
    model: str,
    run_id: str,
    output_dir: Path,
    request_seed: int,
    concurrency: int,
    projected_cost: float,
) -> SeededStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=200.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=projected_cost,
        openrouter_concurrency=concurrency,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return SeededStructuredAdapter(
        ModelSpec(model=model, backend="openrouter", max_model_len=65536),
        config,
        request_seed=request_seed,
    )


def _tree_usage(
    planning: SeededStructuredAdapter,
    target: SeededStructuredAdapter,
) -> dict[str, Any]:
    snapshots = [planning.usage_snapshot(), target.usage_snapshot()]
    summed = {}
    for key in (
        "adapter_requests",
        "http_attempts",
        "retry_count",
        "adapter_reasoning_tokens",
        "forced_exits",
        "forced_final_requests",
        "forced_final_successes",
        "adapter_prompt_tokens",
        "adapter_completion_tokens",
        "adapter_cost_usd",
        "provider_error_retries",
    ):
        summed[key] = sum(float(item.get(key, 0) or 0) for item in snapshots)
    summed["adapter_requests"] = int(summed["adapter_requests"])
    summed["http_attempts"] = int(summed["http_attempts"])
    summed["retry_count"] = int(summed["retry_count"])
    summed["adapter_reasoning_tokens"] = int(
        summed["adapter_reasoning_tokens"]
    )
    summed["forced_exits"] = int(summed["forced_exits"])
    summed["run_cost_usd"] = summed.pop("adapter_cost_usd")
    return summed


def cluster_bootstrap_interval(
    tree_values: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    if not tree_values:
        raise ValueError("tree values are empty")
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(tree_values) for _ in tree_values) / len(tree_values)
        for _ in range(samples)
    )
    return [
        means[int(0.025 * samples)],
        means[min(samples - 1, int(0.975 * samples))],
    ]


def aggregate_tree_comparisons(
    trees: Sequence[dict[str, Any]],
    *,
    baseline: str,
) -> dict[str, Any]:
    differences = [
        tree["comparisons"][baseline][
            "candidate_minus_baseline_brier"
        ]
        for tree in trees
    ]
    hamming_differences = [
        tree["comparisons"][baseline][
            "candidate_minus_baseline_hamming"
        ]
        for tree in trees
    ]
    coverage_differences = [
        tree["comparisons"][baseline]["coverage_difference"]
        for tree in trees
    ]
    candidate_brier = sum(
        tree["endpoint"]["predictive_bayes_risk"][
            "mean_posterior_predictive_brier"
        ]
        for tree in trees
    ) / len(trees)
    baseline_brier = sum(
        tree["endpoint"][baseline]["mean_posterior_predictive_brier"]
        for tree in trees
    ) / len(trees)
    candidate_hamming = sum(
        tree["endpoint"]["predictive_bayes_risk"][
            "mean_best_hamming_error"
        ]
        for tree in trees
    ) / len(trees)
    baseline_hamming = sum(
        tree["endpoint"][baseline]["mean_best_hamming_error"]
        for tree in trees
    ) / len(trees)
    return {
        "candidate_mean_brier": candidate_brier,
        "baseline_mean_brier": baseline_brier,
        "relative_brier_reduction": (
            (baseline_brier - candidate_brier) / baseline_brier
        ),
        "mean_candidate_minus_baseline_brier": sum(differences)
        / len(differences),
        "tree_cluster_brier_difference_95pct_bootstrap": (
            cluster_bootstrap_interval(differences)
        ),
        "brier_tree_wins": sum(value < 0.0 for value in differences),
        "candidate_mean_hamming": candidate_hamming,
        "baseline_mean_hamming": baseline_hamming,
        "relative_hamming_reduction": (
            (baseline_hamming - candidate_hamming) / baseline_hamming
        ),
        "mean_candidate_minus_baseline_hamming": sum(
            hamming_differences
        )
        / len(hamming_differences),
        "tree_cluster_hamming_difference_95pct_bootstrap": (
            cluster_bootstrap_interval(
                hamming_differences,
                seed=BOOTSTRAP_SEED + 1,
            )
        ),
        "hamming_tree_wins": sum(
            value < 0.0 for value in hamming_differences
        ),
        "mean_coverage_difference": sum(coverage_differences)
        / len(coverage_differences),
    }


def run_tree(
    *,
    tree_index: int,
    tree_seed: int,
    target_seed: int,
    output_dir: Path,
    run_id: str,
    planning_model: str = PLANNING_MODEL_ID,
    target_model: str = TARGET_MODEL_ID,
    planning_concurrency: int = 16,
    target_concurrency: int = 1,
    projected_planning_cost: float = PROJECTED_PLANNING_COST_USD,
    projected_target_cost: float = PROJECTED_TARGET_COST_USD,
) -> tuple[dict[str, Any], dict[str, Any]]:
    planning_adapter = _adapter(
        model=planning_model,
        run_id=f"{run_id}-tree{tree_index}-planning",
        output_dir=output_dir,
        request_seed=tree_seed,
        concurrency=planning_concurrency,
        projected_cost=projected_planning_cost,
    )
    target_adapter = _adapter(
        model=target_model,
        run_id=f"{run_id}-tree{tree_index}-target",
        output_dir=output_dir,
        request_seed=target_seed,
        concurrency=target_concurrency,
        projected_cost=projected_target_cost,
    )
    initial_response = planning_adapter.chat_complete_messages_batched_structured(
        [initial_messages()],
        temperature=TEMPERATURE,
        block_size=1,
        response_format=proposal_response_format(),
        max_new_tokens=MAX_TOKENS,
    )[0]
    initial, initial_diagnostics = parse_proposals(initial_response)
    if len(initial) < MIN_INITIAL_VALID:
        raise ValueError(
            f"tree {tree_index} has only {len(initial)} initial rules"
        )
    roots, candidate_metadata = candidate_roots(initial, seed=tree_seed)
    branch_keys = [
        (root, label) for root in roots for label in (False, True)
    ]
    branch_responses = (
        planning_adapter.chat_complete_messages_batched_structured(
            [
                branch_messages(root, label)
                for root, label in branch_keys
            ],
            temperature=TEMPERATURE,
            block_size=len(branch_keys),
            response_format=proposal_response_format(),
            max_new_tokens=MAX_TOKENS,
        )
    )
    branches = {}
    branch_diagnostics = {}
    for (root, label), response in zip(
        branch_keys, branch_responses, strict=True
    ):
        hypotheses, diagnostics = parse_proposals(
            response, observations=((root, label),)
        )
        branches[(root, label)] = hypotheses
        branch_diagnostics[f"{root}:{int(label)}"] = diagnostics
    target_response = target_adapter.chat_complete_messages_batched_structured(
        [initial_messages()],
        temperature=TEMPERATURE,
        block_size=1,
        response_format=proposal_response_format(),
        max_new_tokens=MAX_TOKENS,
    )[0]
    target_list, target_diagnostics = parse_proposals(target_response)
    targets = {
        f"target_{index:02d}_{hypothesis.name}": hypothesis
        for index, hypothesis in enumerate(target_list)
    }
    initial_extensions = {hypothesis.extension for hypothesis in initial}
    novel_targets = {
        name: hypothesis
        for name, hypothesis in targets.items()
        if hypothesis.extension not in initial_extensions
    }
    source_scores = predictive_bayes_risk_scores(
        support=initial,
        roots=roots,
        branches=branches,
    )
    predictive_root = choose_predictive_bayes_risk_root(source_scores)
    policy_roots = {
        "predictive_bayes_risk": predictive_root,
        "myopic_eig": int(candidate_metadata["myopic_root"]),
        "fixed_support_depth_two": int(
            candidate_metadata["fixed_depth_two_root"]
        ),
    }
    per_root = {
        root: evaluate_policy_root(
            policy=f"root_{root}",
            root=root,
            targets=targets,
            branches=branches,
        )
        for root in roots
    }
    endpoint = {
        policy: per_root[root] | {"policy": policy}
        for policy, root in policy_roots.items()
    }
    endpoint["uniform_random_candidate_root"] = (
        aggregate_random_root_control(per_root)
    )
    pts_roots = [int(root) for root in candidate_metadata["pts_roots"]]
    endpoint["positive_test_strategy"] = aggregate_random_root_control(
        {root: per_root[root] for root in pts_roots}
    ) | {"policy": "uniform_over_two_seeded_pts_roots"}
    comparisons = {
        baseline: policy_comparison(
            endpoint["predictive_bayes_risk"],
            endpoint[baseline],
        )
        for baseline in (
            "myopic_eig",
            "fixed_support_depth_two",
            "uniform_random_candidate_root",
            "positive_test_strategy",
        )
    }
    novel_endpoint = {
        policy: evaluate_policy_root(
            policy=policy,
            root=root,
            targets=novel_targets,
            branches=branches,
        )
        for policy, root in policy_roots.items()
    }
    novel_comparison = policy_comparison(
        novel_endpoint["predictive_bayes_risk"],
        novel_endpoint["myopic_eig"],
    )
    usage = _tree_usage(planning_adapter, target_adapter)
    mechanics = {
        "initial_valid": len(initial),
        "minimum_branch_valid": min(
            item["valid_unique_count"]
            for item in branch_diagnostics.values()
        ),
        "target_valid": len(targets),
        "novel_targets": len(novel_targets),
        "exact_18_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS_PER_TREE
        ),
        "exact_attempt_accounting": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
    }
    tree_result = {
        "tree_index": tree_index,
        "tree_seed": tree_seed,
        "target_seed": target_seed,
        "mechanics": mechanics,
        "initial_diagnostics": initial_diagnostics,
        "branch_diagnostics": branch_diagnostics,
        "target_diagnostics": {
            **target_diagnostics,
            "novel_unique_count": len(novel_targets),
        },
        "selection": {
            **candidate_metadata,
            "predictive_bayes_risk_root": predictive_root,
            "pts_roots": pts_roots,
        },
        "source_predictive_risk": {
            str(root): {
                key: value
                for key, value in source_scores[root].items()
                if key != "targets"
            }
            for root in roots
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
        "novel_endpoint": {
            policy: {
                key: value
                for key, value in result.items()
                if key != "targets"
            }
            for policy, result in novel_endpoint.items()
        },
        "novel_comparison_vs_myopic": novel_comparison,
        "usage": usage,
    }
    raw = {
        "tree_index": tree_index,
        "tree_seed": tree_seed,
        "target_seed": target_seed,
        "initial_response": initial_response,
        "branch_responses": [
            {
                "root": root,
                "label": label,
                "response": response,
            }
            for (root, label), response in zip(
                branch_keys, branch_responses, strict=True
            )
        ],
        "target_response": target_response,
    }
    public = {
        "tree_index": tree_index,
        "tree_seed": tree_seed,
        "target_seed": target_seed,
        "initial": [hypothesis.public_dict() for hypothesis in initial],
        "roots": roots,
        "branches": {
            f"{root}:{int(label)}": [
                hypothesis.public_dict()
                for hypothesis in branches[(root, label)]
            ]
            for root, label in branch_keys
        },
        "targets": [
            {
                **hypothesis.public_dict(),
                "novel_to_planning_support": (
                    hypothesis.extension not in initial_extensions
                ),
            }
            for hypothesis in target_list
        ],
    }
    return tree_result, {"raw": raw, "public": public}


def run_replication(
    *,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"trees": []}
    public_trees: list[dict[str, Any]] = []
    tree_results: list[dict[str, Any]] = []
    try:
        for tree_index, (tree_seed, target_seed) in enumerate(
            zip(TREE_SEEDS, TARGET_SEEDS, strict=True)
        ):
            tree, artifacts = run_tree(
                tree_index=tree_index,
                tree_seed=tree_seed,
                target_seed=target_seed,
                output_dir=output_dir,
                run_id=run_id,
            )
            tree_results.append(tree)
            raw["trees"].append(artifacts["raw"])
            public_trees.append(artifacts["public"])
            checkpoint(raw_path, raw)
        aggregate = {
            baseline: aggregate_tree_comparisons(
                tree_results, baseline=baseline
            )
            for baseline in (
                "myopic_eig",
                "fixed_support_depth_two",
                "uniform_random_candidate_root",
                "positive_test_strategy",
            )
        }
        usage = {
            key: sum(
                tree["usage"].get(key, 0) for tree in tree_results
            )
            for key in (
                "adapter_requests",
                "http_attempts",
                "retry_count",
                "provider_error_retries",
                "adapter_reasoning_tokens",
                "forced_exits",
                "run_cost_usd",
            )
        }
        roots_differ_myopic = sum(
            tree["selection"]["predictive_bayes_risk_root"]
            != tree["selection"]["myopic_root"]
            for tree in tree_results
        )
        roots_differ_fixed = sum(
            tree["selection"]["predictive_bayes_risk_root"]
            != tree["selection"]["fixed_depth_two_root"]
            for tree in tree_results
        )
        novel_brier_differences = [
            tree["novel_comparison_vs_myopic"][
                "candidate_minus_baseline_brier"
            ]
            for tree in tree_results
        ]
        novel_hamming_differences = [
            tree["novel_comparison_vs_myopic"][
                "candidate_minus_baseline_hamming"
            ]
            for tree in tree_results
        ]
        gates = {
            "exact_144_accepted_requests": (
                usage["adapter_requests"] == EXPECTED_REQUESTS
            ),
            "transport_attempt_accounting_exact": (
                usage["http_attempts"]
                == usage["adapter_requests"] + usage["retry_count"]
            ),
            "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
            "zero_forced_exits": usage["forced_exits"] == 0,
            "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
            "all_trees_have_at_least_16_initial_rules": all(
                tree["mechanics"]["initial_valid"] >= MIN_INITIAL_VALID
                for tree in tree_results
            ),
            "all_branches_have_at_least_8_rules": all(
                tree["mechanics"]["minimum_branch_valid"]
                >= MIN_BRANCH_VALID
                for tree in tree_results
            ),
            "all_trees_have_at_least_16_targets_and_8_novel": all(
                tree["mechanics"]["target_valid"] >= MIN_TARGET_VALID
                and tree["mechanics"]["novel_targets"]
                >= MIN_NOVEL_TARGETS
                for tree in tree_results
            ),
            "predictive_root_differs_from_myopic_on_at_least_6_trees": (
                roots_differ_myopic >= 6
            ),
            "predictive_root_differs_from_fixed_on_at_least_6_trees": (
                roots_differ_fixed >= 6
            ),
            "brier_gain_vs_myopic_at_least_5_percent": (
                aggregate["myopic_eig"]["relative_brier_reduction"]
                >= 0.05
            ),
            "brier_cluster_ci_vs_myopic_below_zero": (
                aggregate["myopic_eig"][
                    "tree_cluster_brier_difference_95pct_bootstrap"
                ][1]
                < 0.0
            ),
            "brier_wins_vs_myopic_on_at_least_6_trees": (
                aggregate["myopic_eig"]["brier_tree_wins"] >= 6
            ),
            "hamming_gain_vs_myopic_at_least_5_percent": (
                aggregate["myopic_eig"]["relative_hamming_reduction"]
                >= 0.05
            ),
            "hamming_cluster_ci_vs_myopic_below_zero": (
                aggregate["myopic_eig"][
                    "tree_cluster_hamming_difference_95pct_bootstrap"
                ][1]
                < 0.0
            ),
            "no_mean_coverage_loss_vs_myopic": (
                aggregate["myopic_eig"]["mean_coverage_difference"] >= 0.0
            ),
            "brier_gain_vs_fixed_at_least_5_percent": (
                aggregate["fixed_support_depth_two"][
                    "relative_brier_reduction"
                ]
                >= 0.05
            ),
            "brier_cluster_ci_vs_fixed_below_zero": (
                aggregate["fixed_support_depth_two"][
                    "tree_cluster_brier_difference_95pct_bootstrap"
                ][1]
                < 0.0
            ),
            "brier_gain_vs_random_at_least_5_percent": (
                aggregate["uniform_random_candidate_root"][
                    "relative_brier_reduction"
                ]
                >= 0.05
            ),
            "brier_gain_vs_pts_at_least_5_percent": (
                aggregate["positive_test_strategy"][
                    "relative_brier_reduction"
                ]
                >= 0.05
            ),
            "brier_cluster_ci_vs_pts_below_zero": (
                aggregate["positive_test_strategy"][
                    "tree_cluster_brier_difference_95pct_bootstrap"
                ][1]
                < 0.0
            ),
            "novel_targets_have_mean_brier_and_hamming_gains": (
                sum(novel_brier_differences) < 0.0
                and sum(novel_hamming_differences) < 0.0
            ),
        }
        public = {
            "schema_version": SCHEMA_VERSION,
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "planning_model": PLANNING_MODEL_ID,
                "target_model": TARGET_MODEL_ID,
                "reasoning": "disabled",
                "temperature": TEMPERATURE,
                "tree_seeds": list(TREE_SEEDS),
                "target_seeds": list(TARGET_SEEDS),
                "num_trees": len(TREE_SEEDS),
                "requests_per_tree": EXPECTED_REQUESTS_PER_TREE,
            },
            "raw_responses_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
            "trees": public_trees,
        }
        checkpoint(output_dir / "TREES.json", public)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(gates.values()) else "replication_failed",
            "protocol": public["protocol"],
            "gates": gates,
            "root_differences": {
                "versus_myopic": roots_differ_myopic,
                "versus_fixed_support_depth_two": roots_differ_fixed,
            },
            "aggregate": aggregate,
            "novel_target_mean_differences": {
                "candidate_minus_myopic_brier": sum(
                    novel_brier_differences
                )
                / len(novel_brier_differences),
                "candidate_minus_myopic_hamming": sum(
                    novel_hamming_differences
                )
                / len(novel_hamming_differences),
                "brier_tree_wins": sum(
                    value < 0.0 for value in novel_brier_differences
                ),
                "hamming_tree_wins": sum(
                    value < 0.0 for value in novel_hamming_differences
                ),
            },
            "trees": tree_results,
            "usage": usage,
            "trees_sha256": hashlib.sha256(
                (output_dir / "TREES.json").read_bytes()
            ).hexdigest(),
            "raw_responses_sha256": public["raw_responses_sha256"],
        }
        checkpoint(output_dir / "RESULT.json", result)
        return result
    except Exception as exc:
        checkpoint(raw_path, raw)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "planning_model": PLANNING_MODEL_ID,
                "target_model": TARGET_MODEL_ID,
            },
            "error": f"{type(exc).__name__}: {exc}",
            "completed_trees": len(tree_results),
            "raw_responses_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
        }
        checkpoint(output_dir / "FAILURE.json", failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_replication(
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
