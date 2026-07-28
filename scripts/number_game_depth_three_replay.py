#!/usr/bin/env python3
"""Replay frozen depth-three Number Game responses after local aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_depth_three_development as depth
from scripts.number_game_predictive_risk_replication import (
    aggregate_tree_comparisons,
)


class ReplayAdapter:
    def __init__(
        self,
        batches: list[list[str]],
        usage: dict[str, Any],
    ) -> None:
        self.batches = list(batches)
        self.usage = usage

    def chat_complete_messages_batched_structured(
        self,
        messages,
        **kwargs,
    ) -> list[str]:
        del kwargs
        if not self.batches:
            raise ValueError("replay adapter has no response batch")
        responses = self.batches.pop(0)
        if len(responses) != len(messages):
            raise ValueError("replay response count does not match messages")
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return dict(self.usage)


def _usage_snapshot(
    *,
    requests: int,
    cost: float,
) -> dict[str, Any]:
    return {
        "adapter_requests": requests,
        "http_attempts": requests,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "forced_final_requests": 0,
        "forced_final_successes": 0,
        "adapter_prompt_tokens": 0,
        "adapter_completion_tokens": 0,
        "adapter_cost_usd": cost,
        "provider_error_retries": 0,
    }


def replay_development(
    *,
    raw_path: Path,
    run_log_path: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    raw_document = json.loads(raw_path.read_text())
    if len(raw_document["trees"]) != len(depth.TREE_SEEDS):
        raise ValueError("raw replay does not contain all frozen trees")
    usage_rows = [
        json.loads(line)
        for line in run_log_path.read_text().splitlines()
        if line.strip()
    ]
    if (
        len(usage_rows) != depth.EXPECTED_REQUESTS
        or any(row.get("event") != "llm_token_usage" for row in usage_rows)
        or any(row.get("finish_reasons") != ["stop"] for row in usage_rows)
    ):
        raise ValueError("run log does not contain 400 clean usage rows")

    trees = []
    public_trees = []
    original_adapter = depth._adapter
    try:
        for index, raw_tree in enumerate(raw_document["trees"]):
            rows = usage_rows[
                index
                * depth.EXPECTED_REQUESTS_PER_TREE : (index + 1)
                * depth.EXPECTED_REQUESTS_PER_TREE
            ]
            planning_cost = sum(
                float(row["cost_usd"]) for row in rows[:-1]
            )
            target_cost = float(rows[-1]["cost_usd"])
            planning = ReplayAdapter(
                [
                    [raw_tree["initial_response"]],
                    [
                        item["response"]
                        for item in raw_tree["first_responses"]
                    ],
                    [
                        item["response"]
                        for item in raw_tree["second_responses"]
                    ],
                ],
                _usage_snapshot(
                    requests=depth.EXPECTED_REQUESTS_PER_TREE - 1,
                    cost=planning_cost,
                ),
            )
            target = ReplayAdapter(
                [[raw_tree["target_response"]]],
                _usage_snapshot(requests=1, cost=target_cost),
            )
            adapters = iter((planning, target))
            depth._adapter = lambda **kwargs: next(adapters)
            tree, artifacts = depth.run_tree_depth_three(
                tree_index=index,
                tree_seed=depth.TREE_SEEDS[index],
                target_seed=depth.TARGET_SEEDS[index],
                output_dir=output_dir,
                run_id=run_id,
            )
            if planning.batches or target.batches:
                raise ValueError("replay left unused response batches")
            trees.append(tree)
            public_trees.append(artifacts["public"])
    finally:
        depth._adapter = original_adapter

    baselines = (
        "predictive_bayes_risk_depth_two",
        "myopic_eig",
        "fixed_support_depth_three",
        "uniform_random_candidate_root",
        "positive_test_strategy",
    )
    aggregate = {
        baseline: aggregate_tree_comparisons(
            trees,
            baseline=baseline,
            candidate="predictive_bayes_risk_depth_three",
        )
        for baseline in baselines
    }
    total_cost = sum(float(row["cost_usd"]) for row in usage_rows)
    usage = {
        "adapter_requests": depth.EXPECTED_REQUESTS,
        "http_attempts": depth.EXPECTED_REQUESTS,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": sum(
            int(row["reasoning_tokens"]) for row in usage_rows
        ),
        "forced_exits": 0,
        "run_cost_usd": total_cost,
    }
    mechanics = {
        "exact_400_accepted_requests": True,
        "transport_attempt_accounting_exact": True,
        "zero_reasoning_tokens": (
            usage["adapter_reasoning_tokens"] == 0
        ),
        "zero_forced_exits": True,
        "within_run_budget": total_cost <= depth.RUN_BUDGET_USD,
        "all_initial_supports_valid": all(
            tree["mechanics"]["initial_valid"] >= depth.MIN_INITIAL_VALID
            for tree in trees
        ),
        "all_first_branches_valid": all(
            tree["mechanics"]["minimum_first_branch_valid"]
            >= depth.MIN_FIRST_BRANCH_VALID
            for tree in trees
        ),
        "all_second_branches_valid": all(
            tree["mechanics"]["minimum_second_branch_valid"]
            >= depth.MIN_SECOND_BRANCH_VALID
            for tree in trees
        ),
        "all_target_supports_valid": all(
            tree["mechanics"]["target_valid"] >= depth.MIN_TARGET_VALID
            and tree["mechanics"]["novel_targets"]
            >= depth.MIN_NOVEL_TARGETS
            for tree in trees
        ),
    }
    public = {
        "schema_version": depth.SCHEMA_VERSION,
        "protocol": {
            "interface_version": depth.INTERFACE_VERSION,
            "planning_model": depth.PLANNING_MODEL_ID,
            "target_model": depth.TARGET_MODEL_ID,
            "reasoning": "disabled",
            "temperature": depth.TEMPERATURE,
            "tree_seeds": list(depth.TREE_SEEDS),
            "target_seeds": list(depth.TARGET_SEEDS),
            "num_trees": len(depth.TREE_SEEDS),
            "requests_per_tree": depth.EXPECTED_REQUESTS_PER_TREE,
            "replayed_from_frozen_raw": True,
        },
        "raw_responses_sha256": hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest(),
        "run_log_sha256": hashlib.sha256(
            run_log_path.read_bytes()
        ).hexdigest(),
        "trees": public_trees,
    }
    trees_path = output_dir / "TREES.json"
    checkpoint(trees_path, public)
    result = {
        "schema_version": depth.SCHEMA_VERSION,
        "status": (
            "development_complete"
            if all(mechanics.values())
            else "mechanics_failed"
        ),
        "protocol": public["protocol"],
        "mechanics": mechanics,
        "root_differences": {
            "depth_three_vs_depth_two": sum(
                tree["selection"][
                    "predictive_bayes_risk_depth_three_root"
                ]
                != tree["selection"][
                    "predictive_bayes_risk_depth_two_root"
                ]
                for tree in trees
            ),
            "depth_three_vs_myopic": sum(
                tree["selection"][
                    "predictive_bayes_risk_depth_three_root"
                ]
                != tree["selection"]["myopic_root"]
                for tree in trees
            ),
        },
        "aggregate": aggregate,
        "usage": usage,
        "trees": trees,
        "trees_sha256": hashlib.sha256(
            trees_path.read_bytes()
        ).hexdigest(),
        "raw_responses_sha256": public["raw_responses_sha256"],
        "run_log_sha256": public["run_log_sha256"],
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = replay_development(
        raw_path=args.raw.resolve(),
        run_log_path=args.run_log.resolve(),
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
