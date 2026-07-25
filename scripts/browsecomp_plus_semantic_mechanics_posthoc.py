#!/usr/bin/env python3
"""Reconstruct first-link metrics from the closed BrowseComp-Plus v2 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import browsecomp_plus_semantic_mechanics as mechanics


RAW_SHA256 = (
    "73f93e58e64706332c60477363aeed0b3134387008dd80026acdeab485230dea"
)
FAILURE_SHA256 = (
    "6272008d7da3730ea77dddcf76d3330268ee3a2d765ea86544346ae9492fb1d2"
)


def reconstruct(
    *,
    source_path: Path,
    raw_path: Path,
) -> dict[str, Any]:
    if mechanics.sha256_file(raw_path) != RAW_SHA256:
        raise ValueError("v2 raw response hash changed")
    tasks = mechanics.load_tasks(source_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if {
        key: len(raw.get(key, []))
        for key in (
            "initial_responses",
            "refresh_responses",
            "scorer_responses",
            "terminal_responses",
        )
    } != {
        "initial_responses": 5,
        "refresh_responses": 30,
        "scorer_responses": 10,
        "terminal_responses": 10,
    }:
        raise ValueError("v2 response batch counts changed")

    retrievers = [
        mechanics.TaskBM25(mechanics._task_documents(task))
        for task in tasks
    ]
    initials = [
        mechanics.parse_initial(response)
        for response in raw["initial_responses"]
    ]
    branches: list[list[mechanics.Branch]] = []
    for task_index, (_, strategies) in enumerate(initials):
        branches.append(
            [
                mechanics.Branch(
                    root_index=root_index,
                    strategy=strategy,
                    root_documents=retrievers[task_index].search(
                        strategy.root_query
                    ),
                )
                for root_index, strategy in enumerate(strategies)
            ]
        )
    locations = [
        (task_index, root_index)
        for task_index in range(len(tasks))
        for root_index in range(mechanics.ROOT_COUNT)
    ]
    for response, (task_index, root_index) in zip(
        raw["refresh_responses"],
        locations,
        strict=True,
    ):
        branch = branches[task_index][root_index]
        branch.root_belief, branch.adaptive_query = (
            mechanics.parse_refresh(response)
        )
        branch.adaptive_documents = retrievers[task_index].search(
            branch.adaptive_query
        )
    scores: list[dict[str, list[int]]] = [{} for _ in tasks]
    scorer_locations = [
        (task_index, variant)
        for task_index in range(len(tasks))
        for variant in ("aligned", "shuffled")
    ]
    for response, (task_index, variant) in zip(
        raw["scorer_responses"],
        scorer_locations,
        strict=True,
    ):
        scores[task_index][variant] = mechanics.parse_future_scores(response)
    return {
        "tasks": tasks,
        "initials": initials,
        "branches": branches,
        "scores": scores,
        "terminal_responses": raw["terminal_responses"],
    }


def analyze(reconstructed: dict[str, Any]) -> dict[str, Any]:
    totals = {
        name: {"points": 0.0, "pairs": 0}
        for name in (
            "direct_immediate",
            "future_gain",
            "full_total",
            "direct_total",
            "shuffled_total",
        )
    }
    changed_beliefs = 0
    adaptive_differences = 0
    evidence_gain_branches = 0
    root_differences = 0
    strategy_wins = 0
    strategy_losses = 0
    records = []

    for task_index, task in enumerate(reconstructed["tasks"]):
        evidence = {
            str(document["docid"]) for document in task["evidence_docs"]
        }
        gold = {str(document["docid"]) for document in task["gold_docs"]}
        initial_belief = reconstructed["initials"][task_index][0]
        branches = reconstructed["branches"][task_index]
        direct = [branch.strategy.direct_score for branch in branches]
        future = reconstructed["scores"][task_index]["aligned"]
        shuffled = reconstructed["scores"][task_index]["shuffled"]
        full = [left + right for left, right in zip(direct, future)]
        shuffled_full = [
            left + right for left, right in zip(direct, shuffled)
        ]
        immediate = []
        future_gain = []
        total = []
        total_gold = []
        for branch in branches:
            assert branch.root_belief is not None
            assert branch.adaptive_query is not None
            assert branch.adaptive_documents is not None
            root_ids = {
                str(document["docid"])
                for document in branch.root_documents
            }
            adaptive_ids = {
                str(document["docid"])
                for document in branch.adaptive_documents
            }
            root_value = len(root_ids & evidence)
            total_value = len((root_ids | adaptive_ids) & evidence)
            immediate.append(root_value)
            future_gain.append(total_value - root_value)
            total.append(total_value)
            total_gold.append(len((root_ids | adaptive_ids) & gold))
            changed_beliefs += (
                branch.root_belief.signature != initial_belief.signature
            )
            adaptive_differences += (
                mechanics.normalize_query(branch.adaptive_query)
                != mechanics.normalize_query(branch.strategy.root_query)
            )
            evidence_gain_branches += total_value > root_value

        for name, score_values, endpoint_values in (
            ("direct_immediate", direct, immediate),
            ("future_gain", future, future_gain),
            ("full_total", full, total),
            ("direct_total", direct, total),
            ("shuffled_total", shuffled_full, total),
        ):
            accuracy, pairs = mechanics._pairwise_accuracy(
                score_values,
                endpoint_values,
            )
            totals[name]["points"] += accuracy * pairs
            totals[name]["pairs"] += pairs

        myopic_root = mechanics._argmax(direct)
        strategy_root = mechanics._argmax(full)
        root_differences += strategy_root != myopic_root
        strategy_wins += total[strategy_root] > total[myopic_root]
        strategy_losses += total[strategy_root] < total[myopic_root]
        records.append(
            {
                "task_id": str(task["query_id"]),
                "myopic_root": myopic_root,
                "strategy_root": strategy_root,
                "myopic_total_evidence": total[myopic_root],
                "strategy_total_evidence": total[strategy_root],
                "myopic_total_gold": total_gold[myopic_root],
                "strategy_total_gold": total_gold[strategy_root],
                "direct_scores": direct,
                "aligned_future_scores": future,
                "shuffled_future_scores": shuffled,
                "immediate_evidence": immediate,
                "future_evidence_gain": future_gain,
                "total_evidence": total,
            }
        )

    accuracies = {
        name: (
            values["points"] / values["pairs"]
            if values["pairs"]
            else 0.5
        )
        for name, values in totals.items()
    }
    terminal_failures = []
    for index, response in enumerate(
        reconstructed["terminal_responses"]
    ):
        try:
            mechanics.parse_terminal(response)
        except ValueError as exc:
            terminal_failures.append(
                {"index": index, "error": str(exc)}
            )
    return {
        "schema_version": 1,
        "status": "exploratory_null_closed",
        "protocol": {
            "interface_version": (
                "browsecomp-plus-semantic-mechanics-posthoc-1"
            ),
            "raw_sha256": RAW_SHA256,
            "failure_sha256": FAILURE_SHA256,
            "endpoint_analysis_preregistered": False,
            "development_authorized": False,
            "openrouter_calls": 0,
            "oatml_used": False,
        },
        "summary": {
            "initial_parse_count": 5,
            "refresh_parse_count": 30,
            "scorer_parse_count": 10,
            "terminal_parse_count": 10 - len(terminal_failures),
            "terminal_failure_count": len(terminal_failures),
            "changed_root_belief_count": changed_beliefs,
            "adaptive_query_difference_count": adaptive_differences,
            "evidence_gain_branch_count": evidence_gain_branches,
            "direct_immediate_pair_accuracy": accuracies[
                "direct_immediate"
            ],
            "future_gain_pair_accuracy": accuracies["future_gain"],
            "full_total_pair_accuracy": accuracies["full_total"],
            "direct_total_pair_accuracy": accuracies["direct_total"],
            "shuffled_full_total_pair_accuracy": accuracies[
                "shuffled_total"
            ],
            "full_minus_direct_total_accuracy": (
                accuracies["full_total"] - accuracies["direct_total"]
            ),
            "full_minus_shuffled_total_accuracy": (
                accuracies["full_total"] - accuracies["shuffled_total"]
            ),
            "comparable_pairs": {
                name: values["pairs"] for name, values in totals.items()
            },
            "strategy_root_difference_count": root_differences,
            "strategy_vs_myopic_evidence_wins": strategy_wins,
            "strategy_vs_myopic_evidence_losses": strategy_losses,
        },
        "terminal_failures": terminal_failures,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--failure-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if mechanics.sha256_file(args.failure_path) != FAILURE_SHA256:
        raise ValueError("v2 public failure hash changed")

    result = analyze(
        reconstruct(
            source_path=args.source_path,
            raw_path=args.raw_path,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
