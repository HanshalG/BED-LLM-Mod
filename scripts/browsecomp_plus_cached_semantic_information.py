#!/usr/bin/env python3
"""Audit target-blind semantic information scores on cached BrowseComp beliefs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import browsecomp_plus_semantic_mechanics as mechanics
from scripts import browsecomp_plus_semantic_mechanics_posthoc as posthoc


INTERFACE_VERSION = "browsecomp-plus-cached-semantic-information-1"
MIN_TRUTH_PAIRS = 20
MIN_IMMEDIATE_PAIRS = 30
MIN_TRUTH_ACCURACY = 0.60
MIN_IMMEDIATE_ACCURACY = 0.55


def _alias_tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.lower()))


def alias_equivalent(left: str, right: str) -> bool:
    """Return a deterministic, target-independent short-answer alias match."""
    left_normalized = mechanics.normalize_answer(left)
    right_normalized = mechanics.normalize_answer(right)
    if left_normalized == right_normalized:
        return True
    if (
        re.sub(r"[^a-z0-9]", "", left_normalized)
        == re.sub(r"[^a-z0-9]", "", right_normalized)
    ):
        return True
    left_tokens = _alias_tokens(left_normalized)
    right_tokens = _alias_tokens(right_normalized)
    smaller = min(len(left_tokens), len(right_tokens))
    return (
        smaller >= 2
        and (
            left_tokens.issubset(right_tokens)
            or right_tokens.issubset(left_tokens)
        )
    )


def semantic_cluster_ids(values: Sequence[str]) -> list[int]:
    """Cluster aliases by connected components without using task answers."""
    parents = list(range(len(values)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if alias_equivalent(values[left], values[right]):
                union(left, right)

    root_to_id: dict[int, int] = {}
    cluster_ids = []
    for index in range(len(values)):
        root = find(index)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        cluster_ids.append(root_to_id[root])
    return cluster_ids


def belief_distribution(
    belief: mechanics.Belief,
    *,
    value_to_cluster: dict[str, int],
    cluster_count: int,
) -> list[float]:
    probabilities = [0.0] * cluster_count
    for hypothesis, weight in zip(
        belief.hypotheses,
        belief.weights,
        strict=True,
    ):
        probabilities[value_to_cluster[hypothesis]] += weight / 100.0
    return probabilities


def entropy(probabilities: Iterable[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def js_divergence(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    midpoint = [
        (left_value + right_value) / 2.0
        for left_value, right_value in zip(left, right, strict=True)
    ]

    def kl_divergence(
        values: Sequence[float],
        reference: Sequence[float],
    ) -> float:
        return sum(
            value * math.log(value / reference_value)
            for value, reference_value in zip(
                values,
                reference,
                strict=True,
            )
            if value > 0.0
        )

    return (
        kl_divergence(left, midpoint)
        + kl_divergence(right, midpoint)
    ) / 2.0


def truth_mass(belief: mechanics.Belief, answer: str) -> float:
    return sum(
        weight / 100.0
        for hypothesis, weight in zip(
            belief.hypotheses,
            belief.weights,
            strict=True,
        )
        if alias_equivalent(hypothesis, answer)
    )


def _weighted_pairwise(
    records: Sequence[dict[str, Any]],
    *,
    score_key: str,
    endpoint_key: str,
) -> tuple[float, int]:
    points = 0.0
    pairs = 0
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)
    for task_records in by_task.values():
        accuracy, count = mechanics._pairwise_accuracy(
            [record[score_key] for record in task_records],
            [record[endpoint_key] for record in task_records],
        )
        points += accuracy * count
        pairs += count
    return (points / pairs if pairs else 0.5, pairs)


def analyze(reconstructed: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    dynamic_task_count = 0

    for task_index, task in enumerate(reconstructed["tasks"]):
        initial = reconstructed["initials"][task_index][0]
        branches = reconstructed["branches"][task_index]
        all_values = list(initial.hypotheses)
        for branch in branches:
            assert branch.root_belief is not None
            all_values.extend(branch.root_belief.hypotheses)
        cluster_ids = semantic_cluster_ids(all_values)
        value_to_cluster: dict[str, int] = {}
        for value, cluster_id in zip(all_values, cluster_ids, strict=True):
            value_to_cluster[value] = cluster_id
        cluster_count = max(cluster_ids) + 1
        initial_distribution = belief_distribution(
            initial,
            value_to_cluster=value_to_cluster,
            cluster_count=cluster_count,
        )
        initial_entropy = entropy(initial_distribution)
        initial_truth_mass = truth_mass(initial, task["answer"])
        evidence_ids = {
            str(document["docid"]) for document in task["evidence_docs"]
        }
        gold_ids = {
            str(document["docid"]) for document in task["gold_docs"]
        }
        task_records = []

        for branch in branches:
            assert branch.root_belief is not None
            assert branch.adaptive_documents is not None
            root_distribution = belief_distribution(
                branch.root_belief,
                value_to_cluster=value_to_cluster,
                cluster_count=cluster_count,
            )
            root_ids = {
                str(document["docid"])
                for document in branch.root_documents
            }
            adaptive_ids = {
                str(document["docid"])
                for document in branch.adaptive_documents
            }
            root_truth_mass = truth_mass(
                branch.root_belief,
                task["answer"],
            )
            record = {
                "task_id": str(task["query_id"]),
                "root_index": branch.root_index,
                "direct_score": branch.strategy.direct_score,
                "semantic_entropy_drop": (
                    initial_entropy - entropy(root_distribution)
                ),
                "js_divergence": js_divergence(
                    initial_distribution,
                    root_distribution,
                ),
                "novel_mass": sum(
                    root_value
                    for initial_value, root_value in zip(
                        initial_distribution,
                        root_distribution,
                        strict=True,
                    )
                    if initial_value == 0.0
                ),
                "top_cluster_mass_gain": (
                    max(root_distribution) - max(initial_distribution)
                ),
                "truth_mass_gain": (
                    root_truth_mass - initial_truth_mass
                ),
                "root_truth_mass": root_truth_mass,
                "immediate_evidence": len(root_ids & evidence_ids),
                "future_evidence_gain": (
                    len((root_ids | adaptive_ids) & evidence_ids)
                    - len(root_ids & evidence_ids)
                ),
                "total_evidence": len(
                    (root_ids | adaptive_ids) & evidence_ids
                ),
                "total_gold": len((root_ids | adaptive_ids) & gold_ids),
            }
            records.append(record)
            task_records.append(record)

        entropy_values = [
            record["semantic_entropy_drop"] for record in task_records
        ]
        dynamic_task_count += (
            max(entropy_values) - min(entropy_values) > 1e-12
        )
        entropy_root = mechanics._argmax(entropy_values)
        direct_root = mechanics._argmax(
            [record["direct_score"] for record in task_records]
        )
        selections.append(
            {
                "task_id": str(task["query_id"]),
                "entropy_root": entropy_root,
                "direct_root": direct_root,
                "entropy_root_truth_mass": task_records[entropy_root][
                    "root_truth_mass"
                ],
                "direct_root_truth_mass": task_records[direct_root][
                    "root_truth_mass"
                ],
                "entropy_root_immediate_evidence": task_records[
                    entropy_root
                ]["immediate_evidence"],
                "direct_root_immediate_evidence": task_records[
                    direct_root
                ]["immediate_evidence"],
            }
        )

    metric_names = (
        "semantic_entropy_drop",
        "js_divergence",
        "novel_mass",
        "top_cluster_mass_gain",
    )
    endpoint_names = (
        "truth_mass_gain",
        "immediate_evidence",
        "future_evidence_gain",
        "total_evidence",
        "total_gold",
    )
    pairwise = {
        metric_name: {
            endpoint_name: {
                "accuracy": accuracy,
                "pairs": pairs,
            }
            for endpoint_name in endpoint_names
            for accuracy, pairs in [
                _weighted_pairwise(
                    records,
                    score_key=metric_name,
                    endpoint_key=endpoint_name,
                )
            ]
        }
        for metric_name in metric_names
    }
    selection_wins = sum(
        selection["entropy_root_truth_mass"]
        > selection["direct_root_truth_mass"]
        for selection in selections
    )
    selection_losses = sum(
        selection["entropy_root_truth_mass"]
        < selection["direct_root_truth_mass"]
        for selection in selections
    )
    selection_nonworse = len(selections) - selection_losses
    primary = pairwise["semantic_entropy_drop"]
    gates = {
        "exact_30_roots": len(records) == 30,
        "entropy_score_dynamic_on_5_tasks": dynamic_task_count == 5,
        "at_least_20_truth_pairs": (
            primary["truth_mass_gain"]["pairs"] >= MIN_TRUTH_PAIRS
        ),
        "at_least_30_immediate_pairs": (
            primary["immediate_evidence"]["pairs"]
            >= MIN_IMMEDIATE_PAIRS
        ),
        "truth_mass_pair_accuracy_at_least_0_60": (
            primary["truth_mass_gain"]["accuracy"]
            >= MIN_TRUTH_ACCURACY
        ),
        "immediate_pair_accuracy_at_least_0_55": (
            primary["immediate_evidence"]["accuracy"]
            >= MIN_IMMEDIATE_ACCURACY
        ),
        "entropy_selection_truth_nonworse_on_at_least_4_tasks": (
            selection_nonworse >= 4
        ),
        "entropy_selection_truth_wins_at_least_1_task": (
            selection_wins >= 1
        ),
        "entropy_selection_truth_losses_at_most_1_task": (
            selection_losses <= 1
        ),
    }
    return {
        "schema_version": 1,
        "status": (
            "directional_first_link_pass"
            if all(gates.values())
            else "exploratory_gate_failure"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_sha256": mechanics.SOURCE_SHA256,
            "raw_sha256": posthoc.RAW_SHA256,
            "posthoc_open_mechanics": True,
            "target_blind_scores": list(metric_names),
            "endpoint_only_fields": list(endpoint_names),
            "openrouter_calls": 0,
            "oatml_used": False,
            "development_authorized": False,
        },
        "summary": {
            "task_count": len(selections),
            "root_count": len(records),
            "dynamic_task_count": dynamic_task_count,
            "pairwise": pairwise,
            "entropy_selection_truth_wins": selection_wins,
            "entropy_selection_truth_losses": selection_losses,
            "entropy_selection_truth_nonworse": selection_nonworse,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        },
        "selections": selections,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reconstructed = posthoc.reconstruct(
        source_path=args.source_path,
        raw_path=args.raw_path,
    )
    result = analyze(reconstructed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
