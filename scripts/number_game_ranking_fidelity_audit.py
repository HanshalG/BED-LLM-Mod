#!/usr/bin/env python3
"""Audit Number Game root-ranking fidelity and particle stability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    RuleHypothesis,
    candidate_roots,
    choose_predictive_bayes_risk_root,
    compile_expression,
    evaluate_policy_root,
    predictive_bayes_risk_scores,
)
from scripts.number_game_predictive_risk_replication import BOOTSTRAP_SEED
from scripts.number_game_size_matched_support_control import (
    fast_predictive_bayes_risk_root,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-ranking-fidelity-audit-1"
BOOTSTRAP_SAMPLES = 20_000


def _rule(item: dict[str, Any]) -> RuleHypothesis:
    return RuleHypothesis(
        name=item["name"],
        expression=item["expression"],
        extension=compile_expression(item["expression"]),
    )


def pairwise_concordance(
    estimated: Sequence[float],
    realized: Sequence[float],
) -> float:
    if len(estimated) != len(realized):
        raise ValueError("rank vectors differ in length")
    concordant = 0.0
    comparable = 0
    for left in range(len(estimated)):
        for right in range(left + 1, len(estimated)):
            estimated_delta = estimated[left] - estimated[right]
            realized_delta = realized[left] - realized[right]
            if estimated_delta == 0.0 or realized_delta == 0.0:
                continue
            comparable += 1
            if (estimated_delta > 0.0) == (realized_delta > 0.0):
                concordant += 1.0
    return concordant / comparable if comparable else 0.5


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("rank vectors must have equal positive length")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = _mean(left_ranks)
    right_mean = _mean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(
            left_ranks,
            right_ranks,
            strict=True,
        )
    )
    left_scale = sum(
        (value - left_mean) ** 2 for value in left_ranks
    )
    right_scale = sum(
        (value - right_mean) ** 2 for value in right_ranks
    )
    denominator = (left_scale * right_scale) ** 0.5
    return numerator / denominator if denominator else 0.0


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> list[float]:
    if not values:
        raise ValueError("bootstrap values are empty")
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    return [
        means[int(0.025 * samples)],
        means[min(samples - 1, int(0.975 * samples))],
    ]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def score_public_tree(tree: dict[str, Any]) -> dict[str, Any]:
    initial = [_rule(item) for item in tree["initial"]]
    roots = [int(root) for root in tree["roots"]]
    branches = {}
    for key, items in tree["branches"].items():
        root, label = key.split(":")
        branches[(int(root), bool(int(label)))] = [
            _rule(item) for item in items
        ]
    targets = {
        f"target_{index:02d}_{item['name']}": _rule(item)
        for index, item in enumerate(tree["targets"])
    }

    source_scores = predictive_bayes_risk_scores(
        support=initial,
        roots=roots,
        branches=branches,
    )
    selected_root = choose_predictive_bayes_risk_root(source_scores)
    endpoint = {
        root: evaluate_policy_root(
            policy=f"root_{root}",
            root=root,
            targets=targets,
            branches=branches,
        )
        for root in roots
    }
    oracle_root = min(
        roots,
        key=lambda root: (
            endpoint[root]["mean_posterior_predictive_brier"],
            endpoint[root]["mean_best_hamming_error"],
            -endpoint[root]["truth_extension_coverage_rate"],
            root,
        ),
    )
    reconstructed_roots, metadata = candidate_roots(
        initial,
        seed=int(tree["tree_seed"]),
    )
    if reconstructed_roots != roots:
        raise ValueError("public roots do not match deterministic candidates")

    source_brier = [
        source_scores[root]["mean_posterior_predictive_brier"]
        for root in roots
    ]
    endpoint_brier = [
        endpoint[root]["mean_posterior_predictive_brier"]
        for root in roots
    ]
    endpoint_hamming = [
        endpoint[root]["mean_best_hamming_error"] for root in roots
    ]
    myopic_risk = [
        -float(metadata["immediate_eig_nats"][str(root)])
        for root in roots
    ]
    fixed_risk = [
        -float(metadata["fixed_depth_two_nats"][str(root)])
        for root in roots
    ]
    source_rho = spearman_correlation(source_brier, endpoint_brier)
    myopic_rho = spearman_correlation(myopic_risk, endpoint_brier)
    fixed_rho = spearman_correlation(fixed_risk, endpoint_brier)
    oracle_brier = endpoint[oracle_root][
        "mean_posterior_predictive_brier"
    ]

    myopic_root = int(metadata["myopic_root"])
    fixed_root = int(metadata["fixed_depth_two_root"])
    pts_roots = [int(root) for root in metadata["pts_roots"]]
    random_brier = _mean(
        [
            endpoint[root]["mean_posterior_predictive_brier"]
            for root in roots
        ]
    )
    pts_brier = _mean(
        [
            endpoint[root]["mean_posterior_predictive_brier"]
            for root in pts_roots
        ]
    )
    ranked_roots = sorted(
        roots,
        key=lambda root: (
            endpoint[root]["mean_posterior_predictive_brier"],
            endpoint[root]["mean_best_hamming_error"],
            -endpoint[root]["truth_extension_coverage_rate"],
            root,
        ),
    )

    loo_roots = [
        fast_predictive_bayes_risk_root(
            support=initial[:index] + initial[index + 1 :],
            roots=roots,
            branches=branches,
        )
        for index in range(len(initial))
    ]
    loo_agreement = sum(
        root == selected_root for root in loo_roots
    ) / len(loo_roots)
    return {
        "tree_index": tree["tree_index"],
        "tree_seed": tree["tree_seed"],
        "target_seed": tree["target_seed"],
        "num_initial_rules": len(initial),
        "num_targets": len(targets),
        "selection": {
            "predictive_root": selected_root,
            "oracle_root": oracle_root,
            "myopic_root": myopic_root,
            "fixed_depth_two_root": fixed_root,
            "pts_roots": pts_roots,
            "predictive_is_oracle": selected_root == oracle_root,
            "predictive_endpoint_rank": ranked_roots.index(selected_root) + 1,
        },
        "ranking": {
            "predictive_risk_spearman_brier": source_rho,
            "myopic_spearman_brier": myopic_rho,
            "fixed_depth_two_spearman_brier": fixed_rho,
            "predictive_pairwise_concordance": pairwise_concordance(
                source_brier,
                endpoint_brier,
            ),
            "predictive_risk_spearman_hamming": float(
                spearman_correlation(source_brier, endpoint_hamming)
            ),
        },
        "regret": {
            "predictive_bayes_risk": endpoint[selected_root][
                "mean_posterior_predictive_brier"
            ]
            - oracle_brier,
            "myopic_eig": endpoint[myopic_root][
                "mean_posterior_predictive_brier"
            ]
            - oracle_brier,
            "fixed_depth_two": endpoint[fixed_root][
                "mean_posterior_predictive_brier"
            ]
            - oracle_brier,
            "positive_test_strategy": pts_brier - oracle_brier,
            "uniform_random_candidate_root": random_brier - oracle_brier,
        },
        "leave_one_particle_out": {
            "agreement": loo_agreement,
            "stable_on_all_particles": loo_agreement == 1.0,
            "roots": loo_roots,
        },
        "per_root": {
            str(root): {
                "source_predictive_brier": source_scores[root][
                    "mean_posterior_predictive_brier"
                ],
                "endpoint_brier": endpoint[root][
                    "mean_posterior_predictive_brier"
                ],
                "endpoint_hamming": endpoint[root][
                    "mean_best_hamming_error"
                ],
                "endpoint_coverage": endpoint[root][
                    "truth_extension_coverage_rate"
                ],
                "myopic_risk": myopic_risk[index],
                "fixed_depth_two_risk": fixed_risk[index],
            }
            for index, root in enumerate(roots)
        },
    }


def aggregate_trees(
    trees: Sequence[dict[str, Any]],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    ranking_keys = (
        "predictive_risk_spearman_brier",
        "myopic_spearman_brier",
        "fixed_depth_two_spearman_brier",
        "predictive_pairwise_concordance",
        "predictive_risk_spearman_hamming",
    )
    ranking = {}
    for index, key in enumerate(ranking_keys):
        values = [tree["ranking"][key] for tree in trees]
        ranking[key] = {
            "mean": _mean(values),
            "tree_bootstrap_95pct": bootstrap_mean_interval(
                values,
                seed=BOOTSTRAP_SEED + seed_offset + index,
            ),
        }
    candidate_regret = [
        tree["regret"]["predictive_bayes_risk"] for tree in trees
    ]
    regret = {
        "predictive_bayes_risk": {
            "mean": _mean(candidate_regret),
            "tree_bootstrap_95pct": bootstrap_mean_interval(
                candidate_regret,
                seed=BOOTSTRAP_SEED + seed_offset + 10,
            ),
        }
    }
    for index, baseline in enumerate(
        (
            "myopic_eig",
            "fixed_depth_two",
            "positive_test_strategy",
            "uniform_random_candidate_root",
        )
    ):
        baseline_values = [tree["regret"][baseline] for tree in trees]
        differences = [
            candidate - baseline_value
            for candidate, baseline_value in zip(
                candidate_regret,
                baseline_values,
                strict=True,
            )
        ]
        regret[baseline] = {
            "mean": _mean(baseline_values),
            "predictive_minus_baseline_mean": _mean(differences),
            "predictive_minus_baseline_tree_bootstrap_95pct": (
                bootstrap_mean_interval(
                    differences,
                    seed=BOOTSTRAP_SEED + seed_offset + 20 + index,
                )
            ),
            "predictive_wins": sum(value < 0.0 for value in differences),
        }
    loo = [
        tree["leave_one_particle_out"]["agreement"] for tree in trees
    ]
    ranks = [tree["selection"]["predictive_endpoint_rank"] for tree in trees]
    return {
        "num_trees": len(trees),
        "ranking": ranking,
        "regret": regret,
        "selection": {
            "oracle_root_count": sum(
                tree["selection"]["predictive_is_oracle"] for tree in trees
            ),
            "top_two_count": sum(rank <= 2 for rank in ranks),
            "mean_endpoint_rank": _mean(ranks),
        },
        "leave_one_particle_out": {
            "mean_agreement": _mean(loo),
            "tree_bootstrap_95pct": bootstrap_mean_interval(
                loo,
                seed=BOOTSTRAP_SEED + seed_offset + 30,
            ),
            "fully_stable_tree_count": sum(
                tree["leave_one_particle_out"][
                    "stable_on_all_particles"
                ]
                for tree in trees
            ),
            "agreement_at_least_75pct_count": sum(
                value >= 0.75 for value in loo
            ),
        },
    }


def evaluate_sources(
    *,
    tree_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    sources = {}
    all_trees = []
    for source_index, tree_path in enumerate(tree_paths):
        document = json.loads(tree_path.read_text())
        trees = [score_public_tree(tree) for tree in document["trees"]]
        all_trees.extend(trees)
        sources[tree_path.parent.name] = {
            "source_sha256": hashlib.sha256(
                tree_path.read_bytes()
            ).hexdigest(),
            "source_protocol": document["protocol"],
            "aggregate": aggregate_trees(
                trees,
                seed_offset=100 * source_index,
            ),
            "trees": trees,
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "tree_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "llm_calls": 0,
            "description": (
                "Rank eight common roots using current-particle terminal "
                "risk and independently generated target endpoints."
            ),
        },
        "sources": sources,
        "combined": aggregate_trees(all_trees, seed_offset=1000),
    }
    checkpoint(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_sources(
        tree_paths=[path.resolve() for path in args.trees],
        output_path=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
