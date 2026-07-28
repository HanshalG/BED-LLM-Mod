#!/usr/bin/env python3
"""Develop three-query BED over twice-regenerated Number Game support."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    DOMAIN,
    MAX_TOKENS,
    RuleHypothesis,
    best_query,
    candidate_roots,
    choose_predictive_bayes_risk_root,
    evaluate_policy_root,
    history_messages,
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
    TEMPERATURE,
    _adapter,
    _tree_usage,
    aggregate_tree_comparisons,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-depth-three-development-2"
PLANNING_MODEL_ID = "openai/gpt-5.4-mini"
TARGET_MODEL_ID = "google/gemini-2.5-flash"
TREE_SEEDS = tuple(range(27600, 27608))
TARGET_SEEDS = tuple(range(27700, 27708))
EXPECTED_REQUESTS_PER_TREE = 50
EXPECTED_REQUESTS = len(TREE_SEEDS) * EXPECTED_REQUESTS_PER_TREE
RUN_BUDGET_USD = 1.50
MIN_INITIAL_VALID = 16
MIN_FIRST_BRANCH_VALID = 8
MIN_SECOND_BRANCH_VALID = 4
MIN_TARGET_VALID = 16
MIN_NOVEL_TARGETS = 8
SECOND_SUPPORT_GENERATED_ONLY = "generated_only"
SECOND_SUPPORT_RETAINED_REJUVENATION = "retained_rejuvenation"
SECOND_SUPPORT_MODES = {
    SECOND_SUPPORT_GENERATED_ONLY,
    SECOND_SUPPORT_RETAINED_REJUVENATION,
}


def _terminal_metrics(
    *,
    policy: str,
    root: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "policy": policy,
        "root": root,
        "mean_posterior_predictive_brier": sum(
            row["posterior_predictive_brier"] for row in rows
        )
        / len(rows),
        "mean_best_hamming_error": sum(
            row["best_hamming_error"] for row in rows
        )
        / len(rows),
        "truth_extension_coverage_rate": sum(
            row["truth_extension_covered"] for row in rows
        )
        / len(rows),
        "mean_survivor_count": sum(
            row["survivor_count"] for row in rows
        )
        / len(rows),
        "targets": rows,
    }


def evaluate_policy_root_depth_three(
    *,
    policy: str,
    root: int,
    targets: dict[str, RuleHypothesis],
    first_branches: dict[
        tuple[int, bool], Sequence[RuleHypothesis]
    ],
    second_branches: dict[
        tuple[int, bool, int, bool], Sequence[RuleHypothesis]
    ],
) -> dict[str, Any]:
    rows = []
    for target_name, target in targets.items():
        first_label = target.extension[root]
        first_support = list(first_branches[(root, first_label)])
        second_query, second_eig = best_query(
            first_support,
            excluded=(root,),
        )
        second_label = target.extension[second_query]
        second_support = list(
            second_branches[
                (root, first_label, second_query, second_label)
            ]
        )
        third_query, third_eig = best_query(
            second_support,
            excluded=(root, second_query),
        )
        third_label = target.extension[third_query]
        survivors = [
            hypothesis
            for hypothesis in second_support
            if hypothesis.extension[third_query] == third_label
        ]
        exact_matches = [
            hypothesis
            for hypothesis in survivors
            if hypothesis.extension == target.extension
        ]
        if survivors:
            probabilities = [
                sum(
                    hypothesis.extension[number]
                    for hypothesis in survivors
                )
                / len(survivors)
                for number in DOMAIN
            ]
            queried = {root, second_query, third_query}
            brier = sum(
                (
                    probabilities[number]
                    - float(target.extension[number])
                )
                ** 2
                for number in DOMAIN
                if number not in queried
            ) / (len(DOMAIN) - len(queried))
            best_error = min(
                sum(
                    left != right
                    for left, right in zip(
                        hypothesis.extension,
                        target.extension,
                        strict=True,
                    )
                )
                / len(DOMAIN)
                for hypothesis in survivors
            )
        else:
            brier = 1.0
            best_error = 1.0
        rows.append(
            {
                "target": target_name,
                "root": root,
                "first_label": first_label,
                "second_query": second_query,
                "second_label": second_label,
                "second_eig_nats": second_eig,
                "third_query": third_query,
                "third_label": third_label,
                "third_eig_nats": third_eig,
                "second_support_size": len(second_support),
                "survivor_count": len(survivors),
                "truth_extension_covered": bool(exact_matches),
                "posterior_predictive_brier": brier,
                "best_hamming_error": best_error,
            }
        )
    return _terminal_metrics(policy=policy, root=root, rows=rows)


def depth_three_scores(
    *,
    support: Sequence[RuleHypothesis],
    roots: Sequence[int],
    first_branches: dict[
        tuple[int, bool], Sequence[RuleHypothesis]
    ],
    second_branches: dict[
        tuple[int, bool, int, bool], Sequence[RuleHypothesis]
    ],
) -> dict[int, dict[str, Any]]:
    targets = {
        f"particle_{index:02d}": hypothesis
        for index, hypothesis in enumerate(support)
    }
    return {
        root: evaluate_policy_root_depth_three(
            policy="predictive_bayes_risk_depth_three",
            root=root,
            targets=targets,
            first_branches=first_branches,
            second_branches=second_branches,
        )
        for root in roots
    }


def static_depth_three_branches(
    *,
    support: Sequence[RuleHypothesis],
    roots: Sequence[int],
) -> tuple[
    dict[tuple[int, bool], list[RuleHypothesis]],
    dict[tuple[int, bool, int, bool], list[RuleHypothesis]],
]:
    first = {}
    second = {}
    for root in roots:
        for first_label in (False, True):
            first_support = [
                hypothesis
                for hypothesis in support
                if hypothesis.extension[root] == first_label
            ]
            first[(root, first_label)] = first_support
            second_query, _ = best_query(
                first_support,
                excluded=(root,),
            )
            for second_label in (False, True):
                second[
                    (root, first_label, second_query, second_label)
                ] = [
                    hypothesis
                    for hypothesis in first_support
                    if hypothesis.extension[second_query] == second_label
                ]
    return first, second


def retain_parent_hypotheses(
    *,
    parent_support: Sequence[RuleHypothesis],
    generated_support: Sequence[RuleHypothesis],
    query: int,
    label: bool,
) -> tuple[list[RuleHypothesis], dict[str, int]]:
    retained = [
        hypothesis
        for hypothesis in parent_support
        if hypothesis.extension[query] == label
    ]
    merged = []
    seen = set()
    generated_unique = 0
    for source, hypotheses in (
        ("generated", generated_support),
        ("retained", retained),
    ):
        for hypothesis in hypotheses:
            if hypothesis.extension in seen:
                continue
            seen.add(hypothesis.extension)
            merged.append(hypothesis)
            if source == "generated":
                generated_unique += 1
    return merged, {
        "generated_unique_count": generated_unique,
        "retained_parent_consistent_count": len(retained),
        "retained_parent_novel_count": len(merged) - generated_unique,
        "merged_unique_count": len(merged),
    }


def choose_risk_set_root(
    scores: dict[int, dict[str, Any]],
    *,
    brier_tolerance: float,
) -> int:
    if brier_tolerance < 0.0:
        raise ValueError("brier_tolerance must be non-negative")
    minimum_brier = min(
        score["mean_posterior_predictive_brier"]
        for score in scores.values()
    )
    eligible = [
        root
        for root, score in scores.items()
        if score["mean_posterior_predictive_brier"]
        <= minimum_brier + brier_tolerance
    ]
    return min(
        eligible,
        key=lambda root: (
            scores[root]["mean_best_hamming_error"],
            -scores[root]["truth_extension_coverage_rate"],
            scores[root]["mean_posterior_predictive_brier"],
            root,
        ),
    )


def run_tree_depth_three(
    *,
    tree_index: int,
    tree_seed: int,
    target_seed: int,
    output_dir: Path,
    run_id: str,
    planning_model: str = PLANNING_MODEL_ID,
    target_model: str = TARGET_MODEL_ID,
    planning_concurrency: int = 32,
    target_concurrency: int = 1,
    projected_planning_cost: float = 0.30,
    projected_target_cost: float = 0.02,
    run_budget_usd: float = RUN_BUDGET_USD,
    second_support_mode: str = SECOND_SUPPORT_GENERATED_ONLY,
    brier_tolerance: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if second_support_mode not in SECOND_SUPPORT_MODES:
        raise ValueError(
            f"unsupported second_support_mode {second_support_mode!r}"
        )
    planning = _adapter(
        model=planning_model,
        run_id=f"{run_id}-tree{tree_index}-planning",
        output_dir=output_dir,
        request_seed=tree_seed,
        concurrency=planning_concurrency,
        projected_cost=projected_planning_cost,
        run_budget_usd=run_budget_usd,
    )
    target = _adapter(
        model=target_model,
        run_id=f"{run_id}-tree{tree_index}-target",
        output_dir=output_dir,
        request_seed=target_seed,
        concurrency=target_concurrency,
        projected_cost=projected_target_cost,
        run_budget_usd=run_budget_usd,
    )
    initial_response = planning.chat_complete_messages_batched_structured(
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

    first_keys = [
        (root, label) for root in roots for label in (False, True)
    ]
    first_responses = planning.chat_complete_messages_batched_structured(
        [
            history_messages((key,), enforce_constraints=True)
            for key in first_keys
        ],
        temperature=TEMPERATURE,
        block_size=len(first_keys),
        response_format=proposal_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    first_branches = {}
    first_diagnostics = {}
    for key, response in zip(
        first_keys,
        first_responses,
        strict=True,
    ):
        hypotheses, diagnostics = parse_proposals(
            response,
            observations=(key,),
        )
        first_branches[key] = hypotheses
        first_diagnostics[
            f"{key[0]}:{int(key[1])}"
        ] = diagnostics

    second_keys = []
    for root, first_label in first_keys:
        second_query, _ = best_query(
            first_branches[(root, first_label)],
            excluded=(root,),
        )
        for second_label in (False, True):
            second_keys.append(
                (root, first_label, second_query, second_label)
            )
    second_responses = planning.chat_complete_messages_batched_structured(
        [
            history_messages(
                ((root, first_label), (second_query, second_label)),
                enforce_constraints=True,
            )
            for root, first_label, second_query, second_label in second_keys
        ],
        temperature=TEMPERATURE,
        block_size=len(second_keys),
        response_format=proposal_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    generated_second_branches = {}
    second_branches = {}
    parent_only_second_branches = {}
    second_diagnostics = {}
    for key, response in zip(
        second_keys,
        second_responses,
        strict=True,
    ):
        observations = ((key[0], key[1]), (key[2], key[3]))
        generated, diagnostics = parse_proposals(
            response,
            observations=observations,
        )
        generated_second_branches[key] = generated
        parent_only = [
            hypothesis
            for hypothesis in first_branches[(key[0], key[1])]
            if hypothesis.extension[key[2]] == key[3]
        ]
        parent_only_second_branches[key] = parent_only
        if (
            second_support_mode
            == SECOND_SUPPORT_RETAINED_REJUVENATION
        ):
            hypotheses, retention = retain_parent_hypotheses(
                parent_support=first_branches[(key[0], key[1])],
                generated_support=generated,
                query=key[2],
                label=key[3],
            )
        else:
            hypotheses = generated
            retention = {
                "generated_unique_count": len(generated),
                "retained_parent_consistent_count": len(parent_only),
                "retained_parent_novel_count": 0,
                "merged_unique_count": len(generated),
            }
        second_branches[key] = hypotheses
        second_diagnostics[
            f"{key[0]}:{int(key[1])}:{key[2]}:{int(key[3])}"
        ] = {
            **diagnostics,
            **retention,
            "valid_unique_count": len(hypotheses),
        }

    target_response = target.chat_complete_messages_batched_structured(
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

    depth_three = depth_three_scores(
        support=initial,
        roots=roots,
        first_branches=first_branches,
        second_branches=second_branches,
    )
    depth_three_root = choose_risk_set_root(
        depth_three,
        brier_tolerance=brier_tolerance,
    )
    depth_two = predictive_bayes_risk_scores(
        support=initial,
        roots=roots,
        branches=first_branches,
    )
    depth_two_root = choose_predictive_bayes_risk_root(depth_two)
    static_first, static_second = static_depth_three_branches(
        support=initial,
        roots=roots,
    )
    static_scores = depth_three_scores(
        support=initial,
        roots=roots,
        first_branches=static_first,
        second_branches=static_second,
    )
    static_root = choose_predictive_bayes_risk_root(static_scores)
    generated_only_scores = depth_three_scores(
        support=initial,
        roots=roots,
        first_branches=first_branches,
        second_branches=generated_second_branches,
    )
    generated_only_root = choose_risk_set_root(
        generated_only_scores,
        brier_tolerance=brier_tolerance,
    )
    parent_only_scores = depth_three_scores(
        support=initial,
        roots=roots,
        first_branches=first_branches,
        second_branches=parent_only_second_branches,
    )
    parent_only_root = choose_risk_set_root(
        parent_only_scores,
        brier_tolerance=brier_tolerance,
    )
    policy_roots = {
        "predictive_bayes_risk_depth_three": depth_three_root,
        "predictive_bayes_risk_depth_two": depth_two_root,
        "myopic_eig": int(candidate_metadata["myopic_root"]),
        "fixed_support_depth_three": static_root,
    }
    if (
        second_support_mode
        == SECOND_SUPPORT_RETAINED_REJUVENATION
    ):
        policy_roots.update(
            {
                "generated_only_depth_three": generated_only_root,
                "retained_parent_only_depth_three": parent_only_root,
            }
        )
    per_root = {
        root: evaluate_policy_root_depth_three(
            policy=f"root_{root}",
            root=root,
            targets=targets,
            first_branches=first_branches,
            second_branches=second_branches,
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
    baselines = (
        "predictive_bayes_risk_depth_two",
        "myopic_eig",
        "fixed_support_depth_three",
        "uniform_random_candidate_root",
        "positive_test_strategy",
        *(
            (
                "generated_only_depth_three",
                "retained_parent_only_depth_three",
            )
            if (
                second_support_mode
                == SECOND_SUPPORT_RETAINED_REJUVENATION
            )
            else ()
        ),
    )
    comparisons = {
        baseline: policy_comparison(
            endpoint["predictive_bayes_risk_depth_three"],
            endpoint[baseline],
        )
        for baseline in baselines
    }
    novel_endpoint = {
        policy: evaluate_policy_root_depth_three(
            policy=policy,
            root=root,
            targets=novel_targets,
            first_branches=first_branches,
            second_branches=second_branches,
        )
        for policy, root in policy_roots.items()
    }
    novel_comparison = policy_comparison(
        novel_endpoint["predictive_bayes_risk_depth_three"],
        novel_endpoint["predictive_bayes_risk_depth_two"],
    )
    usage = _tree_usage(planning, target)
    mechanics = {
        "initial_valid": len(initial),
        "minimum_first_branch_valid": min(
            item["valid_unique_count"]
            for item in first_diagnostics.values()
        ),
        "minimum_second_branch_valid": min(
            item["valid_unique_count"]
            for item in second_diagnostics.values()
        ),
        "minimum_generated_second_branch_valid": min(
            item["generated_unique_count"]
            for item in second_diagnostics.values()
        ),
        "minimum_retained_parent_consistent": min(
            item["retained_parent_consistent_count"]
            for item in second_diagnostics.values()
        ),
        "target_valid": len(targets),
        "novel_targets": len(novel_targets),
        "exact_50_requests": (
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
        "selection": {
            **candidate_metadata,
            "predictive_bayes_risk_depth_three_root": depth_three_root,
            "predictive_bayes_risk_depth_two_root": depth_two_root,
            "fixed_support_depth_three_root": static_root,
            "generated_only_depth_three_root": generated_only_root,
            "retained_parent_only_depth_three_root": parent_only_root,
            "pts_roots": pts_roots,
        },
        "initial_diagnostics": initial_diagnostics,
        "first_branch_diagnostics": first_diagnostics,
        "second_branch_diagnostics": second_diagnostics,
        "second_support_mode": second_support_mode,
        "brier_tolerance": brier_tolerance,
        "target_diagnostics": {
            **target_diagnostics,
            "novel_unique_count": len(novel_targets),
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
        "novel_comparison_depth_three_vs_depth_two": novel_comparison,
        "usage": usage,
    }
    raw = {
        "initial_response": initial_response,
        "first_responses": [
            {"key": list(key), "response": response}
            for key, response in zip(
                first_keys, first_responses, strict=True
            )
        ],
        "second_responses": [
            {"key": list(key), "response": response}
            for key, response in zip(
                second_keys, second_responses, strict=True
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
        "first_branches": {
            f"{root}:{int(label)}": [
                hypothesis.public_dict()
                for hypothesis in first_branches[(root, label)]
            ]
            for root, label in first_keys
        },
        "second_branches": {
            f"{root}:{int(first_label)}:{second_query}:{int(second_label)}": [
                hypothesis.public_dict()
                for hypothesis in second_branches[
                    (root, first_label, second_query, second_label)
                ]
            ]
            for root, first_label, second_query, second_label in second_keys
        },
        "generated_second_branches": {
            f"{root}:{int(first_label)}:{second_query}:{int(second_label)}": [
                hypothesis.public_dict()
                for hypothesis in generated_second_branches[
                    (root, first_label, second_query, second_label)
                ]
            ]
            for root, first_label, second_query, second_label in second_keys
        },
        "second_support_mode": second_support_mode,
        "brier_tolerance": brier_tolerance,
        "targets": [
            {
                **hypothesis.public_dict(),
                "novel_to_initial_support": (
                    hypothesis.extension not in initial_extensions
                ),
            }
            for hypothesis in target_list
        ],
    }
    return tree_result, {"raw": raw, "public": public}


def run_development(
    *,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    raw = {"trees": []}
    trees = []
    public_trees = []
    try:
        for index, (tree_seed, target_seed) in enumerate(
            zip(TREE_SEEDS, TARGET_SEEDS, strict=True)
        ):
            tree, artifacts = run_tree_depth_three(
                tree_index=index,
                tree_seed=tree_seed,
                target_seed=target_seed,
                output_dir=output_dir,
                run_id=run_id,
            )
            trees.append(tree)
            raw["trees"].append(artifacts["raw"])
            public_trees.append(artifacts["public"])
            checkpoint(raw_path, raw)
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
        usage = {
            key: sum(tree["usage"].get(key, 0) for tree in trees)
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
        mechanics = {
            "exact_400_accepted_requests": (
                usage["adapter_requests"] == EXPECTED_REQUESTS
            ),
            "transport_attempt_accounting_exact": (
                usage["http_attempts"]
                == usage["adapter_requests"] + usage["retry_count"]
            ),
            "zero_reasoning_tokens": (
                usage["adapter_reasoning_tokens"] == 0
            ),
            "zero_forced_exits": usage["forced_exits"] == 0,
            "within_run_budget": (
                usage["run_cost_usd"] <= RUN_BUDGET_USD
            ),
            "all_initial_supports_valid": all(
                tree["mechanics"]["initial_valid"] >= MIN_INITIAL_VALID
                for tree in trees
            ),
            "all_first_branches_valid": all(
                tree["mechanics"]["minimum_first_branch_valid"]
                >= MIN_FIRST_BRANCH_VALID
                for tree in trees
            ),
            "all_second_branches_valid": all(
                tree["mechanics"]["minimum_second_branch_valid"]
                >= MIN_SECOND_BRANCH_VALID
                for tree in trees
            ),
            "all_target_supports_valid": all(
                tree["mechanics"]["target_valid"] >= MIN_TARGET_VALID
                and tree["mechanics"]["novel_targets"]
                >= MIN_NOVEL_TARGETS
                for tree in trees
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
            "error": f"{type(exc).__name__}: {exc}",
            "completed_trees": len(trees),
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
    result = run_development(
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
