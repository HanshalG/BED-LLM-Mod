#!/usr/bin/env python3
"""Audit Number Game depth-three efficacy outside a classical grammar bank."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_depth_three_development as depth
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    DOMAIN,
    RuleHypothesis,
    compile_expression,
    is_power_of_two,
    is_prime,
    is_square,
)
from scripts.number_game_ranking_fidelity_audit import (
    bootstrap_mean_interval,
)
from scripts.number_game_retained_depth_three import (
    _first_branches,
    _rule,
    _second_branches,
    retained_second_branches,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-classical-grammar-irreducibility-audit-1"
GRAMMAR_VERSION = b"number-game-classical-grammar-v1\0"
DOMAIN_SIZE = len(DOMAIN)
ALL_MASK = (1 << DOMAIN_SIZE) - 1
MASK_BYTES = (DOMAIN_SIZE + 7) // 8
BOOTSTRAP_SEED = 37291
BOOTSTRAP_SAMPLES = 20_000
MIN_BANK_EXTENSIONS = 100_000
MIN_SECOND_UNIQUE_NOVEL_FRACTION = 0.05
MIN_SECOND_NOVEL_TREES = 48
MIN_SECOND_NOVEL_TREES_PER_SOURCE = 20
MIN_ENDPOINT_NOVEL_FRACTION = 0.05
MIN_ENDPOINT_NOVEL_OCCURRENCES = 512
MIN_NONEMPTY_DRAWS_PER_TREE = 8
MIN_ANALYZABLE_TREES = 48
MIN_ANALYZABLE_TREES_PER_SOURCE = 24
MIN_PRIMARY_RELATIVE_BRIER_GAIN = 0.01
MIN_PRIMARY_TREE_WINS = 20

QWEN_RESULT_PATH = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_crossjudge_depth_three"
    / "number-game-qwen-crossjudge-depth-three-20260728"
    / "RESULT.json"
)
QWEN_RESULT_SHA256 = (
    "a7e0549f2f9ff7b1c2394ebe076bdbf1b70799479a3597e01bf66b7003edc1cb"
)
QWEN_ENDPOINTS_PATH = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_crossjudge_depth_three"
    / "number-game-qwen-crossjudge-depth-three-20260728"
    / "ENDPOINTS.json"
)
QWEN_ENDPOINTS_SHA256 = (
    "647de3c6561ff917691dc3c14176dc4007f90b17230bbb6ee690468d978a613d"
)
SOURCE_STUDIES = (
    {
        "name": "fixed_policy_fresh_endpoints",
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_confirmation"
            / "number-game-crossfit-depth-three-confirmation-20260728"
            / "TREES.json"
        ),
        "trees_sha256": (
            "cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9"
        ),
        "result": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_endpoint_precision"
            / "number-game-crossfit-endpoint-precision-20260728"
            / "RESULT.json"
        ),
        "result_sha256": (
            "47e684b11ac5c6340ff4f063ff7a14db8b8d0bd18b184d61f91e315903e4809e"
        ),
    },
    {
        "name": "fresh_tree_replication",
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_fresh_replication"
            / "number-game-crossfit-depth-three-fresh-replication-20260728"
            / "TREES.json"
        ),
        "trees_sha256": (
            "197dcfe3cb48eeb9a4b656d0f9b20ef2e0b45ac8d1df0ec4bd9358e07af18802"
        ),
        "result": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_fresh_replication"
            / "number-game-crossfit-depth-three-fresh-replication-20260728"
            / "RESULT.json"
        ),
        "result_sha256": (
            "25e0939164f6806b30481e48a88372f22639f6065096cb59978600509d43a3d8"
        ),
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extension_mask(extension: Sequence[bool]) -> int:
    if len(extension) != DOMAIN_SIZE:
        raise ValueError(
            f"expected {DOMAIN_SIZE}-value extension, got {len(extension)}"
        )
    return sum(int(value) << index for index, value in enumerate(extension))


def predicate_mask(predicate: Callable[[int], bool]) -> int:
    return extension_mask(tuple(bool(predicate(n)) for n in DOMAIN))


def _nonconstant(masks: Iterable[int]) -> set[int]:
    return {mask for mask in masks if mask not in {0, ALL_MASK}}


def core_atom_masks() -> set[int]:
    masks: set[int] = set()
    for cutoff in range(102):
        masks.add(predicate_mask(lambda n, c=cutoff: n < c))
        masks.add(predicate_mask(lambda n, c=cutoff: n <= c))
    for modulus in range(2, 21):
        for remainder in range(modulus):
            masks.add(
                predicate_mask(
                    lambda n, k=modulus, r=remainder: n % k == r
                )
            )
    for cutoff in range(21):
        masks.add(
            predicate_mask(
                lambda n, c=cutoff: sum(map(int, str(n))) < c
            )
        )
        masks.add(
            predicate_mask(
                lambda n, c=cutoff: sum(map(int, str(n))) == c
            )
        )
    for modulus in range(2, 10):
        for remainder in range(modulus):
            masks.add(
                predicate_mask(
                    lambda n, k=modulus, r=remainder: (
                        sum(map(int, str(n))) % k == r
                    )
                )
            )
    for final_digit in range(10):
        masks.add(
            predicate_mask(
                lambda n, d=final_digit: n % 10 == d
            )
        )
    predicates = (is_square, is_prime, is_power_of_two)
    for shift in range(-12, 13):
        for predicate in predicates:
            masks.add(
                predicate_mask(
                    lambda n, s=shift, p=predicate: p(n + s)
                )
            )
    return _nonconstant(masks)


def extended_atom_masks() -> set[int]:
    masks: set[int] = set()
    for multiplier in range(1, 6):
        for offset in range(-10, 11):
            for modulus in range(2, 16):
                for remainder in range(modulus):
                    masks.add(
                        predicate_mask(
                            lambda n, a=multiplier, b=offset, k=modulus,
                            r=remainder: (a * n + b) % k == r
                        )
                    )
    predicates = (is_square, is_prime, is_power_of_two)
    for multiplier in range(1, 13):
        for offset in range(-50, 51):
            for predicate in predicates:
                masks.add(
                    predicate_mask(
                        lambda n, a=multiplier, b=offset, p=predicate: (
                            p(a * n + b)
                        )
                    )
                )
    for divisor in range(1, 13):
        for offset in range(-30, 31):
            for predicate in predicates:
                masks.add(
                    predicate_mask(
                        lambda n, a=divisor, b=offset, p=predicate: (
                            (n - b) % a == 0 and p((n - b) // a)
                        )
                    )
                )
    return _nonconstant(masks)


def grammar_sha256(bank: Iterable[int]) -> str:
    digest = hashlib.sha256()
    digest.update(GRAMMAR_VERSION)
    for mask in sorted(bank):
        digest.update(mask.to_bytes(MASK_BYTES, "little"))
    return digest.hexdigest()


def build_classical_grammar_bank() -> tuple[set[int], dict[str, Any]]:
    core = sorted(core_atom_masks())
    extended = extended_atom_masks()
    bank = set(core)
    bank.update(extended)
    atom_union_count = len(bank)
    pairwise_candidates = 0
    for left_index, left in enumerate(core):
        for right in core[left_index + 1 :]:
            pairwise_candidates += 6
            bank.add(left & right)
            bank.add(left | right)
            bank.add(left & (ALL_MASK ^ right))
            bank.add(right & (ALL_MASK ^ left))
            bank.add(left ^ right)
            bank.add(ALL_MASK ^ (left ^ right))
    bank = _nonconstant(bank)
    pre_complement_count = len(bank)
    bank.update(ALL_MASK ^ mask for mask in tuple(bank))
    bank = _nonconstant(bank)
    return bank, {
        "grammar_version": GRAMMAR_VERSION.rstrip(b"\0").decode(),
        "domain_size": DOMAIN_SIZE,
        "core_atom_count": len(core),
        "extended_atom_count": len(extended),
        "deduped_atom_union_count": atom_union_count,
        "pairwise_candidate_count": pairwise_candidates,
        "pre_complement_unique_count": pre_complement_count,
        "unique_nonconstant_extension_count": len(bank),
        "sha256": grammar_sha256(bank),
    }


def item_mask(item: dict[str, Any]) -> int:
    extension = compile_expression(str(item["expression"]))
    expected_sha = hashlib.sha256(bytes(extension)).hexdigest()
    if item.get("extension_sha256") != expected_sha:
        raise ValueError(
            f"extension hash mismatch for expression {item['expression']!r}"
        )
    return extension_mask(extension)


def stage_summary(
    items: Sequence[dict[str, Any]],
    *,
    bank: set[int],
) -> dict[str, Any]:
    masks = [item_mask(item) for item in items]
    unique = set(masks)
    novel_occurrences = sum(mask not in bank for mask in masks)
    novel_unique = {mask for mask in unique if mask not in bank}
    return {
        "occurrences": len(masks),
        "unique_extensions": len(unique),
        "grammar_novel_occurrences": novel_occurrences,
        "grammar_novel_occurrence_fraction": (
            novel_occurrences / len(masks) if masks else 0.0
        ),
        "grammar_novel_unique_extensions": len(novel_unique),
        "grammar_novel_unique_fraction": (
            len(novel_unique) / len(unique) if unique else 0.0
        ),
    }


def _flatten_branch_items(
    branches: dict[str, Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        item
        for branch_items in branches.values()
        for item in branch_items
    ]


def load_frozen_evidence() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    if sha256_file(QWEN_RESULT_PATH) != QWEN_RESULT_SHA256:
        raise ValueError("Qwen cross-judge RESULT.json hash changed")
    if sha256_file(QWEN_ENDPOINTS_PATH) != QWEN_ENDPOINTS_SHA256:
        raise ValueError("Qwen cross-judge ENDPOINTS.json hash changed")
    qwen_result = json.loads(QWEN_RESULT_PATH.read_text())
    endpoints = json.loads(QWEN_ENDPOINTS_PATH.read_text())
    sources: dict[str, dict[str, Any]] = {}
    for source in SOURCE_STUDIES:
        if sha256_file(source["trees"]) != source["trees_sha256"]:
            raise ValueError(f"{source['name']} TREES.json hash changed")
        if sha256_file(source["result"]) != source["result_sha256"]:
            raise ValueError(f"{source['name']} RESULT.json hash changed")
        trees = json.loads(source["trees"].read_text())
        result = json.loads(source["result"].read_text())
        if len(trees["trees"]) != 32 or len(result["trees"]) != 32:
            raise ValueError(f"{source['name']} tree count changed")
        sources[source["name"]] = {"trees": trees, "result": result}
    if len(endpoints["trees"]) != 64:
        raise ValueError("Qwen endpoint tree count changed")
    if len(qwen_result["trees"]) != 64:
        raise ValueError("Qwen result tree count changed")
    return sources, endpoints, qwen_result


def support_novelty(
    sources: dict[str, dict[str, Any]],
    *,
    bank: set[int],
) -> dict[str, Any]:
    stage_items = {
        "initial": [],
        "generated_first": [],
        "generated_second": [],
    }
    source_tree_counts = {}
    tree_rows = []
    for source_name, source in sources.items():
        source_novel_trees = 0
        for tree in source["trees"]["trees"]:
            generated_first = _flatten_branch_items(
                tree["generated_first_branches"]
            )
            generated_second = _flatten_branch_items(
                tree["generated_second_branches"]
            )
            stage_items["initial"].extend(tree["initial"])
            stage_items["generated_first"].extend(generated_first)
            stage_items["generated_second"].extend(generated_second)
            second_summary = stage_summary(generated_second, bank=bank)
            has_novel_second = (
                second_summary["grammar_novel_unique_extensions"] > 0
            )
            source_novel_trees += int(has_novel_second)
            tree_rows.append(
                {
                    "source_study": source_name,
                    "tree_index": int(tree["tree_index"]),
                    "tree_seed": int(tree["tree_seed"]),
                    "generated_second": second_summary,
                    "has_grammar_novel_generated_second": has_novel_second,
                }
            )
        source_tree_counts[source_name] = source_novel_trees
    stages = {
        name: stage_summary(items, bank=bank)
        for name, items in stage_items.items()
    }
    novel_tree_count = sum(source_tree_counts.values())
    gates = {
        "second_unique_novel_fraction_at_least_five_percent": (
            stages["generated_second"]["grammar_novel_unique_fraction"]
            >= MIN_SECOND_UNIQUE_NOVEL_FRACTION
        ),
        "at_least_48_trees_have_novel_second_refresh": (
            novel_tree_count >= MIN_SECOND_NOVEL_TREES
        ),
        "each_source_has_at_least_20_novel_second_refresh_trees": all(
            count >= MIN_SECOND_NOVEL_TREES_PER_SOURCE
            for count in source_tree_counts.values()
        ),
    }
    return {
        "stages": stages,
        "trees_with_grammar_novel_generated_second": novel_tree_count,
        "trees_with_grammar_novel_generated_second_by_source": (
            source_tree_counts
        ),
        "trees": tree_rows,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def nearest_bank_distances(
    novel_masks: Iterable[int],
    *,
    bank: set[int],
) -> dict[str, Any]:
    targets = sorted(set(novel_masks))
    bank_by_positive_count: dict[int, list[int]] = defaultdict(list)
    for mask in bank:
        bank_by_positive_count[mask.bit_count()].append(mask)
    distances = []
    for target in targets:
        target_count = target.bit_count()
        best = DOMAIN_SIZE + 1
        for count_delta in range(DOMAIN_SIZE + 1):
            if count_delta >= best:
                break
            counts = {target_count - count_delta, target_count + count_delta}
            for count in counts:
                if not 0 <= count <= DOMAIN_SIZE:
                    continue
                for candidate in bank_by_positive_count.get(count, ()):
                    distance = (target ^ candidate).bit_count()
                    if distance < best:
                        best = distance
                        if best == 1:
                            break
                if best == 1:
                    break
        if best > DOMAIN_SIZE:
            raise RuntimeError("could not find nearest grammar extension")
        distances.append(best)
    return {
        "unique_grammar_novel_extensions": len(targets),
        "minimum_hamming_count": min(distances) if distances else None,
        "median_hamming_count": (
            statistics.median(distances) if distances else None
        ),
        "mean_hamming_count": (
            sum(distances) / len(distances) if distances else None
        ),
        "maximum_hamming_count": max(distances) if distances else None,
        "count_at_least_two": sum(value >= 2 for value in distances),
        "fraction_at_least_two": (
            sum(value >= 2 for value in distances) / len(distances)
            if distances
            else None
        ),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _average_policy_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[str, float]:
    keys = (
        "mean_posterior_predictive_brier",
        "mean_best_hamming_error",
        "truth_extension_coverage_rate",
    )
    return {key: _mean([float(row[key]) for row in rows]) for key in keys}


def _score_tree(
    *,
    source_name: str,
    source_tree: dict[str, Any],
    source_metrics: dict[str, Any],
    endpoint_tree: dict[str, Any],
    bank: set[int],
) -> dict[str, Any]:
    if int(source_tree["tree_seed"]) != int(endpoint_tree["tree_seed"]):
        raise ValueError("source and endpoint tree seeds do not match")
    first = _first_branches(source_tree)
    generated_second = _second_branches(
        source_tree,
        key="generated_second_branches",
    )
    second, _, _ = retained_second_branches(
        first_branches=first,
        generated_second_branches=generated_second,
    )
    selection = source_metrics["selection"]
    roots = {
        "crossfit_depth_three": int(
            selection["crossfit_depth_three_root"]
        ),
        "crossfit_depth_two": int(selection["crossfit_depth_two_root"]),
    }
    draw_rows = []
    novel_occurrences = 0
    total_occurrences = 0
    novel_masks: set[int] = set()
    for draw_index, support in enumerate(endpoint_tree["supports"]):
        total_occurrences += len(support)
        novel_items = [item for item in support if item_mask(item) not in bank]
        novel_occurrences += len(novel_items)
        novel_masks.update(item_mask(item) for item in novel_items)
        if not novel_items:
            continue
        targets = {
            f"draw_{draw_index:02d}_target_{index:02d}_{item['name']}": (
                _rule(item)
            )
            for index, item in enumerate(novel_items)
        }
        policies = {
            policy: depth.evaluate_policy_root_depth_three(
                policy=policy,
                root=root,
                targets=targets,
                first_branches=first,
                second_branches=second,
            )
            for policy, root in roots.items()
        }
        draw_rows.append(
            {
                "draw_index": draw_index,
                "grammar_novel_target_count": len(novel_items),
                "policies": {
                    name: {
                        key: value
                        for key, value in row.items()
                        if key != "targets"
                    }
                    for name, row in policies.items()
                },
            }
        )
    analyzable = len(draw_rows) >= MIN_NONEMPTY_DRAWS_PER_TREE
    row: dict[str, Any] = {
        "source_study": source_name,
        "tree_index": int(source_tree["tree_index"]),
        "tree_seed": int(source_tree["tree_seed"]),
        "selected_roots": roots,
        "root_differs": roots["crossfit_depth_three"]
        != roots["crossfit_depth_two"],
        "endpoint_occurrences": total_occurrences,
        "grammar_novel_endpoint_occurrences": novel_occurrences,
        "grammar_novel_unique_extensions": len(novel_masks),
        "nonempty_draw_count": len(draw_rows),
        "analyzable": analyzable,
        "draws": draw_rows,
    }
    if draw_rows:
        row["endpoint"] = {
            policy: _average_policy_rows(
                [draw["policies"][policy] for draw in draw_rows]
            )
            for policy in roots
        }
    return row


def endpoint_novelty_and_efficacy(
    sources: dict[str, dict[str, Any]],
    endpoints: dict[str, Any],
    *,
    bank: set[int],
) -> dict[str, Any]:
    tree_rows = []
    novel_masks: set[int] = set()
    source_analyzable_counts = defaultdict(int)
    source_endpoint_counts = defaultdict(
        lambda: {"total": 0, "novel": 0}
    )
    for endpoint_tree in endpoints["trees"]:
        source_name = str(endpoint_tree["source_study"])
        local_index = int(endpoint_tree["tree_index"])
        source = sources[source_name]
        row = _score_tree(
            source_name=source_name,
            source_tree=source["trees"]["trees"][local_index],
            source_metrics=source["result"]["trees"][local_index],
            endpoint_tree=endpoint_tree,
            bank=bank,
        )
        tree_rows.append(row)
        source_endpoint_counts[source_name]["total"] += row[
            "endpoint_occurrences"
        ]
        source_endpoint_counts[source_name]["novel"] += row[
            "grammar_novel_endpoint_occurrences"
        ]
        source_analyzable_counts[source_name] += int(row["analyzable"])
        for support in endpoint_tree["supports"]:
            novel_masks.update(
                item_mask(item)
                for item in support
                if item_mask(item) not in bank
            )
    total_occurrences = sum(
        row["endpoint_occurrences"] for row in tree_rows
    )
    novel_occurrences = sum(
        row["grammar_novel_endpoint_occurrences"] for row in tree_rows
    )
    analyzable = [row for row in tree_rows if row["analyzable"]]
    endpoint_gates = {
        "endpoint_novel_fraction_at_least_five_percent": (
            novel_occurrences / total_occurrences
            >= MIN_ENDPOINT_NOVEL_FRACTION
        ),
        "at_least_512_novel_endpoint_occurrences": (
            novel_occurrences >= MIN_ENDPOINT_NOVEL_OCCURRENCES
        ),
        "at_least_48_analyzable_trees": (
            len(analyzable) >= MIN_ANALYZABLE_TREES
        ),
        "each_source_has_at_least_24_analyzable_trees": all(
            source_analyzable_counts[source["name"]]
            >= MIN_ANALYZABLE_TREES_PER_SOURCE
            for source in SOURCE_STUDIES
        ),
    }
    efficacy = _aggregate_efficacy(analyzable)
    return {
        "endpoint_occurrences": total_occurrences,
        "grammar_novel_endpoint_occurrences": novel_occurrences,
        "grammar_novel_endpoint_occurrence_fraction": (
            novel_occurrences / total_occurrences
        ),
        "grammar_novel_unique_endpoint_extensions": len(novel_masks),
        "analyzable_tree_count": len(analyzable),
        "analyzable_tree_count_by_source": dict(source_analyzable_counts),
        "endpoint_occurrences_by_source": dict(source_endpoint_counts),
        "nearest_bank_hamming": nearest_bank_distances(
            novel_masks,
            bank=bank,
        ),
        "endpoint_power_gates": endpoint_gates,
        "all_endpoint_power_gates_pass": all(endpoint_gates.values()),
        "efficacy": efficacy,
        "trees": tree_rows,
    }


def _aggregate_efficacy(
    trees: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not trees:
        return {
            "available": False,
            "gates": {},
            "all_gates_pass": False,
        }
    d3 = "crossfit_depth_three"
    d2 = "crossfit_depth_two"
    metrics = (
        "mean_posterior_predictive_brier",
        "mean_best_hamming_error",
        "truth_extension_coverage_rate",
    )
    endpoint = {
        policy: {
            metric: _mean(
                [float(tree["endpoint"][policy][metric]) for tree in trees]
            )
            for metric in metrics
        }
        for policy in (d3, d2)
    }
    brier_differences = [
        float(tree["endpoint"][d3]["mean_posterior_predictive_brier"])
        - float(tree["endpoint"][d2]["mean_posterior_predictive_brier"])
        for tree in trees
    ]
    hamming_difference = (
        endpoint[d3]["mean_best_hamming_error"]
        - endpoint[d2]["mean_best_hamming_error"]
    )
    coverage_difference = (
        endpoint[d3]["truth_extension_coverage_rate"]
        - endpoint[d2]["truth_extension_coverage_rate"]
    )
    relative_brier_reduction = (
        endpoint[d2]["mean_posterior_predictive_brier"]
        - endpoint[d3]["mean_posterior_predictive_brier"]
    ) / endpoint[d2]["mean_posterior_predictive_brier"]
    wins = sum(value < -1e-12 for value in brier_differences)
    losses = sum(value > 1e-12 for value in brier_differences)
    ties = len(brier_differences) - wins - losses
    source_mean_differences = {
        source["name"]: _mean(
            [
                difference
                for tree, difference in zip(
                    trees, brier_differences, strict=True
                )
                if tree["source_study"] == source["name"]
            ]
        )
        for source in SOURCE_STUDIES
        if any(tree["source_study"] == source["name"] for tree in trees)
    }
    brier_interval = bootstrap_mean_interval(
        brier_differences,
        seed=BOOTSTRAP_SEED,
        samples=BOOTSTRAP_SAMPLES,
    )
    gates = {
        "relative_brier_reduction_at_least_one_percent": (
            relative_brier_reduction >= MIN_PRIMARY_RELATIVE_BRIER_GAIN
        ),
        "paired_brier_tree_bootstrap_ci_below_zero": (
            brier_interval[1] < 0.0
        ),
        "at_least_20_tree_wins": wins >= MIN_PRIMARY_TREE_WINS,
        "mean_hamming_does_not_increase": hamming_difference <= 0.0,
        "mean_coverage_does_not_decrease": coverage_difference >= 0.0,
        "both_source_studies_directional": (
            len(source_mean_differences) == len(SOURCE_STUDIES)
            and all(
                difference < 0.0
                for difference in source_mean_differences.values()
            )
        ),
    }
    return {
        "available": True,
        "tree_count": len(trees),
        "endpoint": endpoint,
        "candidate_minus_baseline_brier": _mean(brier_differences),
        "relative_brier_reduction": relative_brier_reduction,
        "paired_brier_tree_bootstrap_95pct": brier_interval,
        "candidate_minus_baseline_hamming": hamming_difference,
        "coverage_difference": coverage_difference,
        "tree_wins_ties_losses": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "source_mean_brier_differences": source_mean_differences,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def audit_status(
    *,
    bank_gate: bool,
    support_gate: bool,
    endpoint_gate: bool,
    efficacy_gate: bool,
) -> str:
    if not bank_gate:
        return "mechanics_failure"
    if not endpoint_gate:
        return "inconclusive"
    if not support_gate:
        return "negative"
    if efficacy_gate:
        return "positive_irreducibility_audit"
    return "mechanism_only"


def run_audit(*, output_dir: Path) -> dict[str, Any]:
    bank, bank_diagnostics = build_classical_grammar_bank()
    sources, endpoints, qwen_result = load_frozen_evidence()
    support = support_novelty(sources, bank=bank)
    endpoint = endpoint_novelty_and_efficacy(
        sources,
        endpoints,
        bank=bank,
    )
    bank_gate = (
        bank_diagnostics["unique_nonconstant_extension_count"]
        >= MIN_BANK_EXTENSIONS
    )
    status = audit_status(
        bank_gate=bank_gate,
        support_gate=support["all_gates_pass"],
        endpoint_gate=endpoint["all_endpoint_power_gates_pass"],
        efficacy_gate=endpoint["efficacy"]["all_gates_pass"],
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": (
                "results/nonmyopic/"
                "NUMBER_GAME_CLASSICAL_GRAMMAR_IRREDUCIBILITY_"
                "PREREGISTRATION.md"
            ),
            "model_calls": 0,
            "cost_usd": 0.0,
            "source_hashes": {
                "qwen_result": QWEN_RESULT_SHA256,
                "qwen_endpoints": QWEN_ENDPOINTS_SHA256,
                **{
                    f"{source['name']}_trees": source["trees_sha256"]
                    for source in SOURCE_STUDIES
                },
                **{
                    f"{source['name']}_result": source["result_sha256"]
                    for source in SOURCE_STUDIES
                },
            },
            "qwen_source_status": qwen_result["status"],
        },
        "status": status,
        "bank": {
            **bank_diagnostics,
            "minimum_required_extensions": MIN_BANK_EXTENSIONS,
            "mechanics_gate_pass": bank_gate,
        },
        "support_novelty": support,
        "endpoint_novelty_and_efficacy": endpoint,
        "all_positive_gates_pass": (
            status == "positive_irreducibility_audit"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/"
            "number_game_classical_grammar_irreducibility_audit/"
            "number-game-classical-grammar-irreducibility-audit-20260729"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(output_dir=args.output_dir)
    summary = {
        "status": result["status"],
        "bank": result["bank"],
        "support_gates": result["support_novelty"]["gates"],
        "endpoint_power_gates": result[
            "endpoint_novelty_and_efficacy"
        ]["endpoint_power_gates"],
        "efficacy": result["endpoint_novelty_and_efficacy"]["efficacy"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
