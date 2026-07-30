#!/usr/bin/env python3
"""Audit whether repeated Number Game proposals deserve more belief mass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    DOMAIN,
    RuleHypothesis,
    best_query,
)
from scripts.number_game_item_isolated_codec import (
    parse_proposals_item_isolated,
)
from scripts.number_game_pooled_support import POOLED_FIELD, POOL_SIZE
from scripts.number_game_qwen_external_canonical_confirmation import (
    canonical_targets,
)
from scripts.number_game_retained_depth_three import _rule


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-proposal-frequency-calibration-audit-1"
TREE_COUNT_PER_STUDY = 32
BOOTSTRAP_SEED = 68_900
BOOTSTRAP_SAMPLES = 20_000
STAGES = ("initial", "after_one_observation", "after_two_observations")
STUDIES = (
    {
        "name": "development_first_link",
        "role": "development",
        "directory": (
            REPO_ROOT
            / "results/nonmyopic/number_game_qwen_pooled_first_link_confirmation32"
            / "number-game-qwen-pooled-first-link-confirmation32-20260729T094455Z"
        ),
        "trees_sha256": (
            "78bbc36bab604c78e31a68e50d31a1f9d92bea02bd295b1f983fe2f2a96a4d68"
        ),
        "raw_sha256": (
            "522ae568b7856f1fb4f8614bf69bef92357276954f8e9ebcbd3c7e71766ec349"
        ),
    },
    {
        "name": "development_second_refresh",
        "role": "development",
        "directory": (
            REPO_ROOT
            / "results/nonmyopic"
            / "number_game_qwen_pooled_second_refresh_confirmation32"
            / "number-game-qwen-pooled-second-refresh-confirmation32-20260729T104705Z"
        ),
        "trees_sha256": (
            "5061327820199d1b27fac36e351708009184a7a0946bfa0ed21201b583c2f709"
        ),
        "raw_sha256": (
            "1571d6555943e3f2e1256690afaeb2086e060366edbc3c7167165e6ac027456f"
        ),
    },
    {
        "name": "heldout_dynamic_fixed",
        "role": "heldout",
        "directory": (
            REPO_ROOT
            / "results/nonmyopic"
            / "number_game_qwen_dynamic_vs_fixed_confirmation32"
            / "number-game-qwen-dynamic-vs-fixed-confirmation32-20260729T215013Z"
        ),
        "trees_sha256": (
            "dff467d9590854a52c80ae0fec4d3a9ef3cedb315a8712258fd25bed1dce2149"
        ),
        "raw_sha256": (
            "112ebca54cd8b3e0a91adda4b97c60a5cb68da51f1fa3c73d9b204e0854c6997"
        ),
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(values: Sequence[float]) -> list[float]:
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def unique_extensions(
    support: Iterable[RuleHypothesis],
) -> set[tuple[bool, ...]]:
    return {hypothesis.extension for hypothesis in support}


def dedupe_particles(
    support: Iterable[RuleHypothesis],
) -> list[RuleHypothesis]:
    unique = []
    seen = set()
    for hypothesis in support:
        if hypothesis.extension in seen:
            continue
        seen.add(hypothesis.extension)
        unique.append(hypothesis)
    return unique


def extension_sha256(extension: Sequence[bool]) -> str:
    return hashlib.sha256(bytes(extension)).hexdigest()


def validate_public_support(
    support: Sequence[RuleHypothesis],
    public_items: Sequence[dict[str, Any]],
    *,
    label: str,
) -> None:
    reconstructed = {
        extension_sha256(hypothesis.extension) for hypothesis in support
    }
    expected = {str(item["extension_sha256"]) for item in public_items}
    if reconstructed != expected or len(support) != len(public_items):
        raise ValueError(f"{label} does not reconstruct public support")


def pooled_particle_multiset(
    response: str,
    *,
    observations: Sequence[tuple[int, bool]] = (),
) -> list[RuleHypothesis]:
    value = json.loads(response)
    if (
        not isinstance(value, dict)
        or set(value) != {POOLED_FIELD}
        or not isinstance(value[POOLED_FIELD], list)
        or len(value[POOLED_FIELD]) != POOL_SIZE
        or not all(isinstance(item, str) for item in value[POOLED_FIELD])
    ):
        raise ValueError("expected an exact two-draw pooled response")
    particles = []
    for raw_response in value[POOLED_FIELD]:
        support, _ = parse_proposals_item_isolated(
            raw_response,
            observations=observations,
        )
        particles.extend(support)
    return particles


def retain_particle_multiset(
    *,
    generated: Sequence[RuleHypothesis],
    parent: Sequence[RuleHypothesis],
    query: int,
    label: bool,
) -> list[RuleHypothesis]:
    return list(generated) + [
        hypothesis
        for hypothesis in parent
        if hypothesis.extension[query] == label
    ]


def posterior_predictive_brier(
    support: Sequence[RuleHypothesis],
    target: RuleHypothesis,
    *,
    excluded: Iterable[int] = (),
) -> float:
    if not support:
        return 1.0
    excluded_set = set(excluded)
    numbers = [number for number in DOMAIN if number not in excluded_set]
    probabilities = [
        sum(hypothesis.extension[number] for hypothesis in support)
        / len(support)
        for number in numbers
    ]
    return _mean(
        [
            (probability - float(target.extension[number])) ** 2
            for number, probability in zip(
                numbers,
                probabilities,
                strict=True,
            )
        ]
    )


def predictive_probabilities(
    support: Sequence[RuleHypothesis],
) -> tuple[float, ...]:
    if not support:
        return (0.5,) * len(DOMAIN)
    return tuple(
        sum(hypothesis.extension[number] for hypothesis in support)
        / len(support)
        for number in DOMAIN
    )


def brier_from_probabilities(
    probabilities: Sequence[float],
    target: RuleHypothesis,
    *,
    excluded: Iterable[int] = (),
) -> float:
    excluded_set = set(excluded)
    return _mean(
        [
            (probability - float(target.extension[number])) ** 2
            for number, probability in enumerate(probabilities)
            if number not in excluded_set
        ]
    )


def _raw_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[Any, ...], str]:
    return {
        tuple(row["key"]): str(row["response"])
        for row in rows
    }


def reconstruct_tree_supports(
    public_tree: dict[str, Any],
    raw_tree: dict[str, Any],
) -> dict[str, Any]:
    roots = [int(root) for root in public_tree["roots"]]
    weighted_initial = pooled_particle_multiset(
        raw_tree["initial_response"]
    )
    uniform_initial = dedupe_particles(weighted_initial)
    validate_public_support(
        uniform_initial,
        public_tree["initial"],
        label="initial multiset",
    )

    raw_first = _raw_rows(raw_tree["first_responses"])
    generated_first = {}
    uniform_first = {}
    weighted_first = {}
    for root in roots:
        for label in (False, True):
            key = (root, label)
            generated = pooled_particle_multiset(
                raw_first[key],
                observations=(key,),
            )
            generated_first[key] = generated
            weighted_first[key] = retain_particle_multiset(
                generated=generated,
                parent=weighted_initial,
                query=root,
                label=label,
            )
            uniform_first[key] = dedupe_particles(weighted_first[key])
            validate_public_support(
                uniform_first[key],
                public_tree["first_branches"][
                    f"{root}:{int(label)}"
                ],
                label=f"first branch {key}",
            )

    raw_second = _raw_rows(raw_tree["second_responses"])
    generated_second = {}
    uniform_second = {}
    weighted_second = {}
    stored_second_queries = {}
    for key, response in raw_second.items():
        root, first_label, second_query, second_label = key
        branch_key = (root, first_label)
        previous = stored_second_queries.setdefault(
            branch_key,
            second_query,
        )
        if previous != second_query:
            raise ValueError("saved branch has inconsistent second queries")
        observations = (
            (root, first_label),
            (second_query, second_label),
        )
        generated = pooled_particle_multiset(
            response,
            observations=observations,
        )
        generated_second[key] = generated
        weighted_second[key] = retain_particle_multiset(
            generated=generated,
            parent=weighted_first[branch_key],
            query=second_query,
            label=second_label,
        )
        uniform_second[key] = dedupe_particles(weighted_second[key])
        public_key = (
            f"{root}:{int(first_label)}:"
            f"{second_query}:{int(second_label)}"
        )
        validate_public_support(
            uniform_second[key],
            public_tree["second_branches"][public_key],
            label=f"second branch {key}",
        )
    return {
        "roots": roots,
        "uniform_initial": uniform_initial,
        "weighted_initial": weighted_initial,
        "uniform_first": uniform_first,
        "generated_first": generated_first,
        "weighted_first": weighted_first,
        "uniform_second": uniform_second,
        "generated_second": generated_second,
        "weighted_second": weighted_second,
        "stored_second_queries": stored_second_queries,
    }


def score_target_bank(
    supports: dict[str, Any],
    targets: Sequence[RuleHypothesis],
) -> dict[str, Any]:
    uniform = {stage: [] for stage in STAGES}
    weighted = {stage: [] for stage in STAGES}
    uniform_probabilities = {
        "initial": predictive_probabilities(supports["uniform_initial"]),
        "first": {
            key: predictive_probabilities(support)
            for key, support in supports["uniform_first"].items()
        },
        "second": {
            key: predictive_probabilities(support)
            for key, support in supports["uniform_second"].items()
        },
    }
    weighted_probabilities = {
        "initial": predictive_probabilities(supports["weighted_initial"]),
        "first": {
            key: predictive_probabilities(support)
            for key, support in supports["weighted_first"].items()
        },
        "second": {
            key: predictive_probabilities(support)
            for key, support in supports["weighted_second"].items()
        },
    }
    for target in targets:
        uniform["initial"].append(
            brier_from_probabilities(
                uniform_probabilities["initial"],
                target,
            )
        )
        weighted["initial"].append(
            brier_from_probabilities(
                weighted_probabilities["initial"],
                target,
            )
        )
        for root in supports["roots"]:
            first_label = target.extension[root]
            first_key = (root, first_label)
            uniform["after_one_observation"].append(
                brier_from_probabilities(
                    uniform_probabilities["first"][first_key],
                    target,
                    excluded=(root,),
                )
            )
            weighted["after_one_observation"].append(
                brier_from_probabilities(
                    weighted_probabilities["first"][first_key],
                    target,
                    excluded=(root,),
                )
            )
            second_query = supports["stored_second_queries"][first_key]
            second_label = target.extension[second_query]
            second_key = (
                root,
                first_label,
                second_query,
                second_label,
            )
            uniform["after_two_observations"].append(
                brier_from_probabilities(
                    uniform_probabilities["second"][second_key],
                    target,
                    excluded=(root, second_query),
                )
            )
            weighted["after_two_observations"].append(
                brier_from_probabilities(
                    weighted_probabilities["second"][second_key],
                    target,
                    excluded=(root, second_query),
                )
            )
    return {
        stage: {
            "uniform_brier": _mean(uniform[stage]),
            "frequency_weighted_brier": _mean(weighted[stage]),
            "weighted_minus_uniform_brier": (
                _mean(weighted[stage]) - _mean(uniform[stage])
            ),
            "evaluated_target_paths": len(uniform[stage]),
        }
        for stage in STAGES
    }


def load_study(study: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = Path(study["directory"])
    trees_path = directory / "TREES.json"
    raw_path = directory / "private/RAW_RESPONSES.json"
    if sha256_file(trees_path) != study["trees_sha256"]:
        raise ValueError(f"{study['name']} public trees hash changed")
    if sha256_file(raw_path) != study["raw_sha256"]:
        raise ValueError(f"{study['name']} private raw hash changed")
    public = json.loads(trees_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if (
        len(public["trees"]) != TREE_COUNT_PER_STUDY
        or len(raw["trees"]) != TREE_COUNT_PER_STUDY
    ):
        raise ValueError(f"{study['name']} tree count changed")
    return public, raw


def score_study(study: dict[str, Any]) -> list[dict[str, Any]]:
    public, raw = load_study(study)
    canonical = canonical_targets()
    rows = []
    for public_tree, raw_tree in zip(
        public["trees"],
        raw["trees"],
        strict=True,
    ):
        supports = reconstruct_tree_supports(public_tree, raw_tree)
        validation_targets = [
            _rule(item)
            for draw in public_tree["validation_supports"]
            for item in draw
        ]
        compatibility = [
            best_query(
                supports["weighted_first"][(root, label)],
                excluded=(root,),
            )[0]
            == supports["stored_second_queries"][(root, label)]
            for root in supports["roots"]
            for label in (False, True)
        ]
        rows.append(
            {
                "study": study["name"],
                "role": study["role"],
                "tree_index": int(public_tree["tree_index"]),
                "tree_seed": int(public_tree["tree_seed"]),
                "canonical": score_target_bank(supports, canonical),
                "independent_llm_validation": score_target_bank(
                    supports,
                    validation_targets,
                ),
                "particle_diagnostics": {
                    "initial_unique_support_size": len(
                        supports["uniform_initial"]
                    ),
                    "initial_particle_count": len(
                        supports["weighted_initial"]
                    ),
                    "second_query_compatibility_count": sum(compatibility),
                    "second_query_branch_count": len(compatibility),
                },
            }
        )
    return rows


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    bank: str,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    summary = {}
    for stage in STAGES:
        uniform = [
            float(row[bank][stage]["uniform_brier"]) for row in rows
        ]
        weighted = [
            float(row[bank][stage]["frequency_weighted_brier"])
            for row in rows
        ]
        differences = [
            weighted_value - uniform_value
            for uniform_value, weighted_value in zip(
                uniform,
                weighted,
                strict=True,
            )
        ]
        bootstrap = [
            _mean([rng.choice(differences) for _ in differences])
            for _ in range(BOOTSTRAP_SAMPLES)
        ]
        uniform_mean = _mean(uniform)
        weighted_mean = _mean(weighted)
        summary[stage] = {
            "tree_count": len(rows),
            "uniform_brier": uniform_mean,
            "frequency_weighted_brier": weighted_mean,
            "weighted_minus_uniform_brier": weighted_mean - uniform_mean,
            "relative_brier_reduction": (
                (uniform_mean - weighted_mean) / uniform_mean
            ),
            "tree_cluster_bootstrap_95pct": _interval(bootstrap),
            "tree_wins": sum(value < -1e-15 for value in differences),
            "tree_ties": sum(abs(value) <= 1e-15 for value in differences),
            "tree_losses": sum(value > 1e-15 for value in differences),
        }
    return summary


def promotion_gates(
    by_study: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    development = [
        study for study in STUDIES if study["role"] == "development"
    ]
    heldout = next(study for study in STUDIES if study["role"] == "heldout")
    heldout_summary = by_study[heldout["name"]]
    return {
        "both_development_cohorts_improve_at_all_stages": all(
            by_study[study["name"]]["canonical"][stage][
                "weighted_minus_uniform_brier"
            ]
            < 0.0
            for study in development
            for stage in STAGES
        ),
        "heldout_improves_at_all_stages": all(
            heldout_summary["canonical"][stage][
                "weighted_minus_uniform_brier"
            ]
            < 0.0
            for stage in STAGES
        ),
        "heldout_one_observation_interval_below_zero": (
            heldout_summary["canonical"]["after_one_observation"][
                "tree_cluster_bootstrap_95pct"
            ][1]
            < 0.0
        ),
        "heldout_two_observation_interval_below_zero": (
            heldout_summary["canonical"]["after_two_observations"][
                "tree_cluster_bootstrap_95pct"
            ][1]
            < 0.0
        ),
        "heldout_llm_validation_improves_after_observations": all(
            heldout_summary["independent_llm_validation"][stage][
                "weighted_minus_uniform_brier"
            ]
            < 0.0
            for stage in STAGES[1:]
        ),
    }


def run_audit(output_dir: Path) -> dict[str, Any]:
    rows = [
        row
        for study in STUDIES
        for row in score_study(study)
    ]
    by_study = {}
    for study_index, study in enumerate(STUDIES):
        study_rows = [row for row in rows if row["study"] == study["name"]]
        by_study[study["name"]] = {
            "role": study["role"],
            "canonical": summarize_rows(
                study_rows,
                bank="canonical",
                seed=BOOTSTRAP_SEED + study_index * 10,
            ),
            "independent_llm_validation": summarize_rows(
                study_rows,
                bank="independent_llm_validation",
                seed=BOOTSTRAP_SEED + study_index * 10 + 1,
            ),
            "particle_diagnostics": {
                "mean_initial_unique_support_size": _mean(
                    [
                        row["particle_diagnostics"][
                            "initial_unique_support_size"
                        ]
                        for row in study_rows
                    ]
                ),
                "mean_initial_particle_count": _mean(
                    [
                        row["particle_diagnostics"][
                            "initial_particle_count"
                        ]
                        for row in study_rows
                    ]
                ),
                "second_query_compatibility_rate": (
                    sum(
                        row["particle_diagnostics"][
                            "second_query_compatibility_count"
                        ]
                        for row in study_rows
                    )
                    / sum(
                        row["particle_diagnostics"][
                            "second_query_branch_count"
                        ]
                        for row in study_rows
                    )
                ),
            },
        }
    gates = promotion_gates(by_study)
    promoted = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "retrospective_calibration_positive"
            if promoted
            else "retrospective_calibration_null"
        ),
        "decision": (
            "authorize_prospective_frequency_weighted_policy_smoke"
            if promoted
            else "close_proposal_frequency_weighting_route"
        ),
        "protocol": {
            "analysis_is_retrospective": True,
            "formal_source_statuses_unchanged": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "planner": "qwen/qwen3.7-plus",
            "pool_size": POOL_SIZE,
            "particle_semantics": (
                "concatenate independently parsed draws; at each refreshed "
                "branch concatenate generated particles with every "
                "consistent parent particle"
            ),
            "uniform_control": (
                "the exact saved extension-deduplicated retained support"
            ),
            "primary_target_bank": (
                "33 equal-weight canonical Number Game concepts"
            ),
            "secondary_target_bank": (
                "all saved independent Gemini validation proposals"
            ),
            "development_studies": [
                study["name"]
                for study in STUDIES
                if study["role"] == "development"
            ],
            "heldout_study": next(
                study["name"]
                for study in STUDIES
                if study["role"] == "heldout"
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "source_hashes": {
                study["name"]: {
                    "trees_sha256": study["trees_sha256"],
                    "private_raw_sha256": study["raw_sha256"],
                }
                for study in STUDIES
            },
            "limitation": (
                "saved second-step generations exist only for the uniform "
                "policy's second query, so this audit tests calibration and "
                "does not replay a frequency-weighted planning policy"
            ),
        },
        "promotion_gates": gates,
        "studies": by_study,
        "trees": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = run_audit(args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "promotion_gates": result["promotion_gates"],
                "studies": result["studies"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
