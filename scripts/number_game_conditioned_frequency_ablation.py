#!/usr/bin/env python3
"""Isolate history-conditioned proposal frequency from initial prior mass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import RuleHypothesis, best_query
from scripts.number_game_proposal_frequency_calibration_audit import (
    BOOTSTRAP_SEED,
    STAGES,
    STUDIES,
    _mean,
    load_study,
    reconstruct_tree_supports,
    retain_particle_multiset,
    score_target_bank,
    summarize_rows,
)
from scripts.number_game_qwen_external_canonical_confirmation import (
    canonical_targets,
)
from scripts.number_game_retained_depth_three import _rule


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-conditioned-frequency-ablation-1"
VARIANTS = ("propagated_conditioned_particles", "local_refresh_particles")
OBSERVED_STAGES = STAGES[1:]


def conditioned_support_variants(
    supports: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    conditioned_first = {
        key: retain_particle_multiset(
            generated=supports["generated_first"][key],
            parent=supports["uniform_initial"],
            query=key[0],
            label=key[1],
        )
        for key in supports["uniform_first"]
    }
    propagated_second = {}
    local_second = {}
    for key, generated in supports["generated_second"].items():
        root, first_label, second_query, second_label = key
        first_key = (root, first_label)
        propagated_second[key] = retain_particle_multiset(
            generated=generated,
            parent=conditioned_first[first_key],
            query=second_query,
            label=second_label,
        )
        local_second[key] = retain_particle_multiset(
            generated=generated,
            parent=supports["uniform_first"][first_key],
            query=second_query,
            label=second_label,
        )

    common = {
        **supports,
        "weighted_initial": supports["uniform_initial"],
        "weighted_first": conditioned_first,
    }
    return {
        "propagated_conditioned_particles": {
            **common,
            "weighted_second": propagated_second,
        },
        "local_refresh_particles": {
            **common,
            "weighted_second": local_second,
        },
    }


def score_study(study: dict[str, Any]) -> list[dict[str, Any]]:
    public, raw = load_study(study)
    canonical = canonical_targets()
    rows = []
    for public_tree, raw_tree in zip(
        public["trees"],
        raw["trees"],
        strict=True,
    ):
        base = reconstruct_tree_supports(public_tree, raw_tree)
        variants = conditioned_support_variants(base)
        validation_targets = [
            _rule(item)
            for draw in public_tree["validation_supports"]
            for item in draw
        ]
        variant_scores = {}
        for variant, supports in variants.items():
            compatibility = [
                best_query(
                    supports["weighted_first"][(root, label)],
                    excluded=(root,),
                )[0]
                == supports["stored_second_queries"][(root, label)]
                for root in supports["roots"]
                for label in (False, True)
            ]
            variant_scores[variant] = {
                "canonical": score_target_bank(supports, canonical),
                "independent_llm_validation": score_target_bank(
                    supports,
                    validation_targets,
                ),
                "second_query_compatibility_count": sum(compatibility),
                "second_query_branch_count": len(compatibility),
            }
        rows.append(
            {
                "study": study["name"],
                "role": study["role"],
                "tree_index": int(public_tree["tree_index"]),
                "tree_seed": int(public_tree["tree_seed"]),
                "variants": variant_scores,
            }
        )
    return rows


def _summary_input(
    rows: Sequence[dict[str, Any]],
    *,
    variant: str,
) -> list[dict[str, Any]]:
    return [
        {
            "canonical": row["variants"][variant]["canonical"],
            "independent_llm_validation": row["variants"][variant][
                "independent_llm_validation"
            ],
        }
        for row in rows
    ]


def summarize_studies(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for study_index, study in enumerate(STUDIES):
        study_rows = [row for row in rows if row["study"] == study["name"]]
        variants = {}
        for variant_index, variant in enumerate(VARIANTS):
            summary_rows = _summary_input(study_rows, variant=variant)
            variants[variant] = {
                "canonical": summarize_rows(
                    summary_rows,
                    bank="canonical",
                    seed=(
                        BOOTSTRAP_SEED
                        + 100
                        + study_index * 20
                        + variant_index * 2
                    ),
                ),
                "independent_llm_validation": summarize_rows(
                    summary_rows,
                    bank="independent_llm_validation",
                    seed=(
                        BOOTSTRAP_SEED
                        + 101
                        + study_index * 20
                        + variant_index * 2
                    ),
                ),
                "second_query_compatibility_rate": (
                    sum(
                        row["variants"][variant][
                            "second_query_compatibility_count"
                        ]
                        for row in study_rows
                    )
                    / sum(
                        row["variants"][variant][
                            "second_query_branch_count"
                        ]
                        for row in study_rows
                    )
                ),
            }
        summaries[study["name"]] = {
            "role": study["role"],
            "variants": variants,
        }
    return summaries


def prospective_gates(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    heldout_name = next(
        study["name"] for study in STUDIES if study["role"] == "heldout"
    )
    heldout = summaries[heldout_name]["variants"]
    return {
        "both_variants_improve_canonical_brier_in_every_cohort_and_stage": all(
            summaries[study["name"]]["variants"][variant]["canonical"][
                stage
            ]["weighted_minus_uniform_brier"]
            < 0.0
            for study in STUDIES
            for variant in VARIANTS
            for stage in OBSERVED_STAGES
        ),
        "heldout_local_refresh_intervals_below_zero": all(
            heldout["local_refresh_particles"]["canonical"][stage][
                "tree_cluster_bootstrap_95pct"
            ][1]
            < 0.0
            for stage in OBSERVED_STAGES
        ),
        "heldout_propagated_intervals_below_zero": all(
            heldout["propagated_conditioned_particles"]["canonical"][stage][
                "tree_cluster_bootstrap_95pct"
            ][1]
            < 0.0
            for stage in OBSERVED_STAGES
        ),
        "heldout_llm_validation_improves_for_both_variants": all(
            heldout[variant]["independent_llm_validation"][stage][
                "weighted_minus_uniform_brier"
            ]
            < 0.0
            for variant in VARIANTS
            for stage in OBSERVED_STAGES
        ),
    }


def run_ablation(output_dir: Path) -> dict[str, Any]:
    rows = [
        row
        for study in STUDIES
        for row in score_study(study)
    ]
    summaries = summarize_studies(rows)
    gates = prospective_gates(summaries)
    authorized = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "retrospective_conditioned_frequency_positive"
            if authorized
            else "retrospective_conditioned_frequency_null"
        ),
        "decision": (
            "authorize_fresh_conditioned_particle_policy_smoke"
            if authorized
            else "close_conditioned_particle_route"
        ),
        "protocol": {
            "analysis_is_post_hoc": True,
            "formal_source_statuses_unchanged": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "initial_prior": "uniform extension-deduplicated support",
            "propagated_variant": (
                "concatenate current generated particles with all consistent "
                "particles from the preceding conditioned support"
            ),
            "local_variant": (
                "concatenate current generated particles with one copy of "
                "each consistent hypothesis in the saved preceding support"
            ),
            "primary_target_bank": (
                "33 equal-weight canonical Number Game concepts"
            ),
            "secondary_target_bank": (
                "all saved independent Gemini validation proposals"
            ),
            "observed_stages_only": list(OBSERVED_STAGES),
            "source_hashes": {
                study["name"]: {
                    "trees_sha256": study["trees_sha256"],
                    "private_raw_sha256": study["raw_sha256"],
                }
                for study in STUDIES
            },
            "limitation": (
                "saved generations condition on the uniform policy's second "
                "query; this ablation tests belief calibration, not a fully "
                "replanned particle policy"
            ),
        },
        "prospective_gates": gates,
        "studies": summaries,
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
    result = run_ablation(args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "prospective_gates": result["prospective_gates"],
                "studies": result["studies"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
