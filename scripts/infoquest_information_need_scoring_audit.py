#!/usr/bin/env python3
"""Audit target-blind score aggregations on frozen information-need outputs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import infoquest_cached_answer_ranking_diagnostic as diagnostic
from scripts import infoquest_cached_partition_eig_gate as cached
from scripts import infoquest_discrete_action_causal_gate as discrete
from scripts import infoquest_information_need_gate as needs
from scripts import infoquest_support_causal_link_gate as base
from scripts import infoquest_target_alignment_audit as alignment


INTERFACE_VERSION = "infoquest-information-need-scoring-audit-1"
MECHANICS_PUBLIC_SHA256 = (
    "e38980e38254a3d1b1c19865cf3bde11c7cccfa43150c38973c2978f098b9312"
)
MECHANICS_RAW_SHA256 = (
    "a12153fcdee1d532d843274b2eb1998a959b60706a6e1beb1293b378fe91d530"
)
POWERS = (1, 2, 3, 4, 6, 8, 12, 16)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_scores(
    belief: needs.InformationNeedBelief,
) -> dict[str, tuple[float, ...]]:
    weights = belief.weights
    profiles = belief.resolution_probabilities
    total_weight = sum(weights)
    variants: dict[str, tuple[float, ...]] = {}
    for power in POWERS:
        variants[f"power_{power}"] = tuple(
            sum(
                weight * (probability / 100.0) ** power
                for weight, probability in zip(weights, profile)
            )
            / total_weight
            for profile in profiles
        )
    variants["max_resolution"] = tuple(
        float(max(profile)) for profile in profiles
    )
    variants["weighted_max_resolution"] = tuple(
        float(
            max(
                weight * probability
                for weight, probability in zip(weights, profile)
            )
        )
        for profile in profiles
    )
    variants["resolution_margin"] = tuple(
        float(
            sorted(profile, reverse=True)[0]
            - sorted(profile, reverse=True)[1]
        )
        for profile in profiles
    )
    variants["max_minus_mean_remainder"] = tuple(
        max(profile) - (sum(profile) - max(profile)) / (needs.NEED_COUNT - 1)
        for profile in profiles
    )
    variants["peak_ratio"] = tuple(
        max(profile) / sum(profile) if sum(profile) else 0.0
        for profile in profiles
    )
    return variants


def load_mechanics_beliefs(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    *,
    raw_path: Path,
    public_path: Path,
) -> list[list[needs.InformationNeedBelief]]:
    if _sha256_path(raw_path) != MECHANICS_RAW_SHA256:
        raise ValueError("information-need private raw SHA-256 mismatch")
    if _sha256_path(public_path) != MECHANICS_PUBLIC_SHA256:
        raise ValueError("information-need public SHA-256 mismatch")
    raw = json.loads(raw_path.read_text())
    public = json.loads(public_path.read_text())
    protocol = public.get("protocol", {})
    if raw.get("interface_version") != needs.INTERFACE_VERSION:
        raise ValueError("information-need raw interface mismatch")
    if protocol.get("interface_version") != needs.INTERFACE_VERSION:
        raise ValueError("information-need public interface mismatch")
    if protocol.get("private_raw_sha256") != MECHANICS_RAW_SHA256:
        raise ValueError("information-need public artifact does not bind raw")
    responses = raw.get("information_need_beliefs")
    if not isinstance(responses, list) or len(responses) != 30:
        raise ValueError("information-need raw does not contain thirty beliefs")
    parsed = []
    for response_index, response in enumerate(responses):
        fixture_index = response_index // base.ROOT_COUNT
        root_index = response_index % base.ROOT_COUNT
        parsed.append(
            needs.parse_information_need_belief(
                response,
                initials[fixtures[fixture_index].record_id],
                root_index,
            )
        )
    return needs._reshape(parsed)


def _rescore(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    beliefs: Sequence[Sequence[needs.InformationNeedBelief]],
    variant: str,
) -> list[list[needs.InformationNeedBelief]]:
    rows = []
    for fixture_index, fixture in enumerate(fixtures):
        initial = initials[fixture.record_id]
        row = []
        for root_index, belief in enumerate(beliefs[fixture_index]):
            scores = variant_scores(belief)[variant]
            selected = max(range(len(scores)), key=scores.__getitem__)
            candidates = discrete.candidate_bank(
                initial,
                root_index,
            )
            row.append(
                replace(
                    belief,
                    scores=scores,
                    selected_action_index=selected,
                    selected_question=candidates[selected],
                )
            )
        rows.append(row)
    return rows


def audit(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    beliefs: Sequence[Sequence[needs.InformationNeedBelief]],
    dynamic_baseline: Sequence[Sequence[Any]],
    fixed_baseline: Sequence[Sequence[Any]],
    judgments: Sequence[base.ChecklistJudgment],
) -> dict[str, Any]:
    variant_names = list(variant_scores(beliefs[0][0]))
    results = {}
    for variant in variant_names:
        rescored = _rescore(
            fixtures,
            initials,
            beliefs,
            variant,
        )
        metrics, fixture_metrics, retrospective_gates = needs._target_metrics(
            fixtures,
            initials,
            rescored,
            dynamic_baseline,
            fixed_baseline,
            judgments,
        )
        results[variant] = {
            "metrics": metrics,
            "fixture_metrics": fixture_metrics,
            "retrospective_v2_thresholds": retrospective_gates,
            "retrospective_all_thresholds_met": all(
                retrospective_gates.values()
            ),
        }

    linear = results["power_1"]["metrics"]
    expected = {
        "mean_information_need_target_spearman": -0.12375393350773627,
        "mean_information_need_selected_target_gain": 0.43333333333333335,
        "information_need_target_optimal_cells": 11,
    }
    for key, value in expected.items():
        observed = linear[key]
        if isinstance(value, float):
            if abs(observed - value) > 1e-12:
                raise ValueError(f"linear audit does not reproduce {key}")
        elif observed != value:
            raise ValueError(f"linear audit does not reproduce {key}")

    ranked = sorted(
        variant_names,
        key=lambda name: (
            results[name]["metrics"][
                "mean_information_need_selected_target_gain"
            ],
            results[name]["metrics"][
                "mean_information_need_target_spearman"
            ],
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "status": "diagnostic_complete",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "post_hoc_development_audit": True,
            "source_mechanics_public_sha256": MECHANICS_PUBLIC_SHA256,
            "source_mechanics_raw_sha256": MECHANICS_RAW_SHA256,
            "source_target_audit_sha256": needs.TARGET_AUDIT_SHA256,
            "variant_family": variant_names,
            "target_or_checklist_content_used_by_scorers": False,
            "private_semantic_text_emitted": False,
            "llm_calls": 0,
            "cost_usd": 0.0,
            "cannot_rescue_v2": True,
        },
        "ranked_by_selected_gain_then_target_spearman": ranked,
        "variants": results,
        "validation": {
            "power_1_exactly_reproduces_banked_v2": True,
            "variants_evaluated": len(variant_names),
            "cells_per_variant": 30,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cached-raw", type=Path, required=True)
    parser.add_argument("--cached-public", type=Path, required=True)
    parser.add_argument("--v3-raw", type=Path, required=True)
    parser.add_argument("--v3-public", type=Path, required=True)
    parser.add_argument("--diagnostic-raw", type=Path, required=True)
    parser.add_argument("--diagnostic-public", type=Path, required=True)
    parser.add_argument("--mechanics-raw", type=Path, required=True)
    parser.add_argument("--mechanics-public", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    needs._verify_target_audit(args.target_audit)
    fixtures, public_fixture = base.build_fixtures(
        settings_path=args.settings,
        baseline_path=args.baseline,
    )
    initials, _ = cached.load_cached_histories(
        fixtures,
        raw_path=args.cached_raw,
        public_path=args.cached_public,
    )
    dynamic, fixed = diagnostic.load_v3_beliefs(
        fixtures,
        initials,
        raw_path=args.v3_raw,
        public_path=args.v3_public,
    )
    judgments, fixture_ids = alignment.load_diagnostic_judgments(
        raw_path=args.diagnostic_raw,
        public_path=args.diagnostic_public,
    )
    if [fixture.fixture_id for fixture in fixtures] != fixture_ids:
        raise ValueError("fixture order differs across target artifacts")
    beliefs = load_mechanics_beliefs(
        fixtures,
        initials,
        raw_path=args.mechanics_raw,
        public_path=args.mechanics_public,
    )
    result = audit(
        fixtures,
        initials,
        beliefs,
        dynamic,
        fixed,
        judgments,
    )
    result["protocol"]["fixture_sha256"] = public_fixture[
        "private_fixture_sha256"
    ]
    base._checkpoint(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
