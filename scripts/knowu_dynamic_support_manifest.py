#!/usr/bin/env python3
"""Freeze task-family splits for KnowU dynamic-support BED."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

try:
    from scripts import knowu_bench_source_audit as source_audit
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    import knowu_bench_source_audit as source_audit


SELECTION_SEED = 24_415
SPLIT_SIZES = {
    "mechanics": 2,
    "opportunity": 3,
    "development": 2,
}
MINIMUM_ELIGIBLE_FAMILIES = 10

# Filled after the first content-sealed invocation and enforced on replay.
EXPECTED_ELIGIBLE_FAMILIES: int | None = None
EXPECTED_SPLIT_HASHES: dict[str, str] | None = None
EXPECTED_MECHANICS_IDS: tuple[str, ...] | None = None


def ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(values).encode("utf-8")
    ).hexdigest()


def eligible_families(root: Path) -> dict[str, dict[str, Any]]:
    return {
        family["class_name"]: family
        for family in source_audit.preference_task_families(root)
        if (
            family["hard"]
            and family["agent_user_interaction"]
            and family["profile_variants"] >= 3
            and family["static_nonempty_goal"]
        )
    }


def split_families(
    eligible: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    family_ids = sorted(eligible)
    random.Random(SELECTION_SEED).shuffle(family_ids)
    cursor = 0
    splits: dict[str, list[str]] = {}
    for name, size in SPLIT_SIZES.items():
        splits[name] = family_ids[cursor : cursor + size]
        cursor += size
    splits["holdout"] = family_ids[cursor:]
    return splits


def build_manifest(
    source_root: Path,
    *,
    enforce_frozen: bool = True,
) -> dict[str, Any]:
    audit = source_audit.build_audit(source_root)
    eligible = eligible_families(source_root)
    splits = split_families(eligible)
    split_hashes = {
        name: ordered_hash(family_ids)
        for name, family_ids in splits.items()
    }
    mechanics_ids = tuple(splits["mechanics"])

    if enforce_frozen:
        if any(
            value is None
            for value in (
                EXPECTED_ELIGIBLE_FAMILIES,
                EXPECTED_SPLIT_HASHES,
                EXPECTED_MECHANICS_IDS,
            )
        ):
            raise ValueError("KnowU manifest constants are not frozen")
        if len(eligible) != EXPECTED_ELIGIBLE_FAMILIES:
            raise ValueError("KnowU eligible family count changed")
        if split_hashes != EXPECTED_SPLIT_HASHES:
            raise ValueError("KnowU split hashes changed")
        if mechanics_ids != EXPECTED_MECHANICS_IDS:
            raise ValueError("KnowU mechanics IDs changed")

    gates = {
        "source_audit_authorizes_manifest": audit["decision"][
            "derived_dynamic_support_manifest_authorized"
        ],
        "at_least_10_eligible_families": (
            len(eligible) >= MINIMUM_ELIGIBLE_FAMILIES
        ),
        "all_splits_nonempty": all(splits.values()),
        "every_selected_family_has_at_least_three_worlds": all(
            eligible[family_id]["profile_variants"] >= 3
            for family_ids in splits.values()
            for family_id in family_ids
        ),
    }
    return {
        "interface_version": "knowu-dynamic-support-manifest-1",
        "source": audit["source"],
        "construction": {
            "selection_seed": SELECTION_SEED,
            "minimum_eligible_families": MINIMUM_ELIGIBLE_FAMILIES,
            "eligible_families": len(eligible),
            "world_prior": "uniform_over_supported_official_profiles",
            "profile_labels_hidden_from_policy": True,
            "initial_observation": (
                "task_relevant_clean_behavior_logs_only"
            ),
            "question_interface": "one_atomic_task_preference_per_turn",
            "belief_support": "llm_generated_and_history_regenerated",
        },
        "splits": {
            name: {
                "family_ids": family_ids,
                "ordered_sha256": split_hashes[name],
                "family_count": len(family_ids),
                "profile_world_count": sum(
                    eligible[family_id]["profile_variants"]
                    for family_id in family_ids
                ),
                "profile_variant_histogram": dict(
                    sorted(
                        Counter(
                            eligible[family_id]["profile_variants"]
                            for family_id in family_ids
                        ).items()
                    )
                ),
            }
            for name, family_ids in splits.items()
        },
        "gates": gates,
        "passed": all(gates.values()),
        "goal_content_emitted": False,
        "profile_content_emitted": False,
        "log_content_emitted": False,
        "endpoint_content_emitted": False,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show-unfrozen-constants", action="store_true")
    args = parser.parse_args()
    result = build_manifest(
        args.source_root.resolve(),
        enforce_frozen=not args.show_unfrozen_constants,
    )
    if not args.show_unfrozen_constants:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
