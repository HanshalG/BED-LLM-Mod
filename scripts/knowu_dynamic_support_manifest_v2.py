#!/usr/bin/env python3
"""Freeze KnowU splits with profile-independent static goal expressions."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

try:
    from scripts import knowu_bench_source_audit as source_audit
    from scripts import knowu_dynamic_support_manifest as v1
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    import knowu_bench_source_audit as source_audit
    import knowu_dynamic_support_manifest as v1


EXPECTED_ELIGIBLE_FAMILIES = 10
EXPECTED_SPLIT_HASHES = {
    "mechanics": (
        "b01108b17cd92ec30e7d0c532de0275c02cdcf9f02bd6a120a3a2810e90a09d4"
    ),
    "opportunity": (
        "0826daa1505c46993d718e95496780ca7c1e83c4cff93c7a17fa9f5ab0c23e15"
    ),
    "development": (
        "df33c350f984653da9d3f196ef41d16ee724f6bf489b07de2acead19c2aaddcc"
    ),
    "holdout": (
        "0e3c4321fefd73b42399ad9bb3c25a25b702204a5093c7a00d4341d2331154f7"
    ),
}
EXPECTED_MECHANICS_IDS = (
    "BuyComputerPreferenceAskUserTask",
    "MattermostLeaveNoticeTask",
)


def static_string_expression(node: ast.AST) -> str | None:
    """Resolve strings whose AST contains no runtime interpolation."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if not (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                return None
            parts.append(value.value)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = static_string_expression(node.left)
        right = static_string_expression(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def static_goals_by_class(root: Path) -> dict[str, str]:
    task_root = (
        root / "src" / "knowu_bench" / "tasks" / "definitions"
        / "preference"
    )
    goals: dict[str, str] = {}
    for path in sorted(task_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for class_node in tree.body:
            if not isinstance(class_node, ast.ClassDef):
                continue
            for statement in class_node.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if not any(
                    isinstance(target, ast.Name)
                    and target.id == "GOAL_REQUEST"
                    for target in statement.targets
                ):
                    continue
                resolved = static_string_expression(statement.value)
                if resolved is not None and resolved.strip():
                    goals[class_node.name] = resolved
    return goals


def eligible_families_v2(root: Path) -> dict[str, dict[str, Any]]:
    static_goals = static_goals_by_class(root)
    return {
        family["class_name"]: family
        for family in source_audit.preference_task_families(root)
        if (
            family["hard"]
            and family["agent_user_interaction"]
            and family["profile_variants"] >= 3
            and family["class_name"] in static_goals
        )
    }


def build_manifest(
    source_root: Path,
    *,
    enforce_frozen: bool = True,
) -> dict[str, Any]:
    audit = source_audit.build_audit(source_root)
    eligible = eligible_families_v2(source_root)
    splits = v1.split_families(eligible)
    split_hashes = {
        name: v1.ordered_hash(family_ids)
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
            raise ValueError("KnowU V2 manifest constants are not frozen")
        if len(eligible) != EXPECTED_ELIGIBLE_FAMILIES:
            raise ValueError("KnowU V2 eligible family count changed")
        if split_hashes != EXPECTED_SPLIT_HASHES:
            raise ValueError("KnowU V2 split hashes changed")
        if mechanics_ids != EXPECTED_MECHANICS_IDS:
            raise ValueError("KnowU V2 mechanics IDs changed")

    gates = {
        "source_audit_authorizes_manifest": audit["decision"][
            "derived_dynamic_support_manifest_authorized"
        ],
        "at_least_10_eligible_families": (
            len(eligible) >= v1.MINIMUM_ELIGIBLE_FAMILIES
        ),
        "all_splits_nonempty": all(splits.values()),
        "every_selected_family_has_at_least_three_worlds": all(
            eligible[family_id]["profile_variants"] >= 3
            for family_ids in splits.values()
            for family_id in family_ids
        ),
    }
    result = {
        "interface_version": "knowu-dynamic-support-manifest-2",
        "amendment_from_v1": {
            "scope": (
                "Accept profile-independent GOAL_REQUEST expressions made "
                "only of constant strings and zero-slot f-strings."
            ),
            "selection_seed_changed": False,
            "split_sizes_changed": False,
            "eligibility_semantics_changed": False,
            "profile_or_endpoint_content_read": False,
        },
        "source": audit["source"],
        "construction": {
            "selection_seed": v1.SELECTION_SEED,
            "minimum_eligible_families": v1.MINIMUM_ELIGIBLE_FAMILIES,
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
    return result


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
