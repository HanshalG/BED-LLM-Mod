#!/usr/bin/env python3
"""Audit KnowU-Bench's released latent-profile interaction structure."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


SOURCE_REPOSITORY = "https://github.com/ZJU-REAL/KnowU-Bench"
SOURCE_COMMIT = "c03a825991ede13add6631f2ed19b90755930dc6"
PREFERENCE_SOURCE_SHA256 = (
    "a67717f0f4f665e15e941625acfe21225481824282eba760d516df131b80f846"
)
PROFILE_SOURCE_SHA256 = (
    "d77840606416c3b94c3a2de294ff4564a30e5cfda537081f6427799c93712a3e"
)
LOG_SOURCE_SHA256 = (
    "d5acd87e9d2b8528bc388b0dc7eea48363514de4a1c09a01d1bec83368f1d2d8"
)
PROFILE_IDS = ("developer", "grandma", "student", "user")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_sha256(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{sha256_file(path)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _literal_assignments(node: ast.ClassDef) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for statement in node.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                values[target.id] = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                continue
    return values


def preference_task_families(root: Path) -> list[dict[str, Any]]:
    task_root = (
        root / "src" / "knowu_bench" / "tasks" / "definitions"
        / "preference"
    )
    families: list[dict[str, Any]] = []
    for path in sorted(task_root.glob("*.py")):
        if path.name.startswith(("base_", "__")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            values = _literal_assignments(node)
            if "task_tags" not in values:
                continue
            tags = set(values["task_tags"])
            profiles = set(values.get("supported_profiles", PROFILE_IDS))
            goal = values.get("GOAL_REQUEST")
            families.append(
                {
                    "class_name": node.name,
                    "file_name": path.name,
                    "tags": sorted(tags),
                    "supported_profiles": sorted(profiles),
                    "profile_variants": len(profiles),
                    "hard": "hard" in tags,
                    "agent_user_interaction": (
                        "agent-user-interaction" in tags
                    ),
                    "static_nonempty_goal": (
                        isinstance(goal, str) and bool(goal.strip())
                    ),
                }
            )
    return families


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_leaf_paths(child, path))
        return result
    if isinstance(value, list):
        return {prefix} if prefix else set()
    return {prefix} if prefix else set()


def profile_summary(root: Path) -> dict[str, Any]:
    profile_root = root / "src" / "knowu_bench" / "user_profile"
    log_root = root / "src" / "knowu_bench" / "user_logs"
    leaf_counts: dict[str, int] = {}
    log_counts: dict[str, int] = {}
    top_level_sections: Counter[str] = Counter()
    for profile_id in PROFILE_IDS:
        profile = yaml.safe_load(
            (profile_root / f"{profile_id}.yaml").read_text(
                encoding="utf-8"
            )
        )
        user_profile = profile.get("user_profile", {})
        leaf_counts[profile_id] = len(_leaf_paths(user_profile))
        top_level_sections.update(user_profile.keys())
        logs = json.loads(
            (log_root / f"{profile_id}.json").read_text(
                encoding="utf-8"
            )
        )
        log_counts[profile_id] = len(logs)
    return {
        "profile_ids": list(PROFILE_IDS),
        "profile_leaf_counts": leaf_counts,
        "clean_log_entry_counts": log_counts,
        "top_level_section_coverage": dict(
            sorted(top_level_sections.items())
        ),
    }


def build_audit(root: Path) -> dict[str, Any]:
    preference_root = (
        root / "src" / "knowu_bench" / "tasks" / "definitions"
        / "preference"
    )
    profile_root = root / "src" / "knowu_bench" / "user_profile"
    log_root = root / "src" / "knowu_bench" / "user_logs"
    preference_digest = aggregate_sha256(
        preference_root.glob("*.py"), root
    )
    profile_digest = aggregate_sha256(
        (profile_root / f"{profile_id}.yaml" for profile_id in PROFILE_IDS),
        root,
    )
    log_digest = aggregate_sha256(
        (log_root / f"{profile_id}.json" for profile_id in PROFILE_IDS),
        root,
    )
    if preference_digest != PREFERENCE_SOURCE_SHA256:
        raise ValueError("KnowU preference source hash changed")
    if profile_digest != PROFILE_SOURCE_SHA256:
        raise ValueError("KnowU profile source hash changed")
    if log_digest != LOG_SOURCE_SHA256:
        raise ValueError("KnowU clean-log source hash changed")

    families = preference_task_families(root)
    hard_multi_profile = [
        family
        for family in families
        if (
            family["hard"]
            and family["agent_user_interaction"]
            and family["profile_variants"] >= 3
        )
    ]
    return {
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "preference_source_sha256": preference_digest,
            "profile_source_sha256": profile_digest,
            "clean_log_source_sha256": log_digest,
        },
        "profiles": profile_summary(root),
        "task_structure": {
            "preference_task_families": len(families),
            "official_profile_variants": sum(
                family["profile_variants"] for family in families
            ),
            "hard_multi_profile_families": len(hard_multi_profile),
            "profile_variant_histogram": dict(
                sorted(
                    Counter(
                        family["profile_variants"] for family in families
                    ).items()
                )
            ),
            "hard_multi_profile_family_ids": [
                family["class_name"] for family in hard_multi_profile
            ],
        },
        "released_mechanics": {
            "registry_cross_products_task_profiles": True,
            "profile_hidden_from_gui_agent": True,
            "profile_conditioned_clean_logs_released": True,
            "free_form_ask_user_released": True,
            "simulator_receives_full_dialogue_history": True,
            "task_specific_programmatic_or_hybrid_endpoints": True,
            "reference_nonmyopic_policy_released": False,
            "explicit_profile_prior_released": False,
        },
        "decision": {
            "direct_nonmyopic_claim_authorized": False,
            "reason": (
                "Unrestricted ask_user permits direct terminal-preference "
                "questions, and the release defines neither a profile prior "
                "nor a reference non-myopic policy."
            ),
            "derived_dynamic_support_manifest_authorized": True,
            "derived_route": (
                "Use the official cross-profile task families with a frozen "
                "uniform prior, profile labels hidden, task-relevant clean "
                "logs as the initial observation, atomic clarification "
                "questions, and LLM-generated path-dependent support."
            ),
        },
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_audit(args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
