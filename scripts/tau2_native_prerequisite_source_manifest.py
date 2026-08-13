#!/usr/bin/env python3
"""Freeze fresh native-prerequisite Tau2 telecom BED cohorts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TAU2_ROOT = REPO_ROOT / "external/tau2-bench"
T3_ROOT = REPO_ROOT / "external/T3"
DATA_ROOT = TAU2_ROOT / "data/tau2/domains/telecom"
TASKS_PATH = DATA_ROOT / "tasks_full.json"
SPLITS_PATH = DATA_ROOT / "split_tasks.json"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/TAU2_NATIVE_PREREQUISITE_SOURCE_PROTOCOL_20260813.md"
)
OUTPUT_DIR = REPO_ROOT / "results/nonmyopic/tau2_native_prerequisite_source"
TAU2_COMMIT = "1d244f5dca42944b67a379b44bfeb9f5748f189d"
T3_COMMIT = "492f31fa05d2065c750a72d5e798385af282fa5d"
TASKS_SHA256 = "37e562e1ae3242577407e1303b1548bc64e7ea68e37d36173e6747990ceaf8a4"
SPLITS_SHA256 = "605b488bb9a6acb3c7f4505240a855fdc8681d09aadb16a8f38b2efcfc5c3aec"
SALT = "tau2-native-prerequisite-v1|"
EXPECTED_TASKS = 2285
EXPECTED_BASE = 114
EXPECTED_PRIOR = 90
EXPECTED_FRESH = 2092
EXPECTED_ELIGIBLE = {"mms_abroad": 297, "mms_home": 77, "mobile_abroad": 29}
QUOTAS = {
    "mms_abroad": {"mechanics": 2, "opportunity": 24, "development": 40, "confirmation": 64},
    "mms_home": {"mechanics": 2, "opportunity": 12, "development": 16, "confirmation": 24},
    "mobile_abroad": {"mechanics": 2, "opportunity": 8, "development": 8, "confirmation": 8},
}
EXPECTED_SPLIT_COUNTS = {
    "mechanics": (6, 26),
    "opportunity": (44, 186),
    "development": (64, 267),
    "confirmation": (96, 394),
    "reserve": (193, 777),
}
APP_STATES = frozenset(
    {"break_app_sms_permission", "break_app_storage_permission", "break_app_both_permissions"}
)
ROAM_STATES = frozenset(
    {"user_abroad_roaming_enabled_off", "user_abroad_roaming_disabled_on", "user_abroad_roaming_disabled_off"}
)
MMS_VARIANTS = (
    "bad_network_preference",
    "bad_wifi_calling",
    "break_apn_mms_setting",
    "break_app_sms_permission",
    "break_app_storage_permission",
    "break_app_both_permissions",
)
MMS_BACKBONES = (
    (),
    ("airplane_mode_on",),
    ("data_mode_off",),
    ("data_usage_exceeded",),
    ("unseat_sim_card",),
    ("user_abroad_roaming_disabled_off",),
    ("user_abroad_roaming_disabled_on",),
    ("user_abroad_roaming_enabled_off",),
    ("airplane_mode_on", "data_mode_off"),
    ("airplane_mode_on", "data_usage_exceeded"),
    ("data_mode_off", "unseat_sim_card"),
    ("data_usage_exceeded", "user_abroad_roaming_disabled_off"),
    ("unseat_sim_card", "user_abroad_roaming_disabled_on"),
    ("airplane_mode_on", "data_mode_off", "unseat_sim_card"),
)
ACCOUNT_WORLDS = (
    ("airplane_mode_on", "user_abroad_roaming_enabled_off"),
    ("airplane_mode_on", "user_abroad_roaming_disabled_on"),
    ("airplane_mode_on", "user_abroad_roaming_disabled_off"),
    ("airplane_mode_on", "data_usage_exceeded", "user_abroad_roaming_enabled_off"),
    ("airplane_mode_on", "data_usage_exceeded", "user_abroad_roaming_disabled_on"),
    ("airplane_mode_on", "data_usage_exceeded", "user_abroad_roaming_disabled_off"),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sequence_sha256(values: Iterable[str]) -> str:
    return sha256_bytes("\n".join(values).encode())


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def task_key(task_id: str) -> tuple[str, frozenset[str]]:
    if not task_id.startswith("[") or "]" not in task_id or "[PERSONA:" not in task_id:
        raise ValueError("unexpected Tau2 telecom task ID")
    issue = task_id.split("]", 1)[0][1:]
    body = task_id.split("]", 1)[1].split("[PERSONA:", 1)[0]
    return issue, frozenset(item for item in body.split("|") if item)


def prior_worlds() -> set[tuple[str, frozenset[str]]]:
    worlds = {
        ("mms_issue", frozenset((*backbone, variant)))
        for backbone in MMS_BACKBONES
        for variant in MMS_VARIANTS
    }
    worlds.update(("mobile_data_issue", frozenset(faults)) for faults in ACCOUNT_WORLDS)
    return worlds


def episode_key(
    issue: str, faults: frozenset[str]
) -> tuple[str, frozenset[str], frozenset[str]] | None:
    if issue == "mms_issue":
        family = "mms_abroad" if faults & ROAM_STATES else "mms_home"
        target = faults & APP_STATES
        return family, faults - APP_STATES, target
    if issue == "mobile_data_issue" and faults & ROAM_STATES:
        target = faults & (ROAM_STATES | {"data_usage_exceeded"})
        return "mobile_abroad", faults - target, target
    return None


def select_episodes(
    task_ids: list[str], base_ids: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Tau2 task IDs are not unique")
    physical = {task_key(task_id): task_id for task_id in task_ids}
    if len(physical) != len(task_ids):
        raise ValueError("Tau2 physical fault sets are not unique")
    base = {task_key(task_id) for task_id in base_ids}
    prior = prior_worlds()
    fresh = {key: task_id for key, task_id in physical.items() if key not in base and key not in prior}
    grouped: dict[tuple[str, frozenset[str]], list[dict[str, Any]]] = defaultdict(list)
    for (issue, faults), task_id in fresh.items():
        episode = episode_key(issue, faults)
        if episode is None:
            continue
        family, backbone, target = episode
        grouped[(family, backbone)].append(
            {"task_id": task_id, "issue": issue, "faults": faults, "target": target}
        )
    eligible: dict[str, list[tuple[tuple[str, frozenset[str]], list[dict[str, Any]]]]] = defaultdict(list)
    for key, worlds in grouped.items():
        if len(worlds) < 4 or len({tuple(sorted(row["target"])) for row in worlds}) != len(worlds):
            continue
        family = key[0]
        eligible[family].append((key, worlds))
    for family in eligible:
        eligible[family].sort(
            key=lambda item: (
                sha256_bytes((SALT + family + "|" + "|".join(sorted(item[0][1]))).encode()),
                tuple(sorted(item[0][1])),
            )
        )
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family, quotas in QUOTAS.items():
        offset = 0
        for split, size in quotas.items():
            chosen = eligible[family][offset : offset + size]
            offset += size
            for (selected_family, backbone), worlds in chosen:
                selected[split].append(
                    {"family": selected_family, "backbone": backbone, "worlds": worlds}
                )
        for (selected_family, backbone), worlds in eligible[family][offset:]:
            selected["reserve"].append(
                {"family": selected_family, "backbone": backbone, "worlds": worlds}
            )
    counts = {
        "official_tasks": len(task_ids),
        "official_base_physical_sets": len(base),
        "prior_study_physical_sets": len(prior),
        "fresh_physical_sets": len(fresh),
        **{f"eligible_{family}": len(rows) for family, rows in eligible.items()},
    }
    return dict(selected), counts


def build_manifest(tasks: list[dict[str, Any]], split_payload: dict[str, Any]) -> dict[str, Any]:
    task_ids = [str(task["id"]) for task in tasks]
    selected, counts = select_episodes(task_ids, list(split_payload["base"]))
    splits = {}
    for split, episodes in selected.items():
        public = []
        for episode in episodes:
            world_ids = sorted(row["task_id"] for row in episode["worlds"])
            episode_digest = sha256_bytes(
                (episode["family"] + "|" + "|".join(sorted(episode["backbone"]))).encode()
            )
            public.append(
                {
                    "family": episode["family"],
                    "episode_sha256": episode_digest,
                    "state_count": len(world_ids),
                    "selected_task_ids_sha256": sequence_sha256(world_ids),
                }
            )
        splits[split] = {
            "episode_count": len(public),
            "world_count": sum(row["state_count"] for row in public),
            "family_episode_counts": dict(sorted(Counter(row["family"] for row in public).items())),
            "episodes": public,
        }
    return {
        "schema_version": 1,
        "interface_version": "tau2-native-prerequisite-source-manifest-1",
        "source": {
            "tau2_commit": TAU2_COMMIT,
            "t3_commit": T3_COMMIT,
            "tasks_sha256": TASKS_SHA256,
            "splits_sha256": SPLITS_SHA256,
        },
        "counts": counts,
        "selection_salt": SALT,
        "splits": splits,
        "task_ids_serialized": False,
        "fault_names_serialized": False,
        "public_source_and_ticket_templates_inspected": True,
        "selected_task_identities_serialized": False,
        "selected_initialization_actions_serialized": False,
        "selected_tool_responses_opened": False,
        "evaluation_endpoints_opened": False,
        "model_calls_made": 0,
        "cost_usd": 0.0,
    }


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if git_head(TAU2_ROOT) != TAU2_COMMIT or git_head(T3_ROOT) != T3_COMMIT:
        raise ValueError("Tau2 source commit changed")
    if sha256_file(TASKS_PATH) != TASKS_SHA256 or sha256_file(SPLITS_PATH) != SPLITS_SHA256:
        raise ValueError("Tau2 source files changed")
    tasks = json.loads(TASKS_PATH.read_text())
    splits = json.loads(SPLITS_PATH.read_text())
    manifest = build_manifest(tasks, splits)
    counts = manifest["counts"]
    gates = {
        "exact_source_commits": git_head(TAU2_ROOT) == TAU2_COMMIT and git_head(T3_ROOT) == T3_COMMIT,
        "exact_source_hashes": sha256_file(TASKS_PATH) == TASKS_SHA256 and sha256_file(SPLITS_PATH) == SPLITS_SHA256,
        "exact_2285_unique_tasks": counts["official_tasks"] == EXPECTED_TASKS,
        "exact_114_base_sets_excluded": counts["official_base_physical_sets"] == EXPECTED_BASE,
        "exact_90_prior_sets_excluded": counts["prior_study_physical_sets"] == EXPECTED_PRIOR,
        "exact_2092_fresh_sets": counts["fresh_physical_sets"] == EXPECTED_FRESH,
        "exact_eligible_episode_counts": all(
            counts[f"eligible_{family}"] == expected for family, expected in EXPECTED_ELIGIBLE.items()
        ),
        "exact_split_counts": all(
            (manifest["splits"][split]["episode_count"], manifest["splits"][split]["world_count"])
            == expected
            for split, expected in EXPECTED_SPLIT_COUNTS.items()
        ),
        "no_selected_response_or_outcome_opened": not any(
            manifest[field]
            for field in (
                "selected_task_identities_serialized",
                "selected_initialization_actions_serialized",
                "selected_tool_responses_opened",
                "evaluation_endpoints_opened",
            )
        ),
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema_version": 1,
        "status": "source_manifest_frozen" if gates["all_pass"] else "source_manifest_invalid",
        "authorizes": "zero_call_opportunity_audit_only" if gates["all_pass"] else "nothing",
        "protocol_sha256": sha256_file(PROTOCOL),
        "manifest_sha256": sha256_bytes(canonical_json(manifest).encode()),
        "gates": gates,
        "model_calls_made": 0,
        "cost_usd": 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not gates["all_pass"]:
        raise ValueError("Tau2 native prerequisite source gates failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
