#!/usr/bin/env python3
"""Audit the frozen fresh cohort for the Tau2 split-partition interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tau2_native_prerequisite_source_manifest as source
from scripts import tau2_native_prerequisite_semantic_verify as independent

MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_split_partition_calibration/CALIBRATION_MANIFEST.json"
PRIOR_MANIFESTS = (
    REPO_ROOT / "results/nonmyopic/tau2_mms_array_semantic_calibration/CALIBRATION_MANIFEST.json",
    REPO_ROOT / "results/nonmyopic/tau2_mms_checkpointed_semantic_calibration/CALIBRATION_MANIFEST.json",
    REPO_ROOT / "results/nonmyopic/tau2_mms_partition_semantic_calibration/CALIBRATION_MANIFEST.json",
)


def episode_hash(episode):
    return hashlib.sha256(
        (episode["family"] + "|" + "|".join(sorted(episode["backbone"]))).encode()
    ).hexdigest()


def load_episodes():
    tasks = json.loads(source.TASKS_PATH.read_text())
    splits = json.loads(source.SPLITS_PATH.read_text())
    selected, _ = source.select_episodes(
        [str(row["id"]) for row in tasks], list(splits["base"])
    )
    by_hash = {episode_hash(row): row for row in selected["reserve"]}
    rows = json.loads(MANIFEST.read_text())["episodes"]
    prior = {
        row["episode_sha256"]
        for path in PRIOR_MANIFESTS
        for row in json.loads(path.read_text())["episodes"]
    }
    episodes = []
    for row in rows:
        episode = by_hash.get(row["episode_sha256"])
        if (
            episode is None
            or row["episode_sha256"] in prior
            or row["family"] != episode["family"]
            or row["state_count"] != len(episode["worlds"])
        ):
            raise ValueError("split-partition cohort binding changed")
        episodes.append(episode)
    return episodes


def audit():
    episodes = load_episodes()
    task_bank, get_environment = independent.exact.initialize_tau2()
    rows = []
    for episode in episodes:
        observations = independent.official_episode(episode, task_bank, get_environment)
        canonical = [
            {
                action: independent.canonical_json(value)
                for action, value in world.items()
            }
            for world in observations
        ]
        digest = episode_hash(episode)
        metric = independent.exact.episode_metrics(
            episode["family"], digest, canonical
        )
        root_constant = all(
            len({world[action] for world in canonical}) == 1
            for action in independent.MMS_ROOTS
        )
        native_unique = len(
            {world["messaging_permissions"] for world in canonical}
        ) == 4
        rows.append(
            {
                "episode_sha256": digest,
                "family": episode["family"],
                "root_actions_all_constant": root_constant,
                "native_partition_has_four_groups": native_unique,
                "greedy_first_action": metric["greedy_first_action"],
                "depth_two_first_action": metric["depth_two_first_action"],
                "horizon_gain_nats": metric["horizon_gain_nats"],
            }
        )
    gates = {
        "exact_six_fresh_episodes": len(rows) == 6,
        "family_balance_three_three": [row["family"] for row in rows]
        == ["mms_abroad"] * 3 + ["mms_home"] * 3,
        "all_root_actions_constant": all(
            row["root_actions_all_constant"] for row in rows
        ),
        "all_native_partitions_four_way": all(
            row["native_partition_has_four_groups"] for row in rows
        ),
        "all_greedy_avoid_prerequisite": all(
            row["greedy_first_action"] != "installed_apps" for row in rows
        ),
        "all_depth_two_select_prerequisite": all(
            row["depth_two_first_action"] == "installed_apps" for row in rows
        ),
        "all_horizon_gains_at_least_one_nat": all(
            row["horizon_gain_nats"] >= 1.0 for row in rows
        ),
    }
    gates["all_source_gates_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "interface_version": "tau2-mms-split-partition-source-audit-1",
        "status": "source_pass" if gates["all_source_gates_pass"] else "source_null",
        "authorizes": "implementation_only" if gates["all_source_gates_pass"] else "nothing",
        "gates": gates,
        "episodes": rows,
        "model_calls_made": 0,
        "cost_usd": 0.0,
        "selected_tool_responses_opened_for_source_audit_only": True,
        "repair_or_task_success_endpoints_opened": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
