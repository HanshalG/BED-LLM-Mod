#!/usr/bin/env python3
"""Audit exact native-prerequisite horizon opportunity in fresh Tau2 worlds."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tau2_native_prerequisite_source_manifest as source


PROTOCOL = source.PROTOCOL
PROTOCOL_SHA256 = "cc44807b74199d4aff5ec86b60ff8beae3ab123156750cb2cbcc1f1a1f092f78"
SOURCE_MANIFEST = source.OUTPUT_DIR / "SOURCE_MANIFEST.json"
SOURCE_MANIFEST_SHA256 = "4fff94c86fbc4bd5633d945d56ea072e3fef31523ff6598a608b3278562eae2e"
OUTPUT_DIR = REPO_ROOT / "results/nonmyopic/tau2_native_prerequisite_opportunity"
OPENED_SPLITS = ("mechanics", "opportunity")
MMS_ROOT_ACTIONS = (
    "apn_settings",
    "installed_apps",
    "mms_probe",
    "network_mode",
    "network_status",
    "speed_test",
    "status_bar",
    "wifi_calling",
)
MOBILE_ROOT_ACTIONS = (
    "customer_lookup",
    "network_status",
    "payment_request",
    "sim_status",
    "speed_test",
    "status_bar",
)
MMS_UNLOCKED_ACTION = "messaging_permissions"
MOBILE_UNLOCKED_ACTIONS = ("customer_bills", "data_usage", "line_details")
ID_RE = re.compile(r"\b([BCDLPT])\d+\b")
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_bindings() -> dict[str, Any]:
    if source.sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Tau2 native prerequisite protocol changed")
    if source.sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Tau2 native prerequisite source manifest changed")
    if source.git_head(source.TAU2_ROOT) != source.TAU2_COMMIT:
        raise ValueError("Tau2 source commit changed")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if manifest["selected_tool_responses_opened"]:
        raise ValueError("source manifest says selected responses were already opened")
    return manifest


def plain_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return plain_value(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return plain_value(value.value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_response(value: Any) -> str:
    text = canonical_json(plain_value(value))
    text = UUID_RE.sub("<uuid>", text)
    text = PHONE_RE.sub("<phone>", text)
    text = ID_RE.sub(lambda match: f"<{match.group(1).lower()}_id>", text)
    return text


def response_hash(value: Any) -> str:
    return sha256_bytes(canonical_response(value).encode())


def information_gain(values: Iterable[str]) -> float:
    observations = list(values)
    if not observations:
        return 0.0
    counts = Counter(observations)
    total = len(observations)
    return math.log(total) - sum(
        (count / total) * math.log(count) for count in counts.values()
    )


def legal_followups(family: str, root_action: str) -> tuple[str, ...]:
    if family.startswith("mms_"):
        actions = [action for action in MMS_ROOT_ACTIONS if action != root_action]
        if root_action == "installed_apps":
            actions.append(MMS_UNLOCKED_ACTION)
        return tuple(sorted(actions))
    if family == "mobile_abroad":
        actions = [action for action in MOBILE_ROOT_ACTIONS if action != root_action]
        if root_action == "customer_lookup":
            actions.extend(MOBILE_UNLOCKED_ACTIONS)
        return tuple(sorted(actions))
    raise ValueError(f"unknown Tau2 prerequisite family: {family}")


def root_actions(family: str) -> tuple[str, ...]:
    if family.startswith("mms_"):
        return MMS_ROOT_ACTIONS
    if family == "mobile_abroad":
        return MOBILE_ROOT_ACTIONS
    raise ValueError(f"unknown Tau2 prerequisite family: {family}")


def prerequisite_action(family: str) -> str:
    return "installed_apps" if family.startswith("mms_") else "customer_lookup"


def partition_indices(
    observations: list[dict[str, str]], action: str, indices: tuple[int, ...]
) -> list[tuple[int, ...]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[observations[index][action]].append(index)
    return [tuple(groups[value]) for value in sorted(groups)]


def subset_information(
    observations: list[dict[str, str]], action: str, indices: tuple[int, ...]
) -> float:
    return information_gain(observations[index][action] for index in indices)


def two_step_information(
    family: str,
    observations: list[dict[str, str]],
    first_action: str,
) -> float:
    indices = tuple(range(len(observations)))
    value = subset_information(observations, first_action, indices)
    continuation = 0.0
    for branch in partition_indices(observations, first_action, indices):
        options = legal_followups(family, first_action)
        best = max(
            subset_information(observations, action, branch) for action in options
        )
        continuation += (len(branch) / len(indices)) * best
    return value + continuation


def episode_metrics(
    family: str, episode_sha256: str, observations: list[dict[str, str]]
) -> dict[str, Any]:
    roots = root_actions(family)
    root_values = {
        action: information_gain(row[action] for row in observations) for action in roots
    }
    best_root = max(root_values.values())
    greedy = max(
        action for action, value in root_values.items() if abs(value - best_root) <= 1e-12
    )
    depth_values = {
        action: two_step_information(family, observations, action) for action in roots
    }
    best_depth = max(depth_values.values())
    depth_two = max(
        action for action, value in depth_values.items() if abs(value - best_depth) <= 1e-12
    )
    gain = depth_values[depth_two] - depth_values[greedy]
    if -1e-12 <= gain < 0.0:
        gain = 0.0
    prior_entropy = math.log(len(observations))
    return {
        "episode_sha256": episode_sha256,
        "family": family,
        "state_count": len(observations),
        "prior_entropy_nats": prior_entropy,
        "maximum_root_information_nats": best_root,
        "maximum_root_information_fraction": best_root / prior_entropy,
        "greedy_first_action": greedy,
        "greedy_two_step_information_nats": depth_values[greedy],
        "depth_two_first_action": depth_two,
        "depth_two_information_nats": depth_values[depth_two],
        "depth_two_changes_first_action": depth_two != greedy,
        "depth_two_selects_native_prerequisite": depth_two
        == prerequisite_action(family),
        "horizon_gain_nats": gain,
    }


def load_selected_episodes() -> dict[str, list[dict[str, Any]]]:
    tasks_payload = json.loads(source.TASKS_PATH.read_text())
    split_payload = json.loads(source.SPLITS_PATH.read_text())
    selected, _ = source.select_episodes(
        [str(task["id"]) for task in tasks_payload], list(split_payload["base"])
    )
    return {split: selected[split] for split in OPENED_SPLITS}


def initialize_tau2() -> tuple[dict[str, Any], Callable[..., Any]]:
    from tau2.domains.telecom.environment import get_environment, get_tasks

    tasks = {str(task.id): task for task in get_tasks(None)}
    return tasks, get_environment


def initialized_environment(task: Any, get_environment: Callable[..., Any]) -> Any:
    environment = get_environment(solo_mode=False)
    environment.set_state(
        initialization_data=task.initial_state.initialization_data,
        initialization_actions=task.initial_state.initialization_actions,
        message_history=[],
    )
    return environment


def mms_observations(environment: Any) -> dict[str, str]:
    calls = {
        "apn_settings": lambda: environment.user_tools.check_apn_settings(),
        "installed_apps": lambda: environment.user_tools.check_installed_apps(),
        "mms_probe": lambda: environment.user_tools.can_send_mms(),
        "network_mode": lambda: environment.user_tools.check_network_mode_preference(),
        "network_status": lambda: environment.user_tools.check_network_status(),
        "speed_test": lambda: environment.user_tools.run_speed_test(),
        "status_bar": lambda: environment.user_tools.check_status_bar(),
        "wifi_calling": lambda: environment.user_tools.check_wifi_calling_status(),
        "messaging_permissions": lambda: environment.user_tools.check_app_permissions(
            "messaging"
        ),
    }
    return {action: response_hash(call()) for action, call in calls.items()}


def mobile_observations(environment: Any) -> dict[str, str]:
    phone = str(environment.user_tools.surroundings.phone_number)
    customer = environment.tools.get_customer_by_phone(phone)
    customer_id = str(customer.customer_id)
    line = environment.tools._get_line_by_phone(phone)
    line_id = str(line.line_id)
    calls = {
        "customer_lookup": lambda: customer,
        "network_status": lambda: environment.user_tools.check_network_status(),
        "payment_request": lambda: environment.user_tools.check_payment_request(),
        "sim_status": lambda: environment.user_tools.check_sim_status(),
        "speed_test": lambda: environment.user_tools.run_speed_test(),
        "status_bar": lambda: environment.user_tools.check_status_bar(),
        "customer_bills": lambda: environment.tools.get_bills_for_customer(customer_id),
        "data_usage": lambda: environment.tools.get_data_usage(customer_id, line_id),
        "line_details": lambda: environment.tools.get_details_by_id(line_id),
    }
    return {action: response_hash(call()) for action, call in calls.items()}


def execute_selected_episodes(
    selected: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    tasks, get_environment = initialize_tau2()
    metrics: dict[str, list[dict[str, Any]]] = {}
    for split, episodes in selected.items():
        split_metrics = []
        for episode in episodes:
            observations = []
            for world in episode["worlds"]:
                task = tasks[world["task_id"]]
                environment = initialized_environment(task, get_environment)
                if episode["family"].startswith("mms_"):
                    observations.append(mms_observations(environment))
                else:
                    observations.append(mobile_observations(environment))
            episode_sha = sha256_bytes(
                (
                    episode["family"]
                    + "|"
                    + "|".join(sorted(episode["backbone"]))
                ).encode()
            )
            split_metrics.append(
                episode_metrics(episode["family"], episode_sha, observations)
            )
        metrics[split] = split_metrics
    return metrics


def public_manifest_rows(manifest: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    return {
        row["episode_sha256"]: row for row in manifest["splits"][split]["episodes"]
    }


def family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = [float(row["horizon_gain_nats"]) for row in rows]
    return {
        "episode_count": len(rows),
        "world_count": sum(int(row["state_count"]) for row in rows),
        "not_root_saturated_count": sum(
            float(row["maximum_root_information_fraction"]) < 0.95 for row in rows
        ),
        "changed_first_action_count": sum(
            bool(row["depth_two_changes_first_action"]) for row in rows
        ),
        "gain_at_least_0_10_count": sum(gain >= 0.10 for gain in gains),
        "native_prerequisite_selected_count": sum(
            bool(row["depth_two_selects_native_prerequisite"]) for row in rows
        ),
        "mean_horizon_gain_nats": sum(gains) / len(gains),
        "maximum_horizon_gain_nats": max(gains),
    }


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    manifest = validate_bindings()
    selected = load_selected_episodes()
    metrics = execute_selected_episodes(selected)
    for split in OPENED_SPLITS:
        public = public_manifest_rows(manifest, split)
        observed = {row["episode_sha256"]: row for row in metrics[split]}
        if set(public) != set(observed):
            raise ValueError(f"Tau2 {split} episode selection changed")
        for episode_sha, row in observed.items():
            if int(row["state_count"]) != int(public[episode_sha]["state_count"]):
                raise ValueError(f"Tau2 {split} state count changed")
    opportunity = metrics["opportunity"]
    gains = [float(row["horizon_gain_nats"]) for row in opportunity]
    families = {
        family: family_summary([row for row in opportunity if row["family"] == family])
        for family in source.EXPECTED_ELIGIBLE
    }
    gates = {
        "exact_6_mechanics_episodes": len(metrics["mechanics"]) == 6,
        "exact_26_mechanics_worlds": sum(row["state_count"] for row in metrics["mechanics"])
        == 26,
        "exact_44_opportunity_episodes": len(opportunity) == 44,
        "exact_186_opportunity_worlds": sum(row["state_count"] for row in opportunity)
        == 186,
        "at_least_36_not_root_saturated": sum(
            row["maximum_root_information_fraction"] < 0.95 for row in opportunity
        )
        >= 36,
        "at_least_30_change_first_action": sum(
            row["depth_two_changes_first_action"] for row in opportunity
        )
        >= 30,
        "at_least_30_gain_0_10_nats": sum(gain >= 0.10 for gain in gains) >= 30,
        "mean_gain_at_least_0_15_nats": sum(gains) / len(gains) >= 0.15,
        "each_family_half_gain_0_10_nats": all(
            row["gain_at_least_0_10_count"] * 2 >= row["episode_count"]
            for row in families.values()
        ),
        "native_prerequisite_selected_at_least_30": sum(
            row["depth_two_selects_native_prerequisite"] for row in opportunity
        )
        >= 30,
        "all_gains_finite_and_nonnegative": all(
            math.isfinite(gain) and gain >= 0.0 for gain in gains
        ),
        "development_confirmation_reserve_unopened": set(metrics) == set(OPENED_SPLITS),
        "no_task_ids_faults_or_raw_responses_serialized": True,
        "zero_model_calls_and_cost": True,
    }
    scientific = (
        "at_least_36_not_root_saturated",
        "at_least_30_change_first_action",
        "at_least_30_gain_0_10_nats",
        "mean_gain_at_least_0_15_nats",
        "each_family_half_gain_0_10_nats",
        "native_prerequisite_selected_at_least_30",
        "all_gains_finite_and_nonnegative",
    )
    gates["all_integrity_gates_pass"] = all(
        value for name, value in gates.items() if name not in scientific
    )
    gates["source_opportunity_pass"] = gates["all_integrity_gates_pass"] and all(
        gates[name] for name in scientific
    )
    result = {
        "schema_version": 1,
        "interface_version": "tau2-native-prerequisite-opportunity-1",
        "status": "source_opportunity_pass"
        if gates["source_opportunity_pass"]
        else "source_opportunity_null",
        "authorizes": "semantic_mechanics_protocol_only"
        if gates["source_opportunity_pass"]
        else "nothing",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "opened_splits": list(OPENED_SPLITS),
        "development_confirmation_reserve_opened": False,
        "summary": family_summary(opportunity),
        "family_summaries": families,
        "gates": gates,
        "episode_metrics": metrics,
        "task_ids_serialized": False,
        "fault_names_serialized": False,
        "raw_tool_responses_serialized": False,
        "evaluation_endpoints_opened": False,
        "model_calls_made": 0,
        "cost_usd": 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
