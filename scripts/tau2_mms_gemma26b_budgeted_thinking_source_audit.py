#!/usr/bin/env python3
"""Audit the frozen Tau2 cohort and public tool contract for Gemma26B budgeted thinking."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tau2_native_prerequisite_source_manifest as source
from scripts import tau2_native_prerequisite_semantic_verify as independent

PROTOCOL = REPO_ROOT / "results/nonmyopic/TAU2_MMS_GEMMA26B_BUDGETED_THINKING_CALIBRATION_PROTOCOL_20260813.md"
MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_gemma26b_budgeted_thinking_calibration/CALIBRATION_MANIFEST.json"
USER_TOOLS = REPO_ROOT / "external/tau2-bench/src/tau2/domains/telecom/user_tools.py"
USER_TOOLS_SHA256 = "03fa751eeea3734313a3f5274223d42750fd5c1a28f2624265a678f271ac309d"
PRIOR_MANIFESTS = (
    REPO_ROOT / "results/nonmyopic/tau2_mms_array_semantic_calibration/CALIBRATION_MANIFEST.json",
    REPO_ROOT / "results/nonmyopic/tau2_mms_checkpointed_semantic_calibration/CALIBRATION_MANIFEST.json",
    REPO_ROOT / "results/nonmyopic/tau2_mms_partition_semantic_calibration/CALIBRATION_MANIFEST.json",
    REPO_ROOT / "results/nonmyopic/tau2_mms_split_partition_calibration/CALIBRATION_MANIFEST.json",
    REPO_ROOT / "results/nonmyopic/tau2_mms_documented_split_calibration/CALIBRATION_MANIFEST.json",
    REPO_ROOT / "results/nonmyopic/tau2_mms_documented_split_resilient_calibration/CALIBRATION_MANIFEST.json",
    REPO_ROOT / "results/nonmyopic/tau2_mms_gemma26b_thinking_calibration/CALIBRATION_MANIFEST.json",
)

ROOT_TOOL_CONTRACT: tuple[dict[str, Any], ...] = (
    {"action_id": "apn_settings", "method": "check_apn_settings", "visible_semantics": "Shows the current APN name and MMSC URL for picture messaging."},
    {"action_id": "installed_apps", "method": "check_installed_apps", "visible_semantics": "Lists the names of all installed phone apps."},
    {"action_id": "mms_probe", "method": "can_send_mms", "visible_semantics": "Shows whether the default messaging app can send MMS messages."},
    {"action_id": "network_mode", "method": "check_network_mode_preference", "visible_semantics": "Shows the current preferred cellular network mode."},
    {"action_id": "network_status", "method": "check_network_status", "visible_semantics": "Shows airplane mode, SIM status, cellular connection, signal, network type, mobile data, data roaming, and Wi-Fi status."},
    {"action_id": "speed_test", "method": "run_speed_test", "visible_semantics": "Shows download speed and connection quality, or a visible failure."},
    {"action_id": "status_bar", "method": "check_status_bar", "visible_semantics": "Shows status-bar network signal, mobile-data, Wi-Fi, airplane-mode, and battery indicators."},
    {"action_id": "wifi_calling", "method": "check_wifi_calling_status", "visible_semantics": "Shows whether Wi-Fi Calling is enabled."},
)
NATIVE_TOOL_CONTRACT = {
    "action_id": "messaging_permissions",
    "method": "check_app_permissions",
    "argument": "messaging",
    "visible_semantics": "Lists the names of permissions currently granted to the messaging app. Relevant visible names are sms, storage, and phone; an absent name is not granted.",
}
EXPECTED_DOCSTRINGS = {
    "check_status_bar": "Shows what icons are currently visible in your phone's status bar (the area at the top of the screen). Displays network signal strength, mobile data status (enabled, disabled, data saver), Wi-Fi status, and battery level.",
    "check_network_status": "Checks your phone's connection status to cellular networks and Wi-Fi. Shows airplane mode status, signal strength, network type, whether mobile data is enabled, and whether data roaming is enabled.",
    "check_network_mode_preference": "Shows the current network mode preference.",
    "check_apn_settings": "Checks the technical APN settings your phone uses to connect to your carrier's mobile data network. Shows current APN name and MMSC URL for picture messaging.",
    "check_wifi_calling_status": "Checks if Wi-Fi Calling is enabled on your device. This feature allows you to make and receive calls over a Wi-Fi network instead of using the cellular network.",
    "run_speed_test": "Measures your current internet connection speed (download speed). Provides information about connection quality and what activities it can support.",
    "can_send_mms": "Checks if the default messaging app can send MMS messages.",
    "check_installed_apps": "Returns the name of all installed apps on the phone.",
    "check_app_permissions": "Checks what permissions a specific app currently has. Shows if the app has access to features like storage, camera, location, etc.",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_hash(episode: dict[str, Any]) -> str:
    return hashlib.sha256((episode["family"] + "|" + "|".join(sorted(episode["backbone"]))).encode()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def load_episodes() -> list[dict[str, Any]]:
    manifest = load_object(MANIFEST)
    tasks = json.loads(source.TASKS_PATH.read_text())
    splits = json.loads(source.SPLITS_PATH.read_text())
    selected, _ = source.select_episodes([str(row["id"]) for row in tasks], list(splits["base"]))
    reserve = {
        family: [row for row in selected["reserve"] if row["family"] == family]
        for family in ("mms_abroad", "mms_home")
    }
    prior = {
        row["episode_sha256"]
        for path in PRIOR_MANIFESTS
        for row in load_object(path)["episodes"]
    }
    episodes = []
    for row in manifest.get("episodes", []):
        family = row.get("family")
        position = row.get("reserve_family_position")
        if family not in reserve or isinstance(position, bool) or not isinstance(position, int) or position not in range(len(reserve[family])):
            raise ValueError("Gemma26B budgeted thinking reserve identity changed")
        episode = reserve[family][position]
        digest = episode_hash(episode)
        if digest != row.get("episode_sha256") or digest in prior or row.get("state_count") != len(episode["worlds"]):
            raise ValueError("Gemma26B budgeted thinking cohort binding changed")
        episodes.append(episode)
    if len(episodes) != 6 or [row["family"] for row in episodes] != ["mms_abroad"] * 4 + ["mms_home"] * 2:
        raise ValueError("Gemma26B budgeted thinking cohort balance changed")
    return episodes


def _read_tool_methods(path: Path = USER_TOOLS) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TelecomUserTools"]
    if len(classes) != 1:
        raise ValueError("Tau2 TelecomUserTools class changed")
    return {node.name: node for node in classes[0].body if isinstance(node, ast.FunctionDef)}


def validate_public_tool_contract(path: Path = USER_TOOLS) -> dict[str, Any]:
    if source.git_head(source.TAU2_ROOT) != source.TAU2_COMMIT or sha256_file(path) != USER_TOOLS_SHA256:
        raise ValueError("Gemma26B budgeted thinking Tau2 source binding changed")
    methods = _read_tool_methods(path)
    names = [row["method"] for row in ROOT_TOOL_CONTRACT] + [NATIVE_TOOL_CONTRACT["method"]]
    for name in names:
        method = methods.get(name)
        if method is None or ast.get_docstring(method) != EXPECTED_DOCSTRINGS[name]:
            raise ValueError(f"Gemma26B budgeted thinking public docstring changed: {name}")
        decorators = {ast.unparse(value) for value in method.decorator_list}
        if "is_tool(ToolType.READ)" not in decorators:
            raise ValueError(f"Gemma26B budgeted thinking tool is no longer read-only: {name}")
    source_text = path.read_text()
    required_native_logic = (
        "permissions.model_dump().items()",
        "if allowed",
        "has permission for:",
        "name.replace(\"_\", \" \" ).lower()",
    )
    normalized = source_text.replace("name.replace(\"_\", \" \"  )", "name.replace(\"_\", \" \" )")
    if not all(fragment in normalized for fragment in required_native_logic[:-1]):
        raise ValueError("Gemma26B budgeted thinking granted-permission semantics changed")
    permission_method = methods["check_app_permissions"]
    if [arg.arg for arg in permission_method.args.args] != ["self", "app_name"]:
        raise ValueError("Gemma26B budgeted thinking permission argument changed")
    return {
        "tau2_commit": source.TAU2_COMMIT,
        "user_tools_sha256": sha256_file(path),
        "root_tool_count": len(ROOT_TOOL_CONTRACT),
        "native_method": NATIVE_TOOL_CONTRACT["method"],
        "all_tools_read_only": True,
        "exact_docstrings": True,
        "granted_names_are_visible": True,
    }


def audit() -> dict[str, Any]:
    contract = validate_public_tool_contract()
    episodes = load_episodes()
    task_bank, get_environment = independent.exact.initialize_tau2()
    rows = []
    for episode in episodes:
        observations = independent.official_episode(episode, task_bank, get_environment)
        canonical = [{action: independent.canonical_json(value) for action, value in world.items()} for world in observations]
        metric = independent.exact.episode_metrics(episode["family"], episode_hash(episode), canonical)
        rows.append({
            "episode_sha256": episode_hash(episode),
            "family": episode["family"],
            "root_actions_all_constant": all(len({world[action] for world in canonical}) == 1 for action in independent.MMS_ROOTS),
            "native_partition_has_four_groups": len({world["messaging_permissions"] for world in canonical}) == 4,
            "greedy_first_action": metric["greedy_first_action"],
            "depth_two_first_action": metric["depth_two_first_action"],
            "horizon_gain_nats": metric["horizon_gain_nats"],
        })
    gates = {
        "exact_public_tool_contract": contract["root_tool_count"] == 8 and contract["native_method"] == "check_app_permissions" and contract["all_tools_read_only"] and contract["exact_docstrings"] and contract["granted_names_are_visible"],
        "exact_six_fresh_episodes": len(rows) == 6,
        "availability_bound_family_balance_four_two": [row["family"] for row in rows] == ["mms_abroad"] * 4 + ["mms_home"] * 2,
        "all_root_actions_constant": all(row["root_actions_all_constant"] for row in rows),
        "all_native_partitions_four_way": all(row["native_partition_has_four_groups"] for row in rows),
        "all_greedy_avoid_prerequisite": all(row["greedy_first_action"] != "installed_apps" for row in rows),
        "all_depth_two_select_prerequisite": all(row["depth_two_first_action"] == "installed_apps" for row in rows),
        "all_horizon_gains_at_least_one_nat": all(row["horizon_gain_nats"] >= 1.0 for row in rows),
    }
    gates["all_source_gates_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "interface_version": "tau2-mms-gemma26b-budgeted-thinking-source-audit-1",
        "status": "source_pass" if gates["all_source_gates_pass"] else "source_null",
        "authorizes": "implementation_only" if gates["all_source_gates_pass"] else "nothing",
        "public_tool_contract": contract,
        "gates": gates,
        "episodes": rows,
        "model_calls_made": 0,
        "cost_usd": 0.0,
        "selected_tool_responses_opened_for_source_audit_only": True,
        "repair_or_task_success_endpoints_opened": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
