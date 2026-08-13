#!/usr/bin/env python3
"""Independently replay the Tau2 MMS Gemma26B budgeted thinking calibration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tau2_native_prerequisite_semantic_mechanics as public_math
from scripts import tau2_native_prerequisite_semantic_verify as mathlib
from scripts import tau2_native_prerequisite_source_manifest as source

MODEL_ID = "google/gemma-4-26b-a4b-it"
INTERFACE_VERSION = "tau2-mms-gemma26b-budgeted-thinking-calibration-1"
ROOT_SEEDS = tuple(range(202608132100, 202608132106))
NATIVE_SEEDS = tuple(range(202608132200, 202608132206))
REASONING_MAX_TOKENS = 4096
REQUEST_MAX_TOKENS = 4608
ROOTS = tuple(mathlib.MMS_ROOTS)
NATIVE = "messaging_permissions"
PROTOCOL = REPO_ROOT / "results/nonmyopic/TAU2_MMS_GEMMA26B_BUDGETED_THINKING_CALIBRATION_PROTOCOL_20260813.md"
MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_gemma26b_budgeted_thinking_calibration/CALIBRATION_MANIFEST.json"
PRIOR_MANIFESTS = tuple(REPO_ROOT / path for path in (
    "results/nonmyopic/tau2_mms_array_semantic_calibration/CALIBRATION_MANIFEST.json",
    "results/nonmyopic/tau2_mms_checkpointed_semantic_calibration/CALIBRATION_MANIFEST.json",
    "results/nonmyopic/tau2_mms_partition_semantic_calibration/CALIBRATION_MANIFEST.json",
    "results/nonmyopic/tau2_mms_split_partition_calibration/CALIBRATION_MANIFEST.json",
    "results/nonmyopic/tau2_mms_documented_split_calibration/CALIBRATION_MANIFEST.json",
    "results/nonmyopic/tau2_mms_documented_split_resilient_calibration/CALIBRATION_MANIFEST.json",
    "results/nonmyopic/tau2_mms_gemma26b_thinking_calibration/CALIBRATION_MANIFEST.json",
))
ROOT_CONTRACT = (
    {"action_id": "apn_settings", "method": "check_apn_settings", "visible_semantics": "Shows the current APN name and MMSC URL for picture messaging."},
    {"action_id": "installed_apps", "method": "check_installed_apps", "visible_semantics": "Lists the names of all installed phone apps."},
    {"action_id": "mms_probe", "method": "can_send_mms", "visible_semantics": "Shows whether the default messaging app can send MMS messages."},
    {"action_id": "network_mode", "method": "check_network_mode_preference", "visible_semantics": "Shows the current preferred cellular network mode."},
    {"action_id": "network_status", "method": "check_network_status", "visible_semantics": "Shows airplane mode, SIM status, cellular connection, signal, network type, mobile data, data roaming, and Wi-Fi status."},
    {"action_id": "speed_test", "method": "run_speed_test", "visible_semantics": "Shows download speed and connection quality, or a visible failure."},
    {"action_id": "status_bar", "method": "check_status_bar", "visible_semantics": "Shows status-bar network signal, mobile-data, Wi-Fi, airplane-mode, and battery indicators."},
    {"action_id": "wifi_calling", "method": "check_wifi_calling_status", "visible_semantics": "Shows whether Wi-Fi Calling is enabled."},
)
NATIVE_CONTRACT = {"action_id": "messaging_permissions", "method": "check_app_permissions", "argument": "messaging", "visible_semantics": "Lists the names of permissions currently granted to the messaging app. Relevant visible names are sms, storage, and phone; an absent name is not granted."}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def episode_hash(episode: Mapping[str, Any]) -> str:
    return hashlib.sha256((episode["family"] + "|" + "|".join(sorted(episode["backbone"]))).encode()).hexdigest()


def load_episodes() -> list[dict[str, Any]]:
    tasks = json.loads(source.TASKS_PATH.read_text())
    splits = json.loads(source.SPLITS_PATH.read_text())
    selected, _ = source.select_episodes([str(row["id"]) for row in tasks], list(splits["base"]))
    reserve = {family: [row for row in selected["reserve"] if row["family"] == family] for family in ("mms_abroad", "mms_home")}
    prior = {row["episode_sha256"] for path in PRIOR_MANIFESTS for row in load_object(path)["episodes"]}
    rows = load_object(MANIFEST).get("episodes")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("independent documented manifest changed")
    episodes = []
    for row in rows:
        family = row.get("family")
        position = row.get("reserve_family_position")
        if family not in reserve or isinstance(position, bool) or not isinstance(position, int) or position >= len(reserve[family]):
            raise ValueError("independent documented reserve changed")
        episode = reserve[family][position]
        digest = episode_hash(episode)
        if digest != row.get("episode_sha256") or digest in prior or row.get("state_count") != len(episode["worlds"]):
            raise ValueError("independent documented cohort changed")
        episodes.append(episode)
    if [row["family"] for row in episodes] != ["mms_abroad"] * 4 + ["mms_home"] * 2:
        raise ValueError("independent documented balance changed")
    return episodes


def public_episode(episode: Mapping[str, Any], index: int) -> dict[str, Any]:
    base = public_math.public_episode(episode, index)
    return {"episode_index": index, "family": episode["family"], "worlds": base["worlds"], "legal_unlock": base["legal_unlock"]}


def root_messages(public: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = {**public, "root_tool_contract": list(ROOT_CONTRACT)}
    return [
        {"role": "system", "content": "You predict observational equivalence for documented telecom read-only tools. Return only strict JSON without reasoning."},
        {"role": "user", "content": "For each root tool in the supplied exact order, decide whether all four candidate worlds would produce exactly the same visible output under its documented semantics. Infer outputs from each world description. Predict observations, not repair success. Preserve action IDs. Confidence is the probability that the boolean is correct. Do not include the unlocked native action.\n" + canonical_json(payload)},
    ]


def native_messages(public: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = {**public, "native_tool_contract": NATIVE_CONTRACT}
    return [
        {"role": "system", "content": "You predict the documented messaging-permission tool output across candidate telecom worlds. Return only strict JSON without reasoning."},
        {"role": "user", "content": "The installed_apps read has already revealed the messaging app and unlocked messaging_permissions. Under the supplied public tool contract, partition the four candidate worlds by whether messaging_permissions would return exactly the same visible granted-permission names. Infer granted names from each world description: absent permission names are not granted. Use canonical first-occurrence labels: world 0 is group 0; later worlds reuse a label exactly for identical visible output, otherwise use the next integer. Return only the supplied native action ID, four labels, and confidence.\n" + canonical_json(payload)},
    ]


def root_response_format() -> dict[str, Any]:
    item = {"type": "object", "additionalProperties": False, "required": ["action_id", "all_worlds_same", "confidence"], "properties": {"action_id": {"type": "string"}, "all_worlds_same": {"type": "boolean"}, "confidence": {"type": "number", "minimum": .5, "maximum": .95}}}
    return {"type": "json_schema", "json_schema": {"name": "tau2_mms_documented_root_equivalence", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["actions"], "properties": {"actions": {"type": "array", "minItems": 8, "maxItems": 8, "items": item}}}}}


def native_response_format() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "tau2_mms_documented_native_partition", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["action_id", "world_groups", "confidence"], "properties": {"action_id": {"type": "string"}, "world_groups": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer", "minimum": 0, "maximum": 3}}, "confidence": {"type": "number", "minimum": .5, "maximum": .95}}}}}


def payload_hash(messages, seed: int, max_tokens: int, response_format) -> str:
    payload = {"model": MODEL_ID, "messages": messages, "temperature": 0.0, "top_p": .95, "top_k": 50, "max_tokens": max_tokens, "n": 1, "reasoning": {"max_tokens": REASONING_MAX_TOKENS, "exclude": False}, "seed": seed, "response_format": response_format, "provider": {"require_parameters": False}}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def expected_requests() -> list[dict[str, Any]]:
    episodes = load_episodes()
    public = [public_episode(row, index) for index, row in enumerate(episodes)]
    requests = []
    for index, row in enumerate(public):
        messages = root_messages(row)
        requests.append({"episode_index": index, "kind": "root", "seed": ROOT_SEEDS[index], "prompt_sha256": hashlib.sha256(canonical_json(messages).encode()).hexdigest(), "payload_sha256": payload_hash(messages, ROOT_SEEDS[index], REQUEST_MAX_TOKENS, root_response_format())})
    for index, row in enumerate(public):
        messages = native_messages(row)
        requests.append({"episode_index": index, "kind": "native", "seed": NATIVE_SEEDS[index], "prompt_sha256": hashlib.sha256(canonical_json(messages).encode()).hexdigest(), "payload_sha256": payload_hash(messages, NATIVE_SEEDS[index], REQUEST_MAX_TOKENS, native_response_format())})
    return requests


def canonical_groups(values):
    labels = {}
    result = []
    for value in values:
        key = value if isinstance(value, str) else canonical_json(value)
        labels.setdefault(key, len(labels))
        result.append(labels[key])
    return result


def confidence(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not .5 <= float(value) <= .95:
        raise ValueError("independent documented confidence changed")
    return float(value)


def parse_root(raw):
    payload = json.loads(raw)
    rows = payload.get("actions") if isinstance(payload, dict) and set(payload) == {"actions"} else None
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("independent documented root array changed")
    result = {}
    for action, row in zip(ROOTS, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"action_id", "all_worlds_same", "confidence"} or row["action_id"] != action or not isinstance(row["all_worlds_same"], bool):
            raise ValueError("independent documented root row changed")
        result[action] = {"all_worlds_same": row["all_worlds_same"], "confidence": confidence(row["confidence"])}
    return result


def parse_native(raw):
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"action_id", "world_groups", "confidence"} or payload["action_id"] != NATIVE:
        raise ValueError("independent documented native root changed")
    groups = payload["world_groups"]
    if not isinstance(groups, list) or len(groups) != 4 or any(isinstance(value, bool) or not isinstance(value, int) or value not in range(4) for value in groups) or canonical_groups(groups) != groups:
        raise ValueError("independent documented native groups changed")
    return {"groups": groups, "confidence": confidence(payload["confidence"])}


def table(groups, conf):
    categories = [str(value) for value in sorted(set(groups))] + ["OTHER"]
    rest = (1.0 - conf) / (len(categories) - 1)
    result = {f"w{index}": {category: conf if category == str(group) else rest for category in categories} for index, group in enumerate(groups)}
    if any(not math.isclose(sum(row.values()), 1.0, abs_tol=1e-12) for row in result.values()):
        raise ValueError("independent documented likelihood changed")
    return result


def score(episodes, roots, natives):
    task_bank, get_environment = mathlib.exact.initialize_tau2()
    observations = [mathlib.official_episode(row, task_bank, get_environment) for row in episodes]
    root_ok = root_n = native_exact = pair_ok = pair_n = native_top = native_n = 0
    root_brier = native_part_brier = native_mass = native_post_brier = 0.0
    tvs = []
    semantic = []
    source_values = []
    episode_metrics = []
    for episode, root, native, truth in zip(episodes, roots, natives, observations, strict=True):
        family = episode["family"]
        prior = [.25] * 4
        all_tables = {}
        for action in ROOTS:
            expected = len({canonical_json(world[action]) for world in truth}) == 1
            predicted = root[action]["all_worlds_same"]
            conf = root[action]["confidence"]
            root_ok += predicted == expected
            root_n += 1
            probability_true = conf if predicted else 1 - conf
            root_brier += 2 * (probability_true - float(expected)) ** 2
            groups = [0, 0, 0, 0] if predicted else [0, 1, 2, 3]
            all_tables[action] = table(groups, conf)
            if expected:
                for left in range(4):
                    for right in range(left + 1, 4):
                        tvs.append(.5 * sum(abs(all_tables[action][f"w{left}"][key] - all_tables[action][f"w{right}"][key]) for key in all_tables[action][f"w{left}"]))
        expected_native = canonical_groups([world[NATIVE] for world in truth])
        native_exact += native["groups"] == expected_native
        native_table = table(native["groups"], native["confidence"])
        all_tables[NATIVE] = native_table
        for left in range(4):
            for right in range(left + 1, 4):
                pair_ok += (native["groups"][left] == native["groups"][right]) == (expected_native[left] == expected_native[right])
                pair_n += 1
        for world_index, outcome in enumerate(expected_native):
            category = str(outcome) if str(outcome) in native_table[f"w{world_index}"] else "OTHER"
            native_part_brier += sum((probability - float(key == category)) ** 2 for key, probability in native_table[f"w{world_index}"].items())
            posterior = mathlib.posterior(prior, native_table, category)
            native_mass += posterior[world_index]
            native_top += posterior[world_index] >= max(posterior) - 1e-12
            native_post_brier += sum((probability - float(index == world_index)) ** 2 for index, probability in enumerate(posterior))
            native_n += 1
        root_values = {action: mathlib.information(prior, all_tables[action]) for action in ROOTS}
        depth_values = {action: mathlib.depth_two(family, all_tables, action) for action in ROOTS}
        greedy = max(ROOTS, key=lambda action: (root_values[action], action))
        planned = max(ROOTS, key=lambda action: (depth_values[action], action))
        canonical = [{action: canonical_json(value) for action, value in world.items()} for world in truth]
        exact_values = {action: mathlib.exact.two_step_information(family, canonical, action) for action in ROOTS}
        semantic.extend(depth_values.values())
        source_values.extend(exact_values.values())
        episode_metrics.append({"family": family, "greedy_first_action": greedy, "depth_two_first_action": planned, "horizon_gain_nats": depth_values[planned] - depth_values[greedy], "root_information_nats": root_values, "two_step_information_nats": depth_values, "exact_two_step_information_nats": exact_values})
    metrics = {"root_decision_count": root_n, "root_exact_count": root_ok, "root_mean_brier": root_brier / root_n, "native_partition_count": 6, "native_partition_exact_count": native_exact, "native_pair_relation_count": pair_n, "native_pair_relation_accuracy": pair_ok / pair_n, "native_mean_partition_brier": native_part_brier / 24, "native_answer_count": native_n, "native_truth_top_rank_count": native_top, "native_mean_truth_posterior": native_mass / native_n, "native_mean_posterior_brier": native_post_brier / native_n, "equivalent_pair_count": len(tvs), "equivalent_mean_total_variation": sum(tvs) / len(tvs), "equivalent_max_total_variation": max(tvs), "semantic_source_two_step_spearman": mathlib.spearman(semantic, source_values), "mean_horizon_gain_nats": sum(row["horizon_gain_nats"] for row in episode_metrics) / 6}
    gates = {"exact_48_root_decisions": root_n == 48, "at_least_47_exact_root_decisions": root_ok >= 47, "root_brier_at_most_0_08": metrics["root_mean_brier"] <= .08, "all_six_native_partitions_exact": native_exact == 6, "exact_36_native_pair_relations": pair_n == 36, "native_pair_accuracy_one": metrics["native_pair_relation_accuracy"] == 1.0, "native_partition_brier_at_most_0_08": metrics["native_mean_partition_brier"] <= .08, "exact_24_native_answers": native_n == 24, "all_24_native_truth_top_rank": native_top == 24, "native_truth_mass_at_least_0_65": metrics["native_mean_truth_posterior"] >= .65, "native_posterior_brier_at_most_0_18": metrics["native_mean_posterior_brier"] <= .18, "equivalent_mean_tv_at_most_0_03": metrics["equivalent_mean_total_variation"] <= .03, "equivalent_max_tv_at_most_0_10": metrics["equivalent_max_total_variation"] <= .10, "all_six_greedy_avoid_installed_apps": all(row["greedy_first_action"] != "installed_apps" for row in episode_metrics), "all_six_depth_two_choose_installed_apps": all(row["depth_two_first_action"] == "installed_apps" for row in episode_metrics), "all_six_horizon_gain_at_least_0_50": all(row["horizon_gain_nats"] >= .5 for row in episode_metrics), "semantic_source_spearman_at_least_0_90": metrics["semantic_source_two_step_spearman"] >= .9}
    gates["all_calibration_gates_pass"] = all(gates.values())
    return {"metrics": metrics, "episode_metrics": episode_metrics, "calibration_gates": gates}


def validate_partial(partial):
    expected = expected_requests()
    expected_identity = [{"index": index, **{key: row[key] for key in ("episode_index", "kind", "seed", "prompt_sha256")}} for index, row in enumerate(expected)]
    if partial.get("interface_version") != INTERFACE_VERSION or partial.get("model_id") != MODEL_ID or partial.get("requests") != expected_identity:
        raise ValueError("independent documented partial identity changed")
    rows = partial.get("completed")
    indexes = [row.get("index") for row in rows] if isinstance(rows, list) else []
    if indexes != list(range(12)) or partial.get("complete") is not True:
        raise ValueError("independent documented partial incomplete")
    for index, row in enumerate(rows):
        if set(row) != {"index", "episode_index", "kind", "seed", "response"} or {key: row[key] for key in ("episode_index", "kind", "seed")} != {key: expected[index][key] for key in ("episode_index", "kind", "seed")} or not isinstance(row["response"], str):
            raise ValueError("independent documented partial row changed")
    attempts = partial.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 12:
        raise ValueError("independent documented attempt count changed")
    expected_first = [{"index": index, "episode_index": row["episode_index"], "kind": row["kind"], "attempt_number": 1, "seed": row["seed"], "payload_sha256": row["payload_sha256"]} for index, row in enumerate(expected)]
    if attempts[:12] != expected_first:
        raise ValueError("independent documented first-attempt identity changed")
    serving_records = partial.get("serving_records")
    expected_keys = {"index", "episode_index", "kind", "seed", "content_nonempty", "finish_reason", "reasoning_tokens", "completion_tokens", "prompt_tokens", "cost_usd"}
    if not isinstance(serving_records, list) or len(serving_records) != 12:
        raise ValueError("independent budgeted serving record count changed")
    for index, row in enumerate(serving_records):
        if set(row) != expected_keys or row["index"] != index or {key: row[key] for key in ("episode_index", "kind", "seed")} != {key: expected[index][key] for key in ("episode_index", "kind", "seed")}:
            raise ValueError("independent budgeted serving record identity changed")
        if not isinstance(row["content_nonempty"], bool) or not isinstance(row["finish_reason"], str):
            raise ValueError("independent budgeted serving record type changed")
        if any(isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0 for key in ("reasoning_tokens", "completion_tokens", "prompt_tokens")):
            raise ValueError("independent budgeted serving token count changed")
        if isinstance(row["cost_usd"], bool) or not isinstance(row["cost_usd"], (int, float)) or not math.isfinite(float(row["cost_usd"])) or row["cost_usd"] < 0:
            raise ValueError("independent budgeted serving cost changed")
    return [row["response"] for row in rows], attempts, serving_records


def replay(raw):
    expected = expected_requests()
    raw_expected = [{key: row[key] for key in ("episode_index", "kind", "seed", "prompt_sha256")} for row in expected]
    if raw.get("model_id") != MODEL_ID or raw.get("requests") != raw_expected or not isinstance(raw.get("responses"), list) or len(raw["responses"]) != 12:
        raise ValueError("independent documented raw bank changed")
    responses = raw["responses"]
    episodes = load_episodes()
    return score(episodes, [parse_root(responses[index]) for index in range(6)], [parse_native(responses[6 + index]) for index in range(6)])


def verify(run_dir: Path, *, output_path: Path | None = None):
    result = load_object(run_dir / "RESULT.json")
    raw = load_object(run_dir / "private/RAW_RESPONSES.json")
    partial = load_object(run_dir / "private/PARTIAL_RAW_RESPONSES.json")
    ordering = load_object(run_dir / "private/ORDERING.json")
    privacy = load_object(run_dir / "private/PROMPT_PRIVACY.json")
    serving_bank = load_object(run_dir / "private/SERVING.json")
    partial_responses, attempts, serving_records = validate_partial(partial)
    if raw["responses"] != partial_responses:
        raise ValueError("independent documented banks differ")
    replayed = replay(raw)
    for key in ("metrics", "episode_metrics", "calibration_gates"):
        if result.get(key) != replayed[key]:
            raise ValueError(f"independent documented {key} differs")
    usage = result.get("usage", {})
    serving = {"exact_twelve_accepted_requests": usage.get("adapter_requests") == 12, "exact_twelve_http_attempts": usage.get("http_attempts") == 12, "zero_retries": usage.get("retry_count") == 0, "zero_provider_error_retries": usage.get("provider_error_retries") == 0, "attempt_identity_complete": len(attempts) == 12 and all(row["attempt_number"] == 1 for row in attempts), "exact_twelve_serving_records": len(serving_records) == 12, "positive_reasoning_tokens": usage.get("adapter_reasoning_tokens", 0) > 0, "every_request_reasoning_within_4096": len(serving_records) == 12 and all(0 <= row["reasoning_tokens"] <= REASONING_MAX_TOKENS for row in serving_records), "all_finish_stop": len(serving_records) == 12 and all(row["finish_reason"] == "stop" for row in serving_records), "all_content_nonempty": len(serving_records) == 12 and all(row["content_nonempty"] for row in serving_records), "zero_forced_exits": usage.get("forced_exits") == 0, "zero_forced_final_requests": usage.get("forced_final_requests") == 0, "zero_forced_final_successes": usage.get("forced_final_successes") == 0, "within_stage_cap": float(usage.get("run_cost_usd", math.inf)) <= .08 + 1e-12}
    passed = all(serving.values()) and replayed["calibration_gates"]["all_calibration_gates_pass"]
    expected_ordering = {"all_partial_responses_banked": True, "complete_raw_bank_created": True, "official_calibration_loaded_after_complete_bank": True}
    expected_privacy = {"prompt_sha256": [row["prompt_sha256"] for row in expected_requests()], "public_tool_contract_in_prompts": True, "selected_task_ids_in_prompts": False, "source_fault_ids_in_prompts": False, "raw_tool_responses_in_prompts": False, "repair_or_endpoint_outcomes_in_prompts": False}
    expected_status = "mms_gemma26b_budgeted_thinking_pass" if passed else "mms_gemma26b_budgeted_thinking_null"
    checks = {
        "result_identity_exact": result.get("schema_version") == 1 and result.get("interface_version") == INTERFACE_VERSION and result.get("protocol_sha256") == sha256_file(PROTOCOL) and result.get("manifest_sha256") == sha256_file(MANIFEST) and result.get("model") == MODEL_ID and result.get("root_seeds") == list(ROOT_SEEDS) and result.get("native_seeds") == list(NATIVE_SEEDS),
        "serving_exact": result.get("serving_gates") == serving,
        "serving_records_exact": result.get("serving_records") == serving_records and serving_bank == {"usage": usage, "attempts": attempts, "serving_records": serving_records, "serving_gates": serving},
        "attempt_identity_exact": result.get("attempt_identity") == {"attempts": attempts},
        "status_exact": result.get("status") == expected_status and result.get("authorizes") == ("prospective_paired_development_protocol_only" if passed else "nothing"),
        "ordering_exact": result.get("ordering") == ordering == expected_ordering,
        "privacy_exact": result.get("privacy") == privacy == expected_privacy,
        "downstream_unopened": result.get("development_confirmation_reserve_opened") is False and result.get("repair_or_task_success_endpoints_opened") is False,
    }
    verification = {"schema_version": 1, "interface_version": "tau2-mms-gemma26b-budgeted-thinking-verification-1", "status": "verified" if all(checks.values()) else "invalid", "expected_result_status": expected_status, "checks": checks, "all_pass": all(checks.values()), "result_sha256": sha256_file(run_dir / "RESULT.json"), "raw_bank_sha256": sha256_file(run_dir / "private/RAW_RESPONSES.json"), "partial_bank_sha256": sha256_file(run_dir / "private/PARTIAL_RAW_RESPONSES.json"), "model_calls_made": 0, "cost_usd": 0.0}
    if not verification["all_pass"]:
        raise ValueError("independent documented verification failed")
    if output_path is not None:
        output_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    return verification
