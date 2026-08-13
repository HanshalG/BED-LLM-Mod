#!/usr/bin/env python3
"""Run the frozen Tau2 native-prerequisite semantic mechanics gate."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import tau2_native_prerequisite_opportunity as exact
from scripts import tau2_native_prerequisite_source_manifest as source
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.number_game_qwen_history_blind_serving_smoke import (
    PerRequestSeedStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-native-prerequisite-semantic-mechanics-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
MODEL_SEEDS = tuple(range(202608130100, 202608130106))
TEMPERATURE = 0.0
MAX_TOKENS = 6_000
EXPECTED_REQUESTS = 6
CONCURRENCY = 6
MAX_RETRIES = 0
RUN_CAP_USD = 0.10
PROJECTED_COST_USD = 0.06
MAX_REQUEST_COST_USD = 0.01
CONFIDENCE_MIN = 0.50
CONFIDENCE_MAX = 0.95
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "TAU2_NATIVE_PREREQUISITE_SEMANTIC_MECHANICS_PROTOCOL_20260813.md"
)
SOURCE_PROTOCOL = source.PROTOCOL
SOURCE_MANIFEST = source.OUTPUT_DIR / "SOURCE_MANIFEST.json"
SOURCE_RESULT = source.OUTPUT_DIR / "RESULT.json"
OPPORTUNITY_RESULT = exact.OUTPUT_DIR / "RESULT.json"


FAULT_TEXT = {
    "airplane_mode_on": "airplane mode is on",
    "bad_network_preference": "preferred network mode is restricted to 2G",
    "bad_vpn": "the active VPN degrades mobile data",
    "bad_wifi_calling": "Wi-Fi Calling is enabled",
    "break_apn_mms_setting": "the MMSC URL is not configured",
    "data_mode_off": "mobile data is switched off",
    "data_saver_mode_on": "data saver is enabled",
    "data_usage_exceeded": "data usage is above the included allowance",
    "unseat_sim_card": "the SIM card is unseated",
    "user_abroad_roaming_disabled_off": (
        "the user is abroad, device data roaming is off, and account roaming is disabled"
    ),
    "user_abroad_roaming_disabled_on": (
        "the user is abroad, device data roaming is on, and account roaming is disabled"
    ),
    "user_abroad_roaming_enabled_off": (
        "the user is abroad, device data roaming is off, and account roaming is enabled"
    ),
    "break_app_sms_permission": "the messaging app lacks SMS permission",
    "break_app_storage_permission": "the messaging app lacks storage permission",
    "break_app_both_permissions": (
        "the messaging app lacks both SMS and storage permissions"
    ),
}

MMS_ACTION_FIELDS: dict[str, dict[str, Any]] = {
    "apn_settings": {"mmsc_configured": [False, True]},
    "installed_apps": {"messaging_installed": [False, True]},
    "mms_probe": {"can_send": [False, True]},
    "network_mode": {
        "mode": ["2g_only", "4g_5g_preferred", "other"]
    },
    "network_status": {
        "airplane_mode": [False, True],
        "sim_active": [False, True],
        "connection": ["no_service", "connected", "other"],
        "signal": ["none", "poor", "excellent", "other"],
        "network_type": ["none", "2g", "5g", "other"],
        "mobile_data": [False, True],
        "data_roaming": [False, True],
        "wifi_radio": [False, True],
        "wifi_connected": [False, True],
    },
    "speed_test": {
        "speed": ["no_connection", "poor", "fair", "good", "excellent"]
    },
    "status_bar": {
        "airplane_mode": [False, True],
        "signal": ["none", "poor", "excellent", "other"],
        "network_type": ["none", "2g", "5g", "other"],
        "data_enabled": [False, True],
    },
    "wifi_calling": {"enabled": [False, True]},
    "messaging_permissions": {
        "sms": [False, True],
        "storage": [False, True],
        "phone": [False, True],
    },
}

MOBILE_ACTION_FIELDS: dict[str, dict[str, Any]] = {
    "customer_lookup": {"account_active": [False, True]},
    "network_status": MMS_ACTION_FIELDS["network_status"],
    "payment_request": {"requested": [False, True]},
    "sim_status": {"active": [False, True]},
    "speed_test": MMS_ACTION_FIELDS["speed_test"],
    "status_bar": MMS_ACTION_FIELDS["status_bar"],
    "customer_bills": {"has_overage_charge": [False, True]},
    "data_usage": {"exhausted": [False, True]},
    "line_details": {
        "roaming_enabled": [False, True],
        "data_exhausted": [False, True],
        "line_active": [False, True],
    },
}


class StructuredAdapter(Protocol):
    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class NonReasoningSeededAdapter(PerRequestSeedStructuredAdapter):
    """Bind request seeds while explicitly disabling model reasoning."""

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=True,
            response_format=response_format,
        )
        payload["reasoning"] = {"enabled": False, "exclude": True}
        return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def action_fields(family: str) -> dict[str, dict[str, Any]]:
    return MMS_ACTION_FIELDS if family.startswith("mms_") else MOBILE_ACTION_FIELDS


def world_description(world: Mapping[str, Any]) -> str:
    faults = sorted(set(world["backbone"]) | set(world["target"]))
    return "; ".join(FAULT_TEXT[item] for item in faults) or "none of the listed faults"


def public_episode(episode: Mapping[str, Any], episode_index: int) -> dict[str, Any]:
    worlds = []
    for world_index, world in enumerate(episode["worlds"]):
        worlds.append(
            {
                "world_id": f"w{world_index}",
                "description": world_description(
                    {"backbone": episode["backbone"], "target": world["target"]}
                ),
            }
        )
    return {
        "episode_index": episode_index,
        "family": episode["family"],
        "worlds": worlds,
        "actions": [
            {"action_id": action, "observable_fields": fields}
            for action, fields in action_fields(episode["family"]).items()
        ],
        "legal_unlock": (
            "installed_apps unlocks messaging_permissions"
            if episode["family"].startswith("mms_")
            else "customer_lookup unlocks customer_bills, data_usage, and line_details"
        ),
    }


def mechanics_episodes() -> list[dict[str, Any]]:
    task_payload = json.loads(source.TASKS_PATH.read_text())
    split_payload = json.loads(source.SPLITS_PATH.read_text())
    selected, _ = source.select_episodes(
        [str(row["id"]) for row in task_payload], list(split_payload["base"])
    )
    return list(selected["mechanics"])


def strict_value_schema(values: Sequence[Any]) -> dict[str, Any]:
    if all(isinstance(value, bool) for value in values):
        return {"type": "boolean"}
    return {"type": "string", "enum": list(values)}


def response_format(family: str, world_count: int) -> dict[str, Any]:
    action_properties: dict[str, Any] = {}
    for action, fields in action_fields(family).items():
        properties = {
            name: strict_value_schema(values) for name, values in fields.items()
        }
        properties["confidence"] = {
            "type": "number",
            "minimum": CONFIDENCE_MIN,
            "maximum": CONFIDENCE_MAX,
        }
        action_properties[action] = {
            "type": "object",
            "additionalProperties": False,
            "required": [*fields, "confidence"],
            "properties": properties,
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"tau2_native_{'mms' if family.startswith('mms_') else 'mobile'}",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["worlds"],
                "properties": {
                    "worlds": {
                        "type": "array",
                        "minItems": world_count,
                        "maxItems": world_count,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["world_id", "predictions"],
                            "properties": {
                                "world_id": {"type": "string"},
                                "predictions": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": list(action_properties),
                                    "properties": action_properties,
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def messages_for_episode(public: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a semantic forward model for telecom diagnostics. Predict "
                "only what each read-only tool would visibly return in each supplied "
                "world. Return strict JSON with no reasoning text. Never report a "
                "diagnosis or choose an action."
            ),
        },
        {
            "role": "user",
            "content": (
                "The candidate worlds are complete and equally possible. Conditions "
                "not mentioned in a world are normal. For each world and action, fill "
                "the supplied typed observable fields. Use confidence to express your "
                "probability that the complete typed signature is correct. The same "
                "physical observation must receive the same field values even when "
                "world descriptions differ. Do not infer hidden faults into a tool "
                "that cannot expose them. Respect this native unlock: "
                f"{public['legal_unlock']}. Unlocked reads are predictions after the "
                "prerequisite has supplied identifiers/app identity. Preserve the "
                "world and action order exactly.\n"
                + canonical_json(public)
            ),
        },
    ]


def parse_response(raw: str, public: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"worlds"}:
        raise ValueError("semantic response root changed")
    rows = value["worlds"]
    expected_worlds = public["worlds"]
    fields = action_fields(str(public["family"]))
    if not isinstance(rows, list) or len(rows) != len(expected_worlds):
        raise ValueError("semantic response world count changed")
    parsed = []
    for expected, row in zip(expected_worlds, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"world_id", "predictions"}:
            raise ValueError("semantic world fields changed")
        if row["world_id"] != expected["world_id"]:
            raise ValueError("semantic world order changed")
        predictions = row["predictions"]
        if not isinstance(predictions, dict) or list(predictions) != list(fields):
            raise ValueError("semantic action order changed")
        clean_predictions = {}
        for action, allowed in fields.items():
            prediction = predictions[action]
            required = {*allowed, "confidence"}
            if not isinstance(prediction, dict) or set(prediction) != required:
                raise ValueError("semantic typed fields changed")
            signature = {}
            for name, options in allowed.items():
                item = prediction[name]
                if item not in options or type(item) is not type(options[0]):
                    raise ValueError("semantic typed value is invalid")
                signature[name] = item
            confidence = prediction["confidence"]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not CONFIDENCE_MIN <= float(confidence) <= CONFIDENCE_MAX
            ):
                raise ValueError("semantic confidence is invalid")
            clean_predictions[action] = {
                "signature": signature,
                "confidence": float(confidence),
            }
        parsed.append(
            {"world_id": row["world_id"], "predictions": clean_predictions}
        )
    return {"worlds": parsed}


def signature_key(signature: Mapping[str, Any]) -> str:
    return canonical_json(dict(signature))


def likelihood_tables(parsed: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    worlds = parsed["worlds"]
    actions = list(worlds[0]["predictions"])
    tables: dict[str, dict[str, dict[str, float]]] = {}
    for action in actions:
        predicted = [
            signature_key(row["predictions"][action]["signature"]) for row in worlds
        ]
        categories = sorted(set(predicted)) + ["OTHER"]
        action_table = {}
        for row, own in zip(worlds, predicted, strict=True):
            confidence = float(row["predictions"][action]["confidence"])
            remainder = (1.0 - confidence) / (len(categories) - 1)
            action_table[row["world_id"]] = {
                category: confidence if category == own else remainder
                for category in categories
            }
        tables[action] = action_table
    return tables


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def posterior(
    prior: Sequence[float], table: Mapping[str, Mapping[str, float]], outcome: str
) -> list[float]:
    world_ids = list(table)
    values = [prior[index] * float(table[world][outcome]) for index, world in enumerate(world_ids)]
    total = sum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("semantic posterior has zero or invalid mass")
    return [value / total for value in values]


def expected_information(prior: Sequence[float], table: Mapping[str, Mapping[str, float]]) -> float:
    world_ids = list(table)
    outcomes = list(next(iter(table.values())))
    start = entropy(prior)
    expected = 0.0
    for outcome in outcomes:
        probability = sum(
            prior[index] * float(table[world][outcome])
            for index, world in enumerate(world_ids)
        )
        if probability > 0.0:
            expected += probability * entropy(posterior(prior, table, outcome))
    return start - expected


def legal_followups_for(family: str, action: str) -> tuple[str, ...]:
    return exact.legal_followups(family, action)


def two_step_information(
    family: str,
    tables: Mapping[str, Mapping[str, Mapping[str, float]]],
    first_action: str,
) -> float:
    world_ids = list(next(iter(tables.values())))
    prior = [1.0 / len(world_ids)] * len(world_ids)
    first = tables[first_action]
    total = expected_information(prior, first)
    outcomes = list(next(iter(first.values())))
    continuation = 0.0
    for outcome in outcomes:
        probability = sum(
            prior[index] * float(first[world][outcome])
            for index, world in enumerate(world_ids)
        )
        if probability <= 0.0:
            continue
        branch_prior = posterior(prior, first, outcome)
        best = max(
            expected_information(branch_prior, tables[action])
            for action in legal_followups_for(family, first_action)
        )
        continuation += probability * best
    return total + continuation


def semantic_plan(family: str, parsed: Mapping[str, Any]) -> dict[str, Any]:
    tables = likelihood_tables(parsed)
    roots = exact.root_actions(family)
    prior = [1.0 / len(parsed["worlds"])] * len(parsed["worlds"])
    root_values = {
        action: expected_information(prior, tables[action]) for action in roots
    }
    depth_values = {
        action: two_step_information(family, tables, action) for action in roots
    }
    greedy = max(roots, key=lambda action: (root_values[action], action))
    depth_two = max(roots, key=lambda action: (depth_values[action], action))
    return {
        "greedy_first_action": greedy,
        "depth_two_first_action": depth_two,
        "horizon_gain_nats": depth_values[depth_two] - depth_values[greedy],
        "root_information_nats": root_values,
        "two_step_information_nats": depth_values,
    }


def _enum_text(text: str, options: Sequence[str]) -> str:
    folded = text.casefold()
    for option in options:
        if option.casefold() in folded:
            return option
    return "other"


def typed_mms_observations(environment: Any) -> dict[str, dict[str, Any]]:
    apn = str(environment.user_tools.check_apn_settings())
    apps = str(environment.user_tools.check_installed_apps())
    probe = str(environment.user_tools.can_send_mms())
    mode = str(environment.user_tools.check_network_mode_preference()).casefold()
    network = str(environment.user_tools.check_network_status())
    speed = str(environment.user_tools.run_speed_test())
    status = str(environment.user_tools.check_status_bar())
    wifi = str(environment.user_tools.check_wifi_calling_status())
    permissions = str(environment.user_tools.check_app_permissions("messaging")).casefold()
    return {
        "apn_settings": {"mmsc_configured": "not set" not in apn.casefold()},
        "installed_apps": {"messaging_installed": "messaging" in apps.casefold()},
        "mms_probe": {"can_send": "cannot send" not in probe.casefold()},
        "network_mode": {
            "mode": "2g_only" if "2g_only" in mode else "4g_5g_preferred" if "4g_5g_preferred" in mode else "other"
        },
        "network_status": parse_network_status(network),
        "speed_test": {"speed": parse_speed(speed)},
        "status_bar": parse_status_bar(status),
        "wifi_calling": {"enabled": "turned on" in wifi.casefold()},
        "messaging_permissions": {
            "sms": "sms" in permissions,
            "storage": "storage" in permissions,
            "phone": "phone" in permissions,
        },
    }


def parse_network_status(text: str) -> dict[str, Any]:
    folded = text.casefold()
    return {
        "airplane_mode": "airplane mode: on" in folded,
        "sim_active": "sim card status: active" in folded,
        "connection": _enum_text(folded, ("no_service", "connected")),
        "signal": _enum_text(folded, ("none", "poor", "excellent")),
        "network_type": _enum_text(folded, ("none", "2g", "5g")),
        "mobile_data": "mobile data enabled: yes" in folded,
        "data_roaming": "data roaming enabled: yes" in folded,
        "wifi_radio": "wi-fi radio: on" in folded,
        "wifi_connected": "wi-fi connected: yes" in folded,
    }


def parse_speed(text: str) -> str:
    folded = text.casefold()
    if "no connection" in folded or "failed" in folded:
        return "no_connection"
    for value in ("poor", "fair", "good", "excellent"):
        if value in folded:
            return value
    raise ValueError("unknown Tau2 speed result")


def parse_status_bar(text: str) -> dict[str, Any]:
    folded = text.casefold()
    return {
        "airplane_mode": "airplane mode" in folded,
        "signal": _enum_text(folded, ("poor", "excellent")) if "airplane mode" not in folded else "none",
        "network_type": _enum_text(folded, ("2g", "5g")) if "airplane mode" not in folded else "none",
        "data_enabled": "data enabled" in folded,
    }


def typed_mobile_observations(environment: Any) -> dict[str, dict[str, Any]]:
    phone = str(environment.user_tools.surroundings.phone_number)
    customer = environment.tools.get_customer_by_phone(phone)
    line = environment.tools._get_line_by_phone(phone)
    customer_id, line_id = str(customer.customer_id), str(line.line_id)
    network = str(environment.user_tools.check_network_status())
    payment = str(environment.user_tools.check_payment_request())
    sim = str(environment.user_tools.check_sim_status())
    speed = str(environment.user_tools.run_speed_test())
    status = str(environment.user_tools.check_status_bar())
    bills = exact.plain_value(environment.tools.get_bills_for_customer(customer_id))
    usage = exact.plain_value(environment.tools.get_data_usage(customer_id, line_id))
    details = exact.plain_value(environment.tools.get_details_by_id(line_id))
    return {
        "customer_lookup": {"account_active": str(customer.account_status).casefold() == "active"},
        "network_status": parse_network_status(network),
        "payment_request": {"requested": "no payment request" not in payment.casefold()},
        "sim_status": {"active": "active and working" in sim.casefold()},
        "speed_test": {"speed": parse_speed(speed)},
        "status_bar": parse_status_bar(status),
        "customer_bills": {
            "has_overage_charge": any(
                str(item.get("item_type", "")).casefold() == "overage"
                for bill in bills
                for item in bill.get("line_items", [])
            )
        },
        "data_usage": {"exhausted": float(usage["data_used_gb"]) > float(usage["data_limit_gb"])},
        "line_details": {
            "roaming_enabled": bool(details["roaming_enabled"]),
            "data_exhausted": float(details["data_used_gb"]) > 15.0,
            "line_active": str(details["status"]).casefold() == "active",
        },
    }


def official_observations(episodes: Sequence[Mapping[str, Any]]) -> list[list[dict[str, dict[str, Any]]]]:
    tasks, get_environment = exact.initialize_tau2()
    result = []
    for episode in episodes:
        rows = []
        for world in episode["worlds"]:
            environment = exact.initialized_environment(tasks[world["task_id"]], get_environment)
            rows.append(
                typed_mms_observations(environment)
                if str(episode["family"]).startswith("mms_")
                else typed_mobile_observations(environment)
            )
        result.append(rows)
    return result


def exact_source_values(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, float]:
    canonical = [
        {action: signature_key(signature) for action, signature in row.items()}
        for row in rows
    ]
    return {
        action: exact.two_step_information(family, canonical, action)
        for action in exact.root_actions(family)
    }


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(values[order[end]] - values[order[start]]) <= 1e-12:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = average_ranks(left), average_ranks(right)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    scale_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    scale_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    value = covariance / (scale_a * scale_b) if scale_a and scale_b else 0.0
    return round(max(-1.0, min(1.0, value)), 15)


def score_responses(
    episodes: Sequence[Mapping[str, Any]],
    parsed: Sequence[Mapping[str, Any]],
    observed: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    episode_rows = []
    signature_correct = 0
    cell_count = 0
    brier_total = 0.0
    family_briers: dict[str, list[float]] = defaultdict(list)
    native_exact = 0
    native_top = 0
    native_true_mass = 0.0
    native_posterior_brier = 0.0
    native_count = 0
    equivalent_tvs = []
    semantic_values, source_values = [], []
    for episode, prediction, actual_rows in zip(episodes, parsed, observed, strict=True):
        family = str(episode["family"])
        plan = semantic_plan(family, prediction)
        tables = likelihood_tables(prediction)
        for world_index, actual in enumerate(actual_rows):
            for action, signature in actual.items():
                predicted_signature = prediction["worlds"][world_index]["predictions"][action]["signature"]
                correct = signature_key(predicted_signature) == signature_key(signature)
                signature_correct += int(correct)
                cell_count += 1
                category = signature_key(signature)
                row = tables[action][f"w{world_index}"]
                realized = category if category in row else "OTHER"
                brier = sum((prob - float(outcome == realized)) ** 2 for outcome, prob in row.items())
                brier_total += brier
                family_briers[family].append(brier)
            native = "messaging_permissions" if family.startswith("mms_") else "line_details"
            signature = actual[native]
            predicted_signature = prediction["worlds"][world_index]["predictions"][native]["signature"]
            native_exact += int(signature_key(signature) == signature_key(predicted_signature))
            table = tables[native]
            category = signature_key(signature)
            outcome = category if category in next(iter(table.values())) else "OTHER"
            prior = [1.0 / len(actual_rows)] * len(actual_rows)
            post = posterior(prior, table, outcome)
            truth_mass = post[world_index]
            native_true_mass += truth_mass
            native_top += int(truth_mass >= max(post) - 1e-12)
            native_posterior_brier += sum(
                (mass - float(index == world_index)) ** 2 for index, mass in enumerate(post)
            )
            native_count += 1
        for action in action_fields(family):
            table = tables[action]
            for left in range(len(actual_rows)):
                for right in range(left + 1, len(actual_rows)):
                    if signature_key(actual_rows[left][action]) != signature_key(actual_rows[right][action]):
                        continue
                    categories = list(table[f"w{left}"])
                    equivalent_tvs.append(
                        0.5 * sum(abs(table[f"w{left}"][cat] - table[f"w{right}"][cat]) for cat in categories)
                    )
        exact_values = exact_source_values(actual_rows, family)
        for action in exact.root_actions(family):
            semantic_values.append(plan["two_step_information_nats"][action])
            source_values.append(exact_values[action])
        episode_rows.append(
            {
                "family": family,
                "state_count": len(actual_rows),
                "greedy_first_action": plan["greedy_first_action"],
                "depth_two_first_action": plan["depth_two_first_action"],
                "native_prerequisite": exact.prerequisite_action(family),
                "horizon_gain_nats": plan["horizon_gain_nats"],
                "root_information_nats": plan["root_information_nats"],
                "two_step_information_nats": plan["two_step_information_nats"],
                "exact_two_step_information_nats": exact_values,
            }
        )
    mean_brier = brier_total / cell_count
    family_mean_brier = {
        family: sum(values) / len(values) for family, values in family_briers.items()
    }
    mean_tv = sum(equivalent_tvs) / len(equivalent_tvs)
    max_tv = max(equivalent_tvs)
    rank = spearman(semantic_values, source_values)
    gains = [row["horizon_gain_nats"] for row in episode_rows]
    metrics = {
        "cell_count": cell_count,
        "typed_signature_accuracy": signature_correct / cell_count,
        "mean_multiclass_brier": mean_brier,
        "family_mean_multiclass_brier": family_mean_brier,
        "native_followup_cell_count": native_count,
        "native_signature_exact_count": native_exact,
        "native_truth_top_rank_count": native_top,
        "native_mean_truth_posterior": native_true_mass / native_count,
        "native_mean_posterior_brier": native_posterior_brier / native_count,
        "equivalent_pair_count": len(equivalent_tvs),
        "equivalent_mean_total_variation": mean_tv,
        "equivalent_max_total_variation": max_tv,
        "semantic_source_two_step_spearman": rank,
        "mean_horizon_gain_nats": sum(gains) / len(gains),
    }
    gates = {
        "exact_234_world_action_cells": cell_count == 234,
        "typed_signature_accuracy_at_least_0_80": metrics["typed_signature_accuracy"] >= 0.80,
        "mean_multiclass_brier_at_most_0_20": mean_brier <= 0.20,
        "each_family_brier_at_most_0_25": set(family_mean_brier) == set(source.EXPECTED_ELIGIBLE) and all(value <= 0.25 for value in family_mean_brier.values()),
        "exact_26_native_followup_cells": native_count == 26,
        "native_signature_exact_at_least_24": native_exact >= 24,
        "native_truth_top_rank_at_least_24": native_top >= 24,
        "native_mean_truth_posterior_at_least_0_60": metrics["native_mean_truth_posterior"] >= 0.60,
        "native_mean_posterior_brier_at_most_0_20": metrics["native_mean_posterior_brier"] <= 0.20,
        "equivalent_mean_tv_at_most_0_05": mean_tv <= 0.05,
        "equivalent_max_tv_at_most_0_15": max_tv <= 0.15,
        "all_six_greedy_avoid_prerequisite": all(row["greedy_first_action"] != row["native_prerequisite"] for row in episode_rows),
        "all_six_depth_two_choose_prerequisite": all(row["depth_two_first_action"] == row["native_prerequisite"] for row in episode_rows),
        "all_six_horizon_gain_at_least_0_05": all(value >= 0.05 for value in gains),
        "mean_horizon_gain_at_least_0_15": metrics["mean_horizon_gain_nats"] >= 0.15,
        "semantic_source_spearman_at_least_0_70": rank >= 0.70,
    }
    gates["all_semantic_and_planning_gates_pass"] = all(gates.values())
    return {"metrics": metrics, "episode_metrics": episode_rows, "gates": gates}


def usage_result(adapter: StructuredAdapter) -> tuple[dict[str, Any], dict[str, bool]]:
    usage = summarize_usage(adapter.usage_snapshot())
    gates = {
        "exact_six_accepted_requests": usage["adapter_requests"] == EXPECTED_REQUESTS,
        "exact_six_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_stage_cap": usage["run_cost_usd"] <= RUN_CAP_USD + 1e-12,
    }
    return usage, gates


def validate_bindings() -> dict[str, str]:
    opportunity = json.loads(OPPORTUNITY_RESULT.read_text())
    source_result = json.loads(SOURCE_RESULT.read_text())
    if opportunity.get("status") != "source_opportunity_pass" or opportunity.get("authorizes") != "semantic_mechanics_protocol_only":
        raise ValueError("Tau2 opportunity does not authorize semantic mechanics")
    if source_result.get("status") != "source_manifest_frozen":
        raise ValueError("Tau2 source manifest is not frozen")
    return {
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_protocol_sha256": sha256_file(SOURCE_PROTOCOL),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "source_result_sha256": sha256_file(SOURCE_RESULT),
        "opportunity_result_sha256": sha256_file(OPPORTUNITY_RESULT),
    }


def build_adapter(*, run_id: str, output_dir: Path) -> NonReasoningSeededAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=245.0,
        openrouter_run_budget_usd=RUN_CAP_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=MAX_RETRIES,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return NonReasoningSeededAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65_536), config
    )


def run(
    *,
    output_dir: Path,
    adapter: StructuredAdapter,
    daily_budget_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bindings = validate_bindings()
    episodes = mechanics_episodes()
    public = [public_episode(episode, index) for index, episode in enumerate(episodes)]
    prompts = [messages_for_episode(row) for row in public]
    privacy = {
        "prompt_sha256": [structural_hash(row) for row in prompts],
        "selected_task_ids_in_prompts": False,
        "source_fault_ids_in_prompts": False,
        "raw_tool_responses_in_prompts": False,
        "repair_or_endpoint_outcomes_in_prompts": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    checkpoint(private / "PROMPT_PRIVACY.json", privacy)
    def dispatch(start: int, stop: int) -> list[str]:
        return list(
            adapter.chat_complete_seeded_messages_batched_structured(
                prompts[start:stop],
                MODEL_SEEDS[start:stop],
                temperature=TEMPERATURE,
                response_format=response_format(
                    public[start]["family"], len(public[start]["worlds"])
                ),
                max_new_tokens=MAX_TOKENS,
            )
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        mms_future = executor.submit(dispatch, 0, 4)
        mobile_six_future = executor.submit(dispatch, 4, 5)
        mobile_four_future = executor.submit(dispatch, 5, 6)
        raw = (
            mms_future.result()
            + mobile_six_future.result()
            + mobile_four_future.result()
        )
    if len(raw) != EXPECTED_REQUESTS:
        raise ValueError("Tau2 semantic mechanics response count changed")
    checkpoint(
        private / "RAW_RESPONSES.json",
        {"model_id": MODEL_ID, "seeds": list(MODEL_SEEDS), "responses": raw},
    )
    ordering = {
        "all_raw_responses_banked": True,
        "official_mechanics_loaded_after_raw_bank": False,
    }
    checkpoint(private / "ORDERING.json", ordering)
    parsed = [
        parse_response(response, row) for response, row in zip(raw, public, strict=True)
    ]
    observed = official_observations(episodes)
    ordering["official_mechanics_loaded_after_raw_bank"] = True
    checkpoint(private / "ORDERING.json", ordering)
    scores = score_responses(episodes, parsed, observed)
    usage, serving_gates = usage_result(adapter)
    all_pass = all(serving_gates.values()) and scores["gates"]["all_semantic_and_planning_gates_pass"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "semantic_mechanics_pass" if all_pass else "semantic_mechanics_null",
        "authorizes": "prospective_development_protocol_only" if all_pass else "nothing",
        "bindings": bindings,
        "model": MODEL_ID,
        "model_seeds": list(MODEL_SEEDS),
        "privacy": privacy,
        "ordering": ordering,
        "usage": usage,
        "serving_gates": serving_gates,
        **scores,
        "daily_budget_status": dict(daily_budget_status or {}),
        "development_confirmation_reserve_opened": False,
        "repair_or_task_success_endpoints_opened": False,
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result
