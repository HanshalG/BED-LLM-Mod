#!/usr/bin/env python3
"""Independently replay the fresh Tau2 MMS array semantic calibration."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tau2_native_prerequisite_semantic_verify as independent
from scripts import tau2_native_prerequisite_source_manifest as source


SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-mms-array-semantic-verification-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
MODEL_SEEDS = tuple(range(202608130200, 202608130206))
MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_array_semantic_calibration/CALIBRATION_MANIFEST.json"


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
    return hashlib.sha256(
        (episode["family"] + "|" + "|".join(sorted(episode["backbone"]))).encode()
    ).hexdigest()


def selected_episodes() -> list[dict[str, Any]]:
    task_payload = json.loads(source.TASKS_PATH.read_text())
    split_payload = json.loads(source.SPLITS_PATH.read_text())
    selected, _ = source.select_episodes(
        [str(row["id"]) for row in task_payload], list(split_payload["base"])
    )
    by_hash = {episode_hash(row): row for row in selected["reserve"]}
    manifest = load_object(MANIFEST)
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("independent MMS manifest changed")
    episodes = []
    for row in rows:
        digest = row.get("episode_sha256")
        episode = by_hash.get(digest)
        if (
            episode is None
            or row.get("family") != episode["family"]
            or row.get("state_count") != len(episode["worlds"])
            or not episode["family"].startswith("mms_")
        ):
            raise ValueError("independent MMS episode binding changed")
        episodes.append(episode)
    return episodes


def parse_raw(raw: str) -> list[dict[str, Any]]:
    payload = json.loads(raw)
    worlds = payload.get("worlds") if isinstance(payload, dict) and set(payload) == {"worlds"} else None
    if not isinstance(worlds, list) or len(worlds) != 4:
        raise ValueError("independent MMS world array changed")
    parsed = []
    fields = independent.MMS_ACTION_FIELDS
    for world_index, world in enumerate(worlds):
        if not isinstance(world, dict) or set(world) != {"world_id", "actions"} or world["world_id"] != f"w{world_index}":
            raise ValueError("independent MMS world order changed")
        rows = world["actions"]
        if not isinstance(rows, list) or len(rows) != len(fields):
            raise ValueError("independent MMS action count changed")
        predictions = {}
        for (action_id, action_fields), row in zip(fields.items(), rows, strict=True):
            if not isinstance(row, dict) or set(row) != {"action_id", "fields", "confidence"} or row["action_id"] != action_id:
                raise ValueError("independent MMS action order changed")
            if not isinstance(row["fields"], list) or len(row["fields"]) != len(action_fields):
                raise ValueError("independent MMS field count changed")
            signature = {}
            for (field_id, options), item in zip(action_fields.items(), row["fields"], strict=True):
                if not isinstance(item, dict) or set(item) != {"field_id", "value"} or item["field_id"] != field_id:
                    raise ValueError("independent MMS field order changed")
                allowed = [str(value).lower() if isinstance(value, bool) else str(value) for value in options]
                if item["value"] not in allowed:
                    raise ValueError("independent MMS field value changed")
                value: Any = item["value"]
                if allowed == ["false", "true"]:
                    value = value == "true"
                signature[field_id] = value
            confidence = row["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0.5 <= float(confidence) <= 0.95:
                raise ValueError("independent MMS confidence changed")
            predictions[action_id] = {"signature": signature, "confidence": float(confidence)}
        parsed.append({"world_id": f"w{world_index}", "predictions": predictions})
    return parsed


def signature(value: Mapping[str, Any]) -> str:
    return canonical_json(dict(value))


def score(episodes: Sequence[Mapping[str, Any]], parsed: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    task_bank, get_environment = independent.exact.initialize_tau2()
    actual = [independent.official_episode(row, task_bank, get_environment) for row in episodes]
    cells = correct = native_exact = native_top = native_count = 0
    total_brier = native_mass = native_brier = 0.0
    family_brier: dict[str, list[float]] = defaultdict(list)
    tvs: list[float] = []
    semantic_values: list[float] = []
    source_values: list[float] = []
    episode_metrics = []
    for episode, rows, truths in zip(episodes, parsed, actual, strict=True):
        family = episode["family"]
        all_tables = independent.tables(rows)
        prior = [1.0 / len(rows)] * len(rows)
        root_values = {action: independent.information(prior, all_tables[action]) for action in independent.MMS_ROOTS}
        depth_values = {action: independent.depth_two(family, all_tables, action) for action in independent.MMS_ROOTS}
        greedy = max(independent.MMS_ROOTS, key=lambda action: (root_values[action], action))
        planned = max(independent.MMS_ROOTS, key=lambda action: (depth_values[action], action))
        for world_index, observations in enumerate(truths):
            for action, truth in observations.items():
                predicted = rows[world_index]["predictions"][action]["signature"]
                correct += signature(predicted) == signature(truth)
                cells += 1
                distribution = all_tables[action][f"w{world_index}"]
                category = signature(truth)
                outcome = category if category in distribution else "OTHER"
                brier = sum((probability - float(key == outcome)) ** 2 for key, probability in distribution.items())
                total_brier += brier
                family_brier[family].append(brier)
            native = "messaging_permissions"
            truth = observations[native]
            predicted = rows[world_index]["predictions"][native]["signature"]
            native_exact += signature(predicted) == signature(truth)
            distribution = all_tables[native][f"w{world_index}"]
            category = signature(truth)
            outcome = category if category in distribution else "OTHER"
            post = independent.posterior(prior, all_tables[native], outcome)
            native_mass += post[world_index]
            native_top += post[world_index] >= max(post) - 1e-12
            native_brier += sum((probability - float(index == world_index)) ** 2 for index, probability in enumerate(post))
            native_count += 1
        for action in independent.MMS_ACTION_FIELDS:
            table = all_tables[action]
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    if signature(truths[left][action]) == signature(truths[right][action]):
                        tvs.append(0.5 * sum(abs(table[f"w{left}"][key] - table[f"w{right}"][key]) for key in table[f"w{left}"]))
        canonical = [{action: signature(value) for action, value in row.items()} for row in truths]
        exact_values = {action: independent.exact.two_step_information(family, canonical, action) for action in independent.MMS_ROOTS}
        for action in independent.MMS_ROOTS:
            semantic_values.append(depth_values[action]); source_values.append(exact_values[action])
        episode_metrics.append({
            "family": family, "state_count": len(rows), "greedy_first_action": greedy,
            "depth_two_first_action": planned, "native_prerequisite": "installed_apps",
            "horizon_gain_nats": depth_values[planned] - depth_values[greedy],
            "root_information_nats": root_values, "two_step_information_nats": depth_values,
            "exact_two_step_information_nats": exact_values,
        })
    family_means = {key: sum(values) / len(values) for key, values in family_brier.items()}
    metrics = {
        "cell_count": cells, "typed_signature_accuracy": correct / cells,
        "mean_multiclass_brier": total_brier / cells,
        "family_mean_multiclass_brier": family_means,
        "native_followup_cell_count": native_count,
        "native_signature_exact_count": native_exact,
        "native_truth_top_rank_count": native_top,
        "native_mean_truth_posterior": native_mass / native_count,
        "native_mean_posterior_brier": native_brier / native_count,
        "equivalent_pair_count": len(tvs),
        "equivalent_mean_total_variation": sum(tvs) / len(tvs),
        "equivalent_max_total_variation": max(tvs),
        "semantic_source_two_step_spearman": independent.spearman(semantic_values, source_values),
        "mean_horizon_gain_nats": sum(row["horizon_gain_nats"] for row in episode_metrics) / 6,
    }
    gates = {
        "exact_216_world_action_cells": cells == 216,
        "typed_signature_accuracy_at_least_0_90": metrics["typed_signature_accuracy"] >= 0.90,
        "mean_multiclass_brier_at_most_0_18": metrics["mean_multiclass_brier"] <= 0.18,
        "each_family_brier_at_most_0_25": set(family_means) == {"mms_abroad", "mms_home"} and all(value <= 0.25 for value in family_means.values()),
        "exact_24_native_followup_cells": native_count == 24,
        "native_signature_exact_at_least_23": native_exact >= 23,
        "native_truth_top_rank_at_least_23": native_top >= 23,
        "native_mean_truth_posterior_at_least_0_65": metrics["native_mean_truth_posterior"] >= 0.65,
        "native_mean_posterior_brier_at_most_0_18": metrics["native_mean_posterior_brier"] <= 0.18,
        "equivalent_mean_tv_at_most_0_03": metrics["equivalent_mean_total_variation"] <= 0.03,
        "equivalent_max_tv_at_most_0_10": metrics["equivalent_max_total_variation"] <= 0.10,
        "all_six_greedy_avoid_installed_apps": all(row["greedy_first_action"] != "installed_apps" for row in episode_metrics),
        "all_six_depth_two_choose_installed_apps": all(row["depth_two_first_action"] == "installed_apps" for row in episode_metrics),
        "all_six_horizon_gain_at_least_0_10": all(row["horizon_gain_nats"] >= 0.10 for row in episode_metrics),
        "mean_horizon_gain_at_least_0_50": metrics["mean_horizon_gain_nats"] >= 0.50,
        "semantic_source_spearman_at_least_0_80": metrics["semantic_source_two_step_spearman"] >= 0.80,
    }
    gates["all_calibration_gates_pass"] = all(gates.values())
    return {"metrics": metrics, "episode_metrics": episode_metrics, "calibration_gates": gates}


def replay(raw_bank: Mapping[str, Any]) -> dict[str, Any]:
    if raw_bank.get("model_id") != MODEL_ID or tuple(raw_bank.get("seeds", ())) != MODEL_SEEDS or len(raw_bank.get("responses", ())) != 6:
        raise ValueError("independent MMS raw bank identity changed")
    episodes = selected_episodes()
    parsed = [parse_raw(raw) for raw in raw_bank["responses"]]
    return score(episodes, parsed)


def verify(run_dir: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    result = load_object(run_dir / "RESULT.json")
    raw = load_object(run_dir / "private/RAW_RESPONSES.json")
    ordering = load_object(run_dir / "private/ORDERING.json")
    privacy = load_object(run_dir / "private/PROMPT_PRIVACY.json")
    replayed = replay(raw)
    for key in ("metrics", "episode_metrics", "calibration_gates"):
        if result.get(key) != replayed[key]:
            raise ValueError(f"independent MMS {key} differs")
    usage = result.get("usage", {})
    serving = {
        "exact_six_accepted_requests": usage.get("adapter_requests") == 6,
        "exact_six_http_attempts": usage.get("http_attempts") == 6,
        "zero_retries": usage.get("retry_count") == 0,
        "zero_provider_error_retries": usage.get("provider_error_retries") == 0,
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "within_stage_cap": float(usage.get("run_cost_usd", math.inf)) <= 0.1 + 1e-12,
    }
    passed = all(serving.values()) and replayed["calibration_gates"]["all_calibration_gates_pass"]
    checks = {
        "serving_exact": result.get("serving_gates") == serving,
        "status_exact": result.get("status") == ("mms_array_semantic_pass" if passed else "mms_array_semantic_null") and result.get("authorizes") == ("prospective_paired_development_protocol_only" if passed else "nothing"),
        "ordering_exact": ordering == {"all_raw_responses_banked": True, "official_calibration_loaded_after_raw_bank": True} and result.get("ordering") == ordering,
        "privacy_exact": all(privacy.get(key) is False for key in ("selected_task_ids_in_prompts", "source_fault_ids_in_prompts", "raw_tool_responses_in_prompts", "repair_or_endpoint_outcomes_in_prompts")),
        "downstream_unopened": result.get("development_confirmation_reserve_opened") is False and result.get("repair_or_task_success_endpoints_opened") is False,
    }
    verification = {
        "schema_version": SCHEMA_VERSION, "interface_version": INTERFACE_VERSION,
        "status": "verified" if all(checks.values()) else "invalid",
        "expected_result_status": "mms_array_semantic_pass" if passed else "mms_array_semantic_null",
        "checks": checks, "all_pass": all(checks.values()),
        "result_sha256": sha256_file(run_dir / "RESULT.json"),
        "raw_bank_sha256": sha256_file(run_dir / "private/RAW_RESPONSES.json"),
        "model_calls_made": 0, "cost_usd": 0.0,
    }
    if not verification["all_pass"]:
        raise ValueError("independent MMS verification failed")
    if output_path:
        output_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    return verification


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir", type=Path); parser.add_argument("--output", type=Path); args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, output_path=args.output), indent=2, sort_keys=True))
