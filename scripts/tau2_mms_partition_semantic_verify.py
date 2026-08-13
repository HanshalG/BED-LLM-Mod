#!/usr/bin/env python3
"""Independently replay the Tau2 MMS observational-partition calibration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tau2_native_prerequisite_semantic_verify as mathlib
from scripts import tau2_native_prerequisite_source_manifest as source

SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-mms-partition-semantic-verification-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
MODEL_SEEDS = tuple(range(202608130400, 202608130406))
MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_partition_semantic_calibration/CALIBRATION_MANIFEST.json"
PRIOR_MANIFESTS = (
    REPO_ROOT / "results/nonmyopic/tau2_mms_array_semantic_calibration/CALIBRATION_MANIFEST.json",
    REPO_ROOT / "results/nonmyopic/tau2_mms_checkpointed_semantic_calibration/CALIBRATION_MANIFEST.json",
)
ACTIONS = tuple(mathlib.MMS_ACTION_FIELDS)
ROOTS = mathlib.MMS_ROOTS


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


def selected_episodes() -> list[dict[str, Any]]:
    tasks = json.loads(source.TASKS_PATH.read_text()); splits = json.loads(source.SPLITS_PATH.read_text())
    selected, _ = source.select_episodes([str(row["id"]) for row in tasks], list(splits["base"]))
    by_hash = {episode_hash(row): row for row in selected["reserve"]}
    rows = load_object(MANIFEST).get("episodes"); prior = {row["episode_sha256"] for path in PRIOR_MANIFESTS for row in load_object(path)["episodes"]}
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("independent partition manifest changed")
    episodes = []
    for row in rows:
        episode = by_hash.get(row.get("episode_sha256"))
        if episode is None or row.get("episode_sha256") in prior or row.get("family") != episode["family"] or len(episode["worlds"]) != 4:
            raise ValueError("independent partition episode changed")
        episodes.append(episode)
    return episodes


def canonical_groups(values: Sequence[Any]) -> list[int]:
    labels: dict[str, int] = {}; result = []
    for value in values:
        key = canonical_json(value) if not isinstance(value, str) else value
        labels.setdefault(key, len(labels)); result.append(labels[key])
    return result


def parse_raw(raw: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(raw); rows = payload.get("actions") if isinstance(payload, dict) and set(payload) == {"actions"} else None
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("independent partition action array changed")
    parsed = {}
    for expected, row in zip(ACTIONS, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"action_id", "world_groups", "confidence"} or row["action_id"] != expected:
            raise ValueError("independent partition action order changed")
        groups = row["world_groups"]
        if not isinstance(groups, list) or len(groups) != 4 or any(isinstance(x, bool) or not isinstance(x, int) or x not in range(4) for x in groups) or canonical_groups(groups) != groups:
            raise ValueError("independent partition labels changed")
        confidence = row["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0.5 <= float(confidence) <= 0.95:
            raise ValueError("independent partition confidence changed")
        parsed[expected] = {"groups": groups, "confidence": float(confidence)}
    return parsed


def tables(prediction: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for action, row in prediction.items():
        categories = [str(value) for value in sorted(set(row["groups"]))] + ["OTHER"]
        rest = (1.0 - row["confidence"]) / (len(categories) - 1)
        result[action] = {f"w{index}": {category: row["confidence"] if category == str(group) else rest for category in categories} for index, group in enumerate(row["groups"])}
        if any(not math.isclose(sum(values.values()), 1.0, abs_tol=1e-12) for values in result[action].values()):
            raise ValueError("independent partition likelihood changed")
    return result


def score(episodes, predictions):
    task_bank, get_environment = mathlib.exact.initialize_tau2()
    observations = [mathlib.official_episode(row, task_bank, get_environment) for row in episodes]
    exact_count = pair_ok = pair_count = native_exact = native_top = native_count = 0
    brier_sum = native_mass = native_brier = 0.0; family_brier = {"mms_abroad": [], "mms_home": []}; tvs = []; sem = []; src = []; episode_metrics = []
    for episode, prediction, truth in zip(episodes, predictions, observations, strict=True):
        family = episode["family"]; all_tables = tables(prediction); prior = [0.25] * 4
        truth_groups = {action: canonical_groups([world[action] for world in truth]) for action in ACTIONS}
        for action, expected in truth_groups.items():
            groups = prediction[action]["groups"]; exact_count += groups == expected
            for left in range(4):
                for right in range(left + 1, 4):
                    pair_ok += (groups[left] == groups[right]) == (expected[left] == expected[right]); pair_count += 1
            table = all_tables[action]
            for world_index, outcome in enumerate(expected):
                category = str(outcome) if str(outcome) in table[f"w{world_index}"] else "OTHER"
                value = sum((prob - float(key == category)) ** 2 for key, prob in table[f"w{world_index}"].items()); brier_sum += value; family_brier[family].append(value)
            for left in range(4):
                for right in range(left + 1, 4):
                    if expected[left] == expected[right]:
                        tvs.append(0.5 * sum(abs(table[f"w{left}"][key] - table[f"w{right}"][key]) for key in table[f"w{left}"]))
        native = "messaging_permissions"; expected = truth_groups[native]; native_exact += prediction[native]["groups"] == expected
        for world_index, outcome in enumerate(expected):
            table = all_tables[native]; category = str(outcome) if str(outcome) in table[f"w{world_index}"] else "OTHER"; post = mathlib.posterior(prior, table, category)
            native_mass += post[world_index]; native_top += post[world_index] >= max(post) - 1e-12; native_brier += sum((prob - float(i == world_index)) ** 2 for i, prob in enumerate(post)); native_count += 1
        roots = {action: mathlib.information(prior, all_tables[action]) for action in ROOTS}; depths = {action: mathlib.depth_two(family, all_tables, action) for action in ROOTS}
        greedy = max(ROOTS, key=lambda action: (roots[action], action)); planned = max(ROOTS, key=lambda action: (depths[action], action))
        canonical = [{action: canonical_json(value) for action, value in world.items()} for world in truth]; exact_values = {action: mathlib.exact.two_step_information(family, canonical, action) for action in ROOTS}
        sem.extend(depths.values()); src.extend(exact_values.values()); episode_metrics.append({"family": family, "greedy_first_action": greedy, "depth_two_first_action": planned, "horizon_gain_nats": depths[planned] - depths[greedy], "root_information_nats": roots, "two_step_information_nats": depths, "exact_two_step_information_nats": exact_values})
    metrics = {"partition_count": 54, "exact_partition_count": exact_count, "pair_relation_count": pair_count, "pair_relation_accuracy": pair_ok / pair_count, "mean_multiclass_brier": brier_sum / 216, "family_mean_multiclass_brier": {key: sum(values) / len(values) for key, values in family_brier.items()}, "native_partition_exact_count": native_exact, "native_answer_count": native_count, "native_truth_top_rank_count": native_top, "native_mean_truth_posterior": native_mass / native_count, "native_mean_posterior_brier": native_brier / native_count, "equivalent_pair_count": len(tvs), "equivalent_mean_total_variation": sum(tvs) / len(tvs), "equivalent_max_total_variation": max(tvs), "semantic_source_two_step_spearman": mathlib.spearman(sem, src), "mean_horizon_gain_nats": sum(row["horizon_gain_nats"] for row in episode_metrics) / 6}
    gates = {"exact_54_action_partitions": True, "at_least_52_exact_partitions": exact_count >= 52, "exact_324_pair_relations": pair_count == 324, "pair_relation_accuracy_at_least_0_98": metrics["pair_relation_accuracy"] >= .98, "mean_brier_at_most_0_08": metrics["mean_multiclass_brier"] <= .08, "each_family_brier_at_most_0_10": all(value <= .10 for value in metrics["family_mean_multiclass_brier"].values()), "all_six_native_partitions_exact": native_exact == 6, "exact_24_native_answers": native_count == 24, "all_24_native_truth_top_rank": native_top == 24, "native_truth_mass_at_least_0_65": metrics["native_mean_truth_posterior"] >= .65, "native_brier_at_most_0_18": metrics["native_mean_posterior_brier"] <= .18, "equivalent_mean_tv_at_most_0_03": metrics["equivalent_mean_total_variation"] <= .03, "equivalent_max_tv_at_most_0_10": metrics["equivalent_max_total_variation"] <= .10, "all_six_greedy_avoid_installed_apps": all(row["greedy_first_action"] != "installed_apps" for row in episode_metrics), "all_six_depth_two_choose_installed_apps": all(row["depth_two_first_action"] == "installed_apps" for row in episode_metrics), "all_six_horizon_gain_at_least_0_50": all(row["horizon_gain_nats"] >= .50 for row in episode_metrics), "semantic_source_spearman_at_least_0_90": metrics["semantic_source_two_step_spearman"] >= .90}
    gates["all_calibration_gates_pass"] = all(gates.values()); return {"metrics": metrics, "episode_metrics": episode_metrics, "calibration_gates": gates}


def validate_partial(partial, require_complete=True):
    if partial.get("interface_version") != "tau2-mms-partition-semantic-calibration-1" or partial.get("model_id") != MODEL_ID or tuple(partial.get("expected_seeds", ())) != MODEL_SEEDS:
        raise ValueError("independent partition partial identity changed")
    rows = partial.get("completed"); indexes = [row.get("index") for row in rows] if isinstance(rows, list) else []
    if indexes != sorted(indexes) or len(set(indexes)) != len(indexes) or any(index not in range(6) for index in indexes):
        raise ValueError("independent partition partial rows changed")
    if any(set(row) != {"index", "seed", "response"} or row["seed"] != MODEL_SEEDS[row["index"]] or not isinstance(row["response"], str) for row in rows):
        raise ValueError("independent partition partial item changed")
    complete = len(rows) == 6
    if partial.get("complete") is not complete or (require_complete and not complete):
        raise ValueError("independent partition partial completeness changed")
    return [row["response"] for row in rows]


def replay(raw):
    if raw.get("model_id") != MODEL_ID or tuple(raw.get("seeds", ())) != MODEL_SEEDS or len(raw.get("responses", ())) != 6:
        raise ValueError("independent partition raw bank changed")
    episodes = selected_episodes(); return score(episodes, [parse_raw(text) for text in raw["responses"]])


def verify(run_dir: Path, *, output_path: Path | None = None):
    result = load_object(run_dir / "RESULT.json"); raw = load_object(run_dir / "private/RAW_RESPONSES.json"); partial = load_object(run_dir / "private/PARTIAL_RAW_RESPONSES.json"); ordering = load_object(run_dir / "private/ORDERING.json"); privacy = load_object(run_dir / "private/PROMPT_PRIVACY.json")
    if raw.get("responses") != validate_partial(partial): raise ValueError("independent partition banks differ")
    replayed = replay(raw)
    for key in ("metrics", "episode_metrics", "calibration_gates"):
        if result.get(key) != replayed[key]: raise ValueError(f"independent partition {key} differs")
    usage = result.get("usage", {}); serving = {"exact_six_accepted_requests": usage.get("adapter_requests") == 6, "exact_six_http_attempts": usage.get("http_attempts") == 6, "zero_retries": usage.get("retry_count") == 0, "zero_provider_error_retries": usage.get("provider_error_retries") == 0, "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0, "zero_forced_exits": usage.get("forced_exits") == 0, "within_stage_cap": float(usage.get("run_cost_usd", math.inf)) <= .06 + 1e-12}
    passed = all(serving.values()) and replayed["calibration_gates"]["all_calibration_gates_pass"]
    expected_ordering = {"all_partial_responses_banked": True, "complete_raw_bank_created": True, "official_calibration_loaded_after_complete_bank": True}
    checks = {"serving_exact": result.get("serving_gates") == serving, "status_exact": result.get("status") == ("mms_partition_semantic_pass" if passed else "mms_partition_semantic_null") and result.get("authorizes") == ("prospective_paired_development_protocol_only" if passed else "nothing"), "ordering_exact": result.get("ordering") == ordering == expected_ordering, "privacy_exact": all(privacy.get(key) is False for key in ("selected_task_ids_in_prompts", "source_fault_ids_in_prompts", "raw_tool_responses_in_prompts", "repair_or_endpoint_outcomes_in_prompts")), "downstream_unopened": result.get("development_confirmation_reserve_opened") is False and result.get("repair_or_task_success_endpoints_opened") is False}
    verification = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "verified" if all(checks.values()) else "invalid", "expected_result_status": "mms_partition_semantic_pass" if passed else "mms_partition_semantic_null", "checks": checks, "all_pass": all(checks.values()), "result_sha256": sha256_file(run_dir / "RESULT.json"), "raw_bank_sha256": sha256_file(run_dir / "private/RAW_RESPONSES.json"), "partial_bank_sha256": sha256_file(run_dir / "private/PARTIAL_RAW_RESPONSES.json"), "model_calls_made": 0, "cost_usd": 0.0}
    if not verification["all_pass"]: raise ValueError("independent partition verification failed")
    if output_path: output_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    return verification
