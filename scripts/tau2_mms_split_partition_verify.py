#!/usr/bin/env python3
"""Independently replay the Tau2 MMS split-partition calibration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tau2_mms_split_partition_source_audit as source_audit
from scripts import tau2_native_prerequisite_semantic_verify as mathlib

MODEL_ID = "deepseek/deepseek-v4-flash-0731"
ROOT_SEEDS = tuple(range(202608130500, 202608130506))
NATIVE_SEEDS = tuple(range(202608130600, 202608130606))
ROOTS = tuple(mathlib.MMS_ROOTS)
NATIVE = "messaging_permissions"


def canonical_json(value): return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
def sha256_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict): raise ValueError(f"expected object: {path}")
    return value


def canonical_groups(values):
    labels = {}; result = []
    for value in values:
        key = value if isinstance(value, str) else canonical_json(value); labels.setdefault(key, len(labels)); result.append(labels[key])
    return result


def confidence(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not .5 <= float(value) <= .95: raise ValueError("independent split confidence changed")
    return float(value)


def parse_root(raw):
    payload = json.loads(raw); rows = payload.get("actions") if isinstance(payload, dict) and set(payload) == {"actions"} else None
    if not isinstance(rows, list) or len(rows) != 8: raise ValueError("independent split root array changed")
    result = {}
    for action, row in zip(ROOTS, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"action_id", "all_worlds_same", "confidence"} or row["action_id"] != action or not isinstance(row["all_worlds_same"], bool): raise ValueError("independent split root row changed")
        result[action] = {"all_worlds_same": row["all_worlds_same"], "confidence": confidence(row["confidence"])}
    return result


def parse_native(raw):
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"action_id", "world_groups", "confidence"} or payload["action_id"] != NATIVE: raise ValueError("independent split native root changed")
    groups = payload["world_groups"]
    if not isinstance(groups, list) or len(groups) != 4 or any(isinstance(x, bool) or not isinstance(x, int) or x not in range(4) for x in groups) or canonical_groups(groups) != groups: raise ValueError("independent split native groups changed")
    return {"groups": groups, "confidence": confidence(payload["confidence"])}


def table(groups, conf):
    categories = [str(x) for x in sorted(set(groups))] + ["OTHER"]; rest = (1 - conf) / (len(categories) - 1)
    result = {f"w{i}": {category: conf if category == str(group) else rest for category in categories} for i, group in enumerate(groups)}
    if any(not math.isclose(sum(row.values()), 1.0, abs_tol=1e-12) for row in result.values()): raise ValueError("independent split likelihood changed")
    return result


def score(episodes, roots, natives):
    task_bank, get_environment = mathlib.exact.initialize_tau2(); observations = [mathlib.official_episode(row, task_bank, get_environment) for row in episodes]
    root_ok = root_n = native_exact = pair_ok = pair_n = native_top = native_n = 0; root_brier = native_part_brier = native_mass = native_post_brier = 0.0; tvs = []; sem = []; src = []; episode_metrics = []
    for episode, root, native, truth in zip(episodes, roots, natives, observations, strict=True):
        family = episode["family"]; prior = [.25] * 4; all_tables = {}
        for action in ROOTS:
            expected = len({canonical_json(world[action]) for world in truth}) == 1; predicted = root[action]["all_worlds_same"]; conf = root[action]["confidence"]
            root_ok += predicted == expected; root_n += 1; p_true = conf if predicted else 1 - conf; root_brier += 2 * (p_true - float(expected)) ** 2
            groups = [0, 0, 0, 0] if predicted else [0, 1, 2, 3]; all_tables[action] = table(groups, conf)
            if expected:
                for left in range(4):
                    for right in range(left + 1, 4): tvs.append(.5 * sum(abs(all_tables[action][f"w{left}"][key] - all_tables[action][f"w{right}"][key]) for key in all_tables[action][f"w{left}"]))
        expected_native = canonical_groups([world[NATIVE] for world in truth]); native_exact += native["groups"] == expected_native; all_tables[NATIVE] = table(native["groups"], native["confidence"]); nt = all_tables[NATIVE]
        for left in range(4):
            for right in range(left + 1, 4): pair_ok += (native["groups"][left] == native["groups"][right]) == (expected_native[left] == expected_native[right]); pair_n += 1
        for world_index, outcome in enumerate(expected_native):
            category = str(outcome) if str(outcome) in nt[f"w{world_index}"] else "OTHER"; native_part_brier += sum((prob - float(key == category)) ** 2 for key, prob in nt[f"w{world_index}"].items()); post = mathlib.posterior(prior, nt, category); native_mass += post[world_index]; native_top += post[world_index] >= max(post) - 1e-12; native_post_brier += sum((prob - float(i == world_index)) ** 2 for i, prob in enumerate(post)); native_n += 1
        root_values = {action: mathlib.information(prior, all_tables[action]) for action in ROOTS}; depth_values = {action: mathlib.depth_two(family, all_tables, action) for action in ROOTS}; greedy = max(ROOTS, key=lambda action: (root_values[action], action)); planned = max(ROOTS, key=lambda action: (depth_values[action], action)); canonical = [{action: canonical_json(value) for action, value in world.items()} for world in truth]; exact_values = {action: mathlib.exact.two_step_information(family, canonical, action) for action in ROOTS}; sem.extend(depth_values.values()); src.extend(exact_values.values()); episode_metrics.append({"family": family, "greedy_first_action": greedy, "depth_two_first_action": planned, "horizon_gain_nats": depth_values[planned] - depth_values[greedy], "root_information_nats": root_values, "two_step_information_nats": depth_values, "exact_two_step_information_nats": exact_values})
    metrics = {"root_decision_count": root_n, "root_exact_count": root_ok, "root_mean_brier": root_brier / root_n, "native_partition_count": 6, "native_partition_exact_count": native_exact, "native_pair_relation_count": pair_n, "native_pair_relation_accuracy": pair_ok / pair_n, "native_mean_partition_brier": native_part_brier / 24, "native_answer_count": native_n, "native_truth_top_rank_count": native_top, "native_mean_truth_posterior": native_mass / native_n, "native_mean_posterior_brier": native_post_brier / native_n, "equivalent_pair_count": len(tvs), "equivalent_mean_total_variation": sum(tvs) / len(tvs), "equivalent_max_total_variation": max(tvs), "semantic_source_two_step_spearman": mathlib.spearman(sem, src), "mean_horizon_gain_nats": sum(row["horizon_gain_nats"] for row in episode_metrics) / 6}
    gates = {"exact_48_root_decisions": root_n == 48, "at_least_47_exact_root_decisions": root_ok >= 47, "root_brier_at_most_0_08": metrics["root_mean_brier"] <= .08, "all_six_native_partitions_exact": native_exact == 6, "exact_36_native_pair_relations": pair_n == 36, "native_pair_accuracy_one": metrics["native_pair_relation_accuracy"] == 1.0, "native_partition_brier_at_most_0_08": metrics["native_mean_partition_brier"] <= .08, "exact_24_native_answers": native_n == 24, "all_24_native_truth_top_rank": native_top == 24, "native_truth_mass_at_least_0_65": metrics["native_mean_truth_posterior"] >= .65, "native_posterior_brier_at_most_0_18": metrics["native_mean_posterior_brier"] <= .18, "equivalent_mean_tv_at_most_0_03": metrics["equivalent_mean_total_variation"] <= .03, "equivalent_max_tv_at_most_0_10": metrics["equivalent_max_total_variation"] <= .10, "all_six_greedy_avoid_installed_apps": all(row["greedy_first_action"] != "installed_apps" for row in episode_metrics), "all_six_depth_two_choose_installed_apps": all(row["depth_two_first_action"] == "installed_apps" for row in episode_metrics), "all_six_horizon_gain_at_least_0_50": all(row["horizon_gain_nats"] >= .5 for row in episode_metrics), "semantic_source_spearman_at_least_0_90": metrics["semantic_source_two_step_spearman"] >= .9}; gates["all_calibration_gates_pass"] = all(gates.values()); return {"metrics": metrics, "episode_metrics": episode_metrics, "calibration_gates": gates}


def expected_requests(): return [{"episode_index": i, "kind": "root", "seed": ROOT_SEEDS[i]} for i in range(6)] + [{"episode_index": i, "kind": "native", "seed": NATIVE_SEEDS[i]} for i in range(6)]
def validate_partial(partial):
    if partial.get("interface_version") != "tau2-mms-split-partition-calibration-1" or partial.get("model_id") != MODEL_ID or partial.get("requests") != [{"index": i, **row} for i, row in enumerate(expected_requests())]: raise ValueError("independent split partial identity changed")
    rows = partial.get("completed"); indexes = [row.get("index") for row in rows] if isinstance(rows, list) else []
    if indexes != list(range(12)) or partial.get("complete") is not True: raise ValueError("independent split partial incomplete")
    for i, row in enumerate(rows):
        if set(row) != {"index", "episode_index", "kind", "seed", "response"} or {key: row[key] for key in ("episode_index", "kind", "seed")} != expected_requests()[i] or not isinstance(row["response"], str): raise ValueError("independent split partial row changed")
    return [row["response"] for row in rows]


def replay(raw):
    if raw.get("model_id") != MODEL_ID or raw.get("requests") != expected_requests() or len(raw.get("responses", ())) != 12: raise ValueError("independent split raw bank changed")
    episodes = source_audit.load_episodes(); responses = raw["responses"]; return score(episodes, [parse_root(responses[i]) for i in range(6)], [parse_native(responses[6+i]) for i in range(6)])


def verify(run_dir: Path, *, output_path: Path | None = None):
    result = load_object(run_dir / "RESULT.json"); raw = load_object(run_dir / "private/RAW_RESPONSES.json"); partial = load_object(run_dir / "private/PARTIAL_RAW_RESPONSES.json"); ordering = load_object(run_dir / "private/ORDERING.json"); privacy = load_object(run_dir / "private/PROMPT_PRIVACY.json")
    if raw["responses"] != validate_partial(partial): raise ValueError("independent split banks differ")
    replayed = replay(raw)
    for key in ("metrics", "episode_metrics", "calibration_gates"):
        if result.get(key) != replayed[key]: raise ValueError(f"independent split {key} differs")
    use = result.get("usage", {}); serving = {"exact_twelve_accepted_requests": use.get("adapter_requests") == 12, "exact_twelve_http_attempts": use.get("http_attempts") == 12, "zero_retries": use.get("retry_count") == 0, "zero_provider_error_retries": use.get("provider_error_retries") == 0, "zero_reasoning_tokens": use.get("adapter_reasoning_tokens") == 0, "zero_forced_exits": use.get("forced_exits") == 0, "within_stage_cap": float(use.get("run_cost_usd", math.inf)) <= .06 + 1e-12}; passed = all(serving.values()) and replayed["calibration_gates"]["all_calibration_gates_pass"]; expected_order = {"all_partial_responses_banked": True, "complete_raw_bank_created": True, "official_calibration_loaded_after_complete_bank": True}; checks = {"serving_exact": result.get("serving_gates") == serving, "status_exact": result.get("status") == ("mms_split_partition_pass" if passed else "mms_split_partition_null") and result.get("authorizes") == ("prospective_paired_development_protocol_only" if passed else "nothing"), "ordering_exact": result.get("ordering") == ordering == expected_order, "privacy_exact": all(privacy.get(key) is False for key in ("selected_task_ids_in_prompts", "source_fault_ids_in_prompts", "raw_tool_responses_in_prompts", "repair_or_endpoint_outcomes_in_prompts")), "downstream_unopened": result.get("development_confirmation_reserve_opened") is False and result.get("repair_or_task_success_endpoints_opened") is False}; verification = {"schema_version": 1, "interface_version": "tau2-mms-split-partition-verification-1", "status": "verified" if all(checks.values()) else "invalid", "expected_result_status": "mms_split_partition_pass" if passed else "mms_split_partition_null", "checks": checks, "all_pass": all(checks.values()), "result_sha256": sha256_file(run_dir / "RESULT.json"), "raw_bank_sha256": sha256_file(run_dir / "private/RAW_RESPONSES.json"), "partial_bank_sha256": sha256_file(run_dir / "private/PARTIAL_RAW_RESPONSES.json"), "model_calls_made": 0, "cost_usd": 0.0}
    if not verification["all_pass"]: raise ValueError("independent split verification failed")
    if output_path: output_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    return verification
