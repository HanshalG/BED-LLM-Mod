#!/usr/bin/env python3
"""Run the fresh Tau2 MMS split root/native partition calibration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import tau2_mms_split_partition_source_audit as source_audit
from scripts import tau2_mms_partition_semantic_calibration as partition
from scripts import tau2_native_prerequisite_semantic_verify as mathlib
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage

SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-mms-split-partition-calibration-1"
MODEL_ID = partition.MODEL_ID
ROOT_SEEDS = tuple(range(202608130500, 202608130506))
NATIVE_SEEDS = tuple(range(202608130600, 202608130606))
EXPECTED_REQUESTS = 12
CONCURRENCY = 2
ROOT_MAX_TOKENS = 900
NATIVE_MAX_TOKENS = 300
RUN_CAP_USD = 0.06
PROJECTED_COST_USD = 0.04
MAX_REQUEST_COST_USD = 0.004
PROTOCOL = REPO_ROOT / "results/nonmyopic/TAU2_MMS_SPLIT_PARTITION_CALIBRATION_PROTOCOL_20260813.md"
MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_split_partition_calibration/CALIBRATION_MANIFEST.json"
ROOTS = tuple(mathlib.MMS_ROOTS)
NATIVE = "messaging_permissions"

canonical_json = partition.canonical_json
sha256_file = partition.sha256_file
episode_hash = partition.episode_hash
canonical_groups = partition.canonical_groups


def selected_episodes() -> list[dict[str, Any]]:
    episodes = source_audit.load_episodes()
    if len(episodes) != 6:
        raise ValueError("split-partition episode count changed")
    return episodes


def public_episode(episode: Mapping[str, Any], index: int) -> dict[str, Any]:
    public = partition.absolute.base.public_episode(episode, index)
    return {
        "episode_index": index,
        "family": episode["family"],
        "worlds": public["worlds"],
        "root_actions": list(ROOTS),
        "native_action": NATIVE,
        "legal_unlock": public["legal_unlock"],
    }


def root_response_format() -> dict[str, Any]:
    item = {
        "type": "object", "additionalProperties": False,
        "required": ["action_id", "all_worlds_same", "confidence"],
        "properties": {
            "action_id": {"type": "string"},
            "all_worlds_same": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.5, "maximum": 0.95},
        },
    }
    return {"type": "json_schema", "json_schema": {"name": "tau2_mms_root_equivalence", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["actions"], "properties": {"actions": {"type": "array", "minItems": 8, "maxItems": 8, "items": item}}}}}


def native_response_format() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "tau2_mms_native_partition", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["action_id", "world_groups", "confidence"], "properties": {"action_id": {"type": "string"}, "world_groups": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer", "minimum": 0, "maximum": 3}}, "confidence": {"type": "number", "minimum": 0.5, "maximum": 0.95}}}}}


def root_messages(public: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You predict observational equivalence for telecom read-only diagnostics. Return only strict JSON without reasoning."},
        {"role": "user", "content": "For each supplied root action in exact order, decide whether all four candidate worlds would produce exactly the same visible observation. Predict observations, not repair success. Preserve IDs. Confidence is probability the boolean is correct. Do not include the unlocked native action.\n" + canonical_json(public)},
    ]


def native_messages(public: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You predict the visible messaging-permission observation across candidate telecom worlds. Return only strict JSON without reasoning."},
        {"role": "user", "content": "The installed-apps read has already unlocked messaging_permissions. Partition the four candidate worlds by whether messaging_permissions would return exactly the same visible permission observation. Use canonical first-occurrence labels: world 0 is group 0; later worlds reuse a label exactly for identical observations, otherwise use the next integer. Return only the supplied native action ID, four labels, and confidence.\n" + canonical_json(public)},
    ]


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.5 <= float(value) <= 0.95:
        raise ValueError("split-partition confidence changed")
    return float(value)


def parse_root(raw: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(raw); rows = payload.get("actions") if isinstance(payload, dict) and set(payload) == {"actions"} else None
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("split root action array changed")
    parsed = {}
    for expected, row in zip(ROOTS, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"action_id", "all_worlds_same", "confidence"} or row["action_id"] != expected or not isinstance(row["all_worlds_same"], bool):
            raise ValueError("split root row changed")
        parsed[expected] = {"all_worlds_same": row["all_worlds_same"], "confidence": _confidence(row["confidence"])}
    return parsed


def parse_native(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"action_id", "world_groups", "confidence"} or payload["action_id"] != NATIVE:
        raise ValueError("split native root changed")
    groups = payload["world_groups"]
    if not isinstance(groups, list) or len(groups) != 4 or any(isinstance(x, bool) or not isinstance(x, int) or x not in range(4) for x in groups) or canonical_groups(groups) != groups:
        raise ValueError("split native groups changed")
    return {"groups": groups, "confidence": _confidence(payload["confidence"])}


def partition_table(groups: Sequence[int], confidence: float) -> dict[str, dict[str, float]]:
    categories = [str(value) for value in sorted(set(groups))] + ["OTHER"]
    rest = (1.0 - confidence) / (len(categories) - 1)
    return {f"w{index}": {category: confidence if category == str(group) else rest for category in categories} for index, group in enumerate(groups)}


def tables(root: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for action, row in root.items():
        groups = [0, 0, 0, 0] if row["all_worlds_same"] else [0, 1, 2, 3]
        result[action] = partition_table(groups, row["confidence"])
    result[NATIVE] = partition_table(native["groups"], native["confidence"])
    return result


def score(episodes, roots, natives, observations):
    root_correct = root_count = native_exact = pair_correct = pair_count = native_top = native_count = 0
    root_brier = native_partition_brier = native_mass = native_posterior_brier = 0.0
    tvs = []; semantic = []; source_values = []; episode_metrics = []
    for episode, root, native, truth in zip(episodes, roots, natives, observations, strict=True):
        family = episode["family"]; all_tables = tables(root, native); prior = [0.25] * 4
        for action in ROOTS:
            expected = len({canonical_json(world[action]) for world in truth}) == 1; predicted = root[action]["all_worlds_same"]; confidence = root[action]["confidence"]
            root_correct += predicted == expected; root_count += 1
            probability_true = confidence if predicted else 1.0 - confidence
            root_brier += 2.0 * ((probability_true - float(expected)) ** 2)
            if expected:
                table = all_tables[action]
                for left in range(4):
                    for right in range(left + 1, 4):
                        tvs.append(0.5 * sum(abs(table[f"w{left}"][key] - table[f"w{right}"][key]) for key in table[f"w{left}"]))
        expected_native = canonical_groups([world[NATIVE] for world in truth]); native_exact += native["groups"] == expected_native
        for left in range(4):
            for right in range(left + 1, 4):
                pair_correct += (native["groups"][left] == native["groups"][right]) == (expected_native[left] == expected_native[right]); pair_count += 1
        native_table = all_tables[NATIVE]
        for world_index, outcome in enumerate(expected_native):
            category = str(outcome) if str(outcome) in native_table[f"w{world_index}"] else "OTHER"
            native_partition_brier += sum((prob - float(key == category)) ** 2 for key, prob in native_table[f"w{world_index}"].items())
            post = mathlib.posterior(prior, native_table, category); native_mass += post[world_index]; native_top += post[world_index] >= max(post) - 1e-12; native_posterior_brier += sum((prob - float(index == world_index)) ** 2 for index, prob in enumerate(post)); native_count += 1
        root_values = {action: mathlib.information(prior, all_tables[action]) for action in ROOTS}; depth_values = {action: mathlib.depth_two(family, all_tables, action) for action in ROOTS}
        greedy = max(ROOTS, key=lambda action: (root_values[action], action)); planned = max(ROOTS, key=lambda action: (depth_values[action], action))
        canonical = [{action: canonical_json(value) for action, value in world.items()} for world in truth]; exact_values = {action: mathlib.exact.two_step_information(family, canonical, action) for action in ROOTS}
        semantic.extend(depth_values.values()); source_values.extend(exact_values.values()); episode_metrics.append({"family": family, "greedy_first_action": greedy, "depth_two_first_action": planned, "horizon_gain_nats": depth_values[planned] - depth_values[greedy], "root_information_nats": root_values, "two_step_information_nats": depth_values, "exact_two_step_information_nats": exact_values})
    metrics = {"root_decision_count": root_count, "root_exact_count": root_correct, "root_mean_brier": root_brier / root_count, "native_partition_count": 6, "native_partition_exact_count": native_exact, "native_pair_relation_count": pair_count, "native_pair_relation_accuracy": pair_correct / pair_count, "native_mean_partition_brier": native_partition_brier / 24, "native_answer_count": native_count, "native_truth_top_rank_count": native_top, "native_mean_truth_posterior": native_mass / native_count, "native_mean_posterior_brier": native_posterior_brier / native_count, "equivalent_pair_count": len(tvs), "equivalent_mean_total_variation": sum(tvs) / len(tvs), "equivalent_max_total_variation": max(tvs), "semantic_source_two_step_spearman": mathlib.spearman(semantic, source_values), "mean_horizon_gain_nats": sum(row["horizon_gain_nats"] for row in episode_metrics) / 6}
    gates = {"exact_48_root_decisions": root_count == 48, "at_least_47_exact_root_decisions": root_correct >= 47, "root_brier_at_most_0_08": metrics["root_mean_brier"] <= .08, "all_six_native_partitions_exact": native_exact == 6, "exact_36_native_pair_relations": pair_count == 36, "native_pair_accuracy_one": metrics["native_pair_relation_accuracy"] == 1.0, "native_partition_brier_at_most_0_08": metrics["native_mean_partition_brier"] <= .08, "exact_24_native_answers": native_count == 24, "all_24_native_truth_top_rank": native_top == 24, "native_truth_mass_at_least_0_65": metrics["native_mean_truth_posterior"] >= .65, "native_posterior_brier_at_most_0_18": metrics["native_mean_posterior_brier"] <= .18, "equivalent_mean_tv_at_most_0_03": metrics["equivalent_mean_total_variation"] <= .03, "equivalent_max_tv_at_most_0_10": metrics["equivalent_max_total_variation"] <= .10, "all_six_greedy_avoid_installed_apps": all(row["greedy_first_action"] != "installed_apps" for row in episode_metrics), "all_six_depth_two_choose_installed_apps": all(row["depth_two_first_action"] == "installed_apps" for row in episode_metrics), "all_six_horizon_gain_at_least_0_50": all(row["horizon_gain_nats"] >= .50 for row in episode_metrics), "semantic_source_spearman_at_least_0_90": metrics["semantic_source_two_step_spearman"] >= .90}
    gates["all_calibration_gates_pass"] = all(gates.values()); return {"metrics": metrics, "episode_metrics": episode_metrics, "calibration_gates": gates}


class SplitAdapter(partition.absolute.base.NonReasoningSeededAdapter):
    def __init__(self, *args, partial_path: Path, **kwargs):
        super().__init__(*args, **kwargs); self.partial_path = partial_path
    def run_requests(self, requests: Sequence[Mapping[str, Any]]) -> list[str]:
        def request(index):
            row = requests[index]; self._per_request_seed.value = int(row["seed"])
            try: return self._complete_request(row["messages"], 0.0, 1, row["max_tokens"], response_format=row["response_format"])[0]
            finally: del self._per_request_seed.value
        completed = {}; first_error = None
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(request, index): index for index in range(len(requests))}
            for future in as_completed(futures):
                index = futures[future]
                try: completed[index] = future.result()
                except BaseException as exc:
                    if first_error is None: first_error = exc
                partition.checkpointed.checkpoint_partial(self.partial_path, {"interface_version": INTERFACE_VERSION, "model_id": MODEL_ID, "requests": [{"index": i, "episode_index": row["episode_index"], "kind": row["kind"], "seed": row["seed"]} for i, row in enumerate(requests)], "completed": [{"index": i, "episode_index": requests[i]["episode_index"], "kind": requests[i]["kind"], "seed": requests[i]["seed"], "response": completed[i]} for i in sorted(completed)], "complete": len(completed) == len(requests) and first_error is None})
        if first_error is not None: raise first_error
        if len(completed) != len(requests): raise RuntimeError("split-partition response set incomplete")
        return [completed[i] for i in range(len(requests))]


def build_adapter(*, run_id: str, output_dir: Path):
    config = Config(task="animals", run_id=run_id, log_path=output_dir / "run.log", openrouter_budget_usd=245.0, openrouter_run_budget_usd=RUN_CAP_USD, openrouter_projected_cost_usd=PROJECTED_COST_USD, openrouter_concurrency=CONCURRENCY, openrouter_max_retries=0, openrouter_request_timeout_seconds=300.0, openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD, openrouter_max_output_tokens=ROOT_MAX_TOKENS, openrouter_spend_path="results/path_e/openrouter_spend.json")
    return SplitAdapter(ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65_536), config, partial_path=output_dir / "private/PARTIAL_RAW_RESPONSES.json")


def run(*, output_dir: Path, adapter: Any, daily_budget_status: Mapping[str, Any] | None = None):
    episodes = selected_episodes(); public = [public_episode(row, i) for i, row in enumerate(episodes)]; requests = []
    for i, row in enumerate(public): requests.append({"episode_index": i, "kind": "root", "seed": ROOT_SEEDS[i], "messages": root_messages(row), "response_format": root_response_format(), "max_tokens": ROOT_MAX_TOKENS})
    for i, row in enumerate(public): requests.append({"episode_index": i, "kind": "native", "seed": NATIVE_SEEDS[i], "messages": native_messages(row), "response_format": native_response_format(), "max_tokens": NATIVE_MAX_TOKENS})
    privacy = {"prompt_sha256": [hashlib.sha256(canonical_json(row["messages"]).encode()).hexdigest() for row in requests], "selected_task_ids_in_prompts": False, "source_fault_ids_in_prompts": False, "raw_tool_responses_in_prompts": False, "repair_or_endpoint_outcomes_in_prompts": False}
    output_dir.mkdir(parents=True, exist_ok=True); private = output_dir / "private"; private.mkdir(parents=True, exist_ok=True); checkpoint(private / "PROMPT_PRIVACY.json", privacy)
    raw = adapter.run_requests(requests); checkpoint(private / "RAW_RESPONSES.json", {"model_id": MODEL_ID, "requests": [{"episode_index": row["episode_index"], "kind": row["kind"], "seed": row["seed"]} for row in requests], "responses": raw})
    ordering = {"all_partial_responses_banked": True, "complete_raw_bank_created": True, "official_calibration_loaded_after_complete_bank": False}; checkpoint(private / "ORDERING.json", ordering)
    roots = [parse_root(raw[i]) for i in range(6)]; natives = [parse_native(raw[6 + i]) for i in range(6)]; observations = partition.absolute.base.official_observations(episodes); ordering["official_calibration_loaded_after_complete_bank"] = True; checkpoint(private / "ORDERING.json", ordering)
    scores = score(episodes, roots, natives, observations); use = summarize_usage(adapter.usage_snapshot()); serving = {"exact_twelve_accepted_requests": use["adapter_requests"] == 12, "exact_twelve_http_attempts": use["http_attempts"] == 12, "zero_retries": use["retry_count"] == 0, "zero_provider_error_retries": use["provider_error_retries"] == 0, "zero_reasoning_tokens": use["adapter_reasoning_tokens"] == 0, "zero_forced_exits": use["forced_exits"] == 0, "within_stage_cap": use["run_cost_usd"] <= RUN_CAP_USD + 1e-12}; passed = scores["calibration_gates"]["all_calibration_gates_pass"] and all(serving.values())
    result = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "mms_split_partition_pass" if passed else "mms_split_partition_null", "authorizes": "prospective_paired_development_protocol_only" if passed else "nothing", "protocol_sha256": sha256_file(PROTOCOL), "manifest_sha256": sha256_file(MANIFEST), "model": MODEL_ID, "root_seeds": list(ROOT_SEEDS), "native_seeds": list(NATIVE_SEEDS), "privacy": privacy, "ordering": ordering, "usage": use, "serving_gates": serving, **scores, "daily_budget_status": dict(daily_budget_status or {}), "development_confirmation_reserve_opened": False, "repair_or_task_success_endpoints_opened": False}; checkpoint(output_dir / "RESULT.json", result); return result
