#!/usr/bin/env python3
"""Run the fresh Tau2 MMS observational-partition semantic calibration."""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import tau2_mms_checkpointed_semantic_calibration as checkpointed
from scripts import tau2_mms_array_semantic_calibration as absolute
from scripts import tau2_native_prerequisite_semantic_verify as mathlib
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage

SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-mms-partition-semantic-calibration-1"
MODEL_ID = absolute.MODEL_ID
MODEL_SEEDS = tuple(range(202608130400, 202608130406))
MAX_TOKENS = 2_500
EXPECTED_REQUESTS = 6
CONCURRENCY = 2
RUN_CAP_USD = 0.06
PROJECTED_COST_USD = 0.036
MAX_REQUEST_COST_USD = 0.006
PROTOCOL = REPO_ROOT / "results/nonmyopic/TAU2_MMS_PARTITION_SEMANTIC_CALIBRATION_PROTOCOL_20260813.md"
MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_partition_semantic_calibration/CALIBRATION_MANIFEST.json"

canonical_json = absolute.canonical_json
sha256_file = absolute.sha256_file
episode_hash = absolute.episode_hash


def selected_episodes() -> list[dict[str, Any]]:
    tasks = json.loads(absolute.source.TASKS_PATH.read_text())
    splits = json.loads(absolute.source.SPLITS_PATH.read_text())
    selected, _ = absolute.source.select_episodes([str(row["id"]) for row in tasks], list(splits["base"]))
    by_hash = {episode_hash(row): row for row in selected["reserve"]}
    rows = json.loads(MANIFEST.read_text()).get("episodes", [])
    if len(rows) != 6 or len({row.get("episode_sha256") for row in rows}) != 6:
        raise ValueError("partition MMS manifest changed")
    episodes = []
    for row in rows:
        episode = by_hash.get(row.get("episode_sha256"))
        if episode is None or row.get("family") != episode["family"] or len(episode["worlds"]) != 4:
            raise ValueError("partition MMS episode binding changed")
        episodes.append(episode)
    prior_hashes = {episode_hash(row) for row in absolute.selected_episodes()} | {episode_hash(row) for row in checkpointed.selected_episodes()}
    if prior_hashes & {episode_hash(row) for row in episodes}:
        raise ValueError("partition MMS cohort overlaps predecessor")
    return episodes


def public_episode(episode: Mapping[str, Any], index: int) -> dict[str, Any]:
    public = absolute.base.public_episode(episode, index)
    return {
        "episode_index": index, "family": episode["family"], "worlds": public["worlds"],
        "actions": list(absolute.base.MMS_ACTION_FIELDS), "legal_unlock": public["legal_unlock"],
    }


def response_format() -> dict[str, Any]:
    action = {"type": "object", "additionalProperties": False, "required": ["action_id", "world_groups", "confidence"], "properties": {"action_id": {"type": "string"}, "world_groups": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer", "minimum": 0, "maximum": 3}}, "confidence": {"type": "number", "minimum": 0.5, "maximum": 0.95}}}
    return {"type": "json_schema", "json_schema": {"name": "tau2_mms_observation_partitions", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["actions"], "properties": {"actions": {"type": "array", "minItems": 9, "maxItems": 9, "items": action}}}}}


def messages(public: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a semantic simulator for telecom diagnostic observations. Return only strict JSON, without reasoning or literal tool outputs."},
        {"role": "user", "content": "For each action in supplied order, partition the four candidate worlds by whether that action would return exactly the same visible observation. Encode canonical group labels by first occurrence: world 0 is group 0; each later world reuses an existing group exactly for an identical observation, otherwise uses the next integer. Predict observation equivalence, not whether the action repairs the issue. The installed-apps read unlocks messaging-permissions. Preserve action order and IDs. Confidence is probability the entire four-world partition is correct.\n" + canonical_json(public)},
    ]


def canonical_groups(values: Sequence[Any]) -> list[int]:
    labels: dict[str, int] = {}; result = []
    for value in values:
        key = canonical_json(value) if not isinstance(value, str) else value
        labels.setdefault(key, len(labels)); result.append(labels[key])
    return result


def parse(raw: str, public: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(raw)
    rows = payload.get("actions") if isinstance(payload, dict) and set(payload) == {"actions"} else None
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("partition MMS action array changed")
    parsed = {}
    for expected, row in zip(public["actions"], rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"action_id", "world_groups", "confidence"} or row["action_id"] != expected:
            raise ValueError("partition MMS action order changed")
        groups = row["world_groups"]
        if not isinstance(groups, list) or len(groups) != 4 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x > 3 for x in groups) or canonical_groups(groups) != groups:
            raise ValueError("partition MMS groups are not canonical")
        confidence = row["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0.5 <= float(confidence) <= 0.95:
            raise ValueError("partition MMS confidence changed")
        parsed[expected] = {"groups": groups, "confidence": float(confidence)}
    return parsed


def likelihood_tables(parsed: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    tables = {}
    for action, row in parsed.items():
        categories = [str(value) for value in sorted(set(row["groups"]))] + ["OTHER"]
        rest = (1.0 - row["confidence"]) / (len(categories) - 1)
        tables[action] = {f"w{index}": {category: row["confidence"] if category == str(group) else rest for category in categories} for index, group in enumerate(row["groups"])}
    return tables


def score(episodes: Sequence[Mapping[str, Any]], parsed: Sequence[Mapping[str, Any]], observations: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    partition_exact = pair_correct = pair_count = native_exact = native_top = native_count = 0
    brier_total = native_mass = native_brier = 0.0
    family_brier: dict[str, list[float]] = {"mms_abroad": [], "mms_home": []}
    tvs = []; semantic_values = []; source_values = []; episode_metrics = []
    for episode, prediction, truth in zip(episodes, parsed, observations, strict=True):
        family = episode["family"]; tables = likelihood_tables(prediction); prior = [0.25] * 4
        truth_groups = {action: canonical_groups([world[action] for world in truth]) for action in absolute.base.MMS_ACTION_FIELDS}
        for action, expected in truth_groups.items():
            groups = prediction[action]["groups"]
            partition_exact += groups == expected
            for left in range(4):
                for right in range(left + 1, 4):
                    pair_correct += (groups[left] == groups[right]) == (expected[left] == expected[right]); pair_count += 1
            table = tables[action]
            for world_index, outcome in enumerate(expected):
                category = str(outcome) if str(outcome) in table[f"w{world_index}"] else "OTHER"
                brier = sum((prob - float(key == category)) ** 2 for key, prob in table[f"w{world_index}"].items())
                brier_total += brier; family_brier[family].append(brier)
            for left in range(4):
                for right in range(left + 1, 4):
                    if expected[left] == expected[right]:
                        tvs.append(0.5 * sum(abs(table[f"w{left}"][key] - table[f"w{right}"][key]) for key in table[f"w{left}"]))
        native = "messaging_permissions"; expected = truth_groups[native]; native_exact += prediction[native]["groups"] == expected
        for world_index, outcome in enumerate(expected):
            table = tables[native]; category = str(outcome) if str(outcome) in table[f"w{world_index}"] else "OTHER"
            post = mathlib.posterior(prior, table, category); native_mass += post[world_index]; native_top += post[world_index] >= max(post) - 1e-12; native_brier += sum((prob - float(i == world_index)) ** 2 for i, prob in enumerate(post)); native_count += 1
        root_values = {action: mathlib.information(prior, tables[action]) for action in mathlib.MMS_ROOTS}
        depth_values = {action: mathlib.depth_two(family, tables, action) for action in mathlib.MMS_ROOTS}
        greedy = max(mathlib.MMS_ROOTS, key=lambda action: (root_values[action], action)); planned = max(mathlib.MMS_ROOTS, key=lambda action: (depth_values[action], action))
        canonical = [{action: canonical_json(value) for action, value in world.items()} for world in truth]
        exact_values = {action: mathlib.exact.two_step_information(family, canonical, action) for action in mathlib.MMS_ROOTS}
        semantic_values.extend(depth_values.values()); source_values.extend(exact_values.values())
        episode_metrics.append({"family": family, "greedy_first_action": greedy, "depth_two_first_action": planned, "horizon_gain_nats": depth_values[planned] - depth_values[greedy], "root_information_nats": root_values, "two_step_information_nats": depth_values, "exact_two_step_information_nats": exact_values})
    metrics = {"partition_count": 54, "exact_partition_count": partition_exact, "pair_relation_count": pair_count, "pair_relation_accuracy": pair_correct / pair_count, "mean_multiclass_brier": brier_total / 216, "family_mean_multiclass_brier": {key: sum(values) / len(values) for key, values in family_brier.items()}, "native_partition_exact_count": native_exact, "native_answer_count": native_count, "native_truth_top_rank_count": native_top, "native_mean_truth_posterior": native_mass / native_count, "native_mean_posterior_brier": native_brier / native_count, "equivalent_pair_count": len(tvs), "equivalent_mean_total_variation": sum(tvs) / len(tvs), "equivalent_max_total_variation": max(tvs), "semantic_source_two_step_spearman": mathlib.spearman(semantic_values, source_values), "mean_horizon_gain_nats": sum(row["horizon_gain_nats"] for row in episode_metrics) / 6}
    gates = {"exact_54_action_partitions": metrics["partition_count"] == 54, "at_least_52_exact_partitions": partition_exact >= 52, "exact_324_pair_relations": pair_count == 324, "pair_relation_accuracy_at_least_0_98": metrics["pair_relation_accuracy"] >= 0.98, "mean_brier_at_most_0_08": metrics["mean_multiclass_brier"] <= 0.08, "each_family_brier_at_most_0_10": all(value <= 0.10 for value in metrics["family_mean_multiclass_brier"].values()), "all_six_native_partitions_exact": native_exact == 6, "exact_24_native_answers": native_count == 24, "all_24_native_truth_top_rank": native_top == 24, "native_truth_mass_at_least_0_65": metrics["native_mean_truth_posterior"] >= 0.65, "native_brier_at_most_0_18": metrics["native_mean_posterior_brier"] <= 0.18, "equivalent_mean_tv_at_most_0_03": metrics["equivalent_mean_total_variation"] <= 0.03, "equivalent_max_tv_at_most_0_10": metrics["equivalent_max_total_variation"] <= 0.10, "all_six_greedy_avoid_installed_apps": all(row["greedy_first_action"] != "installed_apps" for row in episode_metrics), "all_six_depth_two_choose_installed_apps": all(row["depth_two_first_action"] == "installed_apps" for row in episode_metrics), "all_six_horizon_gain_at_least_0_50": all(row["horizon_gain_nats"] >= 0.50 for row in episode_metrics), "semantic_source_spearman_at_least_0_90": metrics["semantic_source_two_step_spearman"] >= 0.90}
    gates["all_calibration_gates_pass"] = all(gates.values())
    return {"metrics": metrics, "episode_metrics": episode_metrics, "calibration_gates": gates}


class CheckpointingAdapter(checkpointed.CheckpointingAdapter):
    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages: Sequence[list[dict[str, str]]], seeds: Sequence[int],
        *, temperature: float, response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if len(batch_messages) != len(seeds):
            raise ValueError("message and seed counts differ")

        def request(index: int) -> str:
            self._per_request_seed.value = int(seeds[index])
            try:
                return self._complete_request(
                    batch_messages[index], temperature, 1, max_new_tokens,
                    response_format=response_format,
                )[0]
            finally:
                del self._per_request_seed.value

        completed: dict[int, str] = {}; first_error: BaseException | None = None
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(request, index): index for index in range(len(seeds))}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    completed[index] = future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                checkpointed.checkpoint_partial(self.partial_path, {
                    "interface_version": INTERFACE_VERSION, "model_id": MODEL_ID,
                    "expected_seeds": list(seeds),
                    "completed": [{"index": item, "seed": int(seeds[item]), "response": completed[item]} for item in sorted(completed)],
                    "complete": len(completed) == len(seeds) and first_error is None,
                })
        if first_error is not None:
            raise first_error
        if len(completed) != len(seeds):
            raise RuntimeError("partition MMS response set incomplete")
        return [completed[index] for index in range(len(seeds))]


def build_adapter(*, run_id: str, output_dir: Path) -> CheckpointingAdapter:
    config = Config(task="animals", run_id=run_id, log_path=output_dir / "run.log", openrouter_budget_usd=245.0, openrouter_run_budget_usd=RUN_CAP_USD, openrouter_projected_cost_usd=PROJECTED_COST_USD, openrouter_concurrency=CONCURRENCY, openrouter_max_retries=0, openrouter_request_timeout_seconds=300.0, openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD, openrouter_max_output_tokens=MAX_TOKENS, openrouter_spend_path="results/path_e/openrouter_spend.json")
    return CheckpointingAdapter(ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65_536), config, partial_path=output_dir / "private/PARTIAL_RAW_RESPONSES.json")


def run(*, output_dir: Path, adapter: Any, daily_budget_status: Mapping[str, Any] | None = None) -> dict[str, Any]:
    episodes = selected_episodes(); public = [public_episode(row, i) for i, row in enumerate(episodes)]; prompts = [messages(row) for row in public]
    privacy = {"prompt_sha256": [hashlib.sha256(canonical_json(row).encode()).hexdigest() for row in prompts], "selected_task_ids_in_prompts": False, "source_fault_ids_in_prompts": False, "raw_tool_responses_in_prompts": False, "repair_or_endpoint_outcomes_in_prompts": False}
    output_dir.mkdir(parents=True, exist_ok=True); private = output_dir / "private"; private.mkdir(parents=True, exist_ok=True); checkpoint(private / "PROMPT_PRIVACY.json", privacy)
    raw = adapter.chat_complete_seeded_messages_batched_structured(prompts, MODEL_SEEDS, temperature=0.0, response_format=response_format(), max_new_tokens=MAX_TOKENS)
    checkpoint(private / "RAW_RESPONSES.json", {"model_id": MODEL_ID, "seeds": list(MODEL_SEEDS), "responses": list(raw)})
    ordering = {"all_partial_responses_banked": True, "complete_raw_bank_created": True, "official_calibration_loaded_after_complete_bank": False}; checkpoint(private / "ORDERING.json", ordering)
    parsed = [parse(text, row) for text, row in zip(raw, public, strict=True)]
    observed = absolute.base.official_observations(episodes); ordering["official_calibration_loaded_after_complete_bank"] = True; checkpoint(private / "ORDERING.json", ordering)
    scores = score(episodes, parsed, observed); usage_value = summarize_usage(adapter.usage_snapshot()); serving = {"exact_six_accepted_requests": usage_value["adapter_requests"] == 6, "exact_six_http_attempts": usage_value["http_attempts"] == 6, "zero_retries": usage_value["retry_count"] == 0, "zero_provider_error_retries": usage_value["provider_error_retries"] == 0, "zero_reasoning_tokens": usage_value["adapter_reasoning_tokens"] == 0, "zero_forced_exits": usage_value["forced_exits"] == 0, "within_stage_cap": usage_value["run_cost_usd"] <= RUN_CAP_USD + 1e-12}
    passed = scores["calibration_gates"]["all_calibration_gates_pass"] and all(serving.values())
    result = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "mms_partition_semantic_pass" if passed else "mms_partition_semantic_null", "authorizes": "prospective_paired_development_protocol_only" if passed else "nothing", "protocol_sha256": sha256_file(PROTOCOL), "manifest_sha256": sha256_file(MANIFEST), "model": MODEL_ID, "model_seeds": list(MODEL_SEEDS), "privacy": privacy, "ordering": ordering, "usage": usage_value, "serving_gates": serving, **scores, "daily_budget_status": dict(daily_budget_status or {}), "development_confirmation_reserve_opened": False, "repair_or_task_success_endpoints_opened": False}
    checkpoint(output_dir / "RESULT.json", result); return result
