#!/usr/bin/env python3
"""Run the Tau2 MMS documented root/native split calibration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import tau2_mms_documented_split_resilient_source_audit as source_audit
from scripts import tau2_mms_partition_semantic_calibration as partition
from scripts import tau2_native_prerequisite_semantic_verify as mathlib
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage

SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-mms-documented-split-resilient-calibration-1"
MODEL_ID = partition.MODEL_ID
ROOT_SEEDS = tuple(range(202608131500, 202608131506))
NATIVE_SEEDS = tuple(range(202608131600, 202608131606))
EXPECTED_REQUESTS = 12
MAX_HTTP_ATTEMPTS = 13
MAX_PROVIDER_ERROR_RETRIES = 1
CONCURRENCY = 2
ROOT_MAX_TOKENS = 900
NATIVE_MAX_TOKENS = 300
RUN_CAP_USD = 0.06
PROJECTED_COST_USD = 0.04
MAX_REQUEST_COST_USD = 0.004
PROTOCOL = source_audit.PROTOCOL
MANIFEST = source_audit.MANIFEST
ROOTS = tuple(mathlib.MMS_ROOTS)
NATIVE = "messaging_permissions"

canonical_json = partition.canonical_json
sha256_file = partition.sha256_file
canonical_groups = partition.canonical_groups


def selected_episodes() -> list[dict[str, Any]]:
    result = source_audit.audit()
    if result["status"] != "source_pass":
        raise ValueError("documented-split resilient source gate did not pass")
    return source_audit.load_episodes()


def public_episode(episode: Mapping[str, Any], index: int) -> dict[str, Any]:
    public = partition.absolute.base.public_episode(episode, index)
    return {
        "episode_index": index,
        "family": episode["family"],
        "worlds": public["worlds"],
        "legal_unlock": public["legal_unlock"],
    }


def root_response_format() -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action_id", "all_worlds_same", "confidence"],
        "properties": {
            "action_id": {"type": "string"},
            "all_worlds_same": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.5, "maximum": 0.95},
        },
    }
    return {"type": "json_schema", "json_schema": {"name": "tau2_mms_documented_root_equivalence", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["actions"], "properties": {"actions": {"type": "array", "minItems": 8, "maxItems": 8, "items": item}}}}}


def native_response_format() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "tau2_mms_documented_native_partition", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["action_id", "world_groups", "confidence"], "properties": {"action_id": {"type": "string"}, "world_groups": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer", "minimum": 0, "maximum": 3}}, "confidence": {"type": "number", "minimum": 0.5, "maximum": 0.95}}}}}


def root_messages(public: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = {
        **public,
        "root_tool_contract": list(source_audit.ROOT_TOOL_CONTRACT),
    }
    return [
        {"role": "system", "content": "You predict observational equivalence for documented telecom read-only tools. Return only strict JSON without reasoning."},
        {"role": "user", "content": "For each root tool in the supplied exact order, decide whether all four candidate worlds would produce exactly the same visible output under its documented semantics. Infer outputs from each world description. Predict observations, not repair success. Preserve action IDs. Confidence is the probability that the boolean is correct. Do not include the unlocked native action.\n" + canonical_json(payload)},
    ]


def native_messages(public: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = {
        **public,
        "native_tool_contract": source_audit.NATIVE_TOOL_CONTRACT,
    }
    return [
        {"role": "system", "content": "You predict the documented messaging-permission tool output across candidate telecom worlds. Return only strict JSON without reasoning."},
        {"role": "user", "content": "The installed_apps read has already revealed the messaging app and unlocked messaging_permissions. Under the supplied public tool contract, partition the four candidate worlds by whether messaging_permissions would return exactly the same visible granted-permission names. Infer granted names from each world description: absent permission names are not granted. Use canonical first-occurrence labels: world 0 is group 0; later worlds reuse a label exactly for identical visible output, otherwise use the next integer. Return only the supplied native action ID, four labels, and confidence.\n" + canonical_json(payload)},
    ]


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.5 <= float(value) <= 0.95:
        raise ValueError("documented-split resilient confidence changed")
    return float(value)


def parse_root(raw: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(raw)
    rows = payload.get("actions") if isinstance(payload, dict) and set(payload) == {"actions"} else None
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("documented-split resilient root action array changed")
    parsed = {}
    for expected, row in zip(ROOTS, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"action_id", "all_worlds_same", "confidence"} or row["action_id"] != expected or not isinstance(row["all_worlds_same"], bool):
            raise ValueError("documented-split resilient root row changed")
        parsed[expected] = {"all_worlds_same": row["all_worlds_same"], "confidence": _confidence(row["confidence"])}
    return parsed


def parse_native(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"action_id", "world_groups", "confidence"} or payload["action_id"] != NATIVE:
        raise ValueError("documented-split resilient native root changed")
    groups = payload["world_groups"]
    if not isinstance(groups, list) or len(groups) != 4 or any(isinstance(value, bool) or not isinstance(value, int) or value not in range(4) for value in groups) or canonical_groups(groups) != groups:
        raise ValueError("documented-split resilient native groups changed")
    return {"groups": groups, "confidence": _confidence(payload["confidence"])}


def partition_table(groups: Sequence[int], confidence: float) -> dict[str, dict[str, float]]:
    categories = [str(value) for value in sorted(set(groups))] + ["OTHER"]
    rest = (1.0 - confidence) / (len(categories) - 1)
    result = {f"w{index}": {category: confidence if category == str(group) else rest for category in categories} for index, group in enumerate(groups)}
    if any(not math.isclose(sum(row.values()), 1.0, abs_tol=1e-12) for row in result.values()):
        raise ValueError("documented-split resilient likelihood normalization changed")
    return result


def score(episodes, roots, natives, observations):
    root_correct = root_count = native_exact = pair_correct = pair_count = native_top = native_count = 0
    root_brier = native_partition_brier = native_mass = native_posterior_brier = 0.0
    tvs = []
    semantic = []
    source_values = []
    episode_metrics = []
    for episode, root, native, truth in zip(episodes, roots, natives, observations, strict=True):
        family = episode["family"]
        prior = [0.25] * 4
        all_tables = {}
        for action in ROOTS:
            expected = len({canonical_json(world[action]) for world in truth}) == 1
            predicted = root[action]["all_worlds_same"]
            confidence = root[action]["confidence"]
            root_correct += predicted == expected
            root_count += 1
            probability_true = confidence if predicted else 1.0 - confidence
            root_brier += 2.0 * ((probability_true - float(expected)) ** 2)
            groups = [0, 0, 0, 0] if predicted else [0, 1, 2, 3]
            all_tables[action] = partition_table(groups, confidence)
            if expected:
                for left in range(4):
                    for right in range(left + 1, 4):
                        tvs.append(0.5 * sum(abs(all_tables[action][f"w{left}"][key] - all_tables[action][f"w{right}"][key]) for key in all_tables[action][f"w{left}"]))
        expected_native = canonical_groups([world[NATIVE] for world in truth])
        native_exact += native["groups"] == expected_native
        native_table = partition_table(native["groups"], native["confidence"])
        all_tables[NATIVE] = native_table
        for left in range(4):
            for right in range(left + 1, 4):
                pair_correct += (native["groups"][left] == native["groups"][right]) == (expected_native[left] == expected_native[right])
                pair_count += 1
        for world_index, outcome in enumerate(expected_native):
            category = str(outcome) if str(outcome) in native_table[f"w{world_index}"] else "OTHER"
            native_partition_brier += sum((probability - float(key == category)) ** 2 for key, probability in native_table[f"w{world_index}"].items())
            posterior = mathlib.posterior(prior, native_table, category)
            native_mass += posterior[world_index]
            native_top += posterior[world_index] >= max(posterior) - 1e-12
            native_posterior_brier += sum((probability - float(index == world_index)) ** 2 for index, probability in enumerate(posterior))
            native_count += 1
        root_values = {action: mathlib.information(prior, all_tables[action]) for action in ROOTS}
        depth_values = {action: mathlib.depth_two(family, all_tables, action) for action in ROOTS}
        greedy = max(ROOTS, key=lambda action: (root_values[action], action))
        planned = max(ROOTS, key=lambda action: (depth_values[action], action))
        canonical = [{action: canonical_json(value) for action, value in world.items()} for world in truth]
        exact_values = {action: mathlib.exact.two_step_information(family, canonical, action) for action in ROOTS}
        semantic.extend(depth_values.values())
        source_values.extend(exact_values.values())
        episode_metrics.append({"family": family, "greedy_first_action": greedy, "depth_two_first_action": planned, "horizon_gain_nats": depth_values[planned] - depth_values[greedy], "root_information_nats": root_values, "two_step_information_nats": depth_values, "exact_two_step_information_nats": exact_values})
    metrics = {"root_decision_count": root_count, "root_exact_count": root_correct, "root_mean_brier": root_brier / root_count, "native_partition_count": 6, "native_partition_exact_count": native_exact, "native_pair_relation_count": pair_count, "native_pair_relation_accuracy": pair_correct / pair_count, "native_mean_partition_brier": native_partition_brier / 24, "native_answer_count": native_count, "native_truth_top_rank_count": native_top, "native_mean_truth_posterior": native_mass / native_count, "native_mean_posterior_brier": native_posterior_brier / native_count, "equivalent_pair_count": len(tvs), "equivalent_mean_total_variation": sum(tvs) / len(tvs), "equivalent_max_total_variation": max(tvs), "semantic_source_two_step_spearman": mathlib.spearman(semantic, source_values), "mean_horizon_gain_nats": sum(row["horizon_gain_nats"] for row in episode_metrics) / 6}
    gates = {"exact_48_root_decisions": root_count == 48, "at_least_47_exact_root_decisions": root_correct >= 47, "root_brier_at_most_0_08": metrics["root_mean_brier"] <= .08, "all_six_native_partitions_exact": native_exact == 6, "exact_36_native_pair_relations": pair_count == 36, "native_pair_accuracy_one": metrics["native_pair_relation_accuracy"] == 1.0, "native_partition_brier_at_most_0_08": metrics["native_mean_partition_brier"] <= .08, "exact_24_native_answers": native_count == 24, "all_24_native_truth_top_rank": native_top == 24, "native_truth_mass_at_least_0_65": metrics["native_mean_truth_posterior"] >= .65, "native_posterior_brier_at_most_0_18": metrics["native_mean_posterior_brier"] <= .18, "equivalent_mean_tv_at_most_0_03": metrics["equivalent_mean_total_variation"] <= .03, "equivalent_max_tv_at_most_0_10": metrics["equivalent_max_total_variation"] <= .10, "all_six_greedy_avoid_installed_apps": all(row["greedy_first_action"] != "installed_apps" for row in episode_metrics), "all_six_depth_two_choose_installed_apps": all(row["depth_two_first_action"] == "installed_apps" for row in episode_metrics), "all_six_horizon_gain_at_least_0_50": all(row["horizon_gain_nats"] >= .50 for row in episode_metrics), "semantic_source_spearman_at_least_0_90": metrics["semantic_source_two_step_spearman"] >= .90}
    gates["all_calibration_gates_pass"] = all(gates.values())
    return {"metrics": metrics, "episode_metrics": episode_metrics, "calibration_gates": gates}


PROVIDER_ERROR_MESSAGE = "provider-error response persisted after retries"


class ResilientDocumentedSplitAdapter(partition.absolute.base.NonReasoningSeededAdapter):
    def __init__(self, *args, partial_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_path = partial_path
        self._attempt_lock = threading.Lock()
        self._attempts: list[dict[str, Any]] = []

    def _request_payload(self, row: Mapping[str, Any]) -> dict[str, Any]:
        self._per_request_seed.value = int(row["seed"])
        try:
            return self._payload(
                row["messages"], 0.0, 1, row["max_tokens"],
                response_format=row["response_format"],
            )
        finally:
            del self._per_request_seed.value

    def _record_attempt(self, index: int, row: Mapping[str, Any], attempt_number: int) -> None:
        payload = self._request_payload(row)
        record = {
            "index": index,
            "episode_index": row["episode_index"],
            "kind": row["kind"],
            "attempt_number": attempt_number,
            "seed": row["seed"],
            "payload_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        }
        with self._attempt_lock:
            self._attempts.append(record)

    def attempt_records(self) -> list[dict[str, Any]]:
        with self._attempt_lock:
            return sorted(
                (dict(row) for row in self._attempts),
                key=lambda row: (row["attempt_number"], row["index"]),
            )

    def _checkpoint(self, requests, completed, *, complete: bool) -> None:
        partition.checkpointed.checkpoint_partial(self.partial_path, {
            "interface_version": INTERFACE_VERSION,
            "model_id": MODEL_ID,
            "requests": [{"index": i, "episode_index": row["episode_index"], "kind": row["kind"], "seed": row["seed"], "prompt_sha256": hashlib.sha256(canonical_json(row["messages"]).encode()).hexdigest()} for i, row in enumerate(requests)],
            "attempts": self.attempt_records(),
            "completed": [{"index": i, "episode_index": requests[i]["episode_index"], "kind": requests[i]["kind"], "seed": requests[i]["seed"], "response": completed[i]} for i in sorted(completed)],
            "complete": complete and len(completed) == len(requests),
        })

    def run_requests(self, requests: Sequence[Mapping[str, Any]]) -> list[str]:
        def request(index: int, attempt_number: int = 1) -> str:
            row = requests[index]
            self._record_attempt(index, row, attempt_number)
            self._per_request_seed.value = int(row["seed"])
            try:
                return self._complete_request(row["messages"], 0.0, 1, row["max_tokens"], response_format=row["response_format"])[0]
            finally:
                del self._per_request_seed.value

        completed: dict[int, str] = {}
        errors: dict[int, BaseException] = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(request, index): index for index in range(len(requests))}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    completed[index] = future.result()
                except BaseException as exc:
                    errors[index] = exc
                self._checkpoint(requests, completed, complete=False)
        if len(errors) == 1:
            retry_index, error = next(iter(errors.items()))
            if str(error) == PROVIDER_ERROR_MESSAGE:
                with self._usage_lock:
                    self.retry_count += 1
                    self.provider_error_retries += 1
                try:
                    completed[retry_index] = request(retry_index, 2)
                except BaseException as exc:
                    errors[retry_index] = exc
                else:
                    errors.clear()
                self._checkpoint(requests, completed, complete=not errors)
        if errors:
            raise errors[min(errors)]
        if len(completed) != len(requests):
            raise RuntimeError("documented-split resilient response set incomplete")
        self._checkpoint(requests, completed, complete=True)
        return [completed[index] for index in range(len(requests))]


def build_adapter(*, run_id: str, output_dir: Path):
    config = Config(task="animals", run_id=run_id, log_path=output_dir / "run.log", openrouter_budget_usd=245.0, openrouter_run_budget_usd=RUN_CAP_USD, openrouter_projected_cost_usd=PROJECTED_COST_USD, openrouter_concurrency=CONCURRENCY, openrouter_max_retries=0, openrouter_request_timeout_seconds=300.0, openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD, openrouter_max_output_tokens=ROOT_MAX_TOKENS, openrouter_spend_path="results/path_e/openrouter_spend.json")
    return ResilientDocumentedSplitAdapter(ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65_536), config, partial_path=output_dir / "private/PARTIAL_RAW_RESPONSES.json")


def build_requests(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    public = [public_episode(row, index) for index, row in enumerate(episodes)]
    requests = []
    for index, row in enumerate(public):
        requests.append({"episode_index": index, "kind": "root", "seed": ROOT_SEEDS[index], "messages": root_messages(row), "response_format": root_response_format(), "max_tokens": ROOT_MAX_TOKENS})
    for index, row in enumerate(public):
        requests.append({"episode_index": index, "kind": "native", "seed": NATIVE_SEEDS[index], "messages": native_messages(row), "response_format": native_response_format(), "max_tokens": NATIVE_MAX_TOKENS})
    return requests


def run(*, output_dir: Path, adapter: Any, daily_budget_status: Mapping[str, Any] | None = None):
    episodes = selected_episodes()
    requests = build_requests(episodes)
    prompt_hashes = [hashlib.sha256(canonical_json(row["messages"]).encode()).hexdigest() for row in requests]
    privacy = {"prompt_sha256": prompt_hashes, "public_tool_contract_in_prompts": True, "selected_task_ids_in_prompts": False, "source_fault_ids_in_prompts": False, "raw_tool_responses_in_prompts": False, "repair_or_endpoint_outcomes_in_prompts": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    checkpoint(private / "PROMPT_PRIVACY.json", privacy)
    raw = adapter.run_requests(requests)
    request_identity = [{"episode_index": row["episode_index"], "kind": row["kind"], "seed": row["seed"], "prompt_sha256": prompt_hashes[index]} for index, row in enumerate(requests)]
    checkpoint(private / "RAW_RESPONSES.json", {"model_id": MODEL_ID, "requests": request_identity, "responses": raw})
    ordering = {"all_partial_responses_banked": True, "complete_raw_bank_created": True, "official_calibration_loaded_after_complete_bank": False}
    checkpoint(private / "ORDERING.json", ordering)
    roots = [parse_root(raw[index]) for index in range(6)]
    natives = [parse_native(raw[6 + index]) for index in range(6)]
    observations = partition.absolute.base.official_observations(episodes)
    ordering["official_calibration_loaded_after_complete_bank"] = True
    checkpoint(private / "ORDERING.json", ordering)
    scores = score(episodes, roots, natives, observations)
    usage = summarize_usage(adapter.usage_snapshot())
    attempts = adapter.attempt_records()
    retries = [row for row in attempts if row["attempt_number"] == 2]
    serving = {
        "exact_twelve_accepted_requests": usage["adapter_requests"] == 12,
        "http_attempts_within_bound": usage["http_attempts"] in (12, 13),
        "http_attempts_equal_accepted_plus_retries": usage["http_attempts"] == 12 + usage["retry_count"],
        "total_retries_within_bound": usage["retry_count"] in (0, 1),
        "provider_error_retries_exact": usage["provider_error_retries"] == usage["retry_count"],
        "retry_identity_complete": len(attempts) == usage["http_attempts"] and len(retries) == usage["retry_count"],
        "retry_payload_and_seed_identical": all(any(first["index"] == row["index"] and first["attempt_number"] == 1 and first["seed"] == row["seed"] and first["payload_sha256"] == row["payload_sha256"] for first in attempts) for row in retries),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_stage_cap": usage["run_cost_usd"] <= RUN_CAP_USD + 1e-12,
    }
    passed = scores["calibration_gates"]["all_calibration_gates_pass"] and all(serving.values())
    result = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "mms_documented_split_resilient_pass" if passed else "mms_documented_split_resilient_null", "authorizes": "prospective_paired_development_protocol_only" if passed else "nothing", "protocol_sha256": sha256_file(PROTOCOL), "manifest_sha256": sha256_file(MANIFEST), "model": MODEL_ID, "root_seeds": list(ROOT_SEEDS), "native_seeds": list(NATIVE_SEEDS), "privacy": privacy, "ordering": ordering, "retry_identity": {"attempts": attempts}, "usage": usage, "serving_gates": serving, **scores, "daily_budget_status": dict(daily_budget_status or {}), "development_confirmation_reserve_opened": False, "repair_or_task_success_endpoints_opened": False}
    checkpoint(output_dir / "RESULT.json", result)
    return result
