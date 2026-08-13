#!/usr/bin/env python3
"""Run the frozen fresh-cohort Tau2 MMS array semantic calibration."""

from __future__ import annotations

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
from scripts import tau2_native_prerequisite_semantic_mechanics as base
from scripts import tau2_native_prerequisite_source_manifest as source
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage


SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-mms-array-semantic-calibration-1"
MODEL_ID = base.MODEL_ID
MODEL_SEEDS = tuple(range(202608130200, 202608130206))
MAX_TOKENS = 6_000
EXPECTED_REQUESTS = 6
RUN_CAP_USD = 0.10
PROJECTED_COST_USD = 0.06
MAX_REQUEST_COST_USD = 0.01
PROTOCOL = REPO_ROOT / "results/nonmyopic/TAU2_MMS_ARRAY_SEMANTIC_CALIBRATION_PROTOCOL_20260813.md"
MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_array_semantic_calibration/CALIBRATION_MANIFEST.json"


class Adapter(Protocol):
    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages: Sequence[list[dict[str, str]]], seeds: Sequence[int],
        *, temperature: float, response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...
    def usage_snapshot(self) -> dict[str, Any]: ...


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_hash(episode: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (episode["family"] + "|" + "|".join(sorted(episode["backbone"]))).encode()
    ).hexdigest()


def selected_episodes() -> list[dict[str, Any]]:
    tasks = json.loads(source.TASKS_PATH.read_text())
    splits = json.loads(source.SPLITS_PATH.read_text())
    selected, _ = source.select_episodes(
        [str(row["id"]) for row in tasks], list(splits["base"])
    )
    by_hash = {episode_hash(row): row for row in selected["reserve"]}
    manifest = json.loads(MANIFEST.read_text())
    hashes = [row["episode_sha256"] for row in manifest["episodes"]]
    if len(hashes) != 6 or len(set(hashes)) != 6 or not set(hashes) <= set(by_hash):
        raise ValueError("MMS array calibration manifest changed")
    episodes = [by_hash[digest] for digest in hashes]
    if [row["family"] for row in episodes] != [row["family"] for row in manifest["episodes"]]:
        raise ValueError("MMS array calibration family order changed")
    return episodes


def public_episode(episode: Mapping[str, Any], index: int) -> dict[str, Any]:
    public = base.public_episode(episode, index)
    actions = []
    for action_id, fields in base.MMS_ACTION_FIELDS.items():
        actions.append(
            {
                "action_id": action_id,
                "fields": [
                    {
                        "field_id": field_id,
                        "allowed_values": [
                            str(value).lower() if isinstance(value, bool) else str(value)
                            for value in values
                        ],
                    }
                    for field_id, values in fields.items()
                ],
            }
        )
    return {
        "episode_index": index,
        "family": episode["family"],
        "worlds": public["worlds"],
        "actions": actions,
        "legal_unlock": public["legal_unlock"],
    }


def response_format() -> dict[str, Any]:
    field = {
        "type": "object", "additionalProperties": False,
        "required": ["field_id", "value"],
        "properties": {
            "field_id": {"type": "string"},
            "value": {"type": "string"},
        },
    }
    action = {
        "type": "object", "additionalProperties": False,
        "required": ["action_id", "fields", "confidence"],
        "properties": {
            "action_id": {"type": "string"},
            "fields": {"type": "array", "minItems": 1, "maxItems": 9, "items": field},
            "confidence": {"type": "number", "minimum": 0.5, "maximum": 0.95},
        },
    }
    world = {
        "type": "object", "additionalProperties": False,
        "required": ["world_id", "actions"],
        "properties": {
            "world_id": {"type": "string"},
            "actions": {"type": "array", "minItems": 9, "maxItems": 9, "items": action},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tau2_mms_array_semantic",
            "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "required": ["worlds"],
                "properties": {
                    "worlds": {"type": "array", "minItems": 4, "maxItems": 4, "items": world}
                },
            },
        },
    }


def messages(public: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a semantic forward model for telecom diagnostics. Predict "
                "only visible read-only tool outputs. Return strict JSON with no "
                "reasoning text, diagnosis, or action choice."
            ),
        },
        {
            "role": "user",
            "content": (
                "The candidate worlds are complete and equally possible; unmentioned "
                "conditions are normal. For each world, return exactly nine action rows "
                "in the supplied order. For each action, return its fields in the "
                "supplied order and choose only an allowed string value. Confidence is "
                "the probability that the complete typed signature is correct. Reuse "
                "identical values for physically identical observations. Do not leak a "
                "hidden condition into a tool that cannot expose it. The installed-apps "
                "read unlocks messaging-permissions. Preserve all IDs and array order.\n"
                + canonical_json(public)
            ),
        },
    ]


def parse(raw: str, public: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"worlds"}:
        raise ValueError("MMS array root changed")
    worlds = payload["worlds"]
    if not isinstance(worlds, list) or len(worlds) != 4:
        raise ValueError("MMS array world count changed")
    parsed = []
    for expected_world, row in zip(public["worlds"], worlds, strict=True):
        if not isinstance(row, dict) or set(row) != {"world_id", "actions"} or row["world_id"] != expected_world["world_id"]:
            raise ValueError("MMS array world order changed")
        if not isinstance(row["actions"], list) or len(row["actions"]) != 9:
            raise ValueError("MMS array action count changed")
        predictions = {}
        for expected_action, action_row in zip(public["actions"], row["actions"], strict=True):
            if not isinstance(action_row, dict) or set(action_row) != {"action_id", "fields", "confidence"} or action_row["action_id"] != expected_action["action_id"]:
                raise ValueError("MMS array action order changed")
            field_rows = action_row["fields"]
            if not isinstance(field_rows, list) or len(field_rows) != len(expected_action["fields"]):
                raise ValueError("MMS array field count changed")
            signature = {}
            for expected_field, field_row in zip(expected_action["fields"], field_rows, strict=True):
                if not isinstance(field_row, dict) or set(field_row) != {"field_id", "value"} or field_row["field_id"] != expected_field["field_id"] or field_row["value"] not in expected_field["allowed_values"]:
                    raise ValueError("MMS array field order or value changed")
                value: Any = field_row["value"]
                if value in ("true", "false") and expected_field["allowed_values"] == ["false", "true"]:
                    value = value == "true"
                signature[field_row["field_id"]] = value
            confidence = action_row["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0.5 <= float(confidence) <= 0.95:
                raise ValueError("MMS array confidence changed")
            predictions[action_row["action_id"]] = {"signature": signature, "confidence": float(confidence)}
        parsed.append({"world_id": row["world_id"], "predictions": predictions})
    return {"worlds": parsed}


def calibration_gates(scores: Mapping[str, Any]) -> dict[str, bool]:
    metrics = scores["metrics"]
    episodes = scores["episode_metrics"]
    family = metrics["family_mean_multiclass_brier"]
    gates = {
        "exact_216_world_action_cells": metrics["cell_count"] == 216,
        "typed_signature_accuracy_at_least_0_90": metrics["typed_signature_accuracy"] >= 0.90,
        "mean_multiclass_brier_at_most_0_18": metrics["mean_multiclass_brier"] <= 0.18,
        "each_family_brier_at_most_0_25": set(family) == {"mms_abroad", "mms_home"} and all(value <= 0.25 for value in family.values()),
        "exact_24_native_followup_cells": metrics["native_followup_cell_count"] == 24,
        "native_signature_exact_at_least_23": metrics["native_signature_exact_count"] >= 23,
        "native_truth_top_rank_at_least_23": metrics["native_truth_top_rank_count"] >= 23,
        "native_mean_truth_posterior_at_least_0_65": metrics["native_mean_truth_posterior"] >= 0.65,
        "native_mean_posterior_brier_at_most_0_18": metrics["native_mean_posterior_brier"] <= 0.18,
        "equivalent_mean_tv_at_most_0_03": metrics["equivalent_mean_total_variation"] <= 0.03,
        "equivalent_max_tv_at_most_0_10": metrics["equivalent_max_total_variation"] <= 0.10,
        "all_six_greedy_avoid_installed_apps": all(row["greedy_first_action"] != "installed_apps" for row in episodes),
        "all_six_depth_two_choose_installed_apps": all(row["depth_two_first_action"] == "installed_apps" for row in episodes),
        "all_six_horizon_gain_at_least_0_10": all(row["horizon_gain_nats"] >= 0.10 for row in episodes),
        "mean_horizon_gain_at_least_0_50": metrics["mean_horizon_gain_nats"] >= 0.50,
        "semantic_source_spearman_at_least_0_80": metrics["semantic_source_two_step_spearman"] >= 0.80,
    }
    gates["all_calibration_gates_pass"] = all(gates.values())
    return gates


def usage(adapter: Adapter) -> tuple[dict[str, Any], dict[str, bool]]:
    value = summarize_usage(adapter.usage_snapshot())
    gates = {
        "exact_six_accepted_requests": value["adapter_requests"] == 6,
        "exact_six_http_attempts": value["http_attempts"] == 6,
        "zero_retries": value["retry_count"] == 0,
        "zero_provider_error_retries": value["provider_error_retries"] == 0,
        "zero_reasoning_tokens": value["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": value["forced_exits"] == 0,
        "within_stage_cap": value["run_cost_usd"] <= RUN_CAP_USD + 1e-12,
    }
    return value, gates


def build_adapter(*, run_id: str, output_dir: Path) -> base.NonReasoningSeededAdapter:
    config = Config(
        task="animals", run_id=run_id, log_path=output_dir / "run.log",
        openrouter_budget_usd=245.0, openrouter_run_budget_usd=RUN_CAP_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=6, openrouter_max_retries=0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return base.NonReasoningSeededAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65_536), config
    )


def run(*, output_dir: Path, adapter: Adapter, daily_budget_status: Mapping[str, Any] | None = None) -> dict[str, Any]:
    episodes = selected_episodes()
    public = [public_episode(row, index) for index, row in enumerate(episodes)]
    prompts = [messages(row) for row in public]
    privacy = {
        "prompt_sha256": [hashlib.sha256(canonical_json(row).encode()).hexdigest() for row in prompts],
        "selected_task_ids_in_prompts": False, "source_fault_ids_in_prompts": False,
        "raw_tool_responses_in_prompts": False, "repair_or_endpoint_outcomes_in_prompts": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True); private = output_dir / "private"; private.mkdir(parents=True, exist_ok=True)
    checkpoint(private / "PROMPT_PRIVACY.json", privacy)
    raw = adapter.chat_complete_seeded_messages_batched_structured(
        prompts, MODEL_SEEDS, temperature=0.0, response_format=response_format(), max_new_tokens=MAX_TOKENS
    )
    if len(raw) != 6: raise ValueError("MMS array response count changed")
    checkpoint(private / "RAW_RESPONSES.json", {"model_id": MODEL_ID, "seeds": list(MODEL_SEEDS), "responses": list(raw)})
    ordering = {"all_raw_responses_banked": True, "official_calibration_loaded_after_raw_bank": False}; checkpoint(private / "ORDERING.json", ordering)
    parsed = [parse(text, row) for text, row in zip(raw, public, strict=True)]
    observed = base.official_observations(episodes)
    ordering["official_calibration_loaded_after_raw_bank"] = True; checkpoint(private / "ORDERING.json", ordering)
    scores = base.score_responses(episodes, parsed, observed)
    gates = calibration_gates(scores)
    usage_value, serving = usage(adapter)
    passed = gates["all_calibration_gates_pass"] and all(serving.values())
    result = {
        "schema_version": 1, "interface_version": INTERFACE_VERSION,
        "status": "mms_array_semantic_pass" if passed else "mms_array_semantic_null",
        "authorizes": "prospective_paired_development_protocol_only" if passed else "nothing",
        "protocol_sha256": sha256_file(PROTOCOL), "manifest_sha256": sha256_file(MANIFEST),
        "model": MODEL_ID, "model_seeds": list(MODEL_SEEDS), "privacy": privacy,
        "ordering": ordering, "usage": usage_value, "serving_gates": serving,
        "metrics": scores["metrics"], "episode_metrics": scores["episode_metrics"],
        "calibration_gates": gates, "daily_budget_status": dict(daily_budget_status or {}),
        "development_confirmation_reserve_opened": False,
        "repair_or_task_success_endpoints_opened": False,
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result
