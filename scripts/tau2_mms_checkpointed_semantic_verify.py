#!/usr/bin/env python3
"""Independently replay the fresh checkpointed Tau2 MMS calibration."""

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

from scripts import tau2_mms_array_semantic_verify as independent
from scripts import tau2_native_prerequisite_source_manifest as source

SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-mms-checkpointed-semantic-verification-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
MODEL_SEEDS = tuple(range(202608130300, 202608130306))
MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_checkpointed_semantic_calibration/CALIBRATION_MANIFEST.json"
PREDECESSOR_MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_array_semantic_calibration/CALIBRATION_MANIFEST.json"


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
    tasks = json.loads(source.TASKS_PATH.read_text())
    splits = json.loads(source.SPLITS_PATH.read_text())
    selected, _ = source.select_episodes(
        [str(row["id"]) for row in tasks], list(splits["base"])
    )
    by_hash = {episode_hash(row): row for row in selected["reserve"]}
    rows = load_object(MANIFEST).get("episodes")
    old = {row["episode_sha256"] for row in load_object(PREDECESSOR_MANIFEST)["episodes"]}
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("independent checkpointed manifest changed")
    episodes = []
    for row in rows:
        digest = row.get("episode_sha256")
        episode = by_hash.get(digest)
        if (
            episode is None or digest in old or row.get("family") != episode["family"]
            or row.get("state_count") != len(episode["worlds"])
            or not episode["family"].startswith("mms_")
        ):
            raise ValueError("independent checkpointed episode binding changed")
        episodes.append(episode)
    return episodes


def validate_partial_bank(partial: Mapping[str, Any], *, require_complete: bool) -> list[str]:
    if (
        partial.get("interface_version") != "tau2-mms-checkpointed-semantic-calibration-1"
        or partial.get("model_id") != MODEL_ID
        or tuple(partial.get("expected_seeds", ())) != MODEL_SEEDS
    ):
        raise ValueError("independent partial bank identity changed")
    rows = partial.get("completed")
    if not isinstance(rows, list):
        raise ValueError("independent partial bank rows changed")
    indexes = [row.get("index") for row in rows if isinstance(row, dict)]
    if indexes != sorted(indexes) or len(indexes) != len(set(indexes)) or any(index not in range(6) for index in indexes):
        raise ValueError("independent partial bank indexes changed")
    for row in rows:
        if set(row) != {"index", "seed", "response"} or row["seed"] != MODEL_SEEDS[row["index"]] or not isinstance(row["response"], str):
            raise ValueError("independent partial bank row changed")
    complete = len(rows) == 6
    if partial.get("complete") is not complete:
        raise ValueError("independent partial bank completeness changed")
    if require_complete and not complete:
        raise ValueError("independent partial bank is incomplete")
    return [row["response"] for row in rows]


def replay(raw_bank: Mapping[str, Any]) -> dict[str, Any]:
    if raw_bank.get("model_id") != MODEL_ID or tuple(raw_bank.get("seeds", ())) != MODEL_SEEDS or len(raw_bank.get("responses", ())) != 6:
        raise ValueError("independent checkpointed raw bank identity changed")
    episodes = selected_episodes()
    parsed = [independent.parse_raw(raw) for raw in raw_bank["responses"]]
    return independent.score(episodes, parsed)


def verify(run_dir: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    result = load_object(run_dir / "RESULT.json")
    raw = load_object(run_dir / "private/RAW_RESPONSES.json")
    partial = load_object(run_dir / "private/PARTIAL_RAW_RESPONSES.json")
    ordering = load_object(run_dir / "private/ORDERING.json")
    privacy = load_object(run_dir / "private/PROMPT_PRIVACY.json")
    partial_responses = validate_partial_bank(partial, require_complete=True)
    if raw.get("responses") != partial_responses:
        raise ValueError("complete and partial raw banks differ")
    replayed = replay(raw)
    for key in ("metrics", "episode_metrics", "calibration_gates"):
        if result.get(key) != replayed[key]:
            raise ValueError(f"independent checkpointed {key} differs")
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
    expected_ordering = {"all_partial_responses_banked": True, "complete_raw_bank_created": True, "official_calibration_loaded_after_complete_bank": True}
    checks = {
        "serving_exact": result.get("serving_gates") == serving,
        "status_exact": result.get("status") == ("mms_checkpointed_semantic_pass" if passed else "mms_checkpointed_semantic_null") and result.get("authorizes") == ("prospective_paired_development_protocol_only" if passed else "nothing"),
        "ordering_exact": ordering == expected_ordering and result.get("ordering") == ordering,
        "privacy_exact": all(privacy.get(key) is False for key in ("selected_task_ids_in_prompts", "source_fault_ids_in_prompts", "raw_tool_responses_in_prompts", "repair_or_endpoint_outcomes_in_prompts")),
        "downstream_unopened": result.get("development_confirmation_reserve_opened") is False and result.get("repair_or_task_success_endpoints_opened") is False,
    }
    verification = {
        "schema_version": SCHEMA_VERSION, "interface_version": INTERFACE_VERSION,
        "status": "verified" if all(checks.values()) else "invalid",
        "expected_result_status": "mms_checkpointed_semantic_pass" if passed else "mms_checkpointed_semantic_null",
        "checks": checks, "all_pass": all(checks.values()),
        "result_sha256": sha256_file(run_dir / "RESULT.json"),
        "raw_bank_sha256": sha256_file(run_dir / "private/RAW_RESPONSES.json"),
        "partial_bank_sha256": sha256_file(run_dir / "private/PARTIAL_RAW_RESPONSES.json"),
        "model_calls_made": 0, "cost_usd": 0.0,
    }
    if not verification["all_pass"]:
        raise ValueError("independent checkpointed verification failed")
    if output_path:
        output_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    return verification


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir", type=Path); parser.add_argument("--output", type=Path); args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, output_path=args.output), indent=2, sort_keys=True))
