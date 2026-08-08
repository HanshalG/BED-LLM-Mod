#!/usr/bin/env python3
"""Verify the frozen Bongard Luna naive first-link protocol manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_naive_smoke_migration as migration


INTERFACE_VERSION = "bongard-openworld-luna-naive-first-link-manifest-8"
MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_naive_first_link/"
    "PROTOCOL_MANIFEST_V8.json"
)
MANIFEST_SHA256 = (
    "25db6fd3241d8ffaa4989ffaadfe6bb2bec111d7f1e3d6936dd9e39fd333d454"
)
MAIN_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_development64/"
    "PROTOCOL_MANIFEST_V17.json"
)
MAIN_MANIFEST_SHA256 = (
    "7564ced7755f17be13f254067f013130b4beb277313fc51de16b43611a608676"
)
TRANSPORT_RETRY_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_LUNA_TRANSPORT_RETRY_AMENDMENT.md"
)
TRANSPORT_RETRY_AMENDMENT_SHA256 = (
    "0f6ffc66f9b7d0f45d8913cf4f790d6f0c891e135cf5b9102575bd0a51ef7dd9"
)
ENDPOINT_UTILITY_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_ENDPOINT_PREDICTIVE_UTILITY_AMENDMENT.md"
)
ENDPOINT_UTILITY_AMENDMENT_SHA256 = (
    "2fce4b66696d5b635f8d32c5f968a917aac20ab64305cab2728bc5f9973a55b8"
)
SMOKE_RESULT = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_naive_first_link/"
    "smoke-20260808/RESULT.json"
)
SMOKE_RESULT_SHA256 = (
    "f1070b30ea63f571ffac2e23f7b9cfb5fc97ad42dd95e0e6babefbc7ca387a70"
)
SMOKE_REPLAY_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_NAIVE_SMOKE_REPLAY_AMENDMENT_20260808.md"
)
SMOKE_REPLAY_AMENDMENT_SHA256 = (
    "6b6b2b5f189a3b016f51d94e048d30c9f7d8e99ba46da1793cbfb656d568a464"
)
SMOKE_REPLAY_CERTIFICATE_SHA256 = (
    "725bb7827508b6b20bf34037cc33feb17b78e1c31c4da9a21d9c3b9ea474d1da"
)
ESTIMAND_CLARIFICATION = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_HISTORY_BLIND_ESTIMAND_CLARIFICATION_20260808.md"
)
ESTIMAND_CLARIFICATION_SHA256 = (
    "65a6e901dc815d1603611e180b0abdf728e07d1a452e0c442f48fbb1812aea10"
)
MATCHED_REALIZED_UPDATER_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_MATCHED_REALIZED_UPDATER_AMENDMENT_20260808.md"
)
MATCHED_REALIZED_UPDATER_AMENDMENT_SHA256 = (
    "dfa981153687004c8fb2c1195879d0774a281ca6c231c55d85495f2ac622178b"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    if sha256_file(path) != MANIFEST_SHA256:
        raise RuntimeError("naive first-link protocol manifest changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    smoke_replay = migration.verify_certificate()
    expected_model = {
        "concurrency": 10,
        "id": "openai/gpt-5.6-luna",
        "max_tokens": 8192,
        "reasoning_effort": "medium",
        "reasoning_trace_excluded": True,
        "requests_per_development_block": 16,
        "smoke_requests": 10,
        "temperature": 0.0,
    }
    expected_schedule = {
        "development_blocks": {
            "a": "2026-08-11",
            "b": "2026-08-12",
            "c": "2026-08-13",
            "d": "2026-08-14",
        },
        "main_block_precedes_same_day_naive_block": True,
        "smoke_date": "2026-08-08",
    }
    expected_privacy = {
        "candidate_labels_materialized": False,
        "endpoint_images_exposed": False,
        "endpoint_labels_materialized": False,
        "source_metadata_exposed": False,
    }
    expected_scoring = {
        "bootstrap_replicates": 20_000,
        "bootstrap_seed": 2_026_080_751,
        "continuation": "main_dynamic_branch_endpoint_predictive_second_step",
        "main_all_first_action_cache_only": True,
        "one_choice_per_task": True,
        "score_objective": "endpoint_predictive_information_gain_nats",
        "task_count": 64,
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("interface_version") != INTERFACE_VERSION
        or manifest.get("frozen_before_development_responses") is not True
        or manifest.get("banked_serving_smoke_precedes_amendment") is not True
        or manifest.get("banked_serving_smoke_result_sha256")
        != SMOKE_RESULT_SHA256
        or sha256_file(SMOKE_RESULT) != SMOKE_RESULT_SHA256
        or manifest.get("banked_smoke_current_request_replay_required")
        is not True
        or manifest.get("banked_smoke_replay_certificate_sha256")
        != SMOKE_REPLAY_CERTIFICATE_SHA256
        or smoke_replay.get("certificate_sha256")
        != SMOKE_REPLAY_CERTIFICATE_SHA256
        or manifest.get("model") != expected_model
        or manifest.get("schedule") != expected_schedule
        or manifest.get("privacy") != expected_privacy
        or manifest.get("scoring") != expected_scoring
        or manifest.get("main_development_manifest_sha256")
        != MAIN_MANIFEST_SHA256
        or manifest.get("transport_retry_amendment_sha256")
        != TRANSPORT_RETRY_AMENDMENT_SHA256
        or sha256_file(TRANSPORT_RETRY_AMENDMENT)
        != TRANSPORT_RETRY_AMENDMENT_SHA256
        or manifest.get("endpoint_predictive_utility_amendment_sha256")
        != ENDPOINT_UTILITY_AMENDMENT_SHA256
        or sha256_file(ENDPOINT_UTILITY_AMENDMENT)
        != ENDPOINT_UTILITY_AMENDMENT_SHA256
        or manifest.get("smoke_replay_amendment_sha256")
        != SMOKE_REPLAY_AMENDMENT_SHA256
        or sha256_file(SMOKE_REPLAY_AMENDMENT)
        != SMOKE_REPLAY_AMENDMENT_SHA256
        or manifest.get("history_blind_estimand_clarification_sha256")
        != ESTIMAND_CLARIFICATION_SHA256
        or sha256_file(ESTIMAND_CLARIFICATION)
        != ESTIMAND_CLARIFICATION_SHA256
        or manifest.get("matched_realized_updater_amendment_sha256")
        != MATCHED_REALIZED_UPDATER_AMENDMENT_SHA256
        or sha256_file(MATCHED_REALIZED_UPDATER_AMENDMENT)
        != MATCHED_REALIZED_UPDATER_AMENDMENT_SHA256
        or manifest.get("main_development_result_must_independently_replay")
        is not True
    ):
        raise RuntimeError("naive first-link protocol fields changed")
    if sha256_file(MAIN_MANIFEST) != MAIN_MANIFEST_SHA256:
        raise RuntimeError("main development manifest changed")
    file_hashes = manifest.get("files_sha256") or {}
    if not file_hashes:
        raise RuntimeError("naive first-link file bindings are absent")
    for relative, expected in file_hashes.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"naive first-link bound file changed: {relative}")
    return {
        "verified": True,
        "manifest_sha256": MANIFEST_SHA256,
        "main_development_manifest_sha256": MAIN_MANIFEST_SHA256,
        "smoke_replay_certificate_sha256": SMOKE_REPLAY_CERTIFICATE_SHA256,
        "banked_smoke_replayed_under_current_requests": True,
        "bound_file_count": len(file_hashes),
    }


def main() -> int:
    print(json.dumps(verify_protocol_manifest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
