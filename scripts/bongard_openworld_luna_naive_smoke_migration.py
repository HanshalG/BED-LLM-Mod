#!/usr/bin/env python3
"""Certify the banked V1 naive smoke against the unchanged V2 requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_naive_first_link as naive
from scripts import bongard_openworld_vlm_bed as bed


BANKED_SMOKE_RESULT = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_naive_first_link/"
    "smoke-20260808/RESULT.json"
)
PRIVATE_RAW_RESPONSES = BANKED_SMOKE_RESULT.parent / "private/RAW_RESPONSES.json"
PUBLIC_REPLAY_PAYLOAD = BANKED_SMOKE_RESULT.parents[1] / (
    "SMOKE_V1_REPLAY_PAYLOAD.json"
)
CERTIFICATE = BANKED_SMOKE_RESULT.parents[1] / "SMOKE_V2_REPLAY_CERTIFICATE.json"
BANKED_RESULT_SHA256 = (
    "f1070b30ea63f571ffac2e23f7b9cfb5fc97ad42dd95e0e6babefbc7ca387a70"
)
BANKED_RAW_SHA256 = (
    "281b9159ca3cd9e86f9d972e3da4fbaef0af4ecc0c1a3977f9723eef33e2c747"
)
OLD_INTERFACE_VERSION = "bongard-openworld-luna-naive-first-link-1"
OLD_MAIN_MANIFEST_SHA256 = (
    "a0b70ff8bbe3e36eba56b357e563504f12e4792d92cedee237d4b261d15a7708"
)
CERTIFIED_CURRENT_MAIN_MANIFEST_SHA256 = (
    "377596232d9fda34753bd99914292043ecc80a5584d55d95075e659c40e88011"
)
AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_NAIVE_SMOKE_REPLAY_AMENDMENT_20260808.md"
)
_FROZEN_CASES = tuple(naive.smoke_cases(bed.load_mechanics_tasks()))
_FROZEN_MESSAGES = tuple(naive.build_messages(case) for case in _FROZEN_CASES)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def build_certificate(
    *,
    result_path: Path = BANKED_SMOKE_RESULT,
    raw_path: Path = PUBLIC_REPLAY_PAYLOAD,
) -> dict[str, Any]:
    result_sha256 = sha256_file(result_path)
    raw_sha256 = sha256_file(raw_path)
    if result_sha256 != BANKED_RESULT_SHA256:
        raise RuntimeError("banked naive smoke result changed")
    if raw_sha256 != BANKED_RAW_SHA256:
        raise RuntimeError("banked naive smoke raw responses changed")
    result = _load(result_path)
    raw = _load(raw_path)
    protocol = result.get("protocol") or {}
    usage = result.get("usage") or {}
    gates = result.get("gates") or {}
    cases = _FROZEN_CASES
    messages = _FROZEN_MESSAGES
    responses = raw.get("responses") or []
    choices = result.get("choices") or []
    checks = {
        "original_status_and_gates_pass": (
            result.get("status") == "passed"
            and bool(gates)
            and all(value is True for value in gates.values())
        ),
        "original_interface_is_v1": (
            protocol.get("interface_version") == OLD_INTERFACE_VERSION
        ),
        "current_interface_is_v2": (
            naive.INTERFACE_VERSION
            == "bongard-openworld-luna-naive-first-link-2"
        ),
        "original_model_and_reasoning_match_current": (
            protocol.get("model") == naive.MODEL_ID
            and protocol.get("reasoning_effort") == naive.REASONING_EFFORT
        ),
        "original_manifest_is_registered_predecessor": (
            protocol.get("main_development_manifest_sha256")
            == OLD_MAIN_MANIFEST_SHA256
        ),
        "result_binds_exact_raw_responses": (
            result.get("raw_responses_sha256") == raw_sha256
        ),
        "exact_case_ids": raw.get("case_ids") == [case.case_id for case in cases],
        "exact_task_ids": raw.get("task_ids")
        == [case.task.task_id for case in cases],
        "exact_request_seeds": raw.get("seeds") == [case.seed for case in cases],
        "exact_display_orders": raw.get("display_orders")
        == [list(case.display_order) for case in cases],
        "exact_current_message_hashes": raw.get("message_sha256")
        == [naive.message_sha256(message) for message in messages],
        "current_prompt_privacy_passes": all(
            not naive.prompt_errors(case, message)
            for case, message in zip(cases, messages, strict=True)
        ),
        "candidate_labels_remain_unaccessed": (
            raw.get("candidate_labels_accessed") is False
            and protocol.get("candidate_labels_accessed") is False
        ),
        "endpoint_labels_remain_unaccessed": (
            raw.get("endpoint_labels_accessed") is False
            and protocol.get("endpoint_labels_accessed") is False
        ),
        "exact_response_and_choice_count": (
            len(responses) == naive.SMOKE_REQUESTS
            and len(choices) == naive.SMOKE_REQUESTS
        ),
        "strict_choices_replay_under_current_parser": (
            len(responses) == len(cases) == len(choices)
            and [
                naive.parse_choice(response, case.task)
                for response, case in zip(responses, cases, strict=True)
            ]
            == [row.get("first_image_id") for row in choices]
        ),
        "original_usage_is_exact_clean_ten": (
            usage.get("adapter_requests") == naive.SMOKE_REQUESTS
            and usage.get("http_attempts") == naive.SMOKE_REQUESTS
            and usage.get("retry_count") == 0
            and usage.get("provider_error_retries") == 0
            and usage.get("forced_exits") == 0
            and usage.get("run_cost_usd") == 0.006675399999999999
        ),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if value is not True]
        raise RuntimeError("banked naive smoke does not replay: " + ", ".join(failed))
    return {
        "schema_version": 1,
        "interface_version": "bongard-openworld-luna-naive-smoke-migration-1",
        "status": "verified_zero_call_v1_to_v2_request_replay",
        "banked_result_sha256": result_sha256,
        "banked_raw_responses_sha256": raw_sha256,
        "old_interface_version": OLD_INTERFACE_VERSION,
        "current_interface_version": naive.INTERFACE_VERSION,
        "old_main_manifest_sha256": OLD_MAIN_MANIFEST_SHA256,
        "current_main_manifest_sha256": CERTIFIED_CURRENT_MAIN_MANIFEST_SHA256,
        "request_count": naive.SMOKE_REQUESTS,
        "checks": checks,
        "candidate_labels_accessed": False,
        "endpoint_labels_accessed": False,
        "scientific_endpoint_accessed": False,
        "model_calls_made": 0,
        "cost_usd": 0.0,
    }


def verify_certificate(path: Path = CERTIFICATE) -> dict[str, Any]:
    observed = _load(path)
    expected = build_certificate()
    if _canonical(observed) != _canonical(expected):
        raise RuntimeError("banked naive smoke replay certificate changed")
    return {
        "verified": True,
        "certificate_sha256": sha256_file(path),
        "banked_result_sha256": expected["banked_result_sha256"],
        "banked_raw_responses_sha256": expected["banked_raw_responses_sha256"],
        "model_calls_made": 0,
        "cost_usd": 0.0,
    }


def publish_replay_payload(
    *,
    source: Path = PRIVATE_RAW_RESPONSES,
    output: Path = PUBLIC_REPLAY_PAYLOAD,
) -> dict[str, Any]:
    if sha256_file(source) != BANKED_RAW_SHA256:
        raise RuntimeError("private banked naive smoke raw responses changed")
    payload = _load(source)
    allowed = {
        "candidate_labels_accessed",
        "case_ids",
        "display_orders",
        "endpoint_labels_accessed",
        "message_sha256",
        "responses",
        "seeds",
        "stage",
        "task_ids",
    }
    if (
        set(payload) != allowed
        or payload["candidate_labels_accessed"] is not False
        or payload["endpoint_labels_accessed"] is not False
        or payload["stage"] != "smoke"
    ):
        raise RuntimeError("private smoke payload is not safe to publish")
    source_bytes = source.read_bytes()
    if output.exists():
        if output.read_bytes() != source_bytes:
            raise RuntimeError("public naive smoke replay payload changed")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source_bytes)
    return {
        "published": True,
        "payload_sha256": sha256_file(output),
        "candidate_labels_accessed": False,
        "endpoint_labels_accessed": False,
    }


def write_certificate(path: Path = CERTIFICATE) -> dict[str, Any]:
    certificate = build_certificate()
    if path.exists():
        if _canonical(_load(path)) != _canonical(certificate):
            raise RuntimeError("banked naive smoke replay certificate changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(certificate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return verify_certificate(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CERTIFICATE)
    args = parser.parse_args()
    publish_replay_payload()
    print(json.dumps(write_certificate(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
