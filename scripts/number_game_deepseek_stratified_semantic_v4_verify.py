#!/usr/bin/env python3
"""Independent replay for the DeepSeek stratified semantic V4 bank."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import number_game_deepseek_stratified_semantic_v4 as gate


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object")
    return value


def verify(run_dir: Path, *, output: Path | None = None) -> dict[str, Any]:
    raw_path = run_dir / "private/RAW_RESPONSES.json"
    result_path = run_dir / "LABEL_FREE_RESULT.json"
    raw = load(raw_path)
    result = load(result_path)
    responses = raw.get("responses") or []
    if (
        set(raw) != {"schema_version", "interface_version", "responses"}
        or raw["schema_version"] != 1
        or raw["interface_version"] != gate.INTERFACE_VERSION
        or len(responses) % gate.GROUP_SIZE
        or len(responses) not in {gate.GROUP_SIZE, 2 * gate.GROUP_SIZE, len(gate.MODEL_SEEDS)}
    ):
        raise RuntimeError("raw semantic bank envelope changed")
    expected_rows = []
    for group_index in range(len(responses) // gate.GROUP_SIZE):
        for request in gate.group_requests(group_index):
            expected_rows.append({
                **{key: value for key, value in request.items() if key != "messages"},
                "prompt_sha256": hashlib.sha256(canonical_json(request["messages"]).encode()).hexdigest(),
            })
    identities_match = True
    for actual, expected in zip(responses, expected_rows, strict=True):
        identities_match &= {key: actual.get(key) for key in expected} == expected
        identities_match &= set(actual) == {*expected, "response"} and isinstance(actual.get("response"), str)
    diagnostics = [
        gate.parse_group(responses[start : start + gate.GROUP_SIZE])
        for start in range(0, len(responses), gate.GROUP_SIZE)
    ]
    records = (result.get("transport") or {}).get("records") or []
    record_by_seed = {int(row.get("seed", -1)): row for row in records}
    raw_by_seed = {int(row["seed"]): row for row in responses}
    gates = {
        "request_identities_match": identities_match,
        "raw_hash_matches": result.get("raw_response_sha256") == gate.digest(raw_path),
        "diagnostics_replay": result.get("groups") == diagnostics,
        "transport_records_replay": (
            len(records) == len(responses)
            and len(record_by_seed) == len(responses)
            and all(
                record_by_seed[int(row["seed"])].get("model_requested")
                == record_by_seed[int(row["seed"])].get("model_returned")
                == gate.MODEL_ID
                and record_by_seed[int(row["seed"])].get("finish_reasons") == ["stop"]
                and record_by_seed[int(row["seed"])].get("prompt_sha256")
                == raw_by_seed[int(row["seed"])]["prompt_sha256"]
                for row in responses
            )
        ),
        "outcomes_closed": (
            result.get("canonical_targets_opened") is False
            and result.get("policy_endpoints_opened") is False
            and result.get("development_confirmation_opened") is False
        ),
    }
    verified = {
        "schema_version": 1,
        "interface_version": "number-game-deepseek-stratified-semantic-v4-verifier-1",
        "status": "verification_pass" if all(gates.values()) else "verification_failed",
        "gates": gates,
        "model_calls_made": 0,
    }
    if output:
        output.write_text(json.dumps(verified, indent=2, sort_keys=True) + "\n")
    if verified["status"] != "verification_pass":
        raise RuntimeError("DeepSeek semantic bank replay failed")
    return verified
