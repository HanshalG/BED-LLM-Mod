#!/usr/bin/env python3
"""Independent replay of the fresh Number Game bitmask semantic gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.number_game_bitmask_semantic_codec import HISTORIES, INTERFACE_VERSION, diagnostics, parse_audit, parse_proposal


def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text());
    if not isinstance(value, dict): raise RuntimeError("expected object")
    return value


def verify(run_dir: Path, *, output: Path | None = None) -> dict[str, Any]:
    raw_path = run_dir / "private/RAW_RESPONSES.json"; public_path = run_dir / "LABEL_FREE_RESULT.json"
    raw, public = load(raw_path), load(public_path)
    if set(raw) != {"proposal", "audit"} or len(raw["proposal"]) != 10 or len(raw["audit"]) != 10: raise RuntimeError("raw bank incomplete")
    proposals = [parse_proposal(value, HISTORIES[index // 2]) for index, value in enumerate(raw["proposal"])]
    semantic, replay_gates = diagnostics(proposals, [parse_audit(value) for value in raw["audit"]])
    public_gates = public.get("gates") or {}
    gates = {"exact_interface": public.get("interface_version") == INTERFACE_VERSION, "raw_hash_matches": public.get("raw_response_sha256") == digest(raw_path), "semantic_replays": public.get("semantic") == semantic, "semantic_gates_replay": all(public_gates.get(key) is value for key, value in replay_gates.items()), "transport_gates_present": all(key in public_gates for key in ("exact_20_accepted_and_http", "zero_retries_reasoning_forced", "all_clean_stops", "exact_models_and_seeds", "within_stage_cap")), "status_authority_match": (public.get("status") == "mechanics_pass") is all(bool(value) for value in public_gates.values()) and public.get("number_game_targets_opened") is False and public.get("policy_endpoints_opened") is False}
    result = {"schema_version": 1, "interface_version": "number-game-bitmask-semantic-verifier-1", "status": "verification_pass" if all(gates.values()) else "verification_failed", "gates": gates, "model_calls_made": 0, "number_game_targets_opened": False, "policy_endpoints_opened": False}
    if output: output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "verification_pass": raise RuntimeError("bitmask verification failed")
    return result
