#!/usr/bin/env python3
"""Independent replay of overgenerated factorized Number Game mechanics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.number_game_overgenerated_factorized_codec import (
    HISTORIES,
    diagnostics,
    merge_draw,
    parse_audit,
    parse_and_select_proposals,
    parse_translation,
    translation_chunks,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("expected object")
    return value


def verify(run_dir: Path, *, output: Path | None = None):
    raw_path = run_dir / "private/RAW_RESPONSES.json"
    raw, public = load(raw_path), load(run_dir / "LABEL_FREE_RESULT.json")
    if set(raw) != {"proposal", "translation", "audit"} or len(raw["proposal"]) != 40 or len(raw["translation"]) != 30 or len(raw["audit"]) != 10:
        raise RuntimeError("raw response bank incomplete")
    selected, proposal_diagnostics = parse_and_select_proposals(raw["proposal"])
    chunks = [chunk for draw in selected for chunk in translation_chunks(draw)]
    translated = [parse_translation(value, chunk) for value, chunk in zip(raw["translation"], chunks, strict=True)]
    draws = [merge_draw(translated[3 * d : 3 * d + 3], HISTORIES[d // 2])[0] for d in range(10)]
    semantic, replay = diagnostics(draws, [parse_audit(value) for value in raw["audit"]], proposal_diagnostics)
    public_gates = public.get("gates") or {}
    serialized_public = json.dumps(public, sort_keys=True).casefold()
    gates = {
        "raw_hash_matches": public.get("raw_response_sha256") == digest(raw_path),
        "semantic_replays": public.get("semantic") == semantic,
        "semantic_gates_replay": all(public_gates.get(key) is value for key, value in replay.items()),
        "transport_gates_present": all(key in public_gates for key in ("exact_80_accepted_and_http", "zero_retries_reasoning_forced", "all_clean_stops", "exact_models_and_seeds", "within_stage_cap")),
        "public_privacy": not any(token in serialized_public for token in ('"description"', '"membership_mask"', '"observations"', '"answer"')),
        "status_authority": (
            (public.get("status") == "mechanics_pass") is all(bool(value) for value in public_gates.values())
            and public.get("targets_opened") is False
            and public.get("endpoints_opened") is False
        ),
    }
    result = {
        "schema_version": 1,
        "interface_version": "number-game-overgenerated-factorized-verifier-1",
        "status": "verification_pass" if all(gates.values()) else "verification_failed",
        "gates": gates,
        "model_calls_made": 0,
        "targets_opened": False,
        "endpoints_opened": False,
    }
    if output:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "verification_pass":
        raise RuntimeError("overgenerated factorized verification failed")
    return result
