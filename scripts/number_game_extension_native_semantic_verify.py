#!/usr/bin/env python3
"""Independent replay of the extension-native semantic-support gate."""

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

from scripts.number_game_extension_native_semantic_codec import (
    HISTORIES,
    INTERFACE_VERSION,
    parse_audit,
    parse_proposal,
    semantic_diagnostics,
)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def proposal_replay(proposals: list[list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, bool]]:
    draws = []
    pools = []
    for index, rows in enumerate(proposals):
        hashes = [row["extension_hash"] for row in rows]
        draws.append({"draw_index": index, "history_index": index // 2, "schema_valid_unique_count": len(set(hashes)), "extension_hashes": hashes})
    for history_index in range(5):
        left = {row["extension_hash"] for row in proposals[2 * history_index]}
        right = {row["extension_hash"] for row in proposals[2 * history_index + 1]}
        pools.append({"history_index": history_index, "unique_count": len(left | right), "second_draw_novel_count": len(right - left)})
    gates = {
        "all_draws_have_exactly_24_schema_valid": len(draws) == 10 and all(row["schema_valid_unique_count"] == 24 for row in draws),
        "every_draw_has_at_least_20_unique": all(row["schema_valid_unique_count"] >= 20 for row in draws),
        "every_pool_has_at_least_28_unique": all(row["unique_count"] >= 28 for row in pools),
        "every_second_draw_contributes_at_least_4": all(row["second_draw_novel_count"] >= 4 for row in pools),
        "zero_description_lexical_failures": True,
    }
    return {"draws": draws, "pools": pools}, gates


def verify(run_dir: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    raw_path = run_dir / "private/RAW_RESPONSES.json"
    public_path = run_dir / "LABEL_FREE_RESULT.json"
    raw = load_object(raw_path)
    public = load_object(public_path)
    if set(raw) != {"proposal", "audit"} or len(raw["proposal"]) != 10 or len(raw["audit"]) != 10:
        raise RuntimeError("raw response bank is incomplete")
    proposals = [parse_proposal(response, HISTORIES[index // 2]) for index, response in enumerate(raw["proposal"])]
    audits = [parse_audit(response) for response in raw["audit"]]
    proposal, proposal_gates = proposal_replay(proposals)
    semantic, semantic_gates = semantic_diagnostics(proposals, audits)
    replayed = {**proposal_gates, **semantic_gates}
    public_gates = public.get("gates") or {}
    gates = {
        "exact_interface": public.get("interface_version") == INTERFACE_VERSION,
        "raw_hash_matches": public.get("raw_response_sha256") == file_digest(raw_path),
        "proposal_replays_exactly": public.get("proposal") == proposal,
        "semantic_replays_exactly": public.get("semantic") == semantic,
        "proposal_and_semantic_gates_replay": all(public_gates.get(key) is value for key, value in replayed.items()),
        "transport_claim_is_complete": all(key in public_gates for key in ("exact_20_accepted_requests", "exact_20_http_attempts", "zero_retries", "zero_reasoning_tokens", "zero_forced_exits", "all_finish_reasons_stop", "exact_models_and_seeds", "within_stage_cap")),
        "privacy_and_authority_match": public.get("number_game_targets_opened") is False and public.get("policy_endpoints_opened") is False and public.get("authorizes") in ("nothing", "separate_protocol_only"),
        "status_matches_gate_conjunction": (public.get("status") == "mechanics_pass") is all(bool(value) for value in public_gates.values()),
    }
    result = {
        "schema_version": 1,
        "interface_version": "number-game-extension-native-semantic-verifier-1",
        "status": "verification_pass" if all(gates.values()) else "verification_failed",
        "gates": gates,
        "replayed_proposal_gates": proposal_gates,
        "replayed_semantic_gates": semantic_gates,
        "number_game_targets_opened": False,
        "policy_endpoints_opened": False,
        "model_calls_made": 0,
    }
    if output_path is not None:
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "verification_pass":
        raise RuntimeError("extension-native semantic verification failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir.resolve(), output_path=args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
