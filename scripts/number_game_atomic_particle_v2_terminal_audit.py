#!/usr/bin/env python3
"""Audit the atomic-particle V2 initial-diversity null."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.number_game_atomic_particle_v2_codec import DIVERSITY_CUES, extension_hash, parse_atomic


ROOT = REPO_ROOT / "results/nonmyopic/number_game_atomic_particle_v2_mechanics"
BINDING = ROOT / "EXECUTION_BINDING.json"
FAILURE = ROOT / "DAILY_FAILURE_20260813.json"
RAW = ROOT / "mechanics-20260813/private/RAW_RESPONSES.json"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-number-game-atomic-particle-v2.json"
BINDING_SHA256 = "0fc22fddb43f0c4d73e3195ea47d7efdb5cc4408bae581b29ce38af18cd15a67"
FAILURE_SHA256 = "40f637e693198be4461867c52eb0f44d0bb40f27d96b28fb474aca8906cef97d"
RAW_SHA256 = "46233c0b3d43d6f7c2f0f2c1e9601f4eb461bf3d422dbc650b4037952f3fb6d0"
LEDGER_SHA256 = "0540d54185b4858a02050f30c27e2bcd42c04cecdb7cc509de66e434fa161b2e"
EXPECTED_COST = 0.005144319999999999


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object")
    return value


def audit(*, output: Path | None = None) -> dict[str, Any]:
    failure, raw, ledger = load(FAILURE), load(RAW), load(LEDGER)
    response_rows = raw.get("responses") or []
    parsed = [parse_atomic(row["response"], ()) for row in response_rows]
    valid = [row.hypothesis for row in parsed if row.hypothesis is not None]
    frequencies = Counter(extension_hash(row) for row in valid)
    rejections = Counter(row.rejection for row in parsed if row.rejection)
    cue_rows = []
    for cue_index, cue in enumerate(DIVERSITY_CUES):
        rows = parsed[cue_index::len(DIVERSITY_CUES)]
        hypotheses = [row.hypothesis for row in rows if row.hypothesis is not None]
        cue_rows.append({
            "cue": cue,
            "valid_particles": len(hypotheses),
            "unique_extensions": len({row.extension for row in hypotheses}),
            "rejections": dict(Counter(row.rejection for row in rows if row.rejection)),
        })
    log_path = ROOT / "mechanics-20260813/run.log"
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    gates = {
        "bound_terminal_artifacts_match": (
            digest(BINDING) == BINDING_SHA256
            and digest(FAILURE) == FAILURE_SHA256
            and digest(RAW) == RAW_SHA256
            and digest(LEDGER) == LEDGER_SHA256
        ),
        "exact_first_tree_prefix": (
            len(response_rows) == 64
            and response_rows[0].get("seed") == 202608160000
            and response_rows[-1].get("seed") == 202608160063
            and len(ledger.get("block_authorizations") or []) == 1
        ),
        "clean_nonreasoning_transport": (
            len(events) == 64
            and all(
                row.get("event") == "llm_token_usage"
                and row.get("model") == "qwen/qwen3.7-plus"
                and row.get("finish_reasons") == ["stop"]
                and row.get("reasoning_enabled") is False
                and int(row.get("reasoning_tokens", -1)) == 0
                for row in events
            )
        ),
        "validity_floor_passes_but_uniqueness_fails": (
            len(valid) == 53
            and len(frequencies) == 15
            and len(valid) >= 48
            and len(frequencies) < 24
            and rejections == {"expression": 11}
        ),
        "mode_concentration_replays": (
            sorted(frequencies.values(), reverse=True)[:6] == [17, 8, 6, 6, 4, 2]
            and max(row["unique_extensions"] for row in cue_rows) == 7
            and next(row for row in cue_rows if row["cue"] == "prime or square structure")["unique_extensions"] == 1
        ),
        "cost_reconciles": abs(float(failure.get("actual_cost_usd", -1)) - EXPECTED_COST) < 1e-12 and abs(float(ledger["stage"]["actual_cost_usd"]) - EXPECTED_COST) < 1e-12,
        "ordering_and_authority_closed": (
            failure.get("status") == "failed_closed"
            and failure.get("authorizes") == "nothing"
            and failure.get("topology_exists") is False
            and failure.get("verification_exists") is False
            and failure.get("scientific_result_exists") is False
            and failure.get("canonical_targets_opened") is False
            and failure.get("classical_grammar_opened") is False
            and failure.get("development_opened") is False
            and failure.get("confirmation_opened") is False
        ),
    }
    result = {
        "schema_version": 1,
        "interface_version": "number-game-atomic-particle-v2-terminal-audit-1",
        "status": "terminal_audit_pass" if all(gates.values()) else "terminal_audit_failed",
        "decision": "close_exact_atomic_particle_v2_interface",
        "authorizes": "prospective_structurally_stratified_interface_only" if all(gates.values()) else "nothing",
        "gates": gates,
        "diagnostic": {
            "responses": len(response_rows),
            "valid_particles": len(valid),
            "unique_extensions": len(frequencies),
            "rejections": dict(rejections),
            "largest_extension_multiplicities": sorted(frequencies.values(), reverse=True),
            "by_diversity_cue": cue_rows,
        },
        "actual_cost_usd": EXPECTED_COST,
        "model_calls_made": 64,
        "canonical_targets_opened": False,
        "classical_grammar_opened": False,
        "policy_endpoints_opened": False,
    }
    if output:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "terminal_audit_pass":
        raise RuntimeError("atomic-particle V2 terminal audit failed")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
