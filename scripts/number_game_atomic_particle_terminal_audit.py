#!/usr/bin/env python3
"""Audit the zero-call atomic-particle execution failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "results/nonmyopic/number_game_atomic_particle_mechanics"
BINDING = ROOT / "EXECUTION_BINDING.json"
FAILURE = ROOT / "DAILY_FAILURE_20260813.json"
RUN = ROOT / "mechanics-20260813"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-number-game-atomic-particle.json"
BINDING_SHA256 = "9c1e0dd97ac567e87c1b7560210453b1fb65480a2789efa3a32546c44335fd40"
FAILURE_SHA256 = "e47f99979457e12294d1cb8cf24dfb537e7072d07fe46d33cf652c70970621b3"
LEDGER_SHA256 = "ed97c6b40962d1af5ed4621f8915e95f1cd8e10177add70b4f5fde764108f488"
EXPECTED_USAGE = 220.334124806


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object")
    return value


def audit(*, output: Path | None = None) -> dict[str, Any]:
    binding, failure, ledger = load(BINDING), load(FAILURE), load(LEDGER)
    run_files = [path for path in RUN.rglob("*") if path.is_file()] if RUN.exists() else []
    block_rows = ledger.get("block_authorizations") or []
    gates = {
        "immutable_binding_matches": digest(BINDING) == BINDING_SHA256,
        "failure_and_ledger_hashes_match": digest(FAILURE) == FAILURE_SHA256 and digest(LEDGER) == LEDGER_SHA256,
        "exact_pre_dispatch_name_error": (
            failure.get("error_type") == "NameError"
            and failure.get("error") == "name 'active_lock' is not defined"
        ),
        "zero_cost_and_unchanged_usage": (
            float(failure.get("actual_cost_usd", -1)) == 0.0
            and float(ledger.get("execution_opening_total_usage_usd", -1)) == EXPECTED_USAGE
            and float(ledger.get("closing_total_usage_usd", -1)) == EXPECTED_USAGE
        ),
        "only_first_block_authorized_no_dispatch": (
            len(block_rows) == 1
            and block_rows[0].get("first_seed") == 202608136000
            and block_rows[0].get("last_seed") == 202608136063
            and float(block_rows[0].get("accepted_cost_before_block_usd", -1)) == 0.0
        ),
        "no_run_or_response_artifacts": (
            not run_files
            and failure.get("raw_exists") is False
            and failure.get("topology_exists") is False
            and failure.get("verification_exists") is False
            and failure.get("scientific_result_exists") is False
        ),
        "all_scientific_authority_closed": (
            failure.get("status") == "failed_closed"
            and failure.get("authorizes") == "nothing"
            and failure.get("canonical_targets_opened") is False
            and failure.get("classical_grammar_opened") is False
            and failure.get("development_opened") is False
            and failure.get("confirmation_opened") is False
            and ledger.get("stage", {}).get("status") == "failed_closed"
        ),
        "binding_declares_zero_retries": binding.get("maximum_retries") == 0,
    }
    result = {
        "schema_version": 1,
        "interface_version": "number-game-atomic-particle-terminal-audit-1",
        "status": "terminal_audit_pass" if all(gates.values()) else "terminal_audit_failed",
        "decision": "close_exact_atomic_particle_v1_execution",
        "authorizes": "prospective_distinct_successor_only" if all(gates.values()) else "nothing",
        "gates": gates,
        "model_calls_made": 0,
        "actual_cost_usd": 0.0,
        "canonical_targets_opened": False,
        "classical_grammar_opened": False,
        "policy_endpoints_opened": False,
    }
    if output:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "terminal_audit_pass":
        raise RuntimeError("atomic-particle terminal audit failed")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
