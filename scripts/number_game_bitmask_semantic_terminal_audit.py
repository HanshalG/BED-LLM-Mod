#!/usr/bin/env python3
"""Audit terminal closure of the Number Game bitmask semantic gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


BINDING_SHA256 = "b5a08bf584267e5f8e17ec6e67db515ce17748cee4b852ff16be1bf493b1fab9"
OPENING_USAGE = 220.134128880
EVENT = re.compile(r'^\{.*"event": "llm_token_usage".*\}$')


def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path: Path): return json.loads(path.read_text())


def audit(root: Path, *, output: Path | None = None):
    run = root / "mechanics-20260813"; failure_path = root / "DAILY_FAILURE_20260813.json"; binding_path = root / "EXECUTION_BINDING.json"; raw_path = run / "private/RAW_RESPONSES.json"; ledger_path = root.parent / "openrouter_daily_budget/2026-08-13-number-game-bitmask-semantic.json"
    failure, raw, ledger = load(failure_path), load(raw_path), load(ledger_path)
    events = [json.loads(line) for line in (run / "run.log").read_text().splitlines() if EVENT.match(line)]
    parseable = 0
    for response in raw.get("proposal", []):
        try: json.loads(response); parseable += 1
        except json.JSONDecodeError: pass
    local_cost = sum(float(row.get("cost_usd", 0.0)) for row in events)
    expected_recorded = max(float(ledger["closing_total_usage_usd"]) - OPENING_USAGE, float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE + local_cost)
    gates = {"exact_binding_and_ledger": digest(binding_path) == BINDING_SHA256 and failure.get("ledger_sha256") == digest(ledger_path), "exact_proposal_only_transport": len(events) == 10 and len(raw.get("proposal", [])) == 10 and raw.get("audit") == [], "all_ten_length_stops": all(row.get("finish_reasons") == ["length"] and int(row.get("completion_tokens", -1)) == 3000 and int(row.get("reasoning_tokens", -1)) == 0 for row in events), "zero_parseable_complete_responses": parseable == 0 and failure.get("error_type") == "JSONDecodeError", "cost_reconciles": abs(local_cost - float(failure["actual_cost_usd"])) <= 1e-12 and abs(expected_recorded - float(ledger["recorded_actual_spend_usd"])) <= 1e-12 and expected_recorded <= 5.0, "authority_closed": failure.get("status") == "failed_closed" and failure.get("authorizes") == "nothing" and failure.get("targets_opened") is False and failure.get("endpoints_opened") is False and failure.get("label_exists") is False and failure.get("verification_exists") is False}
    result = {"schema_version": 1, "interface_version": "number-game-bitmask-semantic-terminal-audit-1", "status": "terminal_audit_pass" if all(gates.values()) else "terminal_audit_failed", "decision": "close_exact_bitmask_interface", "authorizes": "prospective_sharded_interface_design_only" if all(gates.values()) else "nothing", "gates": gates, "transport": {"accepted_requests": len(events), "length_stop_count": sum(row.get("finish_reasons") == ["length"] for row in events), "parseable_response_count": parseable, "audit_requests": 0, "actual_cost_usd": local_cost}, "targets_opened": False, "endpoints_opened": False, "model_calls_made": 0}
    if output: output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "terminal_audit_pass": raise RuntimeError("bitmask terminal audit failed")
    return result
