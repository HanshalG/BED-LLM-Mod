#!/usr/bin/env python3
"""Independently audit the terminal extension-native semantic gate failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_BINDING_SHA256 = "3bcd4f741e179d983fd1650ac029c6be1b1e719b9bb6b505936b098da6ed11c5"
EXPECTED_OPENING_USAGE = 220.134128880
EXPECTED_REQUESTS = 10
USAGE_EVENT = re.compile(r'^\{.*"event": "llm_token_usage".*\}$')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def audit(root: Path, *, output: Path | None = None) -> dict[str, Any]:
    run = root / "mechanics-20260813"
    failure_path = root / "DAILY_FAILURE_20260813.json"
    binding_path = root / "EXECUTION_BINDING.json"
    raw_path = run / "private/RAW_RESPONSES.json"
    log_path = run / "run.log"
    ledger_path = root.parent / "openrouter_daily_budget/2026-08-13-number-game-extension-native-semantic.json"
    failure, binding, raw, ledger = map(load, (failure_path, binding_path, raw_path, ledger_path))
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if USAGE_EVENT.match(line)]
    proposals = raw.get("proposal")
    audits = raw.get("audit")
    parseable = []
    for response in proposals if isinstance(proposals, list) else []:
        try:
            json.loads(response)
            parseable.append(True)
        except json.JSONDecodeError:
            parseable.append(False)
    local_cost = sum(float(event.get("cost_usd", 0.0)) for event in events)
    expected_recorded = max(
        float(ledger["closing_total_usage_usd"]) - EXPECTED_OPENING_USAGE,
        float(ledger["execution_opening_total_usage_usd"]) - EXPECTED_OPENING_USAGE + local_cost,
    )
    gates = {
        "exact_binding": digest(binding_path) == EXPECTED_BINDING_SHA256 and failure.get("ledger_sha256") == digest(ledger_path),
        "exact_proposal_only_transport": len(events) == EXPECTED_REQUESTS and isinstance(proposals, list) and len(proposals) == EXPECTED_REQUESTS and audits == [],
        "transport_signature_matches": sum(event.get("finish_reasons") == ["stop"] for event in events) == 7 and sum(event.get("finish_reasons") == ["length"] for event in events) == 3 and all(event.get("model") == "qwen/qwen3.7-plus" and int(event.get("reasoning_tokens", -1)) == 0 for event in events),
        "parse_signature_matches": len(parseable) == EXPECTED_REQUESTS and sum(parseable) == 7 and failure.get("error_type") == "JSONDecodeError",
        "cost_reconciles_conservatively": abs(local_cost - float(failure["actual_cost_usd"])) <= 1e-12 and abs(float(ledger["recorded_actual_spend_usd"]) - expected_recorded) <= 1e-12 and float(ledger["recorded_actual_spend_usd"]) <= 5.0,
        "terminal_authority_is_closed": failure.get("status") == "failed_closed" and failure.get("authorizes") == "nothing" and failure.get("number_game_targets_opened") is False and failure.get("policy_endpoints_opened") is False and failure.get("label_free_result_exists") is False and failure.get("verification_exists") is False,
        "no_audit_or_endpoint_artifacts": not (run / "LABEL_FREE_RESULT.json").exists() and not (run / "VERIFICATION.json").exists(),
    }
    result = {
        "schema_version": 1,
        "interface_version": "number-game-extension-native-semantic-terminal-audit-1",
        "status": "terminal_audit_pass" if all(gates.values()) else "terminal_audit_failed",
        "decision": "close_exact_array_extension_interface",
        "authorizes": "prospective_distinct_interface_design_only" if all(gates.values()) else "nothing",
        "gates": gates,
        "transport": {"accepted_requests": len(events), "clean_stop_count": sum(event.get("finish_reasons") == ["stop"] for event in events), "length_stop_count": sum(event.get("finish_reasons") == ["length"] for event in events), "parseable_response_count": sum(parseable), "audit_requests": 0, "actual_cost_usd": local_cost},
        "number_game_targets_opened": False,
        "policy_endpoints_opened": False,
        "model_calls_made": 0,
    }
    if output is not None:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "terminal_audit_pass":
        raise RuntimeError("terminal semantic-support audit failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve(), output=args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
