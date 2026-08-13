#!/usr/bin/env python3
"""Audit the terminal HiddenBench serving failure without importing its runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "results/nonmyopic/hiddenbench_semantic_query_serving"
FAILURE = OUTPUT_ROOT / "DAILY_FAILURE_20260813.json"
BINDING = OUTPUT_ROOT / "EXECUTION_BINDING.json"
RUN_DIR = OUTPUT_ROOT / "serving-20260813"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-hiddenbench-semantic-query.json"
OPENING_USAGE_USD = 220.134128880


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def audit() -> dict[str, Any]:
    failure = load_object(FAILURE)
    ledger = load_object(LEDGER)
    binding = load_object(BINDING)
    gates = {
        "exact_terminal_failure": (
            failure
            == {
                "actual_cost_usd": 0.0,
                "authorizes": "nothing",
                "error": "openrouter_backoff_seconds must be positive",
                "error_type": "ValueError",
                "interface_version": "hiddenbench-semantic-query-aug13-execute-v1",
                "ledger_sha256": sha256_file(LEDGER),
                "opportunity_endpoints_opened": False,
                "registered_answers_opened": False,
                "schema_version": 1,
                "status": "failed_closed",
            }
        ),
        "exact_zero_cost_accounting": (
            ledger.get("opening_total_usage_usd") == OPENING_USAGE_USD
            and ledger.get("execution_opening_total_usage_usd")
            == ledger.get("closing_total_usage_usd")
            == 220.178352166
            and ledger.get("stage", {}).get("actual_cost_usd") == 0.0
            and ledger.get("stage", {}).get("status") == "failed_closed"
            and ledger.get("recorded_actual_spend_usd")
            == 220.178352166 - OPENING_USAGE_USD
        ),
        "execution_binding_matches_ledger": (
            ledger.get("execution_binding_sha256") == sha256_file(BINDING)
            and binding.get("accepted_requests_authorized") == 10
            and binding.get("maximum_http_attempts") == 10
            and binding.get("maximum_retries") == 0
            and binding.get("registered_answers_authorized") is False
            and binding.get("opportunity_endpoints_authorized") is False
        ),
        "no_model_or_endpoint_artifacts": (
            not (RUN_DIR / "private/RAW_RESPONSES.json").exists()
            and not (RUN_DIR / "RESULT.json").exists()
            and not (RUN_DIR / "VERIFICATION.json").exists()
            and not (OUTPUT_ROOT / "DAILY_RESULT_20260813.json").exists()
        ),
        "no_relaunch_authority": failure.get("authorizes") == "nothing",
    }
    result = {
        "schema_version": 1,
        "interface_version": "hiddenbench-semantic-query-terminal-audit-v1",
        "status": "terminal_audit_pass" if all(gates.values()) else "terminal_audit_failed",
        "decision": "close_exact_hiddenbench_semantic_query_interface",
        "gates": gates,
        "failure_sha256": sha256_file(FAILURE),
        "ledger_sha256": sha256_file(LEDGER),
        "execution_binding_sha256": sha256_file(BINDING),
        "model_calls_made": 0,
        "model_cost_usd": 0.0,
        "source_task_values_loaded": False,
        "registered_answers_opened": False,
        "opportunity_endpoints_opened": False,
        "authorizes": "nothing",
    }
    if result["status"] != "terminal_audit_pass":
        raise RuntimeError("HiddenBench terminal audit failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
