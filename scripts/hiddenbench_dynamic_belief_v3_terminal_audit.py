#!/usr/bin/env python3
"""Independently audit the terminal HiddenBench dynamic-belief V3 transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "results/nonmyopic/hiddenbench_dynamic_belief_v3_mechanics"
RUN_DIR = OUTPUT_ROOT / "mechanics-20260813"
FAILURE = OUTPUT_ROOT / "DAILY_FAILURE_20260813.json"
RAW = RUN_DIR / "private/RAW_RESPONSES.json"
LOG = RUN_DIR / "run.log"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-hiddenbench-dynamic-belief-v3.json"
AUDIT = OUTPUT_ROOT / "TERMINAL_AUDIT_20260813.json"

EXPECTED_SHA256 = {
    "failure": "299173b7c0299239d32685a8541d6527b2fce1bbc304c50267e1997f8b34ca82",
    "raw": "327ebb9bbba772ce929503687c0176db543d82969d13699025a9d32c347c4c9d",
    "log": "bfc5b90b3146f7f31fa8199cc73e95fb79c6e93b834ac846fbca7408a6def5bb",
    "ledger": "c005ded5177bd862cd20c392bba37af2f03a95f099b5eae543ce707a34bdb771",
}
EXPECTED_EXECUTION_BINDING_SHA256 = (
    "2809f954a1924fbcdd757d20f5cef9fbecf725eb0cd9dfd222f6a6053b652518"
)
OPENING_USAGE_USD = 220.134128880
EXPECTED_CLOSING_USAGE_USD = 220.179020166
EXPECTED_COST_USD = 0.000668


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def audit(
    *,
    output_root: Path = OUTPUT_ROOT,
    run_dir: Path = RUN_DIR,
    failure_path: Path = FAILURE,
    raw_path: Path = RAW,
    log_path: Path = LOG,
    ledger_path: Path = LEDGER,
) -> dict[str, Any]:
    paths = {
        "failure": failure_path,
        "raw": raw_path,
        "log": log_path,
        "ledger": ledger_path,
    }
    integrity = {
        name: path.is_file() and sha256_file(path) == EXPECTED_SHA256[name]
        for name, path in paths.items()
    }

    failure = load_object(failure_path)
    raw = load_object(raw_path)
    ledger = load_object(ledger_path)
    log_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    roots = raw.get("roots")
    parsed_roots = [json.loads(value) for value in roots] if isinstance(roots, list) else []
    valid_envelopes = sum(
        isinstance(value, dict) and set(value) == {"prior", "queries"}
        for value in parsed_roots
    )
    schema_echoes = [
        value
        for value in parsed_roots
        if isinstance(value, dict) and set(value) == {"name", "schema"}
    ]
    finish_reasons = [row.get("finish_reasons") for row in log_rows]
    costs = [float(row.get("cost_usd", math.nan)) for row in log_rows]

    forbidden = {
        "label_free_result": run_dir / "LABEL_FREE_RESULT.json",
        "verification": run_dir / "VERIFICATION.json",
        "pass_token": run_dir / "LABEL_FREE_PASS_TOKEN.json",
        "registered_answers": run_dir / "private/ENDPOINTS.json",
        "endpoint_result": run_dir / "ENDPOINT_RESULT.json",
        "daily_result": output_root / "DAILY_RESULT_20260813.json",
    }
    gates = {
        "bound_terminal_artifacts": all(integrity.values()),
        "failure_is_non_authorizing": failure.get("status") == "failed_closed"
        and failure.get("authorizes") == "nothing"
        and failure.get("error_type") == "ValueError"
        and failure.get("error") == "root response has the wrong fields",
        "root_batch_stopped_before_refresh": isinstance(roots, list)
        and len(roots) == 4
        and raw.get("refreshes") == []
        and raw.get("router") == []
        and raw.get("auditor") == [],
        "two_valid_roots_two_identical_schema_echoes": valid_envelopes == 2
        and len(schema_echoes) == 2
        and schema_echoes[0] == schema_echoes[1],
        "transport_trace_matches": len(log_rows) == 4
        and finish_reasons.count(["stop"]) == 2
        and finish_reasons.count(["error"]) == 2
        and sum(cost > 0 for cost in costs) == 2
        and sum(abs(cost) <= 1e-15 for cost in costs) == 2
        and math.isclose(sum(costs), EXPECTED_COST_USD, abs_tol=1e-12),
        "ledger_reconciled": ledger.get("stage", {}).get("status") == "failed_closed"
        and ledger.get("execution_binding_sha256") == EXPECTED_EXECUTION_BINDING_SHA256
        and math.isclose(
            float(ledger.get("stage", {}).get("actual_cost_usd", math.nan)),
            EXPECTED_COST_USD,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(ledger.get("closing_total_usage_usd", math.nan)),
            EXPECTED_CLOSING_USAGE_USD,
            abs_tol=1e-12,
        )
        and float(ledger.get("recorded_actual_spend_usd", math.inf)) <= 5.0,
        "labels_and_endpoints_remain_closed": not any(path.exists() for path in forbidden.values())
        and failure.get("label_free_result_exists") is False
        and failure.get("verification_exists") is False
        and failure.get("pass_token_exists") is False
        and failure.get("registered_answers_opened") is False
        and failure.get("endpoint_result_opened") is False,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "interface_version": "hiddenbench-dynamic-belief-v3-terminal-audit-v1",
        "status": "terminal_audit_pass" if passed else "terminal_audit_failed",
        "decision": "close_hiddenbench_dynamic_belief_v3",
        "authorizes": "nothing",
        "classification": "serving_schema_transport_null",
        "scientific_interpretation": "no semantic-calibration or planning inference",
        "http_response_records": len(log_rows),
        "strict_root_envelopes": valid_envelopes,
        "schema_echoes": len(schema_echoes),
        "charged_responses": sum(cost > 0 for cost in costs),
        "zero_cost_error_responses": finish_reasons.count(["error"]),
        "actual_cost_usd": sum(costs),
        "account_wide_spend_since_boundary_usd": EXPECTED_CLOSING_USAGE_USD
        - OPENING_USAGE_USD,
        "registered_answers_opened": False,
        "endpoint_scores_opened": False,
        "integrity": {
            name: {"path": display_path(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AUDIT)
    args = parser.parse_args()
    result = audit()
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "terminal_audit_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
