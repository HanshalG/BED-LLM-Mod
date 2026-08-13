#!/usr/bin/env python3
"""Independent terminal replay for the R2E-Gym structural null."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(terminal: Path, screen_path: Path, source_path: Path, manifest_path: Path, protocol_path: Path) -> dict:
    value = json.loads(terminal.read_text())
    screen = json.loads(screen_path.read_text())
    gates = {
        "exact_file_bindings": value["structural_screen"]["structural_screen_file_sha256"] == digest(screen_path)
        and value["source_result_file_sha256"] == digest(source_path)
        and value["source_manifest_file_sha256"] == digest(manifest_path)
        and value["structural_protocol_file_sha256"] == digest(protocol_path),
        "registered_structural_failure": screen["status"] == "structural_screen_failed_closed"
        and screen["screen"]["screened_count"] == 16
        and screen["screen"]["qualifying_count"] == 1
        and value["structural_screen"]["minimum_required"] == 8,
        "predicate_privacy_bound": screen["privacy"]["payload_rows_materialized"] == 16
        and screen["privacy"]["nonselected_payload_rows_materialized"] == 0
        and not screen["privacy"]["identifiers_serialized_publicly"]
        and not screen["privacy"]["endpoints_opened"],
        "downstream_sealed": not any(value["downstream"].values()),
        "zero_call_accounting": value["accounting"] == {
            "openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0
        },
        "terminal_disposition": value["status"] == "pre_mechanics_structural_null"
        and value["decision"] == "close_exact_r2e_gym_debug_bed_route"
        and value["authorizes"] == "nothing",
    }
    return {"status": "terminal_audit_pass" if all(gates.values()) else "terminal_audit_failed", "gates": gates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.terminal, args.screen, args.source, args.manifest, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "terminal_audit_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
