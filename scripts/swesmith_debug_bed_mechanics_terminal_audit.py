#!/usr/bin/env python3
"""Verify the fail-closed SWE-smith Debug-BED mechanics terminal prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "mechanics_protocol_sha256": "6a786c696e7288ab54b2fe4b82602ff82d5d7ea85bdb79be6c630aeb6580b206",
    "source_manifest_file_sha256": "9358c865ec1f952560aece523d46b90aba0b9ec26423861f391100ffc53a3c9d",
    "source_audit_file_sha256": "07a3730a4c443fcb5de79f2e536bf6cca38b4afde68834915ff22f5fb666de0e",
    "source_v2_implementation_sha256": "0f65e55306ca7a26e1b0daadaf4497d8a6a72aaf40a9953f679142f38c54653c",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, terminal_result: Path) -> dict:
    result = json.loads(terminal_result.read_text(encoding="utf-8"))
    bindings = {
        "mechanics_protocol_sha256": sha256_file(root / "results/nonmyopic/SWESMITH_DEBUG_BED_EXECUTION_MECHANICS_PROTOCOL_20260813.md"),
        "source_manifest_file_sha256": sha256_file(root / "results/nonmyopic/swesmith_debug_bed_source_v2/MANIFEST.json"),
        "source_audit_file_sha256": sha256_file(root / "results/nonmyopic/swesmith_debug_bed_source_v2/SOURCE_AUDIT.json"),
        "source_v2_implementation_sha256": sha256_file(root / "scripts/swesmith_debug_bed_source_v2_audit.py"),
    }
    gates = {
        "dependencies_match": bindings == EXPECTED,
        "exact_mechanics_population_accounted": result.get("mechanics_task_count") == 8,
        "aggregate_hunk_eligibility_accounted": result.get("structurally_eligible_count") == 6 and result.get("structurally_invalid_count") == 2,
        "first_eligible_two_arm_failure_exact": result.get("attempted_eligible_tasks") == 1 and result.get("fresh_arm_exit_codes") == [133, 133] and result.get("fresh_arm_stdout_bytes") == [0, 0],
        "architecture_mismatch_bound": result.get("mechanics_image_count") == 7 and result.get("mechanics_image_platform_counts") == {"linux/amd64": 7} and result.get("execution_host_platform") == "linux/arm64",
        "scientific_stages_closed": all(result.get(key) is False for key in ("test_matrix_opened", "pdb_handshake_opened", "planner_opened", "patch_endpoint_opened", "opportunity_opened", "development_opened", "confirmation_opened")),
        "zero_calls_and_cost": result.get("openrouter_calls") == 0 and result.get("openrouter_cost_usd") == 0.0 and result.get("cluster_use") == 0,
        "terminal_decision": result.get("status") == "infrastructure_failed_closed" and result.get("decision") == "close_exact_swesmith_debug_bed_mechanics",
    }
    return {"status": "audit_pass" if all(gates.values()) else "audit_failed", "bindings": bindings, "gates": gates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--terminal-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.terminal_result.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "audit_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
