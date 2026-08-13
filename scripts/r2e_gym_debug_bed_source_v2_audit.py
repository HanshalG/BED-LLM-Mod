#!/usr/bin/env python3
"""Independent V2 correction audit for R2E-Gym Debug-BED source admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRODUCER_PATH = ROOT / "scripts/r2e_gym_debug_bed_source_audit.py"
SPEC = importlib.util.spec_from_file_location("r2e_source_producer", PRODUCER_PATH)
assert SPEC and SPEC.loader
PRODUCER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRODUCER)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def audit(
    r2e_root: Path,
    data_root: Path,
    debuggym_root: Path,
    protocol: Path,
    v1_manifest_path: Path,
    v1_result_path: Path,
    v2_result_path: Path | None = None,
) -> dict[str, Any]:
    v1_manifest = json.loads(v1_manifest_path.read_text())
    v1_result = json.loads(v1_result_path.read_text())
    manifest, result = PRODUCER.audit(r2e_root, data_root, debuggym_root, protocol)
    pdb_text = (debuggym_root / PRODUCER.DEBUG_FILES["debuggym_pdb"][0]).read_text()
    manifest_hash = digest(canonical(manifest))
    v2_result = json.loads(v2_result_path.read_text()) if v2_result_path else None
    gates = {
        "v1_was_single_gate_audit_null": v1_result.get("status") == "source_failed_closed"
        and [name for name, passed in v1_result.get("gates", {}).items() if not passed] == ["native_debugger_contract"],
        "v1_manifest_reproduced": manifest_hash == v1_result.get("manifest_sha256")
        == "bc1c4eb4da8adf91f0fcedab3e4ab8c4c20e17dc0c67ee052e59ea72eacf2dbc",
        "released_pdb_contract": all(token in pdb_text for token in (
            "class PDBTool", "def start_pdb(", "def restart_pdb(", "def interact_with_pdb("
        )),
        "producer_source_pass": result.get("status") == "source_pass" and all(result.get("gates", {}).values()),
        "zero_call_privacy_boundary": result.get("accounting") == {
            "openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0
        } and not result.get("privacy", {}).get("task_payload_columns_materialized")
        and not result.get("privacy", {}).get("endpoints_opened"),
    }
    if v2_result is not None:
        gates["v2_was_serialization_only_null"] = (
            v2_result.get("status") == "source_failed_closed"
            and [name for name, ok in v2_result.get("gates", {}).items() if not ok]
            == ["v1_manifest_reproduced"]
            and v2_result.get("v2_manifest_sha256") == v1_result.get("manifest_sha256")
        )
    passed = all(gates.values())
    return {
        "protocol_version": (
            "r2e-gym-debug-bed-source-v3-canonical-replay"
            if v2_result is not None else "r2e-gym-debug-bed-source-v2-correction"
        ),
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "structural_screen_protocol_authorized" if passed else "close_exact_r2e_gym_debug_bed_source_v2",
        "correction_protocol_sha256": digest(protocol.read_bytes()),
        "v1_manifest_file_sha256": digest(v1_manifest_path.read_bytes()),
        "v1_result_file_sha256": digest(v1_result_path.read_bytes()),
        "v2_manifest_sha256": manifest_hash,
        "gates": gates,
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0},
        "authorizes": "separately_frozen_exact_row_structural_screen_only" if passed else "nothing",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2e-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--debuggym-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v1-manifest", type=Path, required=True)
    parser.add_argument("--v1-result", type=Path, required=True)
    parser.add_argument("--v2-result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.r2e_root, args.data_root, args.debuggym_root, args.protocol,
        args.v1_manifest, args.v1_result, args.v2_result,
    )
    manifest, _ = PRODUCER.audit(args.r2e_root, args.data_root, args.debuggym_root, args.protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "SOURCE_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
