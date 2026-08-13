#!/usr/bin/env python3
"""V2 metadata correction for DebugGym effective SWE-smith splits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "scripts" / "swesmith_debug_bed_source_audit.py"
SPEC = importlib.util.spec_from_file_location("swesmith_v1", V1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load V1 source audit")
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)

PROTOCOL_VERSION = "swesmith-debug-bed-source-v2"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def audit(debug_root: Path, data_root: Path, protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    v1_manifest, v1_result = V1.audit(debug_root, data_root, protocol_path)
    config_lists = V1.parse_simple_yaml_lists(debug_root / V1.BOUND_FILES["split_config"][0])
    excluded = set(config_lists["excluded"])
    raw_development = config_lists["train-789"]
    raw_confirmation = config_lists["test-125"]
    development = [value for value in raw_development if value not in excluded]
    confirmation = [value for value in raw_confirmation if value not in excluded]

    old_gate = v1_result["gates"].pop("official_development_confirmation_valid")
    if old_gate:
        raise ValueError("V2 correction is invalid because V1 split gate unexpectedly passed")
    v1_result["gates"]["raw_official_lists_valid"] = (
        len(raw_development) == 789
        and len(raw_confirmation) == 125
        and len(set(raw_development)) == len(raw_development)
        and len(set(raw_confirmation)) == len(raw_confirmation)
        and not (set(raw_development) & set(raw_confirmation))
    )
    v1_result["gates"]["effective_official_lists_match_loader"] = (
        len(development) == 760
        and len(confirmation) == 121
        and not (set(development) & set(confirmation))
        and not (excluded & set(development))
        and not (excluded & set(confirmation))
        and "filter_problems(id2idx, problems, custom_splits, excluded_ids)" in (debug_root / V1.BOUND_FILES["swe_smith_environment"][0]).read_text(encoding="utf-8")
    )
    passed = all(v1_result["gates"].values())
    v1_manifest["protocol_version"] = PROTOCOL_VERSION
    v1_manifest["split_counts"]["development"] = len(development)
    v1_manifest["split_counts"]["confirmation"] = len(confirmation)
    v1_manifest["split_counts"]["raw_development"] = len(raw_development)
    v1_manifest["split_counts"]["raw_confirmation"] = len(raw_confirmation)
    v1_manifest["ordered_split_id_sha256"]["development"] = V1.ordered_hash(development)
    v1_manifest["ordered_split_id_sha256"]["confirmation"] = V1.ordered_hash(confirmation)
    v1_manifest["ordered_split_id_sha256"]["raw_development"] = V1.ordered_hash(raw_development)
    v1_manifest["ordered_split_id_sha256"]["raw_confirmation"] = V1.ordered_hash(raw_confirmation)
    v1_result.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "status": "source_pass" if passed else "source_failed_closed",
            "decision": "execution_mechanics_authorized" if passed else "close_exact_swesmith_debug_bed_source_v2",
            "protocol_sha256": sha256_file(protocol_path),
            "manifest_sha256": sha256_bytes(canonical_bytes(v1_manifest)),
            "authorizes": "zero_model_call_mechanics_only" if passed else "nothing",
            "v1_split_gate_passed": False,
        }
    )
    return v1_manifest, v1_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, result = audit(args.debug_root.resolve(), args.data_root.resolve(), args.protocol.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "SOURCE_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
