#!/usr/bin/env python3
"""Independently audit the public native-amd64 preflight attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "swesmith-debug-bed-native-amd64-preflight-v1"
REPOSITORY = "HanshalG/BED-LLM-Mod"
BRANCH = "codex/location-finding-llmstrategy"
IMAGES = (
    ("alpine", "alpine:3.20@sha256:c64c687cbea9300178b30c95835354e34c4e4febc4badfe27102879de0483b5e"),
    ("ubuntu", "ubuntu:22.04@sha256:0199853f6d6b20b0424f3c5694a72a62764f01e6a771b1eb48a4197848986c7e"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result: dict[str, Any], protocol: Path, workflow: Path, commit: str) -> dict[str, bool]:
    bindings = result.get("bindings")
    controls = result.get("controls")
    expected_controls = []
    for name, image in IMAGES:
        for arm in (1, 2):
            nonce = f"swesmith-amd64-{name}-arm-{arm}"
            expected_controls.append({
                "name": name, "arm": arm, "image": image, "nonce": nonce,
                "stdout_lines": [nonce, "x86_64"], "exit_code": 0,
            })
    expected_gate_names = {
        "exact_public_shape", "exact_identity", "immutable_files", "exact_remote_context",
        "native_amd64_runner", "exact_control_replay", "zero_private_surface",
        "zero_paid_or_cluster_use",
    }
    expected_top = {
        "protocol_version", "status", "decision", "bindings", "runner", "controls",
        "privacy", "accounting", "authorizes", "gates",
    }
    return {
        "exact_shape": set(result) == expected_top
        and isinstance(bindings, dict)
        and set(bindings) == {"protocol_sha256", "workflow_sha256", "repository", "branch", "event", "commit"}
        and isinstance(result.get("gates"), dict)
        and set(result["gates"]) == expected_gate_names
        and all(result["gates"].values()),
        "exact_identity": result.get("protocol_version") == VERSION
        and result.get("status") == "preflight_pass"
        and result.get("decision") == "fresh_native_amd64_protocol_may_be_frozen"
        and result.get("authorizes") == "freeze_new_protocol_only",
        "exact_binding": bindings == {
            "protocol_sha256": digest(protocol), "workflow_sha256": digest(workflow),
            "repository": REPOSITORY, "branch": BRANCH, "event": "push", "commit": commit,
        } and len(commit) == 40,
        "native_runner": result.get("runner") == {
            "host_machine": "x86_64", "docker_os": "linux", "docker_arch": "x86_64"
        },
        "exact_controls": controls == expected_controls,
        "zero_private_surface": result.get("privacy") == {
            "swesmith_rows_read": 0, "instance_ids_read": 0, "task_payloads_read": 0,
            "private_files_read": 0, "endpoints_opened": 0,
        },
        "zero_spend": result.get("accounting") == {
            "openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    gates = audit(json.loads(args.result.read_text()), args.protocol, args.workflow, args.expected_commit)
    print(json.dumps(gates, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
