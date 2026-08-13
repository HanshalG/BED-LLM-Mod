#!/usr/bin/env python3
"""Run or verify the public-image native-amd64 Debug-BED preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "swesmith-debug-bed-native-amd64-preflight-v1"
EXPECTED_REPOSITORY = "HanshalG/BED-LLM-Mod"
EXPECTED_BRANCH = "codex/location-finding-llmstrategy"
EXPECTED_EVENT = "push"
IMAGES = {
    "alpine": "alpine:3.20@sha256:c64c687cbea9300178b30c95835354e34c4e4febc4badfe27102879de0483b5e",
    "ubuntu": "ubuntu:22.04@sha256:0199853f6d6b20b0424f3c5694a72a62764f01e6a771b1eb48a4197848986c7e",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=120).stdout.strip()


def expected_controls() -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for name, image in IMAGES.items():
        for arm in (1, 2):
            nonce = f"swesmith-amd64-{name}-arm-{arm}"
            controls.append({"name": name, "arm": arm, "image": image, "nonce": nonce})
    return controls


def run(protocol: Path, workflow: Path, output: Path) -> dict[str, Any]:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    sha = os.environ.get("GITHUB_SHA", "")
    host_machine = platform.machine()
    docker_os = command("docker", "info", "--format", "{{.OSType}}")
    docker_arch = command("docker", "info", "--format", "{{.Architecture}}")
    observed: list[dict[str, Any]] = []
    for control in expected_controls():
        stdout = command(
            "docker", "run", "--rm", "--platform", "linux/amd64", control["image"],
            "sh", "-c", f"printf '%s\\n' '{control['nonce']}'; uname -m",
        )
        observed.append({**control, "stdout_lines": stdout.splitlines(), "exit_code": 0})
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "preflight_pass",
        "decision": "fresh_native_amd64_protocol_may_be_frozen",
        "bindings": {
            "protocol_sha256": sha256_file(protocol),
            "workflow_sha256": sha256_file(workflow),
            "repository": repository,
            "branch": ref_name,
            "event": event,
            "commit": sha,
        },
        "runner": {"host_machine": host_machine, "docker_os": docker_os, "docker_arch": docker_arch},
        "controls": observed,
        "privacy": {
            "swesmith_rows_read": 0,
            "instance_ids_read": 0,
            "task_payloads_read": 0,
            "private_files_read": 0,
            "endpoints_opened": 0,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0},
        "authorizes": "freeze_new_protocol_only",
    }
    gates = validate(result, protocol, workflow, expected_commit=sha)
    if not all(gates.values()):
        raise RuntimeError(f"preflight self-verification failed: {gates}")
    result["gates"] = gates
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def validate(result: dict[str, Any], protocol: Path, workflow: Path, expected_commit: str) -> dict[str, bool]:
    bindings = result.get("bindings", {})
    runner = result.get("runner", {})
    controls = result.get("controls", [])
    privacy = result.get("privacy", {})
    accounting = result.get("accounting", {})
    expected = expected_controls()
    exact_controls = len(controls) == len(expected)
    if exact_controls:
        for actual, wanted in zip(controls, expected):
            exact_controls &= actual == {
                **wanted,
                "stdout_lines": [wanted["nonce"], "x86_64"],
                "exit_code": 0,
            }
    expected_top_keys = {
        "protocol_version", "status", "decision", "bindings", "runner",
        "controls", "privacy", "accounting", "authorizes",
    }
    if "gates" in result:
        expected_top_keys.add("gates")
    return {
        "exact_public_shape": set(result) == expected_top_keys
        and set(bindings) == {"protocol_sha256", "workflow_sha256", "repository", "branch", "event", "commit"},
        "exact_identity": result.get("protocol_version") == PROTOCOL_VERSION
        and result.get("status") == "preflight_pass"
        and result.get("decision") == "fresh_native_amd64_protocol_may_be_frozen"
        and result.get("authorizes") == "freeze_new_protocol_only",
        "immutable_files": bindings.get("protocol_sha256") == sha256_file(protocol)
        and bindings.get("workflow_sha256") == sha256_file(workflow),
        "exact_remote_context": bindings.get("repository") == EXPECTED_REPOSITORY
        and bindings.get("branch") == EXPECTED_BRANCH
        and bindings.get("event") == EXPECTED_EVENT
        and bindings.get("commit") == expected_commit
        and len(expected_commit) == 40,
        "native_amd64_runner": runner == {
            "host_machine": "x86_64", "docker_os": "linux", "docker_arch": "x86_64"
        },
        "exact_control_replay": bool(exact_controls),
        "zero_private_surface": privacy == {
            "swesmith_rows_read": 0,
            "instance_ids_read": 0,
            "task_payloads_read": 0,
            "private_files_read": 0,
            "endpoints_opened": 0,
        },
        "zero_paid_or_cluster_use": accounting == {
            "openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0
        },
    }


def verify(result_path: Path, protocol: Path, workflow: Path, expected_commit: str) -> dict[str, bool]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return validate(result, protocol, workflow, expected_commit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    if args.mode == "run":
        result = run(args.protocol, args.workflow, args.result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not args.expected_commit:
        parser.error("--expected-commit is required for verify")
    gates = verify(args.result, args.protocol, args.workflow, args.expected_commit)
    print(json.dumps(gates, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
