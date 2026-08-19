#!/usr/bin/env python3
"""Enforce the frozen legacy-medium serialization liveness cutoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.chembench_costed_repeat_disk_runtime import require_pushed_runtime_commit
from scripts.chembench_factored_mopen_oracle import sha256


SCHEMA_VERSION = "chembench-costed-repeat-legacy-medium-cutoff-v1"
FROZEN_PID = 23488
CUTOFF_CPU_SECONDS = 330 * 60.0
GRACE_SECONDS = 30.0
POLL_SECONDS = 10.0
LIVENESS_PATH = Path(
    "results/nonmyopic/"
    "CHEMBENCH_COSTED_REPEAT_CORRIDOR_LEGACY_MEDIUM_LIVENESS_CLARIFICATION_20260819.md"
)
LIVENESS_SHA256 = "9a63e391c16f9b3f33de52772cea30d7a1a2e62654134e4d519e7c6e958574a3"
SHARD_DIR = Path(
    "results/nonmyopic/chembench_costed_repeat_corridor/"
    "costed-v1-sharded-20260816"
)


def parse_cpu_time(value: str) -> float:
    text = value.strip()
    if not text:
        raise ValueError("process CPU time is empty")
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    fields = text.split(":")
    if len(fields) == 3:
        hours, minutes, seconds = int(fields[0]), int(fields[1]), float(fields[2])
        invalid_minutes = minutes >= 60
    elif len(fields) == 2:
        hours, minutes, seconds = 0, int(fields[0]), float(fields[1])
        invalid_minutes = False
    else:
        raise ValueError(f"unrecognized process CPU time: {value!r}")
    if min(days, hours, minutes, seconds) < 0 or invalid_minutes or seconds >= 60:
        raise ValueError(f"invalid process CPU time: {value!r}")
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds


def _run(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def process_snapshot(pid: int) -> dict[str, Any] | None:
    cpu_text = _run(["ps", "-p", str(pid), "-o", "time="])
    if not cpu_text:
        return None
    raw = _run(
        [
            "ps",
            "-p",
            str(pid),
            "-o",
            "pid=,ppid=,state=,%cpu=,%mem=,rss=,vsz=,etime=,time=,command=",
        ]
    )
    return {
        "pid": pid,
        "cpu_time": cpu_text,
        "cpu_seconds": parse_cpu_time(cpu_text),
        "ps": raw,
    }


def validate_legacy_process(snapshot: dict[str, Any]) -> None:
    command = str(snapshot.get("ps", ""))
    required = (
        "scripts/chembench_costed_repeat_corridor.py",
        "--required-commit d9559a2f",
        "--difficulty medium",
    )
    if snapshot.get("pid") != FROZEN_PID or any(item not in command for item in required):
        raise RuntimeError("frozen PID no longer identifies the legacy medium process")


def _file_binding(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "bytes": 0, "sha256": None}
    return {
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _output_binding(path: Path) -> dict[str, Any]:
    binding = _file_binding(path)
    binding["valid_json"] = False
    if binding["exists"] and binding["bytes"] > 0:
        try:
            json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        else:
            binding["valid_json"] = True
    return binding


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite cutoff artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale cutoff temporary exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _base_payload(
    *,
    status: str,
    runtime_commit: str,
    snapshot: dict[str, Any] | None,
    log_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "time": datetime.now().astimezone().isoformat(),
        "frozen_pid": FROZEN_PID,
        "cutoff_cpu_seconds": CUTOFF_CPU_SECONDS,
        "grace_seconds": GRACE_SECONDS,
        "process": snapshot,
        "log": _file_binding(log_path),
        "output": _output_binding(output_path),
        "swap": _run(["sysctl", "vm.swapusage"]),
        "runtime_implementation_commit": runtime_commit,
        "liveness_clarification": {
            "file": str(LIVENESS_PATH),
            "sha256": LIVENESS_SHA256,
        },
        "model_calls": 0,
        "network_calls": 0,
        "cost_usd": 0.0,
    }


def watch(main_root: Path, *, runtime_commit: str) -> dict[str, Any]:
    if sha256(REPO_ROOT / LIVENESS_PATH) != LIVENESS_SHA256:
        raise RuntimeError("legacy-medium liveness protocol binding mismatch")
    shard_dir = main_root / SHARD_DIR
    output_path = shard_dir / "medium.json"
    log_path = shard_dir / "medium.log"
    presignal_path = shard_dir / "medium.legacy-cutoff-presignal.json"
    terminal_path = shard_dir / "medium.legacy-cutoff-terminal.json"
    if presignal_path.exists() or terminal_path.exists():
        raise FileExistsError("legacy-medium cutoff artifacts already exist")

    while True:
        if output_path.exists() and output_path.stat().st_size > 0:
            return {"status": "legacy_output_present", "output": _output_binding(output_path)}
        snapshot = process_snapshot(FROZEN_PID)
        if snapshot is None:
            if output_path.exists() and output_path.stat().st_size > 0:
                return {
                    "status": "legacy_output_present",
                    "output": _output_binding(output_path),
                }
            payload = _base_payload(
                status="unexpected_exit_without_output",
                runtime_commit=runtime_commit,
                snapshot=None,
                log_path=log_path,
                output_path=output_path,
            )
            _write_once(terminal_path, payload)
            return payload
        validate_legacy_process(snapshot)
        if snapshot["cpu_seconds"] < CUTOFF_CPU_SECONDS:
            time.sleep(POLL_SECONDS)
            continue

        if output_path.exists() and output_path.stat().st_size > 0:
            return {"status": "legacy_output_present", "output": _output_binding(output_path)}
        presignal = _base_payload(
            status="cutoff_reached_before_signal",
            runtime_commit=runtime_commit,
            snapshot=snapshot,
            log_path=log_path,
            output_path=output_path,
        )
        _write_once(presignal_path, presignal)
        if output_path.exists() and output_path.stat().st_size > 0:
            terminal = _base_payload(
                status="output_appeared_no_signal",
                runtime_commit=runtime_commit,
                snapshot=process_snapshot(FROZEN_PID),
                log_path=log_path,
                output_path=output_path,
            )
            _write_once(terminal_path, terminal)
            return terminal

        signal_snapshot = process_snapshot(FROZEN_PID)
        if signal_snapshot is None:
            terminal = _base_payload(
                status="process_exited_before_signal",
                runtime_commit=runtime_commit,
                snapshot=None,
                log_path=log_path,
                output_path=output_path,
            )
            _write_once(terminal_path, terminal)
            return terminal
        validate_legacy_process(signal_snapshot)
        os.kill(FROZEN_PID, signal.SIGTERM)
        deadline = time.monotonic() + GRACE_SECONDS
        while process_snapshot(FROZEN_PID) is not None and time.monotonic() < deadline:
            time.sleep(1.0)
        used_sigkill = False
        if process_snapshot(FROZEN_PID) is not None:
            os.kill(FROZEN_PID, signal.SIGKILL)
            used_sigkill = True
            for _ in range(30):
                if process_snapshot(FROZEN_PID) is None:
                    break
                time.sleep(1.0)
        terminal = _base_payload(
            status="legacy_runtime_cutoff",
            runtime_commit=runtime_commit,
            snapshot=process_snapshot(FROZEN_PID),
            log_path=log_path,
            output_path=output_path,
        )
        terminal["sigterm_sent"] = True
        terminal["sigkill_sent"] = used_sigkill
        _write_once(terminal_path, terminal)
        return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    commit = require_pushed_runtime_commit(args.required_commit)
    snapshot = process_snapshot(FROZEN_PID)
    output = args.main_root / SHARD_DIR / "medium.json"
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "output_present"
                    if output.exists()
                    else "watch_required",
                    "runtime_implementation_commit": commit,
                    "process": snapshot,
                    "output": _output_binding(output),
                    "protocol_sha256": sha256(REPO_ROOT / LIVENESS_PATH),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = watch(args.main_root, runtime_commit=commit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
