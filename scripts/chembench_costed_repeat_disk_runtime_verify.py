#!/usr/bin/env python3
"""Produce the frozen disk-runtime equivalence evidence manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.chembench_costed_repeat_corridor import (
    DISK_RUNTIME_SHA256,
    SCIENTIFIC_IMPLEMENTATION_COMMIT,
    _proposal_records_hash,
)
from scripts.chembench_costed_repeat_disk_runtime import (
    DiskRuntimeStores,
    SqliteCanonicalMapping,
)
from scripts.chembench_factored_mopen_oracle import sha256


SCHEMA_VERSION = "chembench-costed-repeat-disk-runtime-equivalence-v1"
MEMORY_RECORDS = 100_000
MAXIMUM_PEAK_RSS_MIB = 192.0
BOUND_FILES = (
    "environments/chembench_mopen/compositional.py",
    "scripts/chembench_costed_repeat_corridor.py",
    "scripts/chembench_costed_repeat_disk_runtime.py",
    "scripts/chembench_costed_repeat_disk_runtime_verify.py",
    "tests/test_chembench_costed_repeat_corridor.py",
)


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / (1024.0**2)


def _memory_child(path: Path) -> dict[str, Any]:
    records = SqliteCanonicalMapping(path, cache_mib=8)
    try:
        for index in range(MEMORY_RECORDS):
            key = hashlib.sha256(str(index).encode()).hexdigest()
            records.record_once(key, (index % 17, (index + 3) % 23, index % 5))
        digest = _proposal_records_hash(records)
        records._connection.commit()
        database_bytes = path.stat().st_size
        return {
            "records": len(records),
            "records_sha256": digest,
            "database_bytes": database_bytes,
            "peak_rss_mib": _peak_rss_mib(),
        }
    finally:
        records.close()


def _run_memory_child(path: Path) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--memory-child", str(path)]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def build_manifest() -> dict[str, Any]:
    test_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_chembench_costed_repeat_corridor.py",
    ]
    tests = subprocess.run(
        test_command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    with tempfile.TemporaryDirectory(prefix="chembench-disk-runtime-verify-") as root:
        memory = _run_memory_child(Path(root) / "records.sqlite3")
    tests_passed = tests.returncode == 0 and "passed" in tests.stdout
    bounded_memory = (
        memory["records"] == MEMORY_RECORDS
        and 0.0 < memory["peak_rss_mib"] <= MAXIMUM_PEAK_RSS_MIB
    )
    conditions = {
        "canonical_hash_exact": tests_passed,
        "duplicate_detection_exact": tests_passed,
        "policy_results_exact": tests_passed,
        "proposal_replay_exact": tests_passed,
        "crn_trajectories_exact": tests_passed,
        "bounded_memory_passed": bounded_memory,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(conditions.values()) else "failed_closed",
        "scientific_implementation_commit": SCIENTIFIC_IMPLEMENTATION_COMMIT,
        "runtime_mode": DiskRuntimeStores.mode,
        "protocol_sha256": DISK_RUNTIME_SHA256,
        "conditions": conditions,
        "test": {
            "command": test_command,
            "returncode": tests.returncode,
            "stdout": tests.stdout.strip(),
            "stderr": tests.stderr.strip(),
        },
        "memory": {
            **memory,
            "maximum_peak_rss_mib": MAXIMUM_PEAK_RSS_MIB,
        },
        "file_sha256": {name: sha256(REPO_ROOT / name) for name in BOUND_FILES},
        "model_calls": 0,
        "network_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--memory-child", type=Path)
    args = parser.parse_args()
    if args.memory_child is not None:
        print(json.dumps(_memory_child(args.memory_child), sort_keys=True))
        return
    if args.output is None:
        raise ValueError("--output is required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output),
                "output_sha256": sha256(args.output),
                "memory": manifest["memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if manifest["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
