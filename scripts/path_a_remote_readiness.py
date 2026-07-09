from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.path_a_launch_commands import DEFAULT_EXCLUDED_NODES
from scripts.path_a_sync_commands import REQUIRED_SYNC_PATHS


DEFAULT_HOST = "oat0"
DEFAULT_REMOTE_DIR = "/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z"
REQUIRED_REMOTE_FILES = REQUIRED_SYNC_PATHS
MAX_ACTIVE_JOBS = 8
SPLIT_MPP30_JOB_COUNT = 6
MAX_ACTIVE_JOBS_BEFORE_SPLIT_LAUNCH = MAX_ACTIVE_JOBS - SPLIT_MPP30_JOB_COUNT


def _expand_slurm_nodelist(text: str) -> set[str]:
    nodes: set[str] = set()
    bracket_pattern = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]*?)\[([0-9,\-]+)\]")
    for match in bracket_pattern.finditer(text):
        prefix = match.group(1)
        for part in match.group(2).split(","):
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                width = max(len(start_text), len(end_text))
                for number in range(int(start_text), int(end_text) + 1):
                    nodes.add(f"{prefix}{number:0{width}d}")
            else:
                nodes.add(f"{prefix}{part}")
    text_without_brackets = bracket_pattern.sub(" ", text)
    nodes.update(re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\d+\b", text_without_brackets))
    return nodes


def _has_usable_idle_gh200(lines: list[str], *, excluded_nodes: tuple[str, ...] = DEFAULT_EXCLUDED_NODES) -> bool:
    excluded = set(excluded_nodes)
    for line in lines:
        if "idle" not in line:
            continue
        nodelist = line.split()[-1] if line.split() else ""
        nodes = _expand_slurm_nodelist(nodelist)
        if not nodes:
            return True
        if any(node not in excluded for node in nodes):
            return True
    return False


@dataclass(frozen=True)
class RemoteReadiness:
    active_jobs: int | None
    queue_lines: list[str]
    gh200_lines: list[str]
    files: dict[str, bool]

    @property
    def ok_to_launch(self) -> bool:
        return not launch_blockers(self)


def launch_blockers(readiness: RemoteReadiness) -> list[str]:
    blockers: list[str] = []
    if readiness.active_jobs is None:
        blockers.append("could not parse active job count")
    elif readiness.active_jobs > MAX_ACTIVE_JOBS_BEFORE_SPLIT_LAUNCH:
        blockers.append(
            f"active job count {readiness.active_jobs} exceeds "
            f"{MAX_ACTIVE_JOBS_BEFORE_SPLIT_LAUNCH} allowed before six-job split launch"
        )
    if not _has_usable_idle_gh200(readiness.gh200_lines):
        blockers.append(
            "no idle usable GH200 node after excluding "
            + ",".join(DEFAULT_EXCLUDED_NODES)
        )
    missing_files = [path for path, exists in readiness.files.items() if not exists]
    if missing_files:
        blockers.append(f"{len(missing_files)} required file(s) missing on remote checkout")
    return blockers


def remote_probe_script(remote_dir: str = DEFAULT_REMOTE_DIR) -> str:
    file_args = " ".join(shlex.quote(path) for path in REQUIRED_REMOTE_FILES)
    return f"""set -euo pipefail
cd {shlex.quote(remote_dir)}
echo ACTIVE_JOBS
squeue -h -u hanyal -t PENDING,RUNNING,CONFIGURING,COMPLETING | wc -l
echo QUEUE
squeue -u hanyal -o "%.18i %.40j %.20P %.2t %.12M %.60R %.50N" || true
echo GH200_SINFO
sinfo -h -p gh200 -o "%P %a %l %D %t %N" | sort | uniq || true
echo REQUIRED_FILES
for p in {file_args}; do
  if [ -e "$p" ]; then
    printf 'OK %s\\n' "$p"
  else
    printf 'MISSING %s\\n' "$p"
  fi
done
"""


def parse_remote_probe(output: str) -> RemoteReadiness:
    sections: dict[str, list[str]] = {
        "ACTIVE_JOBS": [],
        "QUEUE": [],
        "GH200_SINFO": [],
        "REQUIRED_FILES": [],
    }
    current: str | None = None
    for line in output.splitlines():
        if line in sections:
            current = line
            continue
        if current is not None:
            sections[current].append(line)

    active_jobs: int | None = None
    for line in sections["ACTIVE_JOBS"]:
        stripped = line.strip()
        if stripped:
            try:
                active_jobs = int(stripped)
            except ValueError:
                active_jobs = None
            break

    files: dict[str, bool] = {}
    for line in sections["REQUIRED_FILES"]:
        if line.startswith("OK "):
            files[line[3:]] = True
        elif line.startswith("MISSING "):
            files[line[8:]] = False
    for path in REQUIRED_REMOTE_FILES:
        files.setdefault(path, False)

    return RemoteReadiness(
        active_jobs=active_jobs,
        queue_lines=[line for line in sections["QUEUE"] if line.strip()],
        gh200_lines=[line for line in sections["GH200_SINFO"] if line.strip()],
        files=files,
    )


def payload(readiness: RemoteReadiness) -> dict[str, Any]:
    return {
        "ok_to_launch": readiness.ok_to_launch,
        "launch_blockers": launch_blockers(readiness),
        "active_jobs": readiness.active_jobs,
        "queue_lines": readiness.queue_lines,
        "gh200_lines": readiness.gh200_lines,
        "missing_files": [path for path, exists in readiness.files.items() if not exists],
        "present_files": [path for path, exists in readiness.files.items() if exists],
    }


def check_remote(host: str = DEFAULT_HOST, remote_dir: str = DEFAULT_REMOTE_DIR) -> RemoteReadiness:
    result = subprocess.run(
        ["ssh", host, "bash -s"],
        input=remote_probe_script(remote_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ssh {host} failed with code {result.returncode}")
    return parse_remote_probe(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Path A remote launch readiness check.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    readiness = check_remote(args.host, args.remote_dir)
    data = payload(readiness)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"ok_to_launch: {data['ok_to_launch']}")
        print(f"active_jobs: {data['active_jobs']}")
        print("launch_blockers:")
        for blocker in data["launch_blockers"]:
            print(f"  {blocker}")
        print("gh200:")
        for line in data["gh200_lines"]:
            print(f"  {line}")
        print("missing_files:")
        for path in data["missing_files"]:
            print(f"  {path}")
    raise SystemExit(0 if data["ok_to_launch"] else 1)


if __name__ == "__main__":
    main()
