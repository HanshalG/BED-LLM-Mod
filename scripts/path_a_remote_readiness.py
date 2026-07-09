from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any


DEFAULT_HOST = "oat0"
DEFAULT_REMOTE_DIR = "/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z"
REQUIRED_REMOTE_FILES = (
    "configs/config_location_branch_decoy_local_final50_26b_a4b.yaml",
    "configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml",
    "scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh",
    "scripts/location_fixed_root_depth_sweep.py",
    "scripts/combine_location_fixed_root_depth_sweeps.py",
    "scripts/build_path_a_package.py",
    "scripts/path_a_preflight.py",
    "scripts/path_a_launch_commands.py",
    "scripts/validate_path_a_package.py",
)


@dataclass(frozen=True)
class RemoteReadiness:
    active_jobs: int | None
    queue_lines: list[str]
    gh200_lines: list[str]
    files: dict[str, bool]

    @property
    def ok_to_launch(self) -> bool:
        if self.active_jobs is None or self.active_jobs > 6:
            return False
        if not any("idle" in line for line in self.gh200_lines):
            return False
        return all(self.files.values())


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
        print("gh200:")
        for line in data["gh200_lines"]:
            print(f"  {line}")
        print("missing_files:")
        for path in data["missing_files"]:
            print(f"  {path}")
    raise SystemExit(0 if data["ok_to_launch"] else 1)


if __name__ == "__main__":
    main()
