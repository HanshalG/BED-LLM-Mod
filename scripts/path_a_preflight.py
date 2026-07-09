from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers import load_config
from scripts.path_a_launch_commands import build_split_mpp30_commands
from scripts.validate_path_a_package import summary_payload, validate_path_a_package


FINAL_CONFIGS = (
    Path("configs/config_location_branch_decoy_local_final50_26b_a4b.yaml"),
    Path("configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml"),
)
FINAL_LAUNCHER = Path("scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh")
SPLIT_COMBINER = Path("scripts/combine_location_fixed_root_depth_sweeps.py")


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


def _check_configs(root: Path) -> PreflightCheck:
    details: list[str] = []
    for relative_path in FINAL_CONFIGS:
        path = root / relative_path
        if not path.exists():
            return PreflightCheck("configs", False, f"missing {relative_path}")
        config = load_config(str(path))
        if config.task != "location_finding":
            return PreflightCheck("configs", False, f"{relative_path} task={config.task}")
        if config.location_num_trials < 30:
            return PreflightCheck("configs", False, f"{relative_path} has too few trials")
        if config.location_num_rounds != 6:
            return PreflightCheck("configs", False, f"{relative_path} rounds={config.location_num_rounds}")
        if config.location_strategy_num_rollouts < 16:
            return PreflightCheck("configs", False, f"{relative_path} rollouts={config.location_strategy_num_rollouts}")
        model = config.model_pairs[0].questioner.model if config.model_pairs else ""
        if "26B-A4B" not in model:
            return PreflightCheck("configs", False, f"{relative_path} model={model}")
        details.append(f"{relative_path}: {model}, trials={config.location_num_trials}")
    return PreflightCheck("configs", True, "; ".join(details))


def _check_launcher(root: Path) -> PreflightCheck:
    path = root / FINAL_LAUNCHER
    if not path.exists():
        return PreflightCheck("gh200_launcher", False, f"missing {FINAL_LAUNCHER}")
    result = subprocess.run(
        ["bash", "-n", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return PreflightCheck("gh200_launcher", False, result.stderr.strip() or "bash -n failed")
    text = path.read_text(encoding="utf-8")
    required = [
        "docker://vllm/vllm-openai:gemma4",
        "singularity exec --nv",
        "python3 scripts/location_fixed_root_depth_sweep.py",
    ]
    missing = [needle for needle in required if needle not in text]
    if missing:
        return PreflightCheck("gh200_launcher", False, f"missing launcher snippets: {missing}")
    return PreflightCheck("gh200_launcher", True, str(FINAL_LAUNCHER))


def _check_split_tools(root: Path) -> PreflightCheck:
    path = root / SPLIT_COMBINER
    if not path.exists():
        return PreflightCheck("split_tools", False, f"missing {SPLIT_COMBINER}")
    return PreflightCheck("split_tools", True, str(SPLIT_COMBINER))


def _check_commands() -> PreflightCheck:
    commands = build_split_mpp30_commands()
    text = "\n".join(
        [
            *commands.sbatch_commands,
            commands.constrained_combine_command,
            commands.unconstrained_combine_command,
            commands.package_command,
        ]
    )
    required = [
        "BED_LLM_VLLM_KWARGS=",
        "BED_LLM_LOG_REASONING_TRACES=1",
        "run_location_fixed_root_depth_sweep_gh200_singularity.sh",
        "--include-myopic-controls",
        "--strategy-depths 1,3,5",
        "--eval-depths 1,3,5",
        "--myopic-control-depths 3,5",
        "--num-trials 10",
        "--trial-offset 20",
        "--total-trials 30",
        "combine_location_fixed_root_depth_sweeps.py",
        "build_path_a_package.py",
    ]
    missing = [needle for needle in required if needle not in text]
    if missing:
        return PreflightCheck("launch_commands", False, f"missing command snippets: {missing}")
    if len(commands.sbatch_commands) != 6:
        return PreflightCheck("launch_commands", False, f"expected 6 sbatch commands, got {len(commands.sbatch_commands)}")
    return PreflightCheck("launch_commands", True, "split-MPP30 dry-run commands are ready")


def run_preflight(root: Path) -> dict[str, Any]:
    checks = [
        _check_configs(root),
        _check_launcher(root),
        _check_split_tools(root),
        _check_commands(),
    ]
    package_payload = summary_payload(validate_path_a_package(root))
    return {
        "ok": all(check.ok for check in checks),
        "checks": [
            {"name": check.name, "ok": check.ok, "detail": check.detail}
            for check in checks
        ],
        "package_validation": package_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-mutating Path A final-sweep preflight checks.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run_preflight(args.root)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for check in payload["checks"]:
            status = "ok" if check["ok"] else "fail"
            print(f"[{status}] {check['name']}: {check['detail']}")
        print(f"package_validation_ok: {payload['package_validation']['ok']}")
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
