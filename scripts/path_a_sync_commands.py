from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


DEFAULT_REMOTE = "oat0:/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z/"
GENERATED_PREFIXES = ("results/", "plots/", "runs/")
REQUIRED_SYNC_PATHS = (
    "EXPERIMENTS.md",
    "paper/main.tex",
    "paper/references.bib",
    "configs/config_location_branch_decoy_local_final50_26b_a4b.yaml",
    "configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml",
    "configs/config_location_branch_decoy_local_mpp30_norefresh_26b_a4b.yaml",
    "configs/config_location_branch_decoy_local_unconstrained_mpp30_norefresh_26b_a4b.yaml",
    "plots/constrained_oracle_robustness/branch_decoy_local_robustness_heatmap.png",
    "results/constrained_oracle/REPORT.md",
    "results/constrained_oracle_robustness/branch_decoy_local_robustness_REPORT.md",
    "results/constrained_oracle_robustness/branch_decoy_local_robustness_summary.json",
    "plots/constrained_oracle_robustness_smoke/smoke_robustness_heatmap.png",
    "results/constrained_oracle_robustness_smoke/smoke_robustness_REPORT.md",
    "results/cost_vs_depth/path_a_preregistered_cost_vs_depth.json",
    "results/cost_vs_depth/path_a_preregistered_cost_vs_depth.md",
    "results/cost_vs_depth/path_a_preregistered_cost_vs_depth.png",
    "results/ranking_fidelity/PHASE1_26B_A4B_GATE.md",
    "results/ranking_fidelity/REPORT.md",
    "results/ranking_fidelity/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_aggregate_plot.png",
    "scripts/build_path_a_package.py",
    "scripts/combine_location_fixed_root_depth_sweeps.py",
    "scripts/compare_location_depth_sweeps.py",
    "scripts/cost_vs_depth_table.py",
    "scripts/extract_location_qualitative_examples.py",
    "scripts/llm_token_usage.py",
    "scripts/location_fixed_root_depth_sweep.py",
    "scripts/path_a_launch_commands.py",
    "scripts/path_a_preflight.py",
    "scripts/path_a_remote_readiness.py",
    "scripts/path_a_sync_commands.py",
    "scripts/recover_depth_sweep_metrics.py",
    "scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh",
    "scripts/validate_experiments_ledger.py",
    "scripts/validate_path_a_package.py",
    "scripts/validate_paper_draft.py",
)


def parse_git_status_paths(status_output: str, *, include_generated: bool = False) -> list[str]:
    paths: list[str] = []
    for line in status_output.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not include_generated and path.startswith(GENERATED_PREFIXES):
            continue
        paths.append(path)
    return sorted(dict.fromkeys(paths))


def changed_paths(*, include_generated: bool = False) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=True,
    )
    paths = parse_git_status_paths(result.stdout, include_generated=include_generated)
    for path in REQUIRED_SYNC_PATHS:
        if Path(path).exists():
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def build_rsync_command(paths: list[str], *, remote: str = DEFAULT_REMOTE) -> str:
    if not paths:
        return "# no changed files to sync"
    quoted_paths = " ".join(shlex.quote(path) for path in paths)
    return f"rsync -avR {quoted_paths} {shlex.quote(remote)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a non-mutating rsync command for current Path A changes.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--include-generated", action="store_true", help="Include results/, plots/, and runs/ changes")
    parser.add_argument("--list", action="store_true", help="Print one path per line instead of an rsync command")
    args = parser.parse_args()

    paths = changed_paths(include_generated=args.include_generated)
    if args.list:
        for path in paths:
            print(path)
        return
    print("# Run from the repository root after reviewing the file list.")
    print("# This syncs git-status paths plus required Path A launch/package files.")
    print(build_rsync_command(paths, remote=args.remote))


if __name__ == "__main__":
    main()
