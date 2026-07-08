from __future__ import annotations

import argparse
from dataclasses import dataclass
import shlex


@dataclass(frozen=True)
class PathACommands:
    constrained_sbatch: str
    unconstrained_sbatch: str
    package_command: str


def build_path_a_commands(
    *,
    partition: str = "gh200",
    constrained_job_name: str = "loc_branch_constr26_f50",
    unconstrained_job_name: str = "loc_branch_uncon26_f50",
    constrained_config: str = "configs/config_location_branch_decoy_local_final50_26b_a4b.yaml",
    unconstrained_config: str = "configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml",
    constrained_run_name: str = "loc_branch_decoy_local_constrained_final50_26b_a4b",
    unconstrained_run_name: str = "loc_branch_decoy_local_unconstrained_final50_26b_a4b",
    package_run_name: str = "location_branch_decoy_depth_contrast_26b_a4b",
    max_depth: int = 5,
    vllm_kwargs: str = '{"max_num_seqs":100,"enforce_eager":false}',
    log_reasoning_traces: bool = True,
    run_suffix: str = "",
    job_suffix: str = "",
    strategy_depths: str | None = None,
    eval_depths: str | None = None,
    myopic_control_depths: str | None = None,
) -> PathACommands:
    launcher = "scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh"
    if run_suffix:
        constrained_run_name = f"{constrained_run_name}{run_suffix}"
        unconstrained_run_name = f"{unconstrained_run_name}{run_suffix}"
        package_run_name = f"{package_run_name}{run_suffix}"
    if job_suffix:
        constrained_job_name = f"{constrained_job_name}{job_suffix}"
        unconstrained_job_name = f"{unconstrained_job_name}{job_suffix}"
    env_parts = [f"BED_LLM_VLLM_KWARGS={shlex.quote(vllm_kwargs)}"]
    if log_reasoning_traces:
        env_parts.append("BED_LLM_LOG_REASONING_TRACES=1")
    env_prefix = " ".join(env_parts)
    depth_args: list[str] = []
    if strategy_depths:
        depth_args.extend(["--strategy-depths", strategy_depths])
    if eval_depths:
        depth_args.extend(["--eval-depths", eval_depths])
    if myopic_control_depths:
        depth_args.extend(["--myopic-control-depths", myopic_control_depths])

    constrained_sbatch = env_prefix + " " + " ".join(
        [
            "sbatch",
            f"--partition={partition}",
            f"--job-name={constrained_job_name}",
            launcher,
            constrained_config,
            "--run-name",
            constrained_run_name,
            "--max-depth",
            str(max_depth),
            "--include-myopic-controls",
        ]
        + depth_args
    )
    unconstrained_sbatch = env_prefix + " " + " ".join(
        [
            "sbatch",
            f"--partition={partition}",
            f"--job-name={unconstrained_job_name}",
            launcher,
            unconstrained_config,
            "--run-name",
            unconstrained_run_name,
            "--max-depth",
            str(max_depth),
            "--include-myopic-controls",
        ]
        + depth_args
    )
    package_command = " ".join(
        [
            "python",
            "scripts/build_path_a_package.py",
            "--constrained",
            f"runs/{constrained_run_name}/fixed_root_depth_sweep_metrics.json",
            "--unconstrained",
            f"runs/{unconstrained_run_name}/fixed_root_depth_sweep_metrics.json",
            "--output-dir",
            "results/location_depth_sweeps",
            "--cost-dir",
            "results/cost_vs_depth",
            "--plot-dir",
            "plots/location_depth_sweeps",
            "--run-name",
            package_run_name,
        ]
    )
    return PathACommands(
        constrained_sbatch=constrained_sbatch,
        unconstrained_sbatch=unconstrained_sbatch,
        package_command=package_command,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print non-mutating Path A final sweep launch commands.")
    parser.add_argument("--partition", default="gh200")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--constrained-job-name", default="loc_branch_constr26_f50")
    parser.add_argument("--unconstrained-job-name", default="loc_branch_uncon26_f50")
    parser.add_argument(
        "--constrained-config",
        default="configs/config_location_branch_decoy_local_final50_26b_a4b.yaml",
    )
    parser.add_argument(
        "--unconstrained-config",
        default="configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml",
    )
    parser.add_argument(
        "--constrained-run-name",
        default="loc_branch_decoy_local_constrained_final50_26b_a4b",
    )
    parser.add_argument(
        "--unconstrained-run-name",
        default="loc_branch_decoy_local_unconstrained_final50_26b_a4b",
    )
    parser.add_argument("--package-run-name", default="location_branch_decoy_depth_contrast_26b_a4b")
    parser.add_argument("--vllm-kwargs", default='{"max_num_seqs":100,"enforce_eager":false}')
    parser.add_argument("--no-log-reasoning-traces", action="store_true")
    parser.add_argument("--run-suffix", default="", help="Append a suffix to both run names and package run name")
    parser.add_argument("--job-suffix", default="", help="Append a suffix to both Slurm job names")
    parser.add_argument("--strategy-depths", help="Comma-separated StrategyEIG depths to run")
    parser.add_argument("--eval-depths", help="Comma-separated rollout evaluation depths to compute")
    parser.add_argument("--myopic-control-depths", help="Comma-separated myopic-control depths to run")
    args = parser.parse_args()

    commands = build_path_a_commands(
        partition=args.partition,
        constrained_job_name=args.constrained_job_name,
        unconstrained_job_name=args.unconstrained_job_name,
        constrained_config=args.constrained_config,
        unconstrained_config=args.unconstrained_config,
        constrained_run_name=args.constrained_run_name,
        unconstrained_run_name=args.unconstrained_run_name,
        package_run_name=args.package_run_name,
        max_depth=args.max_depth,
        vllm_kwargs=args.vllm_kwargs,
        log_reasoning_traces=not args.no_log_reasoning_traces,
        run_suffix=args.run_suffix,
        job_suffix=args.job_suffix,
        strategy_depths=args.strategy_depths,
        eval_depths=args.eval_depths,
        myopic_control_depths=args.myopic_control_depths,
    )
    print("# Submit after syncing this code to the cluster checkout:")
    print(commands.constrained_sbatch)
    print(commands.unconstrained_sbatch)
    print()
    print("# Build package after both jobs finish:")
    print(commands.package_command)


if __name__ == "__main__":
    main()
