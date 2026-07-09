from __future__ import annotations

import argparse
from dataclasses import dataclass
import shlex


@dataclass(frozen=True)
class PathACommands:
    constrained_sbatch: str
    unconstrained_sbatch: str
    package_command: str
    package_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class PathASplitCommands:
    sbatch_commands: tuple[str, ...]
    constrained_combine_command: str
    unconstrained_combine_command: str
    package_command: str
    package_artifacts: tuple[str, ...]


def _package_artifacts(package_run_name: str) -> tuple[str, ...]:
    return (
        f"results/location_depth_sweeps/{package_run_name}_REPORT.md",
        f"results/location_depth_sweeps/{package_run_name}_summary.json",
        f"plots/location_depth_sweeps/{package_run_name}_depth_contrast.png",
        f"plots/location_depth_sweeps/{package_run_name}_headline_rmse.png",
        f"plots/location_depth_sweeps/{package_run_name}_paired_trial_rmse_deltas.png",
        f"plots/location_depth_sweeps/{package_run_name}_paired_trial_truth_log_probability_deltas.png",
        f"results/location_qualitative/{package_run_name}_constrained_qualitative_examples.md",
        f"results/location_qualitative/{package_run_name}_constrained_qualitative_example_1.png",
        f"results/cost_vs_depth/{package_run_name}_cost_vs_depth.md",
        f"results/cost_vs_depth/{package_run_name}_cost_vs_depth.png",
    )


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
    num_trials: int | None = None,
    trial_offset: int | None = None,
    total_trials: int | None = None,
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
    if num_trials is not None:
        depth_args.extend(["--num-trials", str(num_trials)])
    if trial_offset is not None:
        depth_args.extend(["--trial-offset", str(trial_offset)])
    if total_trials is not None:
        depth_args.extend(["--total-trials", str(total_trials)])

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
        package_artifacts=_package_artifacts(package_run_name),
    )


def _block_suffix(offset: int, block_trials: int) -> str:
    return f"_mpp30_b{offset:02d}_{block_trials}"


def _block_job_suffix(offset: int) -> str:
    return f"_b{offset:02d}"


def _run_name(base: str, suffix: str) -> str:
    return f"{base}{suffix}"


def _combine_command(
    *,
    run_names: list[str],
    output_run_name: str,
) -> str:
    return " ".join(
        [
            "python",
            "scripts/combine_location_fixed_root_depth_sweeps.py",
            *[f"runs/{run_name}" for run_name in run_names],
            "--output",
            f"runs/{output_run_name}/fixed_root_depth_sweep_metrics.json",
            "--report",
            f"runs/{output_run_name}/REPORT.md",
            "--plot",
            f"runs/{output_run_name}/fixed_root_depth_sweep.png",
            "--paired-delta-plot",
            f"runs/{output_run_name}/paired_trial_rmse_deltas.png",
        ]
    )


def build_split_mpp30_commands(
    *,
    partition: str = "gh200",
    offsets: tuple[int, ...] = (0, 10, 20),
    block_trials: int = 10,
    total_trials: int = 30,
    constrained_job_name: str = "loc_branch_constr26_f50",
    unconstrained_job_name: str = "loc_branch_uncon26_f50",
    constrained_run_name: str = "loc_branch_decoy_local_constrained_final50_26b_a4b",
    unconstrained_run_name: str = "loc_branch_decoy_local_unconstrained_final50_26b_a4b",
    combined_constrained_run_name: str = "loc_branch_decoy_local_constrained_mpp30_26b_a4b_split",
    combined_unconstrained_run_name: str = "loc_branch_decoy_local_unconstrained_mpp30_26b_a4b_split",
    package_run_name: str = "location_branch_decoy_depth_contrast_26b_a4b_mpp30_split",
    max_depth: int = 5,
    strategy_depths: str = "1,3,5",
    eval_depths: str = "1,3,5",
    myopic_control_depths: str = "3,5",
    vllm_kwargs: str = '{"max_num_seqs":100,"enforce_eager":false}',
    log_reasoning_traces: bool = True,
) -> PathASplitCommands:
    if not offsets:
        raise ValueError("offsets must contain at least one trial offset")
    sbatch_commands: list[str] = []
    constrained_block_runs: list[str] = []
    unconstrained_block_runs: list[str] = []
    for offset in offsets:
        if offset < 0:
            raise ValueError("offsets must be non-negative")
        suffix = _block_suffix(offset, block_trials)
        commands = build_path_a_commands(
            partition=partition,
            constrained_job_name=constrained_job_name,
            unconstrained_job_name=unconstrained_job_name,
            constrained_run_name=constrained_run_name,
            unconstrained_run_name=unconstrained_run_name,
            max_depth=max_depth,
            vllm_kwargs=vllm_kwargs,
            log_reasoning_traces=log_reasoning_traces,
            run_suffix=suffix,
            job_suffix=_block_job_suffix(offset),
            strategy_depths=strategy_depths,
            eval_depths=eval_depths,
            myopic_control_depths=myopic_control_depths,
            num_trials=block_trials,
            trial_offset=offset,
            total_trials=total_trials,
        )
        sbatch_commands.extend([commands.constrained_sbatch, commands.unconstrained_sbatch])
        constrained_block_runs.append(_run_name(constrained_run_name, suffix))
        unconstrained_block_runs.append(_run_name(unconstrained_run_name, suffix))

    package_command = " ".join(
        [
            "python",
            "scripts/build_path_a_package.py",
            "--constrained",
            f"runs/{combined_constrained_run_name}/fixed_root_depth_sweep_metrics.json",
            "--unconstrained",
            f"runs/{combined_unconstrained_run_name}/fixed_root_depth_sweep_metrics.json",
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
    return PathASplitCommands(
        sbatch_commands=tuple(sbatch_commands),
        constrained_combine_command=_combine_command(
            run_names=constrained_block_runs,
            output_run_name=combined_constrained_run_name,
        ),
        unconstrained_combine_command=_combine_command(
            run_names=unconstrained_block_runs,
            output_run_name=combined_unconstrained_run_name,
        ),
        package_command=package_command,
        package_artifacts=_package_artifacts(package_run_name),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print non-mutating Path A final sweep launch commands.")
    parser.add_argument(
        "--split-mpp30",
        action="store_true",
        help="Print the full split-MPP30 command set: six sbatches, two combines, and package command.",
    )
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
    parser.add_argument("--num-trials", type=int, help="Override number of trials in each printed run")
    parser.add_argument("--trial-offset", type=int, help="Replay and skip this many leading paired trials")
    parser.add_argument("--total-trials", type=int, help="Total intended paired trial count for RNG replay")
    parser.add_argument("--offsets", default="0,10,20", help="Comma-separated offsets for --split-mpp30")
    parser.add_argument("--block-trials", type=int, default=10, help="Trials per split-MPP30 block")
    args = parser.parse_args()

    if args.split_mpp30:
        offsets = tuple(int(part.strip()) for part in args.offsets.split(",") if part.strip())
        split_commands = build_split_mpp30_commands(
            partition=args.partition,
            offsets=offsets,
            block_trials=args.block_trials,
            total_trials=args.total_trials or 30,
            constrained_job_name=args.constrained_job_name,
            unconstrained_job_name=args.unconstrained_job_name,
            constrained_run_name=args.constrained_run_name,
            unconstrained_run_name=args.unconstrained_run_name,
            package_run_name=args.package_run_name
            if args.package_run_name != "location_branch_decoy_depth_contrast_26b_a4b"
            else "location_branch_decoy_depth_contrast_26b_a4b_mpp30_split",
            max_depth=args.max_depth,
            vllm_kwargs=args.vllm_kwargs,
            log_reasoning_traces=not args.no_log_reasoning_traces,
            strategy_depths=args.strategy_depths or "1,3,5",
            eval_depths=args.eval_depths or "1,3,5",
            myopic_control_depths=args.myopic_control_depths or "3,5",
        )
        print("# Submit after syncing this code to the cluster checkout:")
        for command in split_commands.sbatch_commands:
            print(command)
        print()
        print("# Combine completed split blocks:")
        print(split_commands.constrained_combine_command)
        print(split_commands.unconstrained_combine_command)
        print()
        print("# Build package after both combined summaries exist:")
        print(split_commands.package_command)
        print()
        print("# Expected package artifacts:")
        for artifact in split_commands.package_artifacts:
            print(artifact)
        return

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
        num_trials=args.num_trials,
        trial_offset=args.trial_offset,
        total_trials=args.total_trials,
    )
    print("# Submit after syncing this code to the cluster checkout:")
    print(commands.constrained_sbatch)
    print(commands.unconstrained_sbatch)
    print()
    print("# Build package after both jobs finish:")
    print(commands.package_command)
    print()
    print("# Expected package artifacts:")
    for artifact in commands.package_artifacts:
        print(artifact)


if __name__ == "__main__":
    main()
