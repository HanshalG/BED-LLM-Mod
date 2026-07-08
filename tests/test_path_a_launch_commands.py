from scripts.path_a_launch_commands import build_path_a_commands


def test_path_a_launch_commands_use_gh200_singularity_and_package_builder():
    commands = build_path_a_commands()

    assert commands.constrained_sbatch.startswith(
        'BED_LLM_VLLM_KWARGS=\'{"max_num_seqs":100,"enforce_eager":false}\' '
        "BED_LLM_LOG_REASONING_TRACES=1 sbatch"
    )
    assert "--partition=gh200" in commands.constrained_sbatch
    assert "scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh" in commands.constrained_sbatch
    assert "configs/config_location_branch_decoy_local_final50_26b_a4b.yaml" in commands.constrained_sbatch
    assert "--include-myopic-controls" in commands.constrained_sbatch
    assert "--max-depth 5" in commands.constrained_sbatch

    assert "--partition=gh200" in commands.unconstrained_sbatch
    assert "BED_LLM_LOG_REASONING_TRACES=1 sbatch" in commands.unconstrained_sbatch
    assert (
        "configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml"
        in commands.unconstrained_sbatch
    )
    assert "--include-myopic-controls" in commands.unconstrained_sbatch

    assert commands.package_command.startswith("python scripts/build_path_a_package.py")
    assert "runs/loc_branch_decoy_local_constrained_final50_26b_a4b/fixed_root_depth_sweep_metrics.json" in (
        commands.package_command
    )
    assert "results/cost_vs_depth" in commands.package_command


def test_path_a_launch_commands_allow_partition_override():
    commands = build_path_a_commands(partition="msc", max_depth=3)

    assert "--partition=msc" in commands.constrained_sbatch
    assert "--max-depth 3" in commands.constrained_sbatch


def test_path_a_launch_commands_can_disable_trace_env():
    commands = build_path_a_commands(log_reasoning_traces=False, vllm_kwargs='{"max_num_seqs":32}')

    assert 'BED_LLM_VLLM_KWARGS=\'{"max_num_seqs":32}\'' in commands.constrained_sbatch
    assert "BED_LLM_LOG_REASONING_TRACES" not in commands.constrained_sbatch


def test_path_a_launch_commands_can_suffix_rerun_names():
    commands = build_path_a_commands(run_suffix="_sharedopt", job_suffix="_opt")

    assert "--job-name=loc_branch_constr26_f50_opt" in commands.constrained_sbatch
    assert "--job-name=loc_branch_uncon26_f50_opt" in commands.unconstrained_sbatch
    assert "--run-name loc_branch_decoy_local_constrained_final50_26b_a4b_sharedopt" in (
        commands.constrained_sbatch
    )
    assert "runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b_sharedopt" in (
        commands.package_command
    )
    assert "--run-name location_branch_decoy_depth_contrast_26b_a4b_sharedopt" in commands.package_command


def test_path_a_launch_commands_can_select_depth_subsets():
    commands = build_path_a_commands(
        strategy_depths="1,3,5",
        eval_depths="1,3,5",
        myopic_control_depths="3,5",
    )

    for sbatch in (commands.constrained_sbatch, commands.unconstrained_sbatch):
        assert "--strategy-depths 1,3,5" in sbatch
        assert "--eval-depths 1,3,5" in sbatch
        assert "--myopic-control-depths 3,5" in sbatch
