from scripts.path_a_launch_commands import (
    DEFAULT_EXCLUDE_NODES_ARG,
    DEFAULT_EXCLUDED_NODES,
    build_path_a_commands,
    build_split_mpp30_commands,
)


def test_path_a_launch_commands_use_gh200_singularity_and_package_builder():
    commands = build_path_a_commands()

    assert DEFAULT_EXCLUDED_NODES == ("oat12",)
    assert DEFAULT_EXCLUDE_NODES_ARG == "oat12"
    assert commands.constrained_sbatch.startswith(
        'BED_LLM_VLLM_KWARGS=\'{"max_num_seqs":100,"enforce_eager":false}\' '
        "BED_LLM_LOG_REASONING_TRACES=1 sbatch"
    )
    assert "--partition=gh200" in commands.constrained_sbatch
    assert "--exclude=oat12" in commands.constrained_sbatch
    assert "scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh" in commands.constrained_sbatch
    assert "configs/config_location_branch_decoy_local_final50_26b_a4b.yaml" in commands.constrained_sbatch
    assert "--include-myopic-controls" in commands.constrained_sbatch
    assert "--max-depth 5" in commands.constrained_sbatch

    assert "--partition=gh200" in commands.unconstrained_sbatch
    assert "--exclude=oat12" in commands.unconstrained_sbatch
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
    assert "plots/location_depth_sweeps/location_branch_decoy_depth_contrast_26b_a4b_depth_contrast.png" in (
        commands.package_artifacts
    )
    assert "plots/location_depth_sweeps/location_branch_decoy_depth_contrast_26b_a4b_headline_rmse.png" in (
        commands.package_artifacts
    )
    assert (
        "plots/location_depth_sweeps/"
        "location_branch_decoy_depth_contrast_26b_a4b_paired_trial_truth_log_probability_deltas.png"
    ) in commands.package_artifacts
    assert "results/location_qualitative/location_branch_decoy_depth_contrast_26b_a4b_constrained_qualitative_examples.md" in (
        commands.package_artifacts
    )


def test_path_a_launch_commands_allow_partition_override():
    commands = build_path_a_commands(partition="msc", max_depth=3, exclude_nodes="oat19")

    assert "--partition=msc" in commands.constrained_sbatch
    assert "--exclude=oat19" in commands.constrained_sbatch
    assert "--max-depth 3" in commands.constrained_sbatch


def test_path_a_launch_commands_can_disable_node_exclusion():
    commands = build_path_a_commands(exclude_nodes=None)

    assert "--exclude=" not in commands.constrained_sbatch
    assert "--exclude=" not in commands.unconstrained_sbatch


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


def test_path_a_launch_commands_can_select_trial_blocks():
    commands = build_path_a_commands(
        num_trials=10,
        trial_offset=20,
        total_trials=30,
        run_suffix="_b20",
    )

    for sbatch in (commands.constrained_sbatch, commands.unconstrained_sbatch):
        assert "--num-trials 10" in sbatch
        assert "--trial-offset 20" in sbatch
        assert "--total-trials 30" in sbatch
    assert "runs/loc_branch_decoy_local_constrained_final50_26b_a4b_b20" in commands.package_command


def test_split_mpp30_commands_print_all_blocks_combines_and_package():
    commands = build_split_mpp30_commands()

    assert len(commands.sbatch_commands) == 6
    for offset in (0, 10, 20):
        constrained = [
            command for command in commands.sbatch_commands
            if f"--job-name=loc_branch_constr26_f50_b{offset:02d}" in command
        ]
        unconstrained = [
            command for command in commands.sbatch_commands
            if f"--job-name=loc_branch_uncon26_f50_b{offset:02d}" in command
        ]
        assert len(constrained) == 1
        assert len(unconstrained) == 1
        for command in (constrained[0], unconstrained[0]):
            assert "--partition=gh200" in command
            assert "--exclude=oat12" in command
            assert "--strategy-depths 1,3,5" in command
            assert "--eval-depths 1,3,5" in command
            assert "--myopic-control-depths 3,5" in command
            assert "--num-trials 10" in command
            assert f"--trial-offset {offset}" in command
            assert "--total-trials 30" in command

    assert "scripts/combine_location_fixed_root_depth_sweeps.py" in commands.constrained_combine_command
    assert "loc_branch_decoy_local_constrained_final50_26b_a4b_mpp30_b00_10" in (
        commands.constrained_combine_command
    )
    assert "loc_branch_decoy_local_constrained_final50_26b_a4b_mpp30_b10_10" in (
        commands.constrained_combine_command
    )
    assert "loc_branch_decoy_local_constrained_final50_26b_a4b_mpp30_b20_10" in (
        commands.constrained_combine_command
    )
    assert "loc_branch_decoy_local_unconstrained_final50_26b_a4b_mpp30_b20_10" in (
        commands.unconstrained_combine_command
    )
    assert "runs/loc_branch_decoy_local_constrained_mpp30_26b_a4b_split/fixed_root_depth_sweep_metrics.json" in (
        commands.package_command
    )
    assert "--run-name location_branch_decoy_depth_contrast_26b_a4b_mpp30_split" in commands.package_command
    assert (
        "plots/location_depth_sweeps/"
        "location_branch_decoy_depth_contrast_26b_a4b_mpp30_split_depth_contrast.png"
    ) in commands.package_artifacts
    assert (
        "results/cost_vs_depth/location_branch_decoy_depth_contrast_26b_a4b_mpp30_split_cost_vs_depth.png"
    ) in commands.package_artifacts
