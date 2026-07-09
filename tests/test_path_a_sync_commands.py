from scripts.path_a_sync_commands import (
    REQUIRED_SYNC_PATHS,
    build_rsync_command,
    changed_paths,
    parse_git_status_paths,
)


def test_parse_git_status_paths_excludes_generated_by_default():
    status = "\n".join(
        [
            " M helpers.py",
            "?? scripts/path_a_sync_commands.py",
            "?? results/ranking_fidelity/REPORT.md",
            "?? plots/location_depth_sweeps/demo.png",
            " R old.py -> new.py",
        ]
    )

    paths = parse_git_status_paths(status)

    assert paths == ["helpers.py", "new.py", "scripts/path_a_sync_commands.py"]


def test_parse_git_status_paths_can_include_generated():
    status = "?? results/ranking_fidelity/REPORT.md\n"

    assert parse_git_status_paths(status, include_generated=True) == [
        "results/ranking_fidelity/REPORT.md"
    ]


def test_build_rsync_command_quotes_paths_and_remote():
    command = build_rsync_command(
        ["normal.py", "path with spaces/file.py"],
        remote="host:/remote/path/",
    )

    assert command == "rsync -avR normal.py 'path with spaces/file.py' host:/remote/path/"


def test_build_rsync_command_handles_empty_paths():
    assert build_rsync_command([]) == "# no changed files to sync"


def test_changed_paths_includes_required_launch_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    for path in REQUIRED_SYNC_PATHS:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")

    def fake_run(*args, **kwargs):
        class Result:
            stdout = " M helpers.py\n"

        return Result()

    monkeypatch.setattr("scripts.path_a_sync_commands.subprocess.run", fake_run)

    assert changed_paths() == sorted([*REQUIRED_SYNC_PATHS, "helpers.py"])


def test_required_sync_paths_include_package_builder_dependencies():
    for path in (
        "scripts/build_path_a_package.py",
        "scripts/compare_location_depth_sweeps.py",
        "scripts/cost_vs_depth_table.py",
        "scripts/extract_location_qualitative_examples.py",
        "scripts/llm_token_usage.py",
        "scripts/validate_path_a_package.py",
        "scripts/validate_experiments_ledger.py",
        "scripts/validate_paper_draft.py",
    ):
        assert path in REQUIRED_SYNC_PATHS


def test_required_sync_paths_include_preflight_paper_and_ledger_inputs():
    for path in (
        "EXPERIMENTS.md",
        "paper/main.tex",
        "paper/references.bib",
    ):
        assert path in REQUIRED_SYNC_PATHS


def test_required_sync_paths_include_banked_package_evidence():
    for path in (
        "results/ranking_fidelity/REPORT.md",
        "results/ranking_fidelity/PHASE1_26B_A4B_GATE.md",
        "results/ranking_fidelity/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_aggregate_plot.png",
        "results/constrained_oracle/REPORT.md",
        "results/constrained_oracle_robustness/branch_decoy_local_robustness_REPORT.md",
        "plots/constrained_oracle_robustness/branch_decoy_local_robustness_heatmap.png",
        "results/constrained_oracle_robustness_smoke/smoke_robustness_REPORT.md",
        "plots/constrained_oracle_robustness_smoke/smoke_robustness_heatmap.png",
        "results/cost_vs_depth/path_a_preregistered_cost_vs_depth.md",
        "results/cost_vs_depth/path_a_preregistered_cost_vs_depth.png",
    ):
        assert path in REQUIRED_SYNC_PATHS
