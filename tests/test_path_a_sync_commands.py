from scripts.path_a_sync_commands import build_rsync_command, changed_paths, parse_git_status_paths


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
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config_location_branch_decoy_local_final50_26b_a4b.yaml").write_text("")
    (
        tmp_path / "configs" / "config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml"
    ).write_text("")

    def fake_run(*args, **kwargs):
        class Result:
            stdout = " M helpers.py\n"

        return Result()

    monkeypatch.setattr("scripts.path_a_sync_commands.subprocess.run", fake_run)

    assert changed_paths() == [
        "configs/config_location_branch_decoy_local_final50_26b_a4b.yaml",
        "configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml",
        "helpers.py",
    ]
