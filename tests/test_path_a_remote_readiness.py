from scripts.path_a_remote_readiness import (
    DEFAULT_EXCLUDED_NODES,
    REQUIRED_REMOTE_FILES,
    _expand_slurm_nodelist,
    parse_remote_probe,
    payload,
    remote_probe_script,
)


def test_parse_remote_probe_detects_ready_state():
    assert DEFAULT_EXCLUDED_NODES == ("oat12",)
    file_lines = "\n".join(f"OK {path}" for path in REQUIRED_REMOTE_FILES)
    output = f"""ACTIVE_JOBS
0
QUEUE
             JOBID NAME PARTITION ST TIME NODELIST(REASON) NODELIST
GH200_SINFO
gh200 up infinite 3 idle oat[19,21-22]
REQUIRED_FILES
{file_lines}
"""

    readiness = parse_remote_probe(output)
    data = payload(readiness)

    assert readiness.ok_to_launch is True
    assert data["active_jobs"] == 0
    assert data["missing_files"] == []
    assert "gh200 up infinite 3 idle oat[19,21-22]" in data["gh200_lines"]


def test_parse_remote_probe_blocks_on_missing_files_or_many_jobs():
    file_lines = "\n".join(
        f"{'MISSING' if index == 0 else 'OK'} {path}"
        for index, path in enumerate(REQUIRED_REMOTE_FILES)
    )
    output = f"""ACTIVE_JOBS
3
QUEUE
job line
GH200_SINFO
gh200 up infinite 3 idle oat[19,21-22]
REQUIRED_FILES
{file_lines}
"""

    readiness = parse_remote_probe(output)
    data = payload(readiness)

    assert readiness.ok_to_launch is False
    assert data["active_jobs"] == 3
    assert data["missing_files"] == [REQUIRED_REMOTE_FILES[0]]


def test_parse_remote_probe_allows_two_existing_jobs_before_split_launch():
    file_lines = "\n".join(f"OK {path}" for path in REQUIRED_REMOTE_FILES)
    output = f"""ACTIVE_JOBS
2
QUEUE
job line
GH200_SINFO
gh200 up infinite 1 idle oat21
REQUIRED_FILES
{file_lines}
"""

    readiness = parse_remote_probe(output)

    assert readiness.ok_to_launch is True


def test_remote_readiness_blocks_when_only_excluded_gh200_node_is_idle():
    file_lines = "\n".join(f"OK {path}" for path in REQUIRED_REMOTE_FILES)
    output = f"""ACTIVE_JOBS
0
QUEUE
GH200_SINFO
gh200 up infinite 1 idle oat12
REQUIRED_FILES
{file_lines}
"""

    readiness = parse_remote_probe(output)

    assert readiness.ok_to_launch is False


def test_remote_readiness_allows_idle_gh200_range_with_usable_nodes():
    file_lines = "\n".join(f"OK {path}" for path in REQUIRED_REMOTE_FILES)
    output = f"""ACTIVE_JOBS
0
QUEUE
GH200_SINFO
gh200 up infinite 3 idle oat[12,19,21-22]
REQUIRED_FILES
{file_lines}
"""

    readiness = parse_remote_probe(output)

    assert readiness.ok_to_launch is True
    assert _expand_slurm_nodelist("oat[12,19,21-22]") == {"oat12", "oat19", "oat21", "oat22"}


def test_remote_probe_script_is_read_only_and_quotes_remote_dir():
    script = remote_probe_script("/tmp/path with spaces")

    assert "cd '/tmp/path with spaces'" in script
    assert "squeue" in script
    assert "sinfo" in script
    assert "rm " not in script
    assert "sbatch" not in script
