from scripts.path_a_remote_readiness import (
    REQUIRED_REMOTE_FILES,
    parse_remote_probe,
    payload,
    remote_probe_script,
)


def test_parse_remote_probe_detects_ready_state():
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
7
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
    assert data["active_jobs"] == 7
    assert data["missing_files"] == [REQUIRED_REMOTE_FILES[0]]


def test_remote_probe_script_is_read_only_and_quotes_remote_dir():
    script = remote_probe_script("/tmp/path with spaces")

    assert "cd '/tmp/path with spaces'" in script
    assert "squeue" in script
    assert "sinfo" in script
    assert "rm " not in script
    assert "sbatch" not in script
