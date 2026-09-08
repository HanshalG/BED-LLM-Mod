"""macOS policy subprocess isolation; never silently fall back to unsandboxed.

Only a separately prepared public directory and installed system runtimes are
readable. The caller must ensure the public directory contains no hidden files
or hard links to them. This runner grants no model or measurement authorization.
"""

import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import tempfile


class IsolationError(RuntimeError):
    pass


def profile(public_root):
    root = Path(public_root).resolve(strict=True)
    if not root.is_dir() or root == Path("/") or root == Path.home():
        raise ValueError("separate public directory required")
    # Do not allow /System wholesale: its Data volume includes user files.
    runtime_roots = (
        "/usr/bin",
        "/usr/lib",
        "/bin",
        "/System/Library",
        "/System/Volumes/Preboot",
        "/Library/Apple",
        "/private/preboot",
        "/opt/homebrew/Cellar",
    )
    reads = " ".join(f"(subpath {json.dumps(p)})" for p in (*runtime_roots, str(root)))
    return (
        "(version 1)(deny default)(allow process-exec)"
        "(allow sysctl-read)(allow file-read-metadata)"
        f'(allow file-read-data (literal "/") {reads} '
        '(literal "/dev/null") (literal "/dev/random") (literal "/dev/urandom"))'
        '(allow file-write-data (literal "/dev/null"))'
    )


def run_policy(
    command, *, public_root, input_bytes=b"", timeout=5.0, output_limit=65536
):
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise IsolationError("tested macOS sandbox backend unavailable")
    if (
        type(command) not in (list, tuple)
        or not command
        or any(type(x) is not str for x in command)
    ):
        raise ValueError("explicit command vector required")
    if not Path(command[0]).is_absolute():
        raise ValueError("absolute executable required")
    if type(input_bytes) is not bytes or len(input_bytes) > 1048576:
        raise ValueError("input must be at most 1 MiB")
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 60
    ):
        raise ValueError("timeout must be in (0,60] seconds")
    if type(output_limit) is not int or not 1 <= output_limit <= 1048576:
        raise ValueError("output limit must be 1..1048576 bytes")
    policy = profile(public_root)
    root = str(Path(public_root).resolve())
    env = dict(
        PATH="/usr/bin:/bin",
        HOME=root,
        TMPDIR=root,
        LC_ALL="C",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )

    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_limit, output_limit))
        cpu = max(1, math.ceil(timeout))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # Regular inherited output descriptors bound disk exposure at the OS level;
    # capture pipes followed by communicate() would allow unbounded allocation.
    with (
        tempfile.TemporaryFile() as inp,
        tempfile.TemporaryFile() as out,
        tempfile.TemporaryFile() as err,
    ):
        inp.write(input_bytes)
        inp.seek(0)
        process = subprocess.Popen(
            ["/usr/bin/sandbox-exec", "-p", policy, *command],
            cwd=root,
            env=env,
            stdin=inp,
            stdout=out,
            stderr=err,
            start_new_session=True,
            preexec_fn=limits,
            close_fds=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise IsolationError("policy execution timed out") from None
        finally:
            # Also remove descendants left behind by an exited parent.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(output_limit + 1), err.read(output_limit + 1)
        if (
            process.returncode != 0
            or len(stdout) >= output_limit
            or len(stderr) >= output_limit
        ):
            # Deliberately withhold raw exceptions and filesystem paths.
            raise IsolationError("isolated policy failed or exceeded output limit")
        return stdout
