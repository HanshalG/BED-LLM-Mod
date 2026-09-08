import json
from pathlib import Path
import platform
import sys

import pytest

from environments.scilaws.policy_isolation import IsolationError, run_policy

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="macOS isolation backend"
)


def run(tmp_path, code, **kwargs):
    public = tmp_path / "public"
    public.mkdir(exist_ok=True)
    executable = str(Path(sys._base_executable).resolve())
    return run_policy([executable, "-I", "-c", code], public_root=public, **kwargs)


def test_public_input_works_and_environment_is_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "TEST_SECRET_NEVER_FORWARD")
    output = run(
        tmp_path,
        "import json,sys,os; print(json.dumps([json.load(sys.stdin),os.getenv('OPENROUTER_API_KEY')]))",
        input_bytes=b'{"public":1}',
    )
    assert json.loads(output) == [{"public": 1}, None]


def test_hidden_file_and_symlink_are_denied(tmp_path):
    hidden = tmp_path / "hidden.txt"
    hidden.write_text("HIDDEN_FIXTURE")
    public = tmp_path / "public"
    public.mkdir()
    link = public / "pretend-public.txt"
    link.symlink_to(hidden)
    alias = Path("/System/Volumes/Data") / hidden.relative_to("/")
    paths = (hidden, link, alias) if alias.exists() else (hidden, link)
    for path in paths:
        code = f"from pathlib import Path; p=Path({str(path)!r});\ntry: p.read_text()\nexcept PermissionError: print('denied')\nelse: raise RuntimeError('hidden file readable')"
        assert run(tmp_path, code) == b"denied\n"


def test_public_read_allowed_but_write_denied(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "task.json").write_text('{"task":1}')
    assert run(tmp_path, "print(open('task.json').read())") == b'{"task":1}\n'
    with pytest.raises(IsolationError):
        run(tmp_path, "open('task.json','w').write('changed')")
    assert (public / "task.json").read_text() == '{"task":1}'


def test_network_denied(tmp_path):
    code = "import socket\ns=socket.socket()\ntry: s.connect(('127.0.0.1',9))\nexcept PermissionError: print('denied')\nelse: raise RuntimeError('network not denied')"
    assert run(tmp_path, code) == b"denied\n"


def test_cannot_spawn_unbounded_children(tmp_path):
    code = "import os\ntry: os.fork()\nexcept PermissionError: print('denied')\nelse: raise RuntimeError('fork not denied')"
    assert run(tmp_path, code) == b"denied\n"


def test_timeout_and_output_limit(tmp_path):
    with pytest.raises(IsolationError, match="timed out"):
        run(tmp_path, "import time; time.sleep(10)", timeout=0.2)
    with pytest.raises(IsolationError, match="output limit"):
        run(tmp_path, "import os; os.write(1,b'x'*100000)", output_limit=1024)


def test_fail_closed_without_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(IsolationError, match="unavailable"):
        run(tmp_path, "print('must not run')")
