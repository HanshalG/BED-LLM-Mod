import io
from pathlib import Path
import subprocess
import sys
import tarfile

from scripts.swebench_lite_graph_unlock_v2 import (
    build_import_graph,
    frozen_split,
    graph_followup_candidates,
    module_aliases,
    sources_from_tar,
)


def _metadata():
    rows = []
    for repo_index in range(12):
        repo = f"org/repo{repo_index}"
        for item_index in range(4):
            rows.append(
                {
                    "instance_id": f"repo{repo_index}-{item_index}",
                    "repo": repo,
                    "base_commit": f"commit-{repo_index}-{item_index}",
                }
            )
    for item_index in range(252):
        repo_index = item_index % 12
        rows.append(
            {
                "instance_id": f"extra-{item_index}",
                "repo": f"org/repo{repo_index}",
                "base_commit": f"extra-commit-{item_index}",
            }
        )
    return rows


def test_frozen_split_is_deterministic_stratified_and_exhaustive():
    rows = _metadata()

    first = frozen_split(rows)
    second = frozen_split(list(reversed(rows)))

    assert first == second
    assert len(first["opportunity"]) == 24
    assert len(first["development"]) == 12
    assert len(first["holdout"]) == 264
    all_ids = [item for values in first.values() for item in values]
    assert len(all_ids) == len(set(all_ids)) == 300
    assert {
        next(row["repo"] for row in rows if row["instance_id"] == instance_id)
        for instance_id in first["opportunity"]
    } == {f"org/repo{index}" for index in range(12)}


def test_module_aliases_handle_src_and_package_initializers():
    assert module_aliases("src/package/sub/module.py") == {
        "package.sub.module",
        "sub.module",
    }
    assert module_aliases("package/sub/__init__.py") == {
        "package.sub",
    }


def test_import_graph_resolves_absolute_and_relative_imports():
    sources = {
        "src/package/a.py": "from package import b\n",
        "src/package/b.py": "from .sub import c\n",
        "src/package/sub/c.py": "VALUE = 1\n",
    }

    outgoing, incoming = build_import_graph(sources)

    assert "src/package/b.py" in outgoing["src/package/a.py"]
    assert "src/package/sub/c.py" in outgoing["src/package/b.py"]
    assert "src/package/a.py" in incoming["src/package/b.py"]
    assert "src/package/b.py" in incoming["src/package/sub/c.py"]


def test_graph_followup_prefers_multiple_import_edges_then_path_overlap():
    sources = {
        "pkg/root_a.py": "from pkg import target\n",
        "pkg/root_b.py": "from pkg import target\n",
        "pkg/target.py": "VALUE = 1\n",
        "pkg/issue_helper.py": "VALUE = 2\n",
        "other/unrelated.py": "VALUE = 3\n",
    }
    outgoing, incoming = build_import_graph(sources)

    selected = graph_followup_candidates(
        "issue helper behavior",
        sources,
        outgoing,
        incoming,
        ["pkg/root_a.py", "pkg/root_b.py"],
        top_k=2,
    )

    assert selected[0] == "pkg/target.py"
    assert "other/unrelated.py" not in selected


def test_sources_from_tar_reads_python_and_applies_character_cap():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in (
            ("pkg/a.py", b"A" * 200_010),
            ("pkg/readme.txt", b"ignored"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    sources = sources_from_tar(buffer.getvalue())

    assert set(sources) == {"pkg/a.py"}
    assert len(sources["pkg/a.py"]) == 200_000


def test_standalone_help_resolves_repo_imports():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "swebench_lite_graph_unlock_v2.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "graph-conditioned file-retrieval opportunity" in result.stdout
