"""Isolated source-contract reproduction, not a ZendoWorld performance study."""

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess


SOURCE = "c18e24eae479a91f45fa0615d809f022c15281f1"
REPO = Path("external/ZendoWorld-source-audit")


def extract_methods(source):
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ZendoStateGameMaster"
    )
    names = {"__init__", "label_input"}
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if {n.name for n in methods} != names:
        raise ValueError("missing required methods")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ClassDef(
                name="IsolatedMaster",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    namespace = {"strip_trailing_var0": lambda program: None}
    exec(
        compile(ast.fix_missing_locations(module), "<source-contract>", "exec"),
        namespace,
    )
    return namespace["IsolatedMaster"]


class ConstantProgram:
    def __init__(self, value):
        self.value = value

    def eval(self, **kwargs):
        return lambda scene: self.value


def reproduce(master):
    output = io.StringIO()
    scene = object()
    with redirect_stdout(output):
        instance = master(
            ConstantProgram(True),
            0,
            [(scene, False)],
            ["unused-image-path"],
            None,
            None,
            images=False,
        )
    stored = instance.remaining_examples[0][0][1]
    queried = instance.label_input(scene)
    if stored is not False or queried is not True:
        raise AssertionError("source contract differs from inspected behavior")
    return {
        "fixture_only": True,
        "stored_label": stored,
        "membership_label_same_scene": queried,
        "mismatch_warning_emitted": "Label mismatch" in output.getvalue(),
        "disagreement_retained": stored != queried,
        "stubbed": ["program evaluation: constant True", "strip_trailing_var0: no-op"],
    }


def audit(repo):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    if git("rev-parse", "HEAD").decode().strip() != SOURCE:
        raise ValueError("source commit mismatch")
    paths = [
        "zendo/game_master.py",
        "zendo/states.py",
        "zendo/game.py",
        "DSL/zendo.py",
        "DSL/vlp_dsl_symbolic.py",
        "README.md",
    ]
    sources = {p: git("show", f"{SOURCE}:{p}") for p in paths}
    tracked = git("ls-tree", "-r", "--name-only", SOURCE).decode().splitlines()
    return {
        "schema_version": 1,
        "status": "source_contract_reproduced_not_adopted",
        "source_commit": SOURCE,
        "source_sha256": {p: hashlib.sha256(s).hexdigest() for p, s in sources.items()},
        "license_named_paths": [
            p
            for p in tracked
            if any(
                key in Path(p).name.lower() for key in ("license", "copying", "notice")
            )
        ],
        "label_contract": reproduce(extract_methods(sources[paths[0]].decode())),
        "scope": "Two extracted methods with stubs; no DSL or game-loop execution",
        "model_calls": 0,
        "cost_usd": 0,
        "scientific_outcomes_opened": False,
        "paid_authority": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_repo)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
