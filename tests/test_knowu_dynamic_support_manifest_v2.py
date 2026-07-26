from __future__ import annotations

import ast

from scripts import knowu_dynamic_support_manifest_v2 as manifest


def _value(source: str):
    return ast.parse(source, mode="eval").body


def test_static_string_expression_accepts_zero_slot_fstrings():
    assert (
        manifest.static_string_expression(
            _value('("first " f"second " "third")')
        )
        == "first second third"
    )


def test_static_string_expression_rejects_runtime_interpolation():
    assert (
        manifest.static_string_expression(_value('f"value {hidden}"'))
        is None
    )


def test_static_goals_by_class_excludes_runtime_goal(tmp_path):
    task_root = (
        tmp_path / "src" / "knowu_bench" / "tasks" / "definitions"
        / "preference"
    )
    task_root.mkdir(parents=True)
    (task_root / "tasks.py").write_text(
        """
class Static:
    GOAL_REQUEST = ("hello " f"world")

class Dynamic:
    GOAL_REQUEST = f"hello {profile}"
""",
        encoding="utf-8",
    )

    goals = manifest.static_goals_by_class(tmp_path)

    assert goals == {"Static": "hello world"}
