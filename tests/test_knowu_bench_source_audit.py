from __future__ import annotations

import ast
from pathlib import Path

from scripts import knowu_bench_source_audit as audit


def test_literal_assignments_extract_profile_task_metadata():
    node = ast.parse(
        """
class Example:
    task_tags = {"hard", "preference", "agent-user-interaction"}
    supported_profiles = {"user", "student", "developer"}
    GOAL_REQUEST = "Do the personalized task."
"""
    ).body[0]

    values = audit._literal_assignments(node)

    assert values["GOAL_REQUEST"] == "Do the personalized task."
    assert values["supported_profiles"] == {
        "user",
        "student",
        "developer",
    }


def test_preference_task_families_apply_default_profile_cross_product(
    tmp_path: Path,
):
    task_root = (
        tmp_path / "src" / "knowu_bench" / "tasks" / "definitions"
        / "preference"
    )
    task_root.mkdir(parents=True)
    (task_root / "task.py").write_text(
        """
class Example:
    task_tags = {"hard", "preference", "agent-user-interaction"}
    GOAL_REQUEST = "Do it."
""",
        encoding="utf-8",
    )

    families = audit.preference_task_families(tmp_path)

    assert len(families) == 1
    assert families[0]["profile_variants"] == 4
    assert families[0]["hard"]
    assert families[0]["agent_user_interaction"]


def test_leaf_paths_preserve_semantic_profile_structure():
    paths = audit._leaf_paths(
        {
            "preferences": {
                "diet": {"drink": "tea", "allergies": ["peanut"]}
            },
            "identity": {"age": 34},
        }
    )

    assert paths == {
        "preferences.diet.drink",
        "preferences.diet.allergies",
        "identity.age",
    }
