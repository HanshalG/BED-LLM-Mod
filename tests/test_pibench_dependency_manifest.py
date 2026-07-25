from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "pibench_dependency_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pibench_dependency_manifest",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_split_dependency_tasks_reproduces_frozen_balanced_splits():
    dependencies = {
        "Financier": [
            "Financier_task_006",
            "Financier_task_010",
            "Financier_task_014",
            "Financier_task_017",
            "Financier_task_019",
            "Financier_task_020",
        ],
        "law_trainee": [
            "law_trainee_task_007",
            "law_trainee_task_011",
            "law_trainee_task_013",
            "law_trainee_task_016",
            "law_trainee_task_017",
            "law_trainee_task_019",
        ],
        "marketer": [
            "marketer_task_006",
            "marketer_task_003",
            "marketer_task_012",
            "marketer_task_017",
            "marketer_task_019",
            "marketer_task_009",
        ],
        "pharmacist": [
            "pharmacist_task_005",
            "pharmacist_task_007",
            "pharmacist_task_011",
            "pharmacist_task_013",
            "pharmacist_task_015",
            "pharmacist_task_019",
        ],
        "researcher": [
            "researcher_task_006",
            "researcher_task_008",
            "researcher_task_009",
            "researcher_task_011",
            "researcher_task_013",
            "researcher_task_020",
        ],
    }
    splits = MODULE.split_dependency_tasks(dependencies)

    assert {key: len(value) for key, value in splits.items()} == {
        "mechanics": 5,
        "opportunity": 10,
        "development": 5,
        "holdout": 10,
    }
    for user_id in MODULE.USER_IDS:
        assert sum(
            task_id.startswith(f"{user_id}_task_")
            for task_id in splits["mechanics"]
        ) == 1
        assert sum(
            task_id.startswith(f"{user_id}_task_")
            for task_id in splits["opportunity"]
        ) == 2
        assert sum(
            task_id.startswith(f"{user_id}_task_")
            for task_id in splits["development"]
        ) == 1
        assert sum(
            task_id.startswith(f"{user_id}_task_")
            for task_id in splits["holdout"]
        ) == 2


def test_split_dependency_tasks_rejects_changed_source_shape():
    dependencies = {user_id: [] for user_id in MODULE.USER_IDS}
    try:
        MODULE.split_dependency_tasks(dependencies)
    except ValueError as exc:
        assert "six dependency-final tasks" in str(exc)
    else:
        raise AssertionError("changed source shape should fail")
