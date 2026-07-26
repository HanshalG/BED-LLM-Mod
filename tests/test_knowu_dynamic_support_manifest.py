from __future__ import annotations

from scripts import knowu_dynamic_support_manifest as manifest


def _family(
    index: int,
    *,
    hard: bool = True,
    profiles: int = 4,
    goal: bool = True,
) -> dict:
    return {
        "class_name": f"Task{index}",
        "file_name": f"task_{index}.py",
        "tags": ["agent-user-interaction", "hard", "preference"],
        "supported_profiles": [
            f"profile_{profile}" for profile in range(profiles)
        ],
        "profile_variants": profiles,
        "hard": hard,
        "agent_user_interaction": True,
        "static_nonempty_goal": goal,
    }


def test_split_is_reproducible_and_family_disjoint():
    eligible = {
        family["class_name"]: family
        for family in (_family(index) for index in range(12))
    }

    splits = manifest.split_families(eligible)

    assert splits == manifest.split_families(eligible)
    assert {name: len(values) for name, values in splits.items()} == {
        "mechanics": 2,
        "opportunity": 3,
        "development": 2,
        "holdout": 5,
    }
    selected = [item for values in splits.values() for item in values]
    assert len(selected) == len(set(selected))


def test_eligibility_requires_hard_multi_profile_static_goal(
    tmp_path,
    monkeypatch,
):
    families = [
        _family(1),
        _family(2, hard=False),
        _family(3, profiles=2),
        _family(4, goal=False),
    ]
    monkeypatch.setattr(
        manifest.source_audit,
        "preference_task_families",
        lambda root: families,
    )

    eligible = manifest.eligible_families(tmp_path)

    assert set(eligible) == {"Task1"}
