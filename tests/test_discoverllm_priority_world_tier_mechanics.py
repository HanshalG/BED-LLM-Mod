from __future__ import annotations

from types import SimpleNamespace

from scripts import discoverllm_priority_world_mechanics as cardinal
from scripts import discoverllm_priority_world_tier_mechanics as mechanics


def _tier_fixture():
    mapping = {"A": (0, 1, 2, 3), "B": (0, 1, 2, 3)}
    root = {}
    followup = {}
    for action in cardinal.ACTION_LABELS:
        for observation_index, observation in enumerate(
            cardinal.OBSERVATION_LABELS
        ):
            truth = cardinal.WORLD_LABELS[observation_index]
            distractor = cardinal.WORLD_LABELS[(observation_index + 1) % 4]
            key = f"{action}_{observation}"
            root[key] = {}
            followup[key] = {}
            for world in cardinal.WORLD_LABELS:
                if world == truth:
                    root[key][world] = "H"
                elif action == "B" and world == distractor:
                    root[key][world] = "H"
                else:
                    root[key][world] = "L"
                if action == "A":
                    followup[key][world] = "M"
                else:
                    followup[key][world] = "H" if world == truth else "L"
    return mapping, root, followup


def _task(task_id: str) -> cardinal.MechanicsTask:
    worlds = tuple(
        {"criterion": f"Criterion {world}", "subcriteria": []}
        for world in cardinal.WORLD_LABELS
    )
    return cardinal.MechanicsTask(
        key=task_id,
        conversation=({"role": "user", "content": "Help me."},),
        actions=("Action A", "Action B"),
        worlds=worlds,
    )


def test_tier_analysis_detects_delayed_reversal():
    mapping, root, followup = _tier_fixture()
    result = mechanics._analyze_task(
        mapping,
        root,
        followup,
        {"H": 4.0, "M": 2.0, "L": 1.0},
    )
    assert result["myopic_action"] == "A"
    assert result["nonmyopic_action"] == "B"
    assert result["root_changed"] is True
    assert result["delayed_reversal"] is True
    assert (
        result["actions"]["B"]["terminal_truth_log_posterior"]
        > result["actions"]["A"]["terminal_truth_log_posterior"]
    )


def test_fixture_mechanics_passes_all_gates(monkeypatch, tmp_path):
    tasks = [_task(task_id) for task_id in mechanics.TASK_IDS]
    monkeypatch.setattr(
        mechanics,
        "_load_tasks",
        lambda paths, manifest: tasks,
    )
    result = mechanics.run_mechanics(
        SimpleNamespace(openrouter_concurrency=3),
        paths={},
        manifest_path=tmp_path / "manifest.json",
        raw_path=tmp_path / "raw.json",
        model=mechanics.DeterministicFixtureModel(),
    )
    assert result["status"] == "passed"
    assert result["gates"]["all_pass"] is True
    assert result["summary"]["root_changes"] == 3
    assert result["summary"]["weight_robust_delayed_reversals"] == 3
    assert result["calibration"]["root"]["truth_top_tier_rate"] == 1.0
    assert result["protocol"]["semantic_content_emitted"] is False
