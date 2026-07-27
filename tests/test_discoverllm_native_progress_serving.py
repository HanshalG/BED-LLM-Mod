from __future__ import annotations

from pathlib import Path

from scripts import discoverllm_native_progress_serving as serving


def _tree(label: str):
    return {"criterion": f"Criterion {label}", "subcriteria": []}


def _task():
    worlds = tuple(
        serving.ProgressWorld(
            current=_tree(world),
            next=_tree(
                serving.WORLD_LABELS[(index + 1) % 4]
            ),
            after_next=_tree(
                serving.WORLD_LABELS[(index + 2) % 4]
            ),
        )
        for index, world in enumerate(serving.WORLD_LABELS)
    )
    return serving.ProgressTask(
        key=serving.SERVING_TASK_ID,
        conversation=({"role": "user", "content": "Help me revise."},),
        worlds=worlds,
    )


def test_action_parser_requires_exact_slots_and_unique_content():
    text = "\n".join(
        (
            "D1|Broad question?",
            "D2|Specific question?",
            "R1|One draft.",
            "R2|Draft one. Draft two.",
        )
    )
    assert tuple(serving._parse_actions(text)) == serving.ACTION_LABELS
    for invalid in (
        text.replace("D1|", "D2|", 1),
        text.replace("Specific question?", "Broad question?"),
        "\n".join(text.splitlines()[:-1]),
    ):
        try:
            serving._parse_actions(invalid)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("invalid action bank parsed")


def test_transition_parser_and_native_advance_rule():
    keys = {"D1_W1", "R1_W1"}
    parsed = serving._parse_transitions(
        "D1_W1|D|P\nR1_W1|R|S",
        keys,
    )
    assert serving._active_index(parsed["D1_W1"]) == 0
    assert serving._active_index(parsed["R1_W1"]) == 1

    for invalid in (
        "D1_W1|D|S\nR1_W1|R|S",
        "D1_W1|D|P\nR1_W1|R|P",
    ):
        try:
            serving._parse_transitions(invalid, keys)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("invalid transition parsed")


def test_observation_mapping_is_reproducible_and_per_action():
    first = serving._observation_mapping("fixture")
    second = serving._observation_mapping("fixture")
    assert first == second
    assert set(first) == set(serving.ACTION_LABELS)
    assert all(sorted(value) == [0, 1, 2, 3] for value in first.values())


def test_fixture_serving_passes_all_stages(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(serving, "_load_task", lambda paths, manifest: _task())
    result = serving.run_serving(
        object(),
        paths={},
        manifest_path=tmp_path / "manifest.json",
        raw_path=tmp_path / "raw.json",
        model=serving.DeterministicFixtureModel(),
    )
    assert result["status"] == "passed"
    assert result["gates"]["all_pass"] is True
    assert result["parse_counts"] == {
        "actions": 4,
        "root_transitions": 16,
        "root_feedback": 16,
        "root_tiers": 16,
        "followups": 16,
        "followup_transitions": 64,
        "followup_feedback": 16,
        "followup_tiers": 16,
    }
    assert result["protocol"]["semantic_content_emitted"] is False
