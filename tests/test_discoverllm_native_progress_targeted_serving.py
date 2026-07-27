from __future__ import annotations

from pathlib import Path

from scripts import discoverllm_native_progress_serving as base
from scripts import discoverllm_native_progress_targeted_serving as targeted


def _tree(label: str):
    return {"criterion": f"Criterion {label}", "subcriteria": []}


def _task():
    return base.ProgressTask(
        key=targeted.SERVING_TASK_ID,
        conversation=({"role": "user", "content": "Help me revise."},),
        worlds=tuple(
            base.ProgressWorld(
                current=_tree(world),
                next=_tree(base.WORLD_LABELS[(index + 1) % 4]),
                after_next=_tree(base.WORLD_LABELS[(index + 2) % 4]),
            )
            for index, world in enumerate(base.WORLD_LABELS)
        ),
    )


def test_state_delta_parser_maps_only_native_state_effects():
    keys = {"D1_W1", "D2_W1", "R2_W1", "R2_W2"}
    parsed = targeted._parse_state_deltas(
        "D1_W1|V\nD2_W1|C\nR2_W1|A\nR2_W2|T",
        keys,
    )
    assert parsed["D1_W1"] == ("D", "N")
    assert parsed["D2_W1"] == ("R", "N")
    assert parsed["R2_W1"] == ("R", "S")
    assert parsed["R2_W2"] == ("D", "T")


def test_targeted_fixture_passes_causal_serving_gate(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(targeted, "_load_task", lambda paths, manifest: _task())
    result = targeted.run_serving(
        object(),
        paths={},
        manifest_path=tmp_path / "manifest.json",
        raw_path=tmp_path / "raw.json",
        model=targeted.DeterministicFixtureModel(),
    )
    assert result["status"] == "passed"
    assert result["gates"]["all_pass"] is True
    assert result["root_transition_summary"]["R2_advance_cells"] == 4
    assert result["root_transition_summary"]["D2_advance_cells"] == 0
    assert result["protocol"]["interface_version"] == targeted.INTERFACE_VERSION
