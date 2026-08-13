from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import revengebench_execution_opportunity_v2_manifest as manifest


def test_v2_projects_only_exact_callable_arenas(tmp_path: Path) -> None:
    v1 = tmp_path / "v1.json"
    interface = tmp_path / "interface.json"
    v1.write_text(
        json.dumps(
            {
                "source_bindings": {"commit": "abc"},
                "betas": [0.5, 1.0, 2.0],
                "max_target_decisions_per_simulation": 64,
                "arenas": {name: {"name": name} for name in (*manifest.CALLABLE_ARENAS, "robocode")},
            }
        ),
        encoding="utf-8",
    )
    interface.write_text(
        json.dumps(
            {"status": "infrastructure_inconclusive", "callable_arenas": list(manifest.CALLABLE_ARENAS)}
        ),
        encoding="utf-8",
    )

    result = manifest.build(v1, interface)

    assert tuple(result["arenas"]) == manifest.CALLABLE_ARENAS
    assert result["minimum_robust_changed_arenas"] == 2
    assert result["privacy"]["trajectory_or_outcome_opened"] is False


def test_v2_rejects_changed_callable_set(tmp_path: Path) -> None:
    v1 = tmp_path / "v1.json"
    interface = tmp_path / "interface.json"
    v1.write_text(json.dumps({}), encoding="utf-8")
    interface.write_text(
        json.dumps({"status": "infrastructure_inconclusive", "callable_arenas": ["battlesnake"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="callable arena set"):
        manifest.build(v1, interface)
