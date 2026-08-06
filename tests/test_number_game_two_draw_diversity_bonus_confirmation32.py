from __future__ import annotations

from copy import deepcopy

from scripts import number_game_two_draw_diversity_bonus_confirmation32 as component


def test_fresh_seed_configuration_is_scoped_and_customizable() -> None:
    names = (
        "TREE_SEEDS",
        "TARGET_SEEDS",
        "VALIDATION_SEED_START",
        "SOURCE_BOOTSTRAP_SEED",
        "SOURCE_INTERFACE_VERSION",
    )
    original = {name: getattr(component.base, name) for name in names}
    with component.configured_fresh_source(
        tree_seeds=(1, 2),
        target_seeds=(3, 4),
        validation_seed_start=5,
        bootstrap_seed=6,
        source_interface_version="custom-source-1",
    ):
        assert component.base.TREE_SEEDS == (1, 2)
        assert component.base.TARGET_SEEDS == (3, 4)
        assert component.base.VALIDATION_SEED_START == 5
        assert component.base.SOURCE_BOOTSTRAP_SEED == 6
        assert component.base.SOURCE_INTERFACE_VERSION == "custom-source-1"
    assert {name: getattr(component.base, name) for name in names} == original


def test_source_scoring_uses_only_frozen_coefficients(
    tmp_path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "source"
    (source_dir / "private").mkdir(parents=True)
    for path in (
        source_dir / "RESULT.json",
        source_dir / "TREES.json",
        source_dir / "private/RAW_RESPONSES.json",
    ):
        path.write_text("{}", encoding="utf-8")
    calls = {}
    rows = [
        {
            "coefficient_grid_roots": {"0.0": 1, "-0.5": 2},
            "adjusted_scores": {1: 0.0, 2: -1.0},
        }
    ]

    def load_source(spec, *, coefficients):
        calls["coefficients"] = coefficients
        return deepcopy(rows)

    def source_summary(value, *, seed, include_coefficient_grid):
        calls["summary"] = (seed, include_coefficient_grid)
        assert value == rows
        return {"comparisons": {}}

    monkeypatch.setattr(component.audit, "sha256_file", lambda path: "h")
    monkeypatch.setattr(component.audit, "load_source", load_source)
    monkeypatch.setattr(component.audit, "source_summary", source_summary)
    monkeypatch.setattr(component, "_mean_rank_metrics", lambda value: {})

    scored = component.score_source_directory(
        source_dir,
        bootstrap_seed=123,
    )

    assert calls["coefficients"] == (0.0, -0.5)
    assert calls["summary"] == (123, False)
    assert "coefficient_grid_roots" not in scored["rows"][0]
    assert scored["rows"][0]["adjusted_scores"] == {
        "1": 0.0,
        "2": -1.0,
    }
