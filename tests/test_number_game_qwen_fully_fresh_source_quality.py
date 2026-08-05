from __future__ import annotations

from pathlib import Path

from scripts import number_game_dynamic_support_quality96 as quality
from scripts import number_game_qwen_fully_fresh_source_quality as fresh


def test_configured_quality_sets_and_restores_source_bindings() -> None:
    original_tree_count = quality.TREE_COUNT
    original_result_hash = quality.SOURCE_RESULT_SHA256

    with fresh.configured_quality():
        assert quality.TREE_COUNT == 32
        assert quality.BOOTSTRAP_SEED == 100_900
        assert quality.SOURCE_RESULT == fresh.SOURCE_RESULT
        assert quality.SOURCE_RESULT_SHA256 == fresh.SOURCE_RESULT_SHA256

    assert quality.TREE_COUNT == original_tree_count
    assert quality.SOURCE_RESULT_SHA256 == original_result_hash


def test_run_analysis_delegates_with_frozen_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed = {}

    def fake_run(output_dir, **kwargs):
        observed["output_dir"] = output_dir
        observed.update(kwargs)
        observed["tree_count"] = quality.TREE_COUNT
        observed["bootstrap_seed"] = quality.BOOTSTRAP_SEED
        return {"analysis": {"ok": True}}

    monkeypatch.setattr(quality, "run_analysis", fake_run)
    result = fresh.run_analysis(tmp_path, bootstrap_samples=17)

    assert result == {"analysis": {"ok": True}}
    assert observed == {
        "output_dir": tmp_path,
        "result_path": fresh.SOURCE_RESULT,
        "trees_path": fresh.SOURCE_TREES,
        "targets_path": fresh.SOURCE_TARGETS,
        "bootstrap_samples": 17,
        "tree_count": 32,
        "bootstrap_seed": 100_900,
    }


def test_frozen_source_hashes_match_public_artifacts() -> None:
    assert quality.sha256_file(fresh.SOURCE_RESULT) == (
        fresh.SOURCE_RESULT_SHA256
    )
    assert quality.sha256_file(fresh.SOURCE_TREES) == (
        fresh.SOURCE_TREES_SHA256
    )
    assert quality.sha256_file(fresh.SOURCE_TARGETS) == (
        fresh.SOURCE_TARGETS_SHA256
    )
