from __future__ import annotations

import json

from scripts import number_game_qwen_external_canonical_pooled64 as pooled


def test_hash_bound_sources_are_disjoint() -> None:
    sources = [pooled.validate_source(study) for study in pooled.STUDIES]
    seed_sets = [
        {int(tree["tree_seed"]) for tree in source["trees"]}
        for source in sources
    ]

    assert len(sources) == 2
    assert all(len(source["trees"]) == 32 for source in sources)
    assert not seed_sets[0].intersection(seed_sets[1])


def test_zero_call_synthesis_preserves_source_nulls(tmp_path) -> None:
    result = pooled.run_synthesis(tmp_path)

    assert result["status"] == "retrospective_robustness_positive"
    assert result["protocol"]["analysis_is_retrospective"] is True
    assert result["protocol"]["cannot_rescue_source_composite_statuses"] is True
    assert result["protocol"]["model_calls"] == 0
    assert result["protocol"]["cost_usd"] == 0.0
    assert all(source["status"] == "gated_null" for source in result["sources"])
    assert all(result["robustness_checks"].values())
    assert json.loads((tmp_path / "RESULT.json").read_text())["status"] == (
        "retrospective_robustness_positive"
    )
