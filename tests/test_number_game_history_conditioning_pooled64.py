from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from scripts import number_game_history_conditioning_pooled64 as audit


def _stage(mse_difference: float, coverage_difference: float) -> dict:
    return {
        "conditional": {
            "posterior_predictive_mse": 0.08 + mse_difference,
            "truth_extension_coverage": 0.6 + coverage_difference,
            "support_size": 30.0,
        },
        "history_blind": {
            "posterior_predictive_mse": 0.08,
            "truth_extension_coverage": 0.6,
            "support_size": 24.0,
        },
        "differences": {
            "conditional_minus_history_blind_predictive_mse": mse_difference,
            "conditional_minus_history_blind_truth_coverage": coverage_difference,
            "conditional_minus_history_blind_support_size": 6.0,
        },
    }


def _row(index: int, *, block: int) -> dict:
    value = (index + 1 + block * 32) / 10_000
    return {
        "tree_index": index + block * 32,
        "tree_mean": {
            "first": _stage(0.02, 0.1),
            "second": _stage(-0.01, 0.15),
        },
        "selected_roots": {
            "roots_differ": True,
            "dynamic_root_prompt_conditioning_benefit": value,
            "fixed_root_prompt_conditioning_benefit": 0.0,
            "dynamic_minus_fixed_root_prompt_conditioning_benefit": value,
            "realized_advantage": value,
        },
    }


def _blocks() -> list[list[dict]]:
    return [
        [_row(index, block=block) for index in range(32)]
        for block in range(2)
    ]


def test_real_sources_are_hash_bound_and_disjoint() -> None:
    blocks = audit.load_and_validate_sources()

    assert [len(block) for block in blocks] == [32, 32]
    assert [row["tree_index"] for row in blocks[0]] == list(range(32))
    assert [row["tree_index"] for row in blocks[1]] == list(range(32, 64))


def test_summary_preserves_positive_links_and_mean_gate() -> None:
    summary = audit.summarize_blocks(_blocks(), bootstrap_samples=200)

    assert summary["tree_count"] == 64
    assert summary["selected_root_prompt_conditioning"]["changed_root_tree_count"] == 64
    assert summary["retrospective_findings"] == {
        "second_stage_conditioning_mse_ci_below_zero": True,
        "second_stage_conditioning_coverage_ci_above_zero": True,
        "root_specific_benefit_spearman_ci_above_zero": True,
        "selected_root_mean_benefit_ci_above_zero": True,
    }


def test_bootstrap_is_deterministic_and_stratified() -> None:
    blocks = _blocks()

    first = audit.bootstrap_analysis(blocks, seed=17, samples=100)
    second = audit.bootstrap_analysis(blocks, seed=17, samples=100)

    assert first == second
    assert first[
        "second_conditional_minus_history_blind_predictive_mse_95pct"
    ] == pytest.approx([-0.01, -0.01])
    assert first[
        "prompt_benefit_contrast_to_realized_spearman_95pct"
    ] == pytest.approx([1.0, 1.0])


def test_loader_rejects_overlapping_tree_blocks(tmp_path) -> None:
    paths = []
    hashes = []
    for index, status in enumerate(("gated_null", "passed")):
        path = tmp_path / f"source-{index}.json"
        payload = {
            "status": status,
            "protocol": {"analysis_was_preregistered": True},
            "trees": [deepcopy(_row(i, block=0)) for i in range(32)],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())

    with pytest.raises(ValueError, match="tree block changed"):
        audit.load_and_validate_sources(paths=paths, hashes=hashes)


def test_run_writes_only_the_public_zero_call_result(tmp_path) -> None:
    output_dir = tmp_path / "pooled"

    result = audit.run_audit(
        output_dir=output_dir,
        bootstrap_samples=50,
    )

    assert sorted(path.name for path in output_dir.iterdir()) == ["RESULT.json"]
    assert result["protocol"]["model_calls"] == 0
    assert result["protocol"]["cost_usd"] == 0.0
    assert json.loads((output_dir / "RESULT.json").read_text()) == result
