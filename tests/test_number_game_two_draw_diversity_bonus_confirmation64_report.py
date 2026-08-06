from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import number_game_two_draw_diversity_bonus_confirmation64_report as report


def _comparison(candidate: float, baseline: float) -> dict:
    difference = candidate - baseline
    return {
        "candidate_mean_brier": candidate,
        "baseline_mean_brier": baseline,
        "mean_candidate_minus_baseline_brier": difference,
        "candidate_brier_sample_sd": 0.02,
        "baseline_brier_sample_sd": 0.03,
        "paired_difference_sample_sd": 0.01,
        "relative_brier_reduction": (baseline - candidate) / baseline,
        "tree_bootstrap_95pct": [difference - 0.002, difference + 0.002],
        "wins": 40,
        "ties": 4,
        "losses": 20,
        "changed_roots": 32,
    }


def _run(tmp_path: Path, *, status: str = "passed") -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    comparisons = {
        name: _comparison(0.09, 0.10 + index * 0.001)
        for index, (name, _) in enumerate(report.COMPARISON_ORDER)
    }
    result = {
        "status": status,
        "comparisons": comparisons,
        "scientific_gates": {"primary": status == "passed"},
        "rank_metrics": {
            "original_mean_candidate_root_spearman": 0.4,
            "bonus_mean_candidate_root_spearman": 0.5,
            "original_mean_candidate_set_oracle_regret": 0.02,
            "bonus_mean_candidate_set_oracle_regret": 0.01,
        },
        "usage": {"adapter_requests": 7360, "run_cost_usd": 8.4},
    }
    (run_dir / "RESULT.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    (run_dir / "VERIFICATION.json").write_text("{}", encoding="utf-8")
    return run_dir


def _verified(**_) -> dict:
    return {"status": "verified", "checks": {"all": True}}


def test_report_emits_every_frozen_comparison_and_uncertainty(
    tmp_path: Path,
) -> None:
    rendered = report.render_report(
        run_dir=_run(tmp_path), verifier=_verified
    )

    assert "Status: **passed**" in rendered
    for _, label in report.COMPARISON_ORDER:
        assert label in rendered
    assert "mean (SD)" in rendered
    assert "Paired difference 95% CI" in rendered
    assert "descriptive controls, not scientific gates" in rendered


def test_gated_null_language_closes_route_without_tuning(
    tmp_path: Path,
) -> None:
    rendered = report.render_report(
        run_dir=_run(tmp_path, status="gated_null"), verifier=_verified
    )
    assert "route closes without tuning" in rendered


def test_report_refuses_unverified_or_missing_control(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    with pytest.raises(RuntimeError, match="not independently verified"):
        report.render_report(
            run_dir=run_dir,
            verifier=lambda **_: {"status": "verification_failed"},
        )

    result = json.loads((run_dir / "RESULT.json").read_text())
    del result["comparisons"]["uniform_random_candidate_root"]
    (run_dir / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="omitted comparisons"):
        report.render_report(run_dir=run_dir, verifier=_verified)
