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
    comparisons["unadjusted_dynamic_vs_fixed_depth_three"].update(
        {
            "candidate_policy": "unadjusted_dynamic_depth_three",
            "baseline_policy": "fixed_support_depth_three",
            "selector_independent_of_diversity_bonus": True,
            "registered_scientific_gate": False,
        }
    )
    result = {
        "status": status,
        "comparisons": comparisons,
        "scientific_gates": {
            name: status == "passed"
            for name in report.DEPTH_GATES + report.VIABILITY_GATES
        },
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
    (run_dir / "VERIFICATION.json").write_text(
        json.dumps(_verified()), encoding="utf-8"
    )
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
    assert "full_llm_native_dynamic_nonmyopic_confirmation" in rendered
    assert "Diversity-selector superiority: **pass**" in rendered


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

    (run_dir / "VERIFICATION.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match independent replay"):
        report.render_report(run_dir=run_dir, verifier=_verified)
    (run_dir / "VERIFICATION.json").write_text(
        json.dumps(_verified()), encoding="utf-8"
    )

    result = json.loads((run_dir / "RESULT.json").read_text())
    del result["comparisons"]["uniform_random_candidate_root"]
    (run_dir / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="omitted comparisons"):
        report.render_report(run_dir=run_dir, verifier=_verified)


def test_registered_pass_does_not_imply_selector_superiority(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path)
    result = json.loads((run_dir / "RESULT.json").read_text())
    original = result["comparisons"]["original_depth_three"]
    original.update(
        {
            "candidate_mean_brier": 0.1,
            "baseline_mean_brier": 0.1,
            "mean_candidate_minus_baseline_brier": 0.0,
            "relative_brier_reduction": 0.0,
            "tree_bootstrap_95pct": [-0.002, 0.002],
            "wins": 20,
            "ties": 24,
            "losses": 20,
        }
    )
    classification = report.classify_claim_scope(result)
    assert classification["registered_nonmyopic_depth_family"]["pass"]
    assert classification["registered_selector_viability_family"]["pass"]
    assert not classification["diversity_selector_superiority_family"]["pass"]
    assert classification["claim_tier"] == (
        "nonmyopic_with_viable_diversity_selector"
    )
    assert not classification["authorizes_full_llm_native_dynamic_claim"]


def test_depth_and_mechanism_families_are_not_allowed_to_rescue_each_other(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, status="gated_null")
    result = json.loads((run_dir / "RESULT.json").read_text())
    classification = report.classify_claim_scope(result)
    assert not classification["registered_nonmyopic_depth_family"]["pass"]
    assert classification["diversity_selector_superiority_family"]["pass"]
    assert classification["claim_tier"] == (
        "mechanism_only_without_nonmyopic_depth"
    )


def test_confounded_bonus_vs_fixed_cannot_authorize_dynamic_claim(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path)
    result = json.loads((run_dir / "RESULT.json").read_text())
    clean = result["comparisons"][
        "unadjusted_dynamic_vs_fixed_depth_three"
    ]
    clean.update(
        {
            "candidate_mean_brier": 0.1,
            "baseline_mean_brier": 0.1,
            "mean_candidate_minus_baseline_brier": 0.0,
            "relative_brier_reduction": 0.0,
            "tree_bootstrap_95pct": [-0.002, 0.002],
            "wins": 20,
            "ties": 24,
            "losses": 20,
        }
    )
    assert result["comparisons"]["fixed_support_depth_three"][
        "tree_bootstrap_95pct"
    ][1] < 0.0
    classification = report.classify_claim_scope(result)
    assert not classification["dynamic_support_endpoint_family"]["pass"]
    assert classification["claim_tier"] == (
        "nonmyopic_with_diversity_selector_gain"
    )
    assert not classification["authorizes_full_llm_native_dynamic_claim"]


def test_report_banks_json_and_markdown_idempotently(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    output = tmp_path / "RESULT_REPORT.md"
    claim_path = run_dir / "CLAIM_REPORT.json"
    first = report.write_report(
        run_dir=run_dir,
        output_path=output,
        claim_path=claim_path,
        verifier=_verified,
    )
    second = report.write_report(
        run_dir=run_dir,
        output_path=output,
        claim_path=claim_path,
        verifier=_verified,
    )
    assert second == first
    claim_payload = json.loads(claim_path.read_text())
    assert claim_payload["claim_tier"] == (
        "full_llm_native_dynamic_nonmyopic_confirmation"
    )
    output.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="report changed"):
        report.write_report(
            run_dir=run_dir,
            output_path=output,
            claim_path=claim_path,
            verifier=_verified,
        )
