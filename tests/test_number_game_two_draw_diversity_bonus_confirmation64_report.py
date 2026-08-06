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


def _coverage_comparison(
    candidate: float,
    baseline: float,
    *,
    candidate_policy: str,
    baseline_policy: str,
    selector_independent: bool,
) -> dict:
    difference = candidate - baseline
    return {
        "candidate_mean_coverage": candidate,
        "baseline_mean_coverage": baseline,
        "mean_candidate_minus_baseline_coverage": difference,
        "candidate_coverage_sample_sd": 0.1,
        "baseline_coverage_sample_sd": 0.1,
        "paired_difference_sample_sd": 0.1,
        "tree_bootstrap_95pct": [difference - 0.002, difference + 0.002],
        "wins": 20 if difference <= 0.0 else 40,
        "ties": 24 if difference <= 0.0 else 4,
        "losses": 20,
        "changed_roots": 32,
        "candidate_policy": candidate_policy,
        "baseline_policy": baseline_policy,
        "selector_independent_of_diversity_bonus": selector_independent,
        "external_canonical_targets_endpoint_only": True,
        "used_for_policy_selection": False,
        "registered_scientific_gate": False,
    }


def _alignment(*, passes: bool) -> dict:
    rho = 0.4 if passes else -0.1
    interval = [0.1, 0.7] if passes else [-0.4, 0.2]
    comparisons = {
        name: {
            "tree_count": 64,
            "changed_root_count": 32,
            "coverage_uplift_brier_benefit_spearman_changed_roots": rho,
            "changed_root_bootstrap_95pct": interval,
            "unchanged_structural_zero_pairs_excluded": True,
            "positive_brier_benefit_means_candidate_improved": True,
            "registered_scientific_gate": False,
            "candidate_policy": candidate,
            "baseline_policy": baseline,
            "selector_independent_of_diversity_bonus": selector_independent,
        }
        for name, candidate, baseline, selector_independent in (
            (
                "bonus_vs_unadjusted_depth_three",
                "bonus_root",
                "original_root",
                False,
            ),
            (
                "bonus_depth_three_vs_crossfit_depth_two",
                "bonus_root",
                "depth_two_root",
                False,
            ),
            (
                "unadjusted_dynamic_vs_fixed_depth_three",
                "original_root",
                "fixed_depth_three_root",
                True,
            ),
        )
    }
    return {
        "comparisons": comparisons,
        "mean_changed_root_spearman": rho,
        "mean_changed_root_spearman_tree_bootstrap_95pct": interval,
        "family_bootstrap_resamples_trees_jointly": True,
        "association_is_noncausal": True,
        "registered_scientific_gate": False,
        "can_rescue_brier_status": False,
    }


def _run(
    tmp_path: Path,
    *,
    status: str = "passed",
    coverage_pass: bool = False,
) -> Path:
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
    coverage_candidate = 0.7 if coverage_pass else 0.6
    coverage = {
        name: _coverage_comparison(
            coverage_candidate,
            0.6,
            candidate_policy=candidate,
            baseline_policy=baseline,
            selector_independent=selector_independent,
        )
        for name, candidate, baseline, selector_independent in (
            (
                "bonus_vs_unadjusted_depth_three",
                "bonus_root",
                "original_root",
                False,
            ),
            (
                "bonus_depth_three_vs_crossfit_depth_two",
                "bonus_root",
                "depth_two_root",
                False,
            ),
            (
                "unadjusted_dynamic_vs_fixed_depth_three",
                "original_root",
                "fixed_depth_three_root",
                True,
            ),
        )
    }
    result = {
        "status": status,
        "comparisons": comparisons,
        "truth_coverage_comparisons": coverage,
        "truth_coverage_brier_alignment": _alignment(passes=coverage_pass),
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
    assert "Canonical Truth Coverage" in rendered
    assert "Coverage-Brier Alignment" in rendered
    assert "Truth-coverage/Brier alignment: **fail**" in rendered
    assert "not mediation evidence" in rendered


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


def test_stronger_tier_requires_brier_coverage_and_direct_alignment(
    tmp_path: Path,
) -> None:
    result = json.loads(
        (_run(tmp_path, coverage_pass=True) / "RESULT.json").read_text()
    )
    classification = report.classify_claim_scope(result)
    assert classification["truth_coverage_endpoint_family"]["pass"]
    assert classification["truth_coverage_alignment_family"]["pass"]
    assert classification["claim_tier"] == (
        "truth_coverage_aligned_dynamic_nonmyopic_confirmation"
    )
    assert classification["authorizes_truth_coverage_aligned_claim"]
    assert not classification[
        "authorizes_causal_truth_coverage_mediation_claim"
    ]

    result["status"] = "gated_null"
    for name in report.DEPTH_GATES + report.VIABILITY_GATES:
        result["scientific_gates"][name] = False
    classification = report.classify_claim_scope(result)
    assert classification["truth_coverage_endpoint_family"]["pass"]
    assert classification["truth_coverage_alignment_family"]["pass"]
    assert classification["claim_tier"] == (
        "mechanism_only_without_nonmyopic_depth"
    )
    assert not classification["authorizes_truth_coverage_aligned_claim"]


def test_parallel_coverage_without_alignment_cannot_unlock_strongest_tier(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, coverage_pass=True)
    result = json.loads((run_dir / "RESULT.json").read_text())
    result["truth_coverage_brier_alignment"] = _alignment(passes=False)
    classification = report.classify_claim_scope(result)
    assert classification["truth_coverage_endpoint_family"]["pass"]
    assert not classification["truth_coverage_alignment_family"]["pass"]
    assert classification["claim_tier"] == (
        "full_llm_native_dynamic_nonmyopic_confirmation"
    )


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
