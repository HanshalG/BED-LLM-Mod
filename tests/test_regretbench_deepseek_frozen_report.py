from __future__ import annotations

import json

import pytest

from scripts import regretbench_deepseek_frozen_report as report


def _policy_row(brier: float) -> dict:
    truth = 1.0 - brier**0.5
    return {
        "truth_mass_final": truth,
        "brier": brier,
        "log_loss": -__import__("math").log(max(1e-12, truth)),
        "truth_mass_after_first": 0.25,
        "valid_two_action_trajectory": True,
        "truth_consistent_first_reply_likelihood_matched": True,
        "likelihood_aligned_two_action_trajectory": True,
        "first_supported": True,
        "second_supported": True,
        "second_action_novel": True,
        "second_reply_likelihood_matched": True,
        "fresh_truth_mass_final": truth,
        "fresh_brier": brier,
        "fresh_log_loss": -__import__("math").log(max(1e-12, truth)),
    }


def _comparison(baseline: str, difference: float) -> dict:
    return {
        "baseline": baseline,
        "brier_dynamic_minus_baseline": {
            "mean": difference,
            "sample_sd": 0.02,
            "ci95": [difference - 0.01, difference + 0.01],
            "probability_improvement": 0.95,
            "samples": 20_000,
            "seed": 1,
        },
        "log_loss_dynamic_minus_baseline": {
            "mean": -0.03,
            "sample_sd": 0.04,
            "ci95": [-0.05, -0.01],
            "probability_improvement": 0.9,
            "samples": 20_000,
            "seed": 2,
        },
        "wins_ties_losses": {"wins": 40, "ties": 4, "losses": 20},
    }


def _result(
    *, stage: str, status: str, include_naive: bool = False
) -> dict:
    briers = {
        "dynamic_depth2": 0.2,
        "myopic_refresh_brier": 0.32,
        "myopic_brier": 0.31,
        "myopic_width": 0.3,
        "history_blind_depth2": 0.28,
        "fixed_depth2": 0.26,
        "random": 0.35,
    }
    tasks = []
    for index in range(64):
        policies = {name: _policy_row(value) for name, value in briers.items()}
        if include_naive:
            policies["naive_thinking"] = _policy_row(0.32)
        tasks.append(
            {
                "task_id": f"task-{index}",
                "selected_roots": {
                    name: (index + offset) % 4
                    for offset, name in enumerate(report.PRIMARY_POLICIES)
                },
                "policies": policies,
            }
        )
    mechanics = {"all_pass": status != "mechanics_failed", "crn_exact": True}
    science = None
    if status != "mechanics_failed":
        comparisons = {
            baseline: _comparison(baseline, -0.02 - index * 0.005)
            for index, baseline in enumerate(report.BASELINES)
        }
        science = {
            "comparisons": comparisons,
            "root_disagreements": {
                baseline: 16 + index
                for index, baseline in enumerate(report.BASELINES)
            },
            "predicted_to_realized_dynamic_myopic": {
                "spearman": 0.3,
                "ci95": [0.1, 0.5],
                "probability_positive": 0.95,
                "n": 20,
            },
            "predicted_to_realized_dynamic_myopic_brier": {
                "spearman": 0.4,
                "ci95": [0.2, 0.6],
                "probability_positive": 0.97,
                "n": 22,
            },
            "predicted_to_realized_dynamic_myopic_refresh_brier": {
                "spearman": 0.45,
                "ci95": [0.25, 0.65],
                "probability_positive": 0.98,
                "n": 24,
            },
            "fresh_regeneration_comparisons_descriptive": comparisons,
            "gates": {
                "dynamic_myopic_differ_at_least_16": True,
                "all_pass": status == "passed",
            },
        }
    interface = {
        "development": "regretbench-deepseek-dynamic-depth2-policy-1",
        "confirmation": "regretbench-deepseek-dynamic-depth2-confirmation-1",
    }[stage]
    protocol = {
        "stage": stage,
        "model": "deepseek/deepseek-v4-flash-0731",
        "reasoning": "disabled_excluded",
        "task_count": 64,
        "primary_endpoint": "aligned_generated_likelihood_truth_mass",
        "fresh_regeneration_endpoint": "secondary_descriptive",
        "draw_stability_diagnostic_amendment_sha256": (
            report.DRAW_STABILITY_DIAGNOSTIC_AMENDMENT_SHA256
        ),
    }
    if stage == "confirmation":
        protocol["source_split"] = "confirmation"
    return {
        "schema_version": 1,
        "interface_version": interface,
        "status": status,
        "protocol": protocol,
        "confirmation_opened": stage == "confirmation",
        "mechanics_gates": mechanics,
        "crn_diagnostics": {"exact_group_count": 1024},
        "science": science,
        "draw_stability_diagnostic": {
            "label": "non_gating_non_rescuing_draw_stability_diagnostic",
            "draw_agreement_count": 48,
            "draw_agreement_fraction": 0.75,
            "all_draws_match_averaged_selection_count": 44,
            "all_draws_match_averaged_selection_fraction": 0.6875,
            "averaged_winner_brier_margin": {
                "mean": 0.03,
                "median": 0.02,
                "minimum": 0.001,
                "maximum": 0.12,
            },
            "stable_tasks_descriptive": {
                "task_count": 48,
                "mean_dynamic_minus_refresh_myopic_brier": -0.04,
            },
            "unstable_tasks_descriptive": {
                "task_count": 16,
                "mean_dynamic_minus_refresh_myopic_brier": 0.01,
            },
            "can_change_status_authorization_or_claim_tier": False,
        },
        "naive_baseline": {
            "status": "available" if include_naive else "disabled_by_smoke"
        },
        "usage": {
            "deepseek_primary": {
                "adapter_requests": 8300,
                "http_attempts": 8300,
                "retry_count": 0,
                "adapter_reasoning_tokens": 0,
                "forced_exits": 0,
                "run_cost_usd": 1.5,
            },
            "deepseek_naive_endpoint": {"adapter_requests": 0},
            "naive_luna": {"adapter_requests": 0},
            "combined_requests": 8300,
            "combined_http_attempts": 8300,
            "combined_cost_usd": 1.5,
        },
        "tasks": tasks,
    }


def _write_verified(tmp_path, payload: dict) -> None:
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = {
        "schema_version": 1,
        "interface_version": "fixture-independent-verification",
        "status": "verified",
        "result_status": payload["status"],
        "model_calls": 0,
        "cost_usd": 0.0,
        "mismatches": [],
        "checks": {"reported_result_matches_replay": True},
        "artifact_sha256": {"RESULT.json": report.sha256_file(result_path)},
    }
    (tmp_path / "VERIFICATION.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )


def test_reporting_binding_matches_protocol_and_generator() -> None:
    binding_path = (
        report.REPO_ROOT
        / "results/nonmyopic/regretbench_deepseek_reporting/REPORTING_BINDING.json"
    )
    binding = json.loads(binding_path.read_text())

    assert binding["status"] == "prospectively_amended_before_responses"
    assert binding["scientific_contract_changed"] is True
    assert report.sha256_file(
        report.REPO_ROOT / binding["protocol"]["path"]
    ) == binding["protocol"]["sha256"]
    assert report.sha256_file(
        report.REPO_ROOT / binding["generator"]["path"]
    ) == binding["generator"]["sha256"]
    assert report.sha256_file(
        report.REPO_ROOT / binding["endpoint_amendment"]["path"]
    ) == binding["endpoint_amendment"]["sha256"]
    assert report.sha256_file(
        report.REPO_ROOT / binding["matched_utility_myopic_amendment"]["path"]
    ) == binding["matched_utility_myopic_amendment"]["sha256"]
    assert report.sha256_file(
        report.REPO_ROOT / binding["refresh_matched_myopic_amendment"]["path"]
    ) == binding["refresh_matched_myopic_amendment"]["sha256"]
    assert report.sha256_file(
        report.REPO_ROOT / binding["draw_stability_diagnostic_amendment"]["path"]
    ) == binding["draw_stability_diagnostic_amendment"]["sha256"]
    assert binding["requirements"][
        "refresh_matched_myopic_is_headline_horizon_control"
    ]
    assert binding["requirements"][
        "pooled_or_secondary_evidence_can_change_tier"
    ] is False


@pytest.mark.parametrize(
    ("stage", "status", "tier"),
    [
        (
            "development",
            "passed",
            "provisional_development_signal_confirmation_required",
        ),
        (
            "development",
            "gated_null",
            "development_policy_null_confirmation_forbidden",
        ),
        (
            "confirmation",
            "passed",
            "confirmed_llm_native_nonmyopic_signal",
        ),
        (
            "confirmation",
            "gated_null",
            "confirmation_null_development_not_confirmed",
        ),
        (
            "development",
            "mechanics_failed",
            "mechanics_failure_no_scientific_result",
        ),
    ],
)
def test_frozen_claim_tiers(stage, status, tier, tmp_path) -> None:
    _write_verified(tmp_path, _result(stage=stage, status=status))

    value = report.build_report(tmp_path, stage=stage)

    assert value["claim_tier"] == tier
    assert value["claim_tier_is_frozen_and_nonadaptive"] is True
    assert value["pooled_or_secondary_evidence_can_change_tier"] is False
    assert value["draw_stability_diagnostic"][
        "can_change_status_authorization_or_claim_tier"
    ] is False
    if status == "mechanics_failed":
        assert value["paired_primary_comparisons"] is None
        assert "no RegretBench policy-efficacy result" in value["interpretation"]
    else:
        assert set(value["paired_primary_comparisons"]) == set(report.BASELINES)


def test_primary_table_and_optional_naive_are_separated(tmp_path) -> None:
    _write_verified(
        tmp_path,
        _result(stage="development", status="passed", include_naive=True),
    )

    value = report.build_report(tmp_path, stage="development")

    table = value["primary_policy_table"]
    assert table["label"] == (
        "aligned_generated_likelihood_with_action_and_first_reply_penalty"
    )
    assert set(table["policies"]) == set(report.PRIMARY_POLICIES)
    assert table["policies"]["dynamic_depth2"]["brier"] == {
        "mean": pytest.approx(0.2),
        "sample_sd": pytest.approx(0.0),
        "n": 64,
    }
    naive = value["optional_naive_thinking"]
    assert naive["available"] is True
    assert naive["can_change_claim_tier"] is False
    assert naive["metrics"]["fresh_brier"]["mean"] == pytest.approx(0.32)
    diagnostic = value["alignment_complete_diagnostic"]
    assert diagnostic["can_change_result_status_or_claim_tier"] is False
    assert diagnostic["controls"]["myopic_refresh_brier"][
        "eligible_task_count"
    ] == 64
    assert diagnostic["controls"]["myopic_brier"]["eligible_task_count"] == 64
    assert diagnostic["controls"]["myopic_width"]["eligible_task_count"] == 64
    assert diagnostic["alignment_complete_corroboration"]["all_pass"] is True


def test_alignment_complete_diagnostic_excludes_penalized_first_paths(
    tmp_path,
) -> None:
    payload = _result(stage="development", status="passed")
    dynamic = payload["tasks"][0]["policies"]["dynamic_depth2"]
    dynamic["truth_consistent_first_reply_likelihood_matched"] = False
    dynamic["likelihood_aligned_two_action_trajectory"] = False
    dynamic["brier"] = 1.0
    dynamic["log_loss"] = -__import__("math").log(1e-12)
    _write_verified(tmp_path, payload)

    value = report.build_report(tmp_path, stage="development")

    diagnostic = value["alignment_complete_diagnostic"]
    assert diagnostic["controls"]["myopic_refresh_brier"][
        "eligible_task_count"
    ] == 63
    assert diagnostic["controls"]["myopic_brier"]["eligible_task_count"] == 63
    assert diagnostic["controls"]["myopic_width"]["eligible_task_count"] == 63
    assert diagnostic["controls"]["fixed_depth2"]["eligible_task_count"] == 63
    assert value["paired_primary_comparisons"] == payload["science"]["comparisons"]


def test_result_tamper_after_verification_refuses_report(tmp_path) -> None:
    payload = _result(stage="development", status="passed")
    _write_verified(tmp_path, payload)
    payload["tasks"][0]["policies"]["dynamic_depth2"]["brier"] = 0.9
    (tmp_path / "RESULT.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="independently verified"):
        report.build_report(tmp_path, stage="development")


def test_write_report_emits_json_and_markdown_without_calls(tmp_path) -> None:
    _write_verified(tmp_path, _result(stage="confirmation", status="passed"))

    written = report.write_report(tmp_path, stage="confirmation")

    assert written["status"] == "written"
    assert written["model_calls"] == 0
    assert written["cost_usd"] == 0.0
    markdown = (tmp_path / "FROZEN_REPORT.md").read_text()
    assert "confirmed_llm_native_nonmyopic_signal" in markdown
    assert "Primary Aligned Endpoint" in markdown
    assert "Ranking Fidelity" in markdown
    assert "Alignment-Complete Diagnostic" in markdown
    assert "Draw Stability Diagnostic" in markdown
    assert "48/64" in markdown
    assert "Science Gates" in markdown
    assert "Secondary Fresh Regeneration" in markdown
    assert "Optional Naive-Thinking Baseline" in markdown
    assert "Primary DeepSeek transport" in markdown
    assert "secondary_descriptive" in markdown
