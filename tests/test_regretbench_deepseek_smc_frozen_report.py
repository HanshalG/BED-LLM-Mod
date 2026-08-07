from __future__ import annotations

import json

import pytest

from scripts import regretbench_deepseek_smc_frozen_report as report
from tests.test_regretbench_deepseek_frozen_report import (
    _comparison,
    _policy_row,
)


def test_smc_reporting_binding_matches_frozen_generator() -> None:
    path = report.REPO_ROOT / (
        "results/nonmyopic/regretbench_deepseek_smc_dynamic_depth2_policy/"
        "REPORTING_BINDING.json"
    )
    binding = json.loads(path.read_text())

    assert binding["status"] == (
        "prospectively_frozen_before_any_smc_policy_response"
    )
    assert binding["headline_control"] == "smc_myopic_refresh_brier"
    assert binding["development_can_be_called_confirmed"] is False
    assert binding["secondary_or_subgroup_evidence_can_change_tier"] is False
    for field in (
        "reporting_protocol",
        "generator",
        "independent_verifier",
        "execution_binding",
    ):
        row = binding[field]
        assert report.sha256_file(report.REPO_ROOT / row["path"]) == row["sha256"]


def _result(*, status: str, include_naive: bool = False) -> dict:
    briers = {
        "smc_dynamic_depth2": 0.20,
        "smc_myopic_refresh_brier": 0.32,
        "smc_myopic_brier": 0.31,
        "smc_myopic_width": 0.30,
        "smc_history_blind_depth2": 0.28,
        "smc_fixed_depth2": 0.26,
        "random": 0.35,
    }
    tasks = []
    for index in range(64):
        policies = {}
        for name, value in briers.items():
            row = _policy_row(value)
            row["posterior_parent_update_applied"] = True
            policies[name] = row
        if include_naive:
            policies["naive_thinking"] = _policy_row(0.34)
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
    science = None
    if status != "mechanics_failed":
        comparisons = {
            baseline: _comparison(baseline, -0.03)
            for baseline in report.BASELINES
        }
        science = {
            "comparisons": comparisons,
            "root_disagreements": {
                baseline: 20 for baseline in report.BASELINES
            },
            "predicted_to_realized_dynamic_myopic_refresh_brier": {
                "spearman": 0.4,
                "ci95": [0.1, 0.7],
                "probability_positive": 0.95,
                "n": 20,
            },
            "predicted_to_realized_dynamic_myopic_brier": {
                "spearman": 0.3,
                "ci95": [0.0, 0.6],
                "probability_positive": 0.9,
                "n": 20,
            },
            "predicted_to_realized_dynamic_myopic": {
                "spearman": 0.2,
                "ci95": [-0.1, 0.5],
                "probability_positive": 0.8,
                "n": 20,
            },
            "fresh_regeneration_comparisons_descriptive": comparisons,
            "gates": {"headline_gate": True, "all_pass": status == "passed"},
            "primary_claim_gates": {
                "headline_gate": status == "passed",
                "history_blind_gate": status == "passed",
            },
            "primary_claim_all_pass": status == "passed",
            "all_34_diagnostic_gates_pass": False,
        }
    return {
        "interface_version": "regretbench-deepseek-smc-dynamic-depth2-experiment-1",
        "status": status,
        "protocol": {
            "stage": "development",
            "model": "deepseek/deepseek-v4-flash-0731",
            "reasoning": "disabled_excluded",
            "task_count": 64,
            "protocol_sha256": report.SMC_POLICY_PROTOCOL_SHA256,
            "claim_gate_amendment_sha256": report.CLAIM_GATE_AMENDMENT_SHA256,
            "primary_endpoint": "aligned_generated_likelihood_truth_mass",
            "fresh_smc_regeneration_endpoint": "secondary_descriptive",
            "selection_frozen_before_truth_access": True,
            "initial_hypotheses_regenerated": False,
            "conditioned_blind_same_seed": True,
            "hidden_cig_exposed_to_model": False,
            "confirmation_opened": False,
        },
        "mechanics_gates": {
            "all_pass": status != "mechanics_failed",
            "all_actual_transitions_exact_lineage_and_retention": True,
        },
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
            "status": "available" if include_naive else "disabled"
        },
        "usage": {
            "deepseek_primary": {"adapter_requests": 8_300},
            "deepseek_naive_endpoint": {"adapter_requests": 128},
            "naive_luna": {"adapter_requests": 128},
            "combined_requests": 8_556,
            "combined_http_attempts": 8_556,
            "combined_cost_usd": 2.5,
        },
        "tasks": tasks,
    }


def _install_validation(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr(
        report,
        "_validate_verified_result",
        lambda run_dir, primary_dir: (
            payload,
            {"interface_version": "independent-smc-verifier"},
            "result-sha",
            "verification-sha",
        ),
    )


@pytest.mark.parametrize(
    ("status", "tier", "has_comparisons"),
    [
        (
            "passed",
            "smc_provisional_development_signal_confirmation_required",
            True,
        ),
        (
            "gated_null",
            "smc_development_policy_null_confirmation_forbidden",
            True,
        ),
        (
            "mechanics_failed",
            "smc_mechanics_failure_no_scientific_result",
            False,
        ),
    ],
)
def test_frozen_smc_claim_tiers(
    tmp_path, monkeypatch, status, tier, has_comparisons
) -> None:
    payload = _result(status=status)
    _install_validation(monkeypatch, payload)

    value = report.build_report(tmp_path, primary_dir=tmp_path)

    assert value["claim_tier"] == tier
    assert value["claim_tier_is_frozen_and_nonadaptive"] is True
    assert value["pooled_or_secondary_evidence_can_change_tier"] is False
    assert (value["paired_primary_comparisons"] is not None) is has_comparisons
    assert value["draw_stability_diagnostic"][
        "can_change_status_authorization_or_claim_tier"
    ] is False
    if status != "mechanics_failed":
        assert value["primary_claim_all_pass"] is (status == "passed")
        assert value["all_34_diagnostic_gates_pass"] is False


def test_report_centers_llm_owned_smc_mechanism_and_matched_control(
    tmp_path, monkeypatch
) -> None:
    payload = _result(status="passed", include_naive=True)
    _install_validation(monkeypatch, payload)

    value = report.build_report(tmp_path, primary_dir=tmp_path)
    markdown = report.render_markdown(value)

    assert set(value["primary_policy_table"]["policies"]) == set(
        report.PRIMARY_POLICIES
    )
    assert "smc_myopic_refresh_brier" in value["paired_primary_comparisons"]
    assert value["protocol"]["llm_owns_reply_likelihoods"] is True
    assert value["protocol"]["llm_owns_retain_revise_transitions"] is True
    assert value["optional_naive_thinking"]["can_change_claim_tier"] is False
    assert "path-dependent retain/revise particle transitions" in markdown
    assert "SMC refresh-matched myopic" in markdown


def test_write_report_emits_only_derived_zero_call_artifacts(
    tmp_path, monkeypatch
) -> None:
    payload = _result(status="gated_null")
    _install_validation(monkeypatch, payload)

    written = report.write_report(tmp_path, primary_dir=tmp_path)

    assert written["status"] == "written"
    assert written["model_calls"] == 0
    assert written["cost_usd"] == 0.0
    saved = json.loads((tmp_path / "FROZEN_REPORT.json").read_text())
    assert saved["claim_tier"] == (
        "smc_development_policy_null_confirmation_forbidden"
    )
    assert (tmp_path / "FROZEN_REPORT.md").is_file()
