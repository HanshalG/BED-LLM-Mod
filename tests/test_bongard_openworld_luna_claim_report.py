from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from scripts import bongard_openworld_luna_claim_report as claim
from scripts import bongard_openworld_luna_confirmation64 as confirmation
from scripts import bongard_openworld_luna_development_claim_finalize as finalize
from scripts import bongard_openworld_luna_vlm_development as development


def _result(*, failed: tuple[str, ...] = ()) -> dict:
    gates = {name: name not in failed for name in claim.EXPECTED_GATES}
    gates["all_pass"] = all(gates.values())
    signal = gates["all_pass"]
    summary = {
        "n": development.TASKS,
        "mean_difference": -0.01,
        "sample_sd": 0.02,
        "ci95": [-0.02, -0.001],
        "bootstrap_probability_improvement": 0.9,
        "wins": 40,
        "ties": 4,
        "losses": 20,
    }
    return {
        "status": "development_signal" if signal else "development_null",
        "authorizes_confirmation_preregistration": signal,
        "authorizes_confirmation_execution": False,
        "protocol": {
            "interface_version": development.INTERFACE_VERSION,
            "task_count": development.TASKS,
            "confirmation_accessed": False,
            "sealed_test_accessed": False,
        },
        "gates": gates,
        "pooled_policy_metrics": {
            "dynamic_depth2": {"mean_brier": 0.12, "mean_log_loss": 0.4},
            "myopic_width": {"mean_brier": 0.14, "mean_log_loss": 0.45},
        },
        "comparisons_vs_myopic": {
            "dynamic_depth2": {
                "mean_brier": summary,
                "mean_log_loss": summary,
            }
        },
        "dynamic_vs_history_blind": {
            "mean_brier": summary,
            "mean_log_loss": summary,
        },
        "dynamic_vs_compute_matched_myopic": {
            "mean_brier": summary,
            "mean_log_loss": summary,
        },
        "dynamic_vs_fixed_depth2": {
            "mean_brier": summary,
            "mean_log_loss": summary,
        },
        "dynamic_vs_fixed_score_dynamic_update": {
            "mean_brier": summary,
            "mean_log_loss": summary,
        },
        "dynamic_vs_history_blind_update_matched_first": {
            "mean_brier": summary,
            "mean_log_loss": summary,
        },
        "ranking_fidelity": {
            "dynamic_depth2": {"mean_spearman": 0.3, "sample_sd": 0.2},
            "myopic_width": {"mean_spearman": 0.2, "sample_sd": 0.2},
            "history_blind_depth2": {
                "mean_spearman": 0.25,
                "sample_sd": 0.2,
            },
        },
        "dynamic_vs_myopic_relative_brier_improvement": 0.1,
        "dynamic_vs_history_blind_relative_brier_improvement": 0.05,
        "dynamic_vs_compute_matched_myopic_relative_brier_improvement": 0.05,
        "dynamic_vs_fixed_depth2_relative_brier_improvement": 0.05,
        "dynamic_vs_fixed_score_dynamic_update_relative_brier_improvement": 0.05,
        "dynamic_vs_history_blind_update_matched_first_relative_brier_improvement": 0.05,
        "dynamic_vs_myopic_changed_final_histories": 32,
        "dynamic_vs_history_blind_changed_final_histories": 28,
        "dynamic_vs_compute_matched_myopic_changed_final_histories": 29,
        "dynamic_vs_fixed_depth2_changed_final_histories": 30,
        "dynamic_vs_fixed_score_dynamic_update_changed_final_histories": 30,
        "dynamic_vs_history_blind_update_matched_first_changed_final_histories": 30,
        "dynamic_vs_history_blind_update_matched_first_robust_second_action_changes": 30,
    }


def _verification(result: dict, sha256: str = "a" * 64) -> dict:
    return {
        "verified": True,
        "status": result["status"],
        "result_sha256": sha256,
        "authorizes_confirmation_preregistration": result[
            "authorizes_confirmation_preregistration"
        ],
    }


@pytest.mark.parametrize(
    ("failed", "expected_tier", "authorized"),
    [
        ((), "full_path_dependent_llm_native_development_signal", True),
        (
            (claim.PATH_DEPENDENT_GATES[0],),
            (
                "policy_and_matched_regeneration_without_"
                "fixed_support_superiority"
            ),
            False,
        ),
        (
            (claim.PATH_DEPENDENT_GATES[-1],),
            (
                "policy_and_matched_regeneration_without_"
                "fixed_support_superiority"
            ),
            False,
        ),
        (
            (claim.MECHANISM_GATES[0],),
            "policy_signal_without_matched_mechanism",
            False,
        ),
        (
            (claim.POLICY_GATES[0],),
            "matched_mechanism_without_policy_signal",
            False,
        ),
        (
            (claim.POLICY_GATES[0], claim.MECHANISM_GATES[0]),
            "development_null",
            False,
        ),
    ],
)
def test_claim_tiers_are_fixed_by_complete_gate_families(
    failed, expected_tier, authorized
) -> None:
    result = _result(failed=failed)
    report = claim.build_claim_report(
        result,
        result_sha256="a" * 64,
        independent_verification=_verification(result),
    )
    assert report["claim_tier"] == expected_tier
    assert report["authorizes_confirmation_preregistration"] is authorized
    assert report["confirmation_execution_remains_unauthorized"]
    scope = " ".join(report["claim_scope"]["allowed"])
    assert "belief regeneration improves over" not in scope
    if expected_tier in {
        "full_path_dependent_llm_native_development_signal",
        "policy_and_matched_regeneration_without_fixed_support_superiority",
        "matched_mechanism_without_policy_signal",
    }:
        assert "first-query planning" in scope
        assert "common realized" in scope
    if expected_tier == "full_path_dependent_llm_native_development_signal":
        assert "second query" in scope
        assert "history-blind intermediate" in scope


def test_unmatched_fixed_policy_win_cannot_authorize_first_action_claim() -> None:
    matched_gate = (
        "dynamic_brier_relative_improvement_vs_"
        "fixed_score_dynamic_update_at_least_3_percent"
    )
    result = _result(failed=(matched_gate,))
    result["dynamic_vs_fixed_depth2"]["mean_brier"][
        "mean_difference"
    ] = -0.02
    result["dynamic_vs_fixed_score_dynamic_update"]["mean_brier"][
        "mean_difference"
    ] = 0.02

    classification = claim.classify_result(result)

    assert all(
        result["gates"][gate]
        for gate in claim.PATH_DEPENDENT_GATES
        if "fixed_score_dynamic_update" not in gate
    )
    assert classification["claim_tier"] == (
        "policy_and_matched_regeneration_without_fixed_support_superiority"
    )
    assert classification["authorizes_confirmation_preregistration"] is False


def test_claim_report_rejects_gate_or_replay_inconsistency() -> None:
    result = _result()
    result["gates"]["all_pass"] = False
    with pytest.raises(ValueError, match="status disagrees"):
        claim.classify_result(result)

    result = _result()
    verification = _verification(result, sha256="b" * 64)
    with pytest.raises(ValueError, match="replay"):
        claim.build_claim_report(
            result,
            result_sha256="a" * 64,
            independent_verification=verification,
        )


def test_claim_report_rejects_nonfinite_metrics() -> None:
    result = deepcopy(_result())
    result["pooled_policy_metrics"]["dynamic_depth2"]["mean_brier"] = math.nan
    with pytest.raises(ValueError, match="not finite"):
        claim.build_claim_report(
            result,
            result_sha256="a" * 64,
            independent_verification=_verification(result),
        )


def test_claim_report_banks_once_and_rejects_tampering(tmp_path) -> None:
    path = tmp_path / "CLAIM_REPORT.json"
    report = {"claim_tier": "development_null", "value": 1}
    assert claim.bank_claim_report(path, report) == report
    assert claim.bank_claim_report(path, report) == report
    path.write_text('{"claim_tier":"development_null","value":2}')
    with pytest.raises(RuntimeError, match="changed"):
        claim.bank_claim_report(path, report)


def test_full_tier_finalizer_reaches_exact_confirmation_handoff(
    tmp_path, monkeypatch
) -> None:
    result = _result()
    combined_path = tmp_path / "COMBINED_RESULT.json"
    combined_path.write_text(json.dumps(result), encoding="utf-8")
    result_sha256 = development.sha256_file(combined_path)
    verification = _verification(result, sha256=result_sha256)
    report_path = tmp_path / "CLAIM_REPORT.json"
    block_results = {}
    ledgers = {}
    for block_id in development.BLOCK_ORDER:
        block_dir = tmp_path / f"block-{block_id}"
        block_dir.mkdir()
        block_results[block_id] = block_dir / "RESULT.json"
        block_results[block_id].write_text("{}", encoding="utf-8")
        (block_dir / "DAILY_EXECUTION.json").write_text("{}", encoding="utf-8")
        ledgers[block_id] = tmp_path / f"ledger-{block_id}.json"
        ledgers[block_id].write_text("{}", encoding="utf-8")

    monkeypatch.setattr(confirmation, "DEVELOPMENT_CLAIM_REPORT", report_path)
    monkeypatch.setattr(confirmation, "DEVELOPMENT_COMBINED", combined_path)
    monkeypatch.setattr(
        confirmation,
        "DEVELOPMENT_BLOCK_RESULTS",
        tuple(block_results[block_id] for block_id in development.BLOCK_ORDER),
    )
    monkeypatch.setattr(
        confirmation.daily,
        "BLOCK_DIRS",
        {block_id: path.parent for block_id, path in block_results.items()},
    )
    monkeypatch.setattr(confirmation.daily, "LEDGERS", ledgers)
    monkeypatch.setattr(
        confirmation.daily,
        "verify_combined_result",
        lambda **kwargs: verification,
    )
    monkeypatch.setattr(
        confirmation.daily,
        "validate_block_result",
        lambda *, path, block_id, ledger_path: {
            "verified": True,
            "result_sha256": development.sha256_file(path),
        },
    )

    def validate_daily(*, block_id, **kwargs):
        return {"combined_endpoint_accessed": block_id == "d"}

    finalized = finalize.finalize_development_claim(
        combined_result=combined_path,
        claim_report_path=report_path,
        block_results=block_results,
        ledgers=ledgers,
        manifest_validator=lambda path: {"manifest_sha256": "m" * 64},
        aug10_validator=lambda **kwargs: {"verified": True},
        block_validator=lambda *, path, block_id, ledger_path: {
            "verified": True,
            "result_sha256": development.sha256_file(path),
        },
        daily_validator=validate_daily,
        combined_validator=lambda **kwargs: verification,
        authorization_validator=confirmation.verify_development_authorization,
    )
    assert finalized["status"] == "confirmation_handoff_verified"
    assert finalized["claim_tier"] == (
        "full_path_dependent_llm_native_development_signal"
    )
    assert finalized["confirmation_authorization"]["verified"] is True
    assert finalized["model_calls_made"] == 0
