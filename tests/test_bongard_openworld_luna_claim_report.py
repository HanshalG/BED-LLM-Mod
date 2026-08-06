from __future__ import annotations

from copy import deepcopy
import math

import pytest

from scripts import bongard_openworld_luna_claim_report as claim
from scripts import bongard_openworld_luna_vlm_development as development


def _result(*, failed: tuple[str, ...] = ()) -> dict:
    gates = {name: name not in failed for name in claim.EXPECTED_GATES}
    gates["all_pass"] = all(gates.values())
    signal = gates["all_pass"]
    summary = {
        "n": 32,
        "mean_difference": -0.01,
        "sample_sd": 0.02,
        "ci95": [-0.02, -0.001],
        "bootstrap_probability_improvement": 0.9,
        "wins": 20,
        "ties": 2,
        "losses": 10,
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
        "dynamic_vs_myopic_changed_final_histories": 16,
        "dynamic_vs_history_blind_changed_final_histories": 14,
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
        ((), "full_llm_native_development_signal", True),
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
