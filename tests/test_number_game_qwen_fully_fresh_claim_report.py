from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from scripts import number_game_qwen_fully_fresh_claim_report as claim
from scripts import number_game_qwen_fully_fresh_control_verify as verify


def _fixture(
    *,
    policy: bool = True,
    dynamic: bool = True,
    matched: bool = True,
) -> tuple[dict, dict, dict, dict]:
    source = {
        "status": "passed" if policy and dynamic else "gated_null",
        "usage": {"adapter_requests": 3680, "run_cost_usd": 4.0},
        "mechanics_gates": {"mechanics": True},
        "myopic_policy_gates": {
            name: policy for name in claim.SOURCE_MYOPIC_GATES
        },
        "dynamic_support": {
            "gates": {name: dynamic for name in claim.SOURCE_DYNAMIC_GATES}
        },
    }
    control = {
        "status": "passed" if matched else "gated_null",
        "usage": {"adapter_requests": 3072, "run_cost_usd": 3.0},
        "mechanics_gates": {"mechanics": True},
        "analysis": {
            "scientific_gates": {
                name: matched for name in claim.CONTROL_SCIENCE_GATES
            }
        },
    }
    gates = {
        "source_mechanics_passed": True,
        "source_myopic_policy_passed": policy,
        "source_dynamic_support_passed": dynamic,
        "control_mechanics_passed": True,
        "control_science_passed": matched,
        "accepted_request_count_exact": True,
        "within_composite_budget": True,
    }
    composite = {"status": "passed" if all(gates.values()) else "gated_null"}
    composite["composite_gates"] = gates
    summary = {
        "source_depth3_vs_myopic": {
            "relative_brier_reduction": 0.15,
            "paired_brier_difference_95pct": [-0.02, -0.01],
            "wins_ties_losses": [26, 1, 5],
        },
        "source_dynamic_vs_fixed_depth3": {
            "relative_brier_reduction": 0.03,
            "paired_brier_difference_95pct": [-0.01, -0.001],
            "wins_ties_losses": [20, 2, 10],
        },
        "control_second_stage": {
            "differences": {
                "conditional_minus_history_blind_predictive_mse": -0.005
            }
        },
        "control_selected_root_conditioning": {
            "changed_root_tree_count": 24,
            "prompt_benefit_contrast_to_realized_advantage_spearman": 0.4,
        },
        "control_bootstrap": {
            "prompt_benefit_contrast_to_realized_spearman_95pct": [0.1, 0.7]
        },
        "control_scientific_gates": control["analysis"]["scientific_gates"],
    }
    artifacts = {"composite_result": "a" * 64}
    verification = {
        "status": "verified",
        "scientific_status_unchanged": True,
        "provider_calls": 0,
        "verification_cost_usd": 0.0,
        "checks": {"replay": True},
        "artifacts": artifacts,
        "summary": summary,
    }
    return source, control, composite, verification


def test_bound_source_claim_ceiling_is_partial_before_control() -> None:
    result = claim.preflight_source_claim_ceiling()
    assert result["status"] == "claim_ceiling_frozen_without_control"
    assert result["nonmyopic_policy_family"]["pass"]
    assert not result["dynamic_support_endpoint_family"]["pass"]
    assert not result["full_fresh_llm_native_replication_reachable"]
    assert result["maximum_tier_if_control_passes"] == (
        "nonmyopic_policy_with_partial_llm_mechanism"
    )
    assert result["model_calls"] == 0
    assert result["files_written"] == 0


@pytest.mark.parametrize(
    ("policy", "dynamic", "matched", "tier", "upgrade"),
    [
        (True, True, True, "full_fresh_llm_native_replication", True),
        (
            True,
            False,
            True,
            "nonmyopic_policy_with_partial_llm_mechanism",
            False,
        ),
        (
            True,
            True,
            False,
            "nonmyopic_policy_with_partial_llm_mechanism",
            False,
        ),
        (
            True,
            False,
            False,
            "nonmyopic_policy_without_fresh_mechanism",
            False,
        ),
        (False, False, True, "fresh_mechanism_without_policy", False),
        (False, False, False, "fresh_replication_null", False),
    ],
)
def test_claim_tiers_keep_policy_and_mechanism_families_separate(
    policy: bool,
    dynamic: bool,
    matched: bool,
    tier: str,
    upgrade: bool,
) -> None:
    source, control, composite, verification = _fixture(
        policy=policy,
        dynamic=dynamic,
        matched=matched,
    )
    report = claim.build_claim_report(
        source=source,
        control=control,
        composite=composite,
        verification=verification,
        artifact_hashes=verification["artifacts"],
    )
    assert report["claim_tier"] == tier
    assert report["authorizes_full_fresh_claim_upgrade"] is upgrade
    assert report["fresh_control_cannot_relabel_source"]
    assert report["diversity_confirmation_schedule_unchanged"]


def test_actual_known_source_shape_can_only_reach_partial_tier() -> None:
    source, control, composite, verification = _fixture(dynamic=False)
    report = claim.build_claim_report(
        source=source,
        control=control,
        composite=composite,
        verification=verification,
        artifact_hashes=verification["artifacts"],
    )
    assert report["claim_tier"] == (
        "nonmyopic_policy_with_partial_llm_mechanism"
    )
    assert not report["authorizes_full_fresh_claim_upgrade"]


def test_claim_report_rejects_composite_or_verification_mismatch() -> None:
    source, control, composite, verification = _fixture()
    composite["composite_gates"]["source_dynamic_support_passed"] = False
    with pytest.raises(ValueError, match="composite gates disagree"):
        claim.classify_result(
            source=source,
            control=control,
            composite=composite,
        )

    source, control, composite, verification = _fixture()
    verification["artifacts"] = {"composite_result": "b" * 64}
    with pytest.raises(ValueError, match="verification"):
        claim.build_claim_report(
            source=source,
            control=control,
            composite=composite,
            verification=verification,
            artifact_hashes={"composite_result": "a" * 64},
        )


def test_mechanics_and_science_are_recorded_separately() -> None:
    source, control, composite, _ = _fixture()
    control["mechanics_gates"]["mechanics"] = False
    composite["composite_gates"]["control_mechanics_passed"] = False
    composite["status"] = "mechanics_failed"
    classification = claim.classify_result(
        source=source,
        control=control,
        composite=composite,
    )
    assert composite["composite_gates"]["control_science_passed"] is True
    assert classification["control_mechanics_pass"] is False
    assert classification["matched_conditioning_family"]["pass"] is False


def test_claim_report_rejects_nonfinite_metrics() -> None:
    source, control, composite, verification = _fixture()
    bad = deepcopy(verification)
    bad["summary"]["source_depth3_vs_myopic"][
        "relative_brier_reduction"
    ] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        claim.build_claim_report(
            source=source,
            control=control,
            composite=composite,
            verification=bad,
            artifact_hashes=bad["artifacts"],
        )


def test_claim_report_banks_json_and_markdown_once(tmp_path) -> None:
    source, control, composite, verification = _fixture(dynamic=False)
    report = claim.build_claim_report(
        source=source,
        control=control,
        composite=composite,
        verification=verification,
        artifact_hashes=verification["artifacts"],
    )
    json_path = tmp_path / "CLAIM_REPORT.json"
    markdown_path = tmp_path / "CLAIM_REPORT.md"
    first = claim.bank_claim_report(
        json_path=json_path,
        markdown_path=markdown_path,
        report=report,
    )
    second = claim.bank_claim_report(
        json_path=json_path,
        markdown_path=markdown_path,
        report=report,
    )
    assert second == first
    markdown_path.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        claim.bank_claim_report(
            json_path=json_path,
            markdown_path=markdown_path,
            report=report,
        )


def test_write_claim_report_replays_and_banks_idempotently(
    tmp_path, monkeypatch
) -> None:
    source, control, composite, verification = _fixture(dynamic=False)
    run_dir = tmp_path / "run"
    (run_dir / "source").mkdir(parents=True)
    (run_dir / "control").mkdir()
    for path, payload in (
        (run_dir / "source/RESULT.json", source),
        (run_dir / "control/RESULT.json", control),
        (run_dir / "RESULT.json", composite),
        (run_dir / "CONTROL_VERIFICATION.json", verification),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        claim.verify,
        "verify_completed_control",
        lambda **_: deepcopy(verification),
    )

    first = claim.write_claim_report(run_dir=run_dir)
    second = claim.write_claim_report(run_dir=run_dir)

    assert second == first
    assert first["claim_tier"] == (
        "nonmyopic_policy_with_partial_llm_mechanism"
    )
    assert (run_dir / "CLAIM_REPORT.json").exists()
    assert (run_dir / "CLAIM_REPORT.md").exists()
