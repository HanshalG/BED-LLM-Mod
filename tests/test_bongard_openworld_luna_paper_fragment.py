from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from scripts import bongard_openworld_luna_claim_report as claim
from scripts import bongard_openworld_luna_paper_fragment as fragment
from scripts import bongard_openworld_luna_vlm_development as development
from scripts.validate_paper_draft import validate_paper_draft


def _summary(n: int) -> dict:
    return {
        "n": n,
        "mean_difference": -0.02,
        "sample_sd": 0.04,
        "ci95": [-0.03, -0.01],
        "bootstrap_probability_improvement": 0.95,
        "wins": n - 12,
        "ties": 4,
        "losses": 8,
    }


def _metrics(n: int) -> dict:
    summary = _summary(n)
    return {
        "pooled_policy_metrics": {
            "dynamic_depth2": {"mean_brier": 0.12, "mean_log_loss": 0.40},
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
            "dynamic_depth2": {"mean_spearman": 0.30, "sample_sd": 0.20},
            "myopic_width": {"mean_spearman": 0.20, "sample_sd": 0.20},
            "history_blind_depth2": {
                "mean_spearman": 0.25,
                "sample_sd": 0.20,
            },
        },
        "dynamic_vs_myopic_relative_brier_improvement": 0.10,
        "dynamic_vs_history_blind_relative_brier_improvement": 0.06,
        "dynamic_vs_fixed_depth2_relative_brier_improvement": 0.05,
        "dynamic_vs_fixed_score_dynamic_update_relative_brier_improvement": 0.04,
        "dynamic_vs_history_blind_update_matched_first_relative_brier_improvement": 0.04,
        "dynamic_vs_myopic_changed_final_histories": n // 2,
        "dynamic_vs_history_blind_changed_final_histories": n // 2 - 1,
        "dynamic_vs_fixed_depth2_changed_final_histories": n // 2 - 2,
        "dynamic_vs_fixed_score_dynamic_update_changed_final_histories": n // 2 - 3,
        "dynamic_vs_history_blind_update_matched_first_changed_final_histories": n // 2 - 4,
        "dynamic_vs_history_blind_update_matched_first_robust_second_action_changes": n // 2 - 4,
    }


def _development_result(*, failed: tuple[str, ...] = ()) -> dict:
    gates = {name: name not in failed for name in claim.EXPECTED_GATES}
    gates["all_pass"] = all(gates.values())
    signal = gates["all_pass"]
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
        **_metrics(development.TASKS),
    }


def _prepare_development(
    tmp_path: Path, monkeypatch, *, failed: tuple[str, ...] = ()
) -> tuple[Path, Path, list[Path], dict]:
    result = _development_result(failed=failed)
    combined = tmp_path / "COMBINED_RESULT.json"
    combined.write_text(json.dumps(result), encoding="utf-8")
    result_hash = fragment.sha256_file(combined)
    verification = {
        "verified": True,
        "status": result["status"],
        "result_sha256": result_hash,
        "authorizes_confirmation_preregistration": result[
            "authorizes_confirmation_preregistration"
        ],
    }
    monkeypatch.setattr(
        fragment.development_daily,
        "verify_combined_result",
        lambda **kwargs: verification,
    )
    report = claim.build_claim_report(
        result,
        result_sha256=result_hash,
        independent_verification=verification,
    )
    report_path = tmp_path / "CLAIM_REPORT.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    blocks = []
    for index in range(4):
        path = tmp_path / f"block-{index}.json"
        path.write_text("{}", encoding="utf-8")
        blocks.append(path)
    return report_path, combined, blocks, report


@pytest.mark.parametrize(
    ("failed", "expected_tier", "expected_phrase"),
    [
        ((), "full_path_dependent_llm_native_development_signal", "not a confirmed claim"),
        (
            (claim.PATH_DEPENDENT_GATES[0],),
            "policy_and_matched_regeneration_without_fixed_support_superiority",
            "does not establish complete superiority",
        ),
        (
            (claim.MECHANISM_GATES[0],),
            "policy_signal_without_matched_mechanism",
            "does not support causal attribution",
        ),
        (
            (claim.POLICY_GATES[0],),
            "matched_mechanism_without_policy_signal",
            "not a non-myopic policy win",
        ),
        (
            (claim.POLICY_GATES[0], claim.MECHANISM_GATES[0]),
            "development_null",
            "supports neither a positive policy claim",
        ),
    ],
)
def test_development_fragment_obeys_every_frozen_tier(
    tmp_path, monkeypatch, failed, expected_tier, expected_phrase
) -> None:
    report, combined, blocks, _ = _prepare_development(
        tmp_path, monkeypatch, failed=failed
    )

    tex, metadata = fragment.build_development_fragment(
        claim_report_path=report,
        combined_result=combined,
        block_results=blocks,
    )

    assert metadata["claim_tier"] == expected_tier
    assert metadata["confirmation_authorized"] is False
    assert metadata["development_and_confirmation_are_not_pooled"] is True
    assert expected_phrase in tex
    assert "14-image tasks" in tex
    assert "ten history-conditioned predictive particles" in tex
    assert "predictive positive-label probability" in tex
    assert "rule strings were interpretive descriptions" in tex
    assert "calibrated likelihoods" not in tex
    assert "endpoint IDs but no labels" in tex
    assert "Relative reductions" in tex
    assert "fixed-support" in tex
    assert "history-blind" in tex
    assert "matched fixed-score" in tex
    assert "matched realized updater" in tex
    assert "common realized updater" in tex
    assert "belief regeneration improves over" not in tex


def test_development_shared_validity_failure_suppresses_efficacy(
    tmp_path, monkeypatch
) -> None:
    report, combined, blocks, saved = _prepare_development(
        tmp_path, monkeypatch, failed=(claim.SHARED_GATES[0],)
    )

    tex, metadata = fragment.build_development_fragment(
        claim_report_path=report,
        combined_result=combined,
        block_results=blocks,
    )

    assert saved["shared_validity"]["pass"] is False
    assert metadata["shared_validity_pass"] is False
    assert "mechanics-inconclusive" in tex
    assert "Relative reductions" not in tex
    assert "endpoint Brier was" not in tex


def test_development_fragment_refuses_claim_report_tamper(
    tmp_path, monkeypatch
) -> None:
    report, combined, blocks, saved = _prepare_development(tmp_path, monkeypatch)
    tampered = deepcopy(saved)
    tampered["claim_tier"] = "development_null"
    report.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="does not replay exactly"):
        fragment.build_development_fragment(
            claim_report_path=report,
            combined_result=combined,
            block_results=blocks,
        )


def _confirmation_result(*, passed: bool) -> dict:
    tier = "full_llm_native_confirmation" if passed else "confirmation_null"
    status = "confirmation_pass" if passed else "confirmation_null"
    return {
        "schema_version": 1,
        "status": status,
        "claim_tier": tier,
        **_metrics(96),
    }


def _prepare_confirmation(tmp_path: Path, monkeypatch, *, passed: bool):
    result = _confirmation_result(passed=passed)
    result_path = tmp_path / "COMBINED_RESULT.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    result_hash = fragment.sha256_file(result_path)
    monkeypatch.setattr(
        fragment.confirmation,
        "verify_development_authorization",
        lambda: {"verified": True, "claim_tier": "full-development"},
    )
    monkeypatch.setattr(
        fragment.confirmation,
        "verify_combined_result",
        lambda **kwargs: {
            "verified": True,
            "status": result["status"],
            "claim_tier": result["claim_tier"],
            "result_sha256": result_hash,
        },
    )
    blocks = []
    for index in range(4):
        path = tmp_path / f"confirmation-block-{index}.json"
        path.write_text("{}", encoding="utf-8")
        blocks.append(path)
    return result_path, blocks, result


@pytest.mark.parametrize(
    ("passed", "expected_tier", "confirmed", "phrase"),
    [
        (
            True,
            "full_llm_native_confirmation",
            True,
            "confirms the registered multimodal LLM-native",
        ),
        (
            False,
            "confirmation_null",
            False,
            "no Bongard headline claim is authorized",
        ),
    ],
)
def test_confirmation_fragment_obeys_pass_and_null(
    tmp_path, monkeypatch, passed, expected_tier, confirmed, phrase
) -> None:
    result, blocks, _ = _prepare_confirmation(
        tmp_path, monkeypatch, passed=passed
    )

    tex, metadata = fragment.build_confirmation_fragment(
        result_path=result, block_results=blocks
    )

    assert metadata["claim_tier"] == expected_tier
    assert metadata["confirmation_authorized"] is confirmed
    assert phrase in tex
    assert "Relative reductions" in tex
    assert "not pooled" in tex
    assert metadata["headline"]["authorized"] is passed
    assert bool(metadata["headline"]["abstract_tex"]) is passed
    assert bool(metadata["headline"]["contribution_tex"]) is passed
    if passed:
        assert "untouched 96-task visual confirmation" in metadata["headline"][
            "abstract_tex"
        ]
        assert "paired 95\\% CI" in metadata["headline"]["abstract_tex"]


def test_confirmation_fragment_refuses_replay_mismatch(
    tmp_path, monkeypatch
) -> None:
    result, blocks, _ = _prepare_confirmation(tmp_path, monkeypatch, passed=True)
    monkeypatch.setattr(
        fragment.confirmation,
        "verify_combined_result",
        lambda **kwargs: {
            "verified": True,
            "status": "confirmation_null",
            "claim_tier": "confirmation_null",
            "result_sha256": fragment.sha256_file(result),
        },
    )

    with pytest.raises(ValueError, match="replay"):
        fragment.build_confirmation_fragment(
            result_path=result, block_results=blocks
        )


def test_failed_closed_confirmation_has_no_efficacy_metrics(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        fragment.confirmation,
        "verify_development_authorization",
        lambda: {"verified": True, "claim_tier": "full-development"},
    )
    failure_path = tmp_path / "FAILURE.json"
    failure_path.write_text(
        json.dumps(
            {
                "schema_version": fragment.confirmation_daily.SCHEMA_VERSION,
                "interface_version": fragment.confirmation_daily.INTERFACE_VERSION,
                "status": "failed_closed",
                "block_id": "b",
                "error_type": "RuntimeError",
                "error": "frozen mechanics gate failed",
            }
        ),
        encoding="utf-8",
    )
    combined = tmp_path / "COMBINED_RESULT.json"

    tex, metadata = fragment.build_confirmation_mechanics_failure_fragment(
        failure_path=failure_path, combined_result=combined
    )

    assert metadata["claim_tier"] == "confirmation_mechanics_inconclusive"
    assert metadata["confirmation_authorized"] is False
    assert metadata["efficacy_metrics_rendered"] is False
    assert "mechanics-inconclusive" in tex
    assert "Relative reductions" not in tex
    combined.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot coexist"):
        fragment.build_confirmation_mechanics_failure_fragment(
            failure_path=failure_path, combined_result=combined
        )


def test_fragment_writes_tex_and_metadata_without_calls(
    tmp_path, monkeypatch
) -> None:
    report, combined, blocks, _ = _prepare_development(tmp_path, monkeypatch)
    output = tmp_path / "paper/generated/bongard_openworld_result.tex"

    written = fragment.write_fragment(
        stage="development",
        output=output,
        claim_report_path=report,
        combined_result=combined,
        block_results=blocks,
    )

    assert written["status"] == "written"
    assert written["model_calls"] == 0
    assert written["cost_usd"] == 0.0
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["tex_sha256"] == fragment.sha256_file(output)
    headline = output.with_name(fragment.HEADLINE_FILENAME)
    assert written["headline_sha256"] == fragment.sha256_file(headline)
    assert metadata["headline_tex_sha256"] == fragment.sha256_file(headline)
    assert "BongardAbstractResult" in headline.read_text()
    assert metadata["fragment_protocol_sha256"] == (
        fragment.FRAGMENT_PROTOCOL_SHA256
    )
    role = metadata["llm_native_computational_role"]
    assert role["llm_supplies_history_conditioned_probability_matrix"] is True
    assert role["rule_strings_used_numerically"] is False
    assert role["universal_classical_impossibility_claim"] is False


def test_fragment_binding_matches_current_files() -> None:
    path = fragment.REPO_ROOT / (
        "results/nonmyopic/bongard_openworld_paper_fragment/"
        "PAPER_FRAGMENT_BINDING.json"
    )
    binding = json.loads(path.read_text())

    assert binding["status"] == "prospectively_frozen_before_any_bongard_response"
    for row in binding["bound_files"].values():
        assert fragment.sha256_file(fragment.REPO_ROOT / row["path"]) == row[
            "sha256"
        ]
    manuscript = binding["manuscript"]
    assert fragment.sha256_file(fragment.REPO_ROOT / manuscript["path"]) == (
        manuscript["preresult_sha256"]
    )
    assert manuscript["generated_fragment_absent_at_freeze"] is True
    assert manuscript["generated_headline_absent_at_freeze"] is True
    assert manuscript["live_hook_absent_at_freeze"] is False
    assert manuscript["conditional_include_to_be_added_after_verified_result"] is False
    assert fragment.sha256_file(
        fragment.REPO_ROOT / manuscript["references_path"]
    ) == manuscript["references_sha256"]
    assert manuscript[
        "detailed_late_number_game_audit_replaced_after_verified_result"
    ] is True
    assert binding["requirements"]["development_and_confirmation_can_be_pooled"] is False
    assert binding["requirements"]["confirmed_tier_requires_confirmation_pass"] is True
    assert binding["requirements"][
        "nonempty_abstract_and_contribution_require_full_confirmation"
    ] is True


@pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("bibtex") is None,
    reason="LaTeX toolchain is unavailable",
)
@pytest.mark.parametrize("passed", [False, True])
def test_confirmation_fragment_compiles_within_page_budget(
    tmp_path, monkeypatch, passed
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result, blocks, _ = _prepare_confirmation(
        run_dir, monkeypatch, passed=passed
    )
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    original = (fragment.REPO_ROOT / "paper/main.tex").read_text(encoding="utf-8")
    (paper_dir / "main.tex").write_text(original, encoding="utf-8")
    shutil.copy2(
        fragment.REPO_ROOT / "paper/references.bib",
        paper_dir / "references.bib",
    )
    plot_dir = tmp_path / "plots/nonmyopic"
    plot_dir.mkdir(parents=True)
    shutil.copy2(
        fragment.REPO_ROOT
        / "plots/nonmyopic/rocksample_scale_confirmation_entropy.png",
        plot_dir / "rocksample_scale_confirmation_entropy.png",
    )
    fragment.write_fragment(
        stage="confirmation",
        output=paper_dir / "generated/bongard_openworld_result.tex",
        combined_result=result,
        block_results=blocks,
    )

    checks = validate_paper_draft(paper_dir)

    assert all(check.ok for check in checks), [
        (check.name, check.detail) for check in checks if not check.ok
    ]
    rendered = (paper_dir / "generated" / fragment.HEADLINE_FILENAME).read_text()
    assert ("untouched 96-task visual confirmation" in rendered) is passed
