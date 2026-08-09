from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import bongard_openworld_paper_with_classical_suite as paper


def _summary(mean: float) -> dict:
    return {
        "values": [mean, mean],
        "mean": mean,
        "bootstrap_draws": 20_000,
        "bootstrap_seed": 20260808,
        "bootstrap_95pct_ci": [mean - 0.01, mean + 0.01],
    }


def _result() -> dict:
    return {
        "status": "classical_suite_complete",
        "stage_result_sha256": "synthetic-stage-sha256",
        "all_gates_pass": True,
        "authorizes_paid_calls": False,
        "dino": {
            "pooled": {
                "dinov2_myopic": {"mean_brier": 0.24},
                "dinov2_depth2": {"mean_brier": 0.22},
            },
            "paired_depth2_minus_myopic": {
                "mean_brier": _summary(-0.02)
            },
        },
        "siglip": {
            "pooled": {
                "siglip_myopic": {"mean_brier": 0.20},
                "siglip_depth2": {"mean_brier": 0.18},
            },
            "paired_depth2_minus_myopic": {
                "mean_brier": _summary(-0.02)
            },
        },
        "paired_luna_minus_dino": {
            "luna_dynamic_minus_dinov2_depth2": {
                "mean_brier": _summary(-0.03),
                "mean_log_loss": _summary(-0.02),
            }
        },
        "paired_luna_minus_siglip": {
            "luna_dynamic_minus_siglip_depth2": {
                "mean_brier": _summary(0.01),
                "mean_log_loss": _summary(0.00),
            }
        },
    }


def _compute_summary(mean: float, *, n: int = 2) -> dict:
    sample_sd = 0.02
    return {
        "n": n,
        "mean_difference": mean,
        "sample_sd": sample_sd,
        "standard_error": sample_sd / math.sqrt(n),
        "ci95": [mean - 0.01, mean + 0.01],
        "bootstrap_probability_improvement": 1.0 if mean < 0 else 0.0,
        "wins": n if mean < 0 else 0,
        "ties": 0,
        "losses": n if mean > 0 else 0,
        "bootstrap_draws": 20_000,
        "negative_favors": "dynamic_depth2",
    }


def _compute_result(*, stage: str = "development") -> dict:
    return {
        "status": "compute_matched_control_audit_complete",
        "stage": stage,
        "stage_result_sha256": "synthetic-stage-sha256",
        "task_count": 2,
        "strict_compute_matched_control": "shuffled_dynamic_depth2",
        "matched_request_count_control": "history_blind_depth2",
        "online_regeneration_greedy_control": "myopic_width",
        "compute_contract_exact": True,
        "comparisons": {
            "shuffled_dynamic_depth2": {
                "mean_brier": _compute_summary(-0.03),
                "mean_log_loss": _compute_summary(-0.02),
                "first_query_changes": 1,
                "final_history_changes": 2,
            }
        },
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "changes_claim_tier": False,
    }


def test_classical_tex_reports_both_encoders_and_luna_comparisons() -> None:
    tex = "\n".join(paper.classical_tex_lines(_result()))
    assert "DINOv2-small" in tex
    assert "SigLIP2-So400m" in tex
    assert "0.2400" in tex
    assert "0.1800" in tex
    assert "-0.0300" in tex
    assert "0.0100" in tex
    assert "negative values favor Luna" in tex
    assert "reported regardless of direction" in tex


def test_classical_tex_rejects_incomplete_suite() -> None:
    invalid = _result()
    invalid["all_gates_pass"] = False
    with pytest.raises(ValueError, match="complete frozen classical suite"):
        paper.classical_tex_lines(invalid)


def test_compute_matched_tex_reports_strict_shuffled_comparison() -> None:
    lines, metadata = paper.compute_matched_tex_lines(_compute_result())
    tex = "\n".join(lines)
    assert "Compute-matched shuffled continuation" in tex
    assert "root scores and continuation-value multiset" in tex
    assert "-0.0300" in tex
    assert "-0.0200" in tex
    assert "1/2 tasks" in tex
    assert "2/2" in tex
    assert "negative differences favor dynamic" in tex
    assert "descriptive and cannot alter" in tex
    assert metadata["strict_compute_matched_control"] == (
        "shuffled_dynamic_depth2"
    )
    assert metadata["changes_claim_tier"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("control", "complete compute-matched audit"),
        ("calls", "complete compute-matched audit"),
        ("claim", "complete compute-matched audit"),
        ("sample_size", "invalid mean_brier"),
        ("standard_error", "invalid mean_brier"),
        ("action_count", "action-change counts"),
    ],
)
def test_compute_matched_tex_rejects_tamper(
    mutation: str, message: str
) -> None:
    result = deepcopy(_compute_result())
    if mutation == "control":
        result["strict_compute_matched_control"] = "myopic_width"
    elif mutation == "calls":
        result["model_calls"] = 1
    elif mutation == "claim":
        result["changes_claim_tier"] = True
    elif mutation == "sample_size":
        result["comparisons"]["shuffled_dynamic_depth2"]["mean_brier"][
            "n"
        ] = 1
    elif mutation == "standard_error":
        result["comparisons"]["shuffled_dynamic_depth2"]["mean_brier"][
            "standard_error"
        ] += 0.01
    else:
        result["comparisons"]["shuffled_dynamic_depth2"][
            "first_query_changes"
        ] = 3
    with pytest.raises(ValueError, match=message):
        paper.compute_matched_tex_lines(result)


def test_classical_claim_scope_requires_both_challengers() -> None:
    result = _result()
    scope = paper.classical_claim_scope(result)
    assert scope["status"] == "within_luna_only_no_task_level_necessity"
    assert scope["comparisons"]["dinov2_depth2"]["clears_challenge"] is True
    assert scope["comparisons"]["siglip_depth2"]["clears_challenge"] is False

    result["paired_luna_minus_siglip"][
        "luna_dynamic_minus_siglip_depth2"
    ]["mean_brier"] = _summary(-0.03)
    result["paired_luna_minus_siglip"][
        "luna_dynamic_minus_siglip_depth2"
    ]["mean_log_loss"] = _summary(-0.01)
    scope = paper.classical_claim_scope(result)
    assert scope["status"] == "luna_clears_both_fixed_classical_challengers"
    assert scope["clears_both_fixed_classical_challengers"] is True


def test_classical_claim_scope_fails_closed_on_missing_log_loss() -> None:
    result = _result()
    del result["paired_luna_minus_dino"][
        "luna_dynamic_minus_dinov2_depth2"
    ]["mean_log_loss"]
    with pytest.raises(ValueError, match="lacks mean_log_loss"):
        paper.classical_claim_scope(result)


def test_compute_matched_replay_rejects_saved_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    saved = tmp_path / "AUDIT.json"
    saved.write_text(json.dumps({"status": "saved"}), encoding="utf-8")
    monkeypatch.setattr(
        paper.compute_control,
        "run_report",
        lambda **kwargs: {"status": "independent-replay"},
    )
    with pytest.raises(ValueError, match="does not independently replay"):
        paper.replay_compute_matched_audit(
            stage="development",
            saved_path=saved,
            result_path=tmp_path / "COMBINED.json",
            block_results=[],
            output_path=tmp_path / "REPLAY.json",
        )


@pytest.mark.skipif(
    shutil.which("pdflatex") is None,
    reason="LaTeX toolchain is unavailable",
)
def test_classical_tex_compiles(tmp_path: Path) -> None:
    source = tmp_path / "classical.tex"
    source.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        + "\n".join(paper.classical_tex_lines(_result()))
        + "\n"
        + "\n".join(paper.compute_matched_tex_lines(_compute_result())[0])
        + "\n\\end{document}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", source.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_combined_writer_preserves_headline_and_appends_both_encoders(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_luna_writer(*, output: Path, stage: str, **kwargs):
        output.write_text("ORIGINAL LUNA FRAGMENT\n", encoding="utf-8")
        headline = output.with_name(paper.luna_fragment.HEADLINE_FILENAME)
        headline.write_text("ORIGINAL HEADLINE\n", encoding="utf-8")
        metadata = {
            "stage": stage,
            "claim_tier": "development_null",
            "headline": {
                "authorized": False,
                "abstract_tex": "",
                "contribution_tex": "",
            },
        }
        output.with_suffix(".json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return {
            "status": "written",
            "stage": stage,
            "claim_tier": metadata["claim_tier"],
        }

    monkeypatch.setattr(
        paper, "verify_bound_implementations", lambda: {"verified": "yes"}
    )
    monkeypatch.setattr(
        paper.luna_fragment, "write_fragment", fake_luna_writer
    )
    monkeypatch.setattr(
        paper, "replay_classical_suite", lambda **kwargs: _result()
    )
    monkeypatch.setattr(
        paper,
        "replay_compute_matched_audit",
        lambda **kwargs: _compute_result(stage=kwargs["stage"]),
    )
    output = tmp_path / "generated/bongard_openworld_result.tex"
    suite = tmp_path / "CLASSICAL_SUITE_RESULT.json"
    combined = tmp_path / "COMBINED_RESULT.json"
    compute = tmp_path / "COMPUTE_MATCHED_AUDIT.json"
    suite.write_text("{}", encoding="utf-8")
    combined.write_text("{}", encoding="utf-8")
    compute.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="classical and compute-matched suites"):
        paper.write_combined_fragment(
            stage="development",
            output=output,
            classical_suite_path=suite,
            compute_audit_path=None,
            claim_report_path=tmp_path / "CLAIM.json",
            combined_result=combined,
            block_results=[],
        )
    monkeypatch.setattr(
        paper,
        "replay_compute_matched_audit",
        lambda **kwargs: {
            **_compute_result(stage=kwargs["stage"]),
            "stage_result_sha256": "wrong-stage-result",
        },
    )
    with pytest.raises(ValueError, match="does not match the rendered stage result"):
        paper.write_combined_fragment(
            stage="development",
            output=output,
            classical_suite_path=suite,
            compute_audit_path=compute,
            claim_report_path=tmp_path / "CLAIM.json",
            combined_result=combined,
            block_results=[],
        )
    monkeypatch.setattr(
        paper,
        "replay_compute_matched_audit",
        lambda **kwargs: _compute_result(stage=kwargs["stage"]),
    )
    result = paper.write_combined_fragment(
        stage="development",
        output=output,
        classical_suite_path=suite,
        compute_audit_path=compute,
        claim_report_path=tmp_path / "CLAIM.json",
        combined_result=combined,
        block_results=[tmp_path / f"block-{index}.json" for index in range(4)],
    )
    tex = output.read_text(encoding="utf-8")
    assert tex.startswith("ORIGINAL LUNA FRAGMENT")
    assert "DINOv2-small" in tex
    assert "SigLIP2-So400m" in tex
    assert "Compute-matched shuffled continuation" in tex
    assert output.with_name(
        paper.luna_fragment.HEADLINE_FILENAME
    ).read_text(encoding="utf-8") == "ORIGINAL HEADLINE\n"
    assert result["status"] == (
        "written_with_mandatory_classical_and_compute_suites"
    )
    assert result["claim_tier"] == "development_null"
    with pytest.raises(
        ValueError, match="mechanics failure cannot have a compute-matched"
    ):
        paper.write_combined_fragment(
            stage="confirmation-mechanics-failure",
            output=tmp_path / "generated/mechanics-failure.tex",
            classical_suite_path=None,
            compute_audit_path=compute,
            failure_path=tmp_path / "FAILURE.json",
        )


@pytest.mark.parametrize(
    ("siglip_brier", "expected_text", "expected_status"),
    [
        (
            -0.03,
            "clears both frozen fixed semantic-vision challengers",
            "luna_clears_both_fixed_classical_challengers",
        ),
        (
            0.01,
            "does not establish task-level LLM necessity",
            "within_luna_only_no_task_level_necessity",
        ),
    ],
)
def test_confirmation_headline_is_qualified_by_classical_scope(
    tmp_path: Path,
    monkeypatch,
    siglip_brier: float,
    expected_text: str,
    expected_status: str,
) -> None:
    def fake_luna_writer(*, output: Path, stage: str, **kwargs):
        output.write_text("ORIGINAL LUNA FRAGMENT\n", encoding="utf-8")
        headline = output.with_name(paper.luna_fragment.HEADLINE_FILENAME)
        headline.write_text("ORIGINAL HEADLINE\n", encoding="utf-8")
        metadata = {
            "stage": stage,
            "claim_tier": "full_llm_native_confirmation",
            "headline": {
                "authorized": True,
                "abstract_tex": "ORIGINAL ABSTRACT.",
                "contribution_tex": "\\item ORIGINAL CONTRIBUTION;",
            },
        }
        output.with_suffix(".json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return {
            "status": "written",
            "stage": stage,
            "claim_tier": metadata["claim_tier"],
        }

    suite_result = _result()
    suite_result["paired_luna_minus_siglip"][
        "luna_dynamic_minus_siglip_depth2"
    ]["mean_brier"] = _summary(siglip_brier)
    monkeypatch.setattr(
        paper, "verify_bound_implementations", lambda: {"verified": "yes"}
    )
    monkeypatch.setattr(
        paper.luna_fragment, "write_fragment", fake_luna_writer
    )
    monkeypatch.setattr(
        paper, "replay_classical_suite", lambda **kwargs: suite_result
    )
    monkeypatch.setattr(
        paper,
        "replay_compute_matched_audit",
        lambda **kwargs: _compute_result(stage=kwargs["stage"]),
    )
    output = tmp_path / "generated/bongard_openworld_result.tex"
    suite = tmp_path / "CLASSICAL_SUITE_RESULT.json"
    combined = tmp_path / "COMBINED_RESULT.json"
    compute = tmp_path / "COMPUTE_MATCHED_AUDIT.json"
    suite.write_text("{}", encoding="utf-8")
    combined.write_text("{}", encoding="utf-8")
    compute.write_text("{}", encoding="utf-8")
    paper.write_combined_fragment(
        stage="confirmation",
        output=output,
        classical_suite_path=suite,
        compute_audit_path=compute,
        combined_result=combined,
        block_results=[tmp_path / f"block-{index}.json" for index in range(4)],
    )
    headline = output.with_name(
        paper.luna_fragment.HEADLINE_FILENAME
    ).read_text(encoding="utf-8")
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert "ORIGINAL ABSTRACT." in headline
    assert "ORIGINAL CONTRIBUTION" in headline
    assert expected_text in headline
    assert metadata["classical_claim_scope"]["status"] == expected_status
    assert metadata["compute_matched_audit"]["summary"][
        "strict_compute_matched_control"
    ] == "shuffled_dynamic_depth2"
