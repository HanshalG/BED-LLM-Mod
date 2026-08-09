from __future__ import annotations

import json
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
    output = tmp_path / "generated/bongard_openworld_result.tex"
    suite = tmp_path / "CLASSICAL_SUITE_RESULT.json"
    combined = tmp_path / "COMBINED_RESULT.json"
    suite.write_text("{}", encoding="utf-8")
    combined.write_text("{}", encoding="utf-8")
    result = paper.write_combined_fragment(
        stage="development",
        output=output,
        classical_suite_path=suite,
        claim_report_path=tmp_path / "CLAIM.json",
        combined_result=combined,
        block_results=[tmp_path / f"block-{index}.json" for index in range(4)],
    )
    tex = output.read_text(encoding="utf-8")
    assert tex.startswith("ORIGINAL LUNA FRAGMENT")
    assert "DINOv2-small" in tex
    assert "SigLIP2-So400m" in tex
    assert output.with_name(
        paper.luna_fragment.HEADLINE_FILENAME
    ).read_text(encoding="utf-8") == "ORIGINAL HEADLINE\n"
    assert result["status"] == "written_with_mandatory_classical_suite"
    assert result["claim_tier"] == "development_null"


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
    output = tmp_path / "generated/bongard_openworld_result.tex"
    suite = tmp_path / "CLASSICAL_SUITE_RESULT.json"
    combined = tmp_path / "COMBINED_RESULT.json"
    suite.write_text("{}", encoding="utf-8")
    combined.write_text("{}", encoding="utf-8")
    paper.write_combined_fragment(
        stage="confirmation",
        output=output,
        classical_suite_path=suite,
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
