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
                "mean_brier": _summary(-0.03)
            }
        },
        "paired_luna_minus_siglip": {
            "luna_dynamic_minus_siglip_depth2": {
                "mean_brier": _summary(0.01)
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
        metadata = {"stage": stage, "claim_tier": "development_null"}
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
