from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts import regretbench_deepseek_smc_confirmation_paper_fragment as fragment
from scripts.validate_paper_draft import validate_paper_draft


TIERS = {
    "mechanics_failed": "smc_confirmation_mechanics_failure_no_result",
    "gated_null": "smc_confirmation_null_no_headline_result",
    "passed": "smc_confirmed_nonmyopic_semantic_particle_result",
}


def _report(status: str) -> dict:
    paired = None
    disagreements = None
    predicted = None
    if status != "mechanics_failed":
        paired = {
            name: {
                "brier_dynamic_minus_baseline": {
                    "mean": -0.03,
                    "sample_sd": 0.1,
                    "ci95": [-0.05, -0.01],
                    "probability_improvement": 0.97,
                },
                "log_loss_dynamic_minus_baseline": {
                    "mean": -0.02,
                    "sample_sd": 0.1,
                    "ci95": [-0.04, 0.0],
                    "probability_improvement": 0.94,
                },
                "wins_ties_losses": {"wins": 40, "ties": 4, "losses": 20},
            }
            for name in fragment.BASELINES
        }
        disagreements = {name: 24 for name in fragment.BASELINES}
        predicted = {
            "refresh_matched": {
                "spearman": 0.31,
                "ci95": [0.08, 0.51],
                "probability_positive": 0.98,
                "n": 24,
            }
        }
    return {
        "claim_tier": TIERS[status],
        "interpretation": f"Frozen {status} interpretation.",
        "paired_primary_comparisons": paired,
        "root_disagreements": disagreements,
        "predicted_to_realized": predicted,
        "draw_stability_diagnostic": {"draw_agreement_count": 48},
        "result_sha256": "result-sha",
        "verification_sha256": "verification-sha",
    }


def _prepare(tmp_path: Path, monkeypatch, status: str) -> dict:
    saved = _report(status)
    (tmp_path / "FROZEN_REPORT.json").write_text(json.dumps(saved))
    monkeypatch.setattr(
        fragment.frozen_report,
        "build_report",
        lambda run_dir, parent_dir: _report(status),
    )
    return saved


@pytest.mark.parametrize(
    ("status", "has_table", "confirmed"),
    [
        ("mechanics_failed", False, False),
        ("gated_null", True, False),
        ("passed", True, True),
    ],
)
def test_confirmation_fragment_obeys_frozen_tiers(
    tmp_path, monkeypatch, status, has_table, confirmed
) -> None:
    saved = _prepare(tmp_path, monkeypatch, status)

    tex, metadata = fragment.build_fragment(tmp_path, parent_dir=tmp_path)

    assert metadata["claim_tier"] == TIERS[status]
    assert metadata["confirmed_claim_authorized"] is confirmed
    assert metadata["development_and_confirmation_are_not_pooled"] is True
    assert saved["interpretation"] in tex
    assert ("\\begin{table}" in tex) is has_table
    assert "hidden from the planner" in tex
    assert "retained two through six particles" in tex
    assert "cannot change this confirmation tier" in tex
    if has_table:
        assert "Refresh-matched myopic" in tex
        assert "History-blind SMC d2" in tex
        assert "Spearman" in tex
    else:
        assert "development signal remains provisional" in tex


def test_confirmation_fragment_refuses_report_tamper(tmp_path, monkeypatch) -> None:
    saved = _prepare(tmp_path, monkeypatch, "passed")
    saved["claim_tier"] = "smc_confirmation_null_no_headline_result"
    (tmp_path / "FROZEN_REPORT.json").write_text(json.dumps(saved))

    with pytest.raises(ValueError, match="does not replay exactly"):
        fragment.build_fragment(tmp_path, parent_dir=tmp_path)


def test_confirmation_fragment_writes_tex_and_metadata_without_calls(
    tmp_path, monkeypatch
) -> None:
    _prepare(tmp_path, monkeypatch, "gated_null")
    output = tmp_path / "paper/generated/regretbench_result.tex"

    written = fragment.write_fragment(
        tmp_path, parent_dir=tmp_path, output=output
    )

    assert written["status"] == "written"
    assert written["model_calls"] == 0
    assert written["cost_usd"] == 0.0
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["claim_tier"] == TIERS["gated_null"]
    assert metadata["tex_sha256"] == fragment.sha256_file(output)


def test_confirmation_fragment_binding_matches_current_files() -> None:
    path = fragment.REPO_ROOT / (
        "results/nonmyopic/regretbench_deepseek_smc_dynamic_depth2_confirmation/"
        "PAPER_FRAGMENT_BINDING.json"
    )
    binding = json.loads(path.read_text())

    assert binding["status"] == (
        "prospectively_frozen_before_any_smc_policy_response"
    )
    for field in (
        "reporting_protocol",
        "report_generator",
        "fragment_protocol",
        "fragment_generator",
    ):
        row = binding[field]
        assert fragment.sha256_file(fragment.REPO_ROOT / row["path"]) == row[
            "sha256"
        ]
    manuscript = binding["manuscript"]
    assert fragment.sha256_file(fragment.REPO_ROOT / manuscript["path"]) == (
        manuscript["preresult_sha256"]
    )
    assert manuscript["generated_fragment_absent_at_freeze"] is True
    assert binding["requirements"][
        "confirmed_tier_requires_confirmation_pass"
    ] is True
    assert binding["requirements"][
        "development_and_confirmation_can_be_pooled"
    ] is False


@pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("bibtex") is None,
    reason="LaTeX toolchain is unavailable",
)
def test_confirmation_fragment_compiles_within_page_budget(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _prepare(run_dir, monkeypatch, "passed")
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    shutil.copy2(fragment.REPO_ROOT / "paper/main.tex", paper_dir / "main.tex")
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
        run_dir,
        parent_dir=run_dir,
        output=paper_dir / "generated/regretbench_result.tex",
    )

    checks = validate_paper_draft(paper_dir)

    assert all(check.ok for check in checks), [
        (check.name, check.detail) for check in checks if not check.ok
    ]
