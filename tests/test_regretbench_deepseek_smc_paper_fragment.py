from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts import regretbench_deepseek_smc_frozen_report as report
from scripts import regretbench_deepseek_smc_paper_fragment as fragment
from scripts.validate_paper_draft import validate_paper_draft
from tests.test_regretbench_deepseek_smc_frozen_report import (
    _install_validation,
    _result,
)


def _prepare(tmp_path: Path, monkeypatch, *, status: str) -> dict:
    payload = _result(status=status)
    _install_validation(monkeypatch, payload)
    report.write_report(tmp_path, primary_dir=tmp_path)
    return json.loads((tmp_path / "FROZEN_REPORT.json").read_text())


@pytest.mark.parametrize(
    ("status", "tier", "has_table"),
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
def test_fragment_uses_exact_smc_tier_and_table_boundary(
    tmp_path, monkeypatch, status, tier, has_table
) -> None:
    saved = _prepare(tmp_path, monkeypatch, status=status)

    tex, metadata = fragment.build_fragment(tmp_path, primary_dir=tmp_path)

    assert metadata["claim_tier"] == tier
    assert metadata["development_can_be_called_confirmed"] is False
    assert saved["interpretation"] in tex
    assert ("\\begin{table}" in tex) is has_table
    assert "exact banked semantic parent slots" in tex
    assert "retained two through six particles" in tex
    assert "cannot change this development tier" in tex
    if has_table:
        assert "Refresh-matched myopic" in tex
        assert "History-blind SMC d2" in tex
        assert "Spearman" in tex
        assert "amended 13-gate conjunction" in tex
        assert metadata["primary_claim_all_pass"] is (status == "passed")
        assert metadata["all_34_diagnostic_gates_pass"] is False
    else:
        assert "No policy-efficacy table is shown" in tex


def test_fragment_refuses_saved_report_tamper(tmp_path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch, status="passed")
    path = tmp_path / "FROZEN_REPORT.json"
    value = json.loads(path.read_text())
    value["claim_tier"] = "confirmed_llm_native_nonmyopic_signal"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="does not replay exactly"):
        fragment.build_fragment(tmp_path, primary_dir=tmp_path)


def test_write_fragment_banks_tex_and_metadata_without_calls(
    tmp_path, monkeypatch
) -> None:
    _prepare(tmp_path, monkeypatch, status="gated_null")
    output = tmp_path / "paper/generated/regretbench_result.tex"

    written = fragment.write_fragment(
        tmp_path, primary_dir=tmp_path, output=output
    )

    assert written["status"] == "written"
    assert written["model_calls"] == 0
    assert written["cost_usd"] == 0.0
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["claim_tier"] == (
        "smc_development_policy_null_confirmation_forbidden"
    )
    assert metadata["tex_sha256"] == fragment.sha256_file(output)
    assert metadata["claim_gate_amendment_sha256"] == (
        report.CLAIM_GATE_AMENDMENT_SHA256
    )


def test_current_manuscript_remains_unchanged_and_fragment_absent() -> None:
    main = fragment.REPO_ROOT / "paper/main.tex"

    assert fragment.sha256_file(main) == fragment.PRERESULT_MANUSCRIPT_SHA256
    assert not fragment.DEFAULT_OUTPUT.exists()
    assert "\\IfFileExists{generated/regretbench_result.tex}" in main.read_text()


def test_smc_paper_fragment_binding_matches_current_files() -> None:
    path = fragment.REPO_ROOT / (
        "results/nonmyopic/regretbench_deepseek_smc_dynamic_depth2_policy/"
        "PAPER_FRAGMENT_BINDING.json"
    )
    binding = json.loads(path.read_text())

    assert binding["status"] == (
        "prospectively_frozen_before_any_smc_policy_response"
    )
    for field in (
        "reporting_binding",
        "fragment_protocol",
        "report_generator",
        "fragment_generator",
    ):
        row = binding[field]
        assert fragment.sha256_file(fragment.REPO_ROOT / row["path"]) == row["sha256"]
    manuscript = binding["manuscript"]
    assert fragment.sha256_file(fragment.REPO_ROOT / manuscript["path"]) == (
        manuscript["preresult_sha256"]
    )
    assert manuscript["generated_fragment_absent_at_freeze"] is True
    assert binding["requirements"]["development_can_be_called_confirmed"] is False
    assert binding["requirements"][
        "refresh_matched_myopic_is_headline_horizon_control"
    ] is True


@pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("bibtex") is None,
    reason="LaTeX toolchain is unavailable",
)
def test_generated_smc_fragment_compiles_within_page_budget(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _prepare(run_dir, monkeypatch, status="passed")
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
        primary_dir=run_dir,
        output=paper_dir / "generated/regretbench_result.tex",
    )

    checks = validate_paper_draft(paper_dir)

    assert all(check.ok for check in checks), [
        (check.name, check.detail) for check in checks if not check.ok
    ]
