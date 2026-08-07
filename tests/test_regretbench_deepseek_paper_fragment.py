from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts import regretbench_deepseek_frozen_report as frozen_report
from scripts import regretbench_deepseek_paper_fragment as fragment
from scripts.validate_paper_draft import validate_paper_draft
from tests.test_regretbench_deepseek_frozen_report import _result, _write_verified


def _prepare(tmp_path: Path, *, stage: str, status: str) -> dict:
    _write_verified(tmp_path, _result(stage=stage, status=status))
    frozen_report.write_report(tmp_path, stage=stage)
    return json.loads((tmp_path / "FROZEN_REPORT.json").read_text())


@pytest.mark.parametrize(
    ("stage", "status", "tier", "has_table"),
    [
        (
            "development",
            "passed",
            "provisional_development_signal_confirmation_required",
            True,
        ),
        (
            "development",
            "gated_null",
            "development_policy_null_confirmation_forbidden",
            True,
        ),
        (
            "confirmation",
            "passed",
            "confirmed_llm_native_nonmyopic_signal",
            True,
        ),
        (
            "confirmation",
            "gated_null",
            "confirmation_null_development_not_confirmed",
            True,
        ),
        (
            "development",
            "mechanics_failed",
            "mechanics_failure_no_scientific_result",
            False,
        ),
    ],
)
def test_fragment_uses_exact_frozen_tier_and_table_boundary(
    tmp_path, stage, status, tier, has_table
) -> None:
    report = _prepare(tmp_path, stage=stage, status=status)

    tex, metadata = fragment.build_fragment(tmp_path, stage=stage)

    assert metadata["claim_tier"] == tier
    assert report["interpretation"] in tex
    assert "released RegretBench intents, aliases, facets, slots" in tex
    assert ("\\begin{table}" in tex) is has_table
    if has_table:
        assert "Matched-Brier myopic" in tex
        assert "Myopic EIG width" in tex
        assert "History-blind d2" in tex
        assert "Fixed-support d2" in tex
        assert "Random" in tex
        assert "Spearman" in tex
        assert "alignment-complete corroboration" in tex
    else:
        assert "No policy-efficacy table is shown" in tex


def test_fragment_refuses_report_tampering(tmp_path) -> None:
    _prepare(tmp_path, stage="development", status="passed")
    path = tmp_path / "FROZEN_REPORT.json"
    value = json.loads(path.read_text())
    value["claim_tier"] = "confirmed_llm_native_nonmyopic_signal"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="does not replay exactly"):
        fragment.build_fragment(tmp_path, stage="development")


def test_write_fragment_banks_tex_and_metadata_without_calls(tmp_path) -> None:
    _prepare(tmp_path, stage="confirmation", status="passed")
    output = tmp_path / "paper" / "generated" / "regretbench_result.tex"

    written = fragment.write_fragment(
        tmp_path, stage="confirmation", output=output
    )

    assert written["status"] == "written"
    assert written["model_calls"] == 0
    assert written["cost_usd"] == 0.0
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["claim_tier"] == "confirmed_llm_native_nonmyopic_signal"
    assert metadata["tex_sha256"] == fragment.sha256_file(output)
    assert metadata["report_sha256"] == fragment.sha256_file(
        tmp_path / "FROZEN_REPORT.json"
    )


def test_main_has_frozen_conditional_page_substitution() -> None:
    main = (fragment.REPO_ROOT / "paper/main.tex").read_text(encoding="utf-8")
    output = fragment.DEFAULT_OUTPUT

    assert main.count("\\IfFileExists{generated/regretbench_result.tex}") == 2
    assert (
        "\\IfFileExists{generated/regretbench_result.tex}{%\n"
        "  % The frozen RegretBench fragment replaces the detailed late Number Game audit.\n"
        "}{%\n"
        "A zero-call post-hoc ranking audit"
    ) in main
    assert "the mechanism evidence.\n}" in main
    assert (
        "\\IfFileExists{generated/regretbench_result.tex}{%\n"
        "  \\input{generated/regretbench_result.tex}%\n"
        "}{}"
    ) in main
    assert not output.exists()


@pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("bibtex") is None,
    reason="LaTeX toolchain is unavailable",
)
@pytest.mark.parametrize(
    ("stage", "status"),
    [
        ("confirmation", "passed"),
        ("development", "mechanics_failed"),
    ],
)
def test_generated_fragment_compiles_within_page_budget(
    tmp_path: Path, stage: str, status: str
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _prepare(run_dir, stage=stage, status=status)

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
        stage=stage,
        output=paper_dir / "generated/regretbench_result.tex",
    )

    checks = validate_paper_draft(paper_dir)

    assert all(check.ok for check in checks), [
        (check.name, check.detail) for check in checks if not check.ok
    ]
    page_check = next(check for check in checks if check.name == "paper_page_count")
    assert page_check.detail in {"4 pages", "5 pages", "6 pages"}


def test_paper_fragment_binding_matches_current_files() -> None:
    path = (
        fragment.REPO_ROOT
        / "results/nonmyopic/regretbench_deepseek_reporting/"
        "PAPER_FRAGMENT_BINDING.json"
    )
    binding = json.loads(path.read_text())

    assert binding["status"] == "prospectively_amended_before_responses"
    assert binding["scientific_contract_changed"] is True
    assert fragment.sha256_file(
        fragment.REPO_ROOT / binding["fragment"]["protocol_path"]
    ) == binding["fragment"]["protocol_sha256"]
    assert fragment.sha256_file(
        fragment.REPO_ROOT / binding["fragment"]["generator_path"]
    ) == binding["fragment"]["generator_sha256"]
    assert (
        binding["reporting"]["endpoint_amendment_sha256"]
        == frozen_report.REPORTING_ENDPOINT_AMENDMENT_SHA256
    )
    assert (
        binding["reporting"]["matched_utility_myopic_amendment_sha256"]
        == frozen_report.MATCHED_UTILITY_MYOPIC_AMENDMENT_SHA256
    )
    assert fragment.sha256_file(
        fragment.REPO_ROOT / binding["manuscript"]["path"]
    ) == binding["manuscript"]["preresult_sha256"]
    assert binding["manuscript"]["generated_fragment_absent_at_freeze"] is True
    assert binding["requirements"]["matched_brier_myopic_is_headline_control"]
    assert (
        binding["requirements"][
            "late_number_game_detail_replaced_only_when_fragment_exists"
        ]
        is True
    )
