from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import regretbench_deepseek_frozen_report as frozen_report
from scripts import regretbench_deepseek_paper_fragment as fragment
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
        assert "Myopic width" in tex
        assert "History-blind d2" in tex
        assert "Fixed-support d2" in tex
        assert "Random" in tex
        assert "Spearman" in tex
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


def test_main_has_only_conditional_preresult_include() -> None:
    main = (fragment.REPO_ROOT / "paper/main.tex").read_text(encoding="utf-8")
    output = fragment.DEFAULT_OUTPUT

    assert (
        "\\IfFileExists{generated/regretbench_result.tex}{%\n"
        "  \\input{generated/regretbench_result.tex}%\n"
        "}{}"
    ) in main
    assert not output.exists()


def test_paper_fragment_binding_matches_current_files() -> None:
    path = (
        fragment.REPO_ROOT
        / "results/nonmyopic/regretbench_deepseek_reporting/"
        "PAPER_FRAGMENT_BINDING.json"
    )
    binding = json.loads(path.read_text())

    assert binding["status"] == "frozen_before_responses"
    assert binding["scientific_contract_changed"] is False
    assert fragment.sha256_file(
        fragment.REPO_ROOT / binding["fragment"]["protocol_path"]
    ) == binding["fragment"]["protocol_sha256"]
    assert fragment.sha256_file(
        fragment.REPO_ROOT / binding["fragment"]["generator_path"]
    ) == binding["fragment"]["generator_sha256"]
    assert fragment.sha256_file(
        fragment.REPO_ROOT / binding["manuscript"]["path"]
    ) == binding["manuscript"]["preresult_sha256"]
    assert binding["manuscript"]["generated_fragment_absent_at_freeze"] is True
