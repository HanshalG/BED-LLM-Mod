from pathlib import Path
import subprocess

from scripts import validate_paper_draft as vpd


VALID_LIMITATIONS_TEXT = """
This is a positive structured-benchmark result, not yet a positive
external-benchmark claim. Rock Diagnosis has an exact finite simulator and does
not test robustness to learned likelihoods. Animals streams are only partially
paired. Paprika policy counts are endpoint-invalid. MediQ stops before a calibrated
policy comparison. We do not claim that non-myopic BED cannot work. The study uses
one model family, and OpenRouter introduces provider nondeterminism. The iCRAFT
profile-support gate failed before likelihood evaluation. Mushroom and Cleveland
show exact d2 structure but failed proposal gates.
"""

VALID_FIGURE_LABELS = "\\label{fig:validation-chain}\\label{fig:rock-entropy}"

VALID_TEXT_CHECKS = VALID_LIMITATIONS_TEXT + VALID_FIGURE_LABELS


def test_validate_paper_draft_reports_missing_tex(tmp_path):
    results = vpd.validate_paper_draft(tmp_path)
    payload = vpd.summary_payload(results)

    assert payload["ok"] is False
    assert payload["checks"][0]["name"] == "paper_tex_exists"
    assert payload["checks"][0]["ok"] is False


def test_validate_paper_draft_runs_latex_bibtex_and_checks_pages(tmp_path, monkeypatch):
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}x\n"
        + VALID_TEXT_CHECKS
        + "\\end{document}\n"
    )
    commands = []

    def fake_run(command, *, cwd: Path, timeout: int):
        commands.append(command)
        if command[0] == "pdflatex":
            output_dir = Path(command[command.index("-output-directory") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "main.pdf").write_bytes(b"%PDF-1.4 fake")
        return subprocess.CompletedProcess(command, 0, stdout="Output written on main.pdf (5 pages, 1 bytes).\n")

    monkeypatch.setattr(vpd, "_run", fake_run)
    monkeypatch.setattr(vpd, "_pdf_page_count", lambda pdf_path, latex_output="": 5)

    results = vpd.validate_paper_draft(paper_dir, min_pages=4, max_pages=6)
    payload = vpd.summary_payload(results)

    assert payload["ok"] is True
    assert [command[0] for command in commands] == ["pdflatex", "bibtex", "pdflatex", "pdflatex"]
    assert payload["checks"][-1] == {"name": "paper_page_count", "ok": True, "detail": "5 pages"}


def test_validate_paper_draft_rejects_unexpected_todo_before_compile(tmp_path, monkeypatch):
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}\n"
        "\\todo{rewrite this vague section someday}\n"
        + VALID_TEXT_CHECKS
        + "\\end{document}\n"
    )

    def fail_if_called(command, *, cwd: Path, timeout: int):
        raise AssertionError("compile should not run after text validation fails")

    monkeypatch.setattr(vpd, "_run", fail_if_called)

    payload = vpd.summary_payload(vpd.validate_paper_draft(paper_dir))

    assert payload["ok"] is False
    todo_check = next(check for check in payload["checks"] if check["name"] == "paper_todo_scope")
    assert todo_check["ok"] is False
    assert "rewrite this vague section someday" in todo_check["detail"]


def test_validate_paper_draft_rejects_missing_required_limitations(tmp_path, monkeypatch):
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}\n")

    def fail_if_called(command, *, cwd: Path, timeout: int):
        raise AssertionError("compile should not run after text validation fails")

    monkeypatch.setattr(vpd, "_run", fail_if_called)

    payload = vpd.summary_payload(vpd.validate_paper_draft(paper_dir))

    assert payload["ok"] is False
    limitations_check = next(
        check for check in payload["checks"] if check["name"] == "paper_limitations_coverage"
    )
    assert limitations_check["ok"] is False
    assert "structured_positive_scope" in limitations_check["detail"]


def test_validate_paper_draft_requires_validation_chain_figure(tmp_path, monkeypatch):
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}\n"
        + VALID_LIMITATIONS_TEXT
        + "\\end{document}\n"
    )

    def fail_if_called(command, *, cwd: Path, timeout: int):
        raise AssertionError("compile should not run after text validation fails")

    monkeypatch.setattr(vpd, "_run", fail_if_called)

    payload = vpd.summary_payload(vpd.validate_paper_draft(paper_dir))

    assert payload["ok"] is False
    figure_check = next(check for check in payload["checks"] if check["name"] == "paper_required_figures")
    assert figure_check == {
        "name": "paper_required_figures",
        "ok": False,
        "detail": "missing: validation_chain, rock_entropy",
    }


def test_latex_pages_from_output_parses_singular_and_plural():
    assert vpd._latex_pages_from_output("Output written on main.pdf (1 page, 10 bytes).") == 1
    assert vpd._latex_pages_from_output("Output written on main.pdf (6 pages, 10 bytes).") == 6
