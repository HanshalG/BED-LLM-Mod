from pathlib import Path
import subprocess

from scripts import validate_paper_draft as vpd


def test_validate_paper_draft_reports_missing_tex(tmp_path):
    results = vpd.validate_paper_draft(tmp_path)
    payload = vpd.summary_payload(results)

    assert payload["ok"] is False
    assert payload["checks"][0]["name"] == "paper_tex_exists"
    assert payload["checks"][0]["ok"] is False


def test_validate_paper_draft_runs_latex_bibtex_and_checks_pages(tmp_path, monkeypatch):
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}\n")
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


def test_latex_pages_from_output_parses_singular_and_plural():
    assert vpd._latex_pages_from_output("Output written on main.pdf (1 page, 10 bytes).") == 1
    assert vpd._latex_pages_from_output("Output written on main.pdf (6 pages, 10 bytes).") == 6
