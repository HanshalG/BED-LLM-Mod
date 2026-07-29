from pathlib import Path
import subprocess

from scripts import validate_paper_draft as vpd


VALID_LIMITATIONS_TEXT = """
This is a positive structured-benchmark result and a
narrow, unstable simulator-grounded LLM-native result. It uses one frozen
GPT-5.4 support tree, with a mixture
selected on a disclosed development curve and exact rather than learned
likelihoods. The first tree is positive, while a fresh structured tree
reverses the support effect and does not replicate across generations.
It tests fresh physical worlds, not fresh model generation.
Rock Diagnosis has an exact finite simulator and does
not test robustness to learned likelihoods. Animals streams are only partially
paired. Paprika policy counts are endpoint-invalid. MediQ stops before a calibrated
policy comparison. We do not claim that non-myopic BED cannot work. The study uses
one model family, and OpenRouter introduces provider nondeterminism. The iCRAFT
profile-support gate failed before likelihood evaluation. Mushroom, Cleveland, and
Thyroid show exact d2 structure with partial LLM transfer. Thyroid uses native delayed-assay metadata,
fails to beat matched-random continuations, and exposes ungrounded empirical assay utility.
Its utility-card repair stops before endpoints.
The projected repair uses exact local summaries and projected branches, so it is not unaided LLM planning.
Its projection-only ablation collected 0/50.
Cleveland recovered 100.9 with zero projected branches, but its independent truth-log
interval crossed zero, so it is not an all-gates confirmation.
Mushroom collected 50/50 with zero projected branches, exactly matched d2, and its
truth-log intervals agreed.
In a range-gated task, Exact d3 beat exact d2,
and a 26B gate selected it on 16/16 cells. A tied-control identity mismatch blocked
the audit, so there was no trajectory; this is proposal evidence only.
An exact d4 gate beat d3 by .3484 with 500/0/0, but a frontier smoke returned
reasoning-only responses. Semantic target selection recovered the route on 10/10
smoke cells before provider truncation, so this is mechanism evidence, not an
LLM-policy claim.
An exact h5 gate wins 500/500 pairs. A hierarchical target gate selects the
route on 15/16 cells and recovers .937 of the opportunity. An initial 50-pair
run matched exhaustive d5. Three additional preregistered replications gain
.2727 over identical compiled plans at h4 (150/0/0) and .2118 over
matched-random h5 targets (148/2/0), using 51 physical prompts for 1,200 logical
decisions. In total this is 200 positive trajectories across four seeds. This
remains an engineered exact task, not unaided LLM planning.
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


def test_validate_paper_draft_accepts_inline_native_scope(tmp_path, monkeypatch):
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    limitations = VALID_LIMITATIONS_TEXT
    (paper_dir / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}\n"
        + limitations
        + VALID_FIGURE_LABELS
        + "\\end{document}\n"
    )
    commands = []

    def fake_run(command, *, cwd: Path, timeout: int):
        commands.append(command)
        if command[0] == "pdflatex":
            output_dir = Path(command[command.index("-output-directory") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "main.pdf").write_bytes(b"%PDF-1.4 fake")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Output written on main.pdf (5 pages, 1 bytes).\n",
        )

    monkeypatch.setattr(vpd, "_run", fake_run)
    monkeypatch.setattr(vpd, "_pdf_page_count", lambda pdf_path, latex_output="": 5)

    payload = vpd.summary_payload(vpd.validate_paper_draft(paper_dir))

    assert payload["ok"] is True


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
