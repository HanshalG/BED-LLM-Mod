from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


ALLOWED_TODO_KEYWORDS = (
    ("Phase 4", "package validation"),
    ("constrained and unconstrained paired results", "pre-registered primary endpoint"),
    ("evolved strategies", "query trajectories"),
    ("outcome playbook", "truth-log-posterior probability"),
)

REQUIRED_LIMITATION_PATTERNS = {
    "forced_thinking_exit_rate": (r"forced[- ]thinking[- ]exit",),
    "single_environment_family": (r"one constrained\s+environment family",),
    "method_blind_constructed_environment": (r"method[- ]blind", r"deliberately constructed|constructed"),
    "robustness_heatmap_scope": (r"robustness heatmap",),
    "no_mpc_ablation": (r"MPC[- ]style", r"future work"),
    "workshop_scale_trials": (r"workshop[- ]scale", r"limited number of paired\s+trials"),
}

REQUIRED_FIGURE_LABELS = {
    "ranking_fidelity_diagnostics": "fig:ranking-fidelity",
    "main_depth_contrast": "fig:depth-sweep",
    "cost_vs_depth": "fig:cost-depth",
    "qualitative_strategies": "fig:qualitative",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _run(command: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _latex_pages_from_output(output: str) -> int | None:
    match = re.search(r"Output written on .+ \((\d+) pages?,", output)
    return int(match.group(1)) if match else None


def _pdf_page_count(pdf_path: Path, *, latex_output: str = "") -> int | None:
    if pdf_path.exists():
        try:
            from pypdf import PdfReader

            return len(PdfReader(str(pdf_path)).pages)
        except Exception:
            pass
    return _latex_pages_from_output(latex_output)


def _tail(text: str, max_chars: int = 2000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def _todo_bodies(tex: str) -> list[str]:
    return re.findall(r"\\todo\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", tex, flags=re.DOTALL)


def _text_contains_all(text: str, patterns: tuple[str, ...]) -> bool:
    return all(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _paper_text_checks(tex: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    todos = _todo_bodies(tex)
    unexpected_todos = [
        todo.strip()
        for todo in todos
        if not any(all(keyword in todo for keyword in allowed) for allowed in ALLOWED_TODO_KEYWORDS)
    ]
    if unexpected_todos:
        checks.append(
            CheckResult(
                "paper_todo_scope",
                False,
                "unexpected TODO(s): " + "; ".join(unexpected_todos),
            )
        )
    else:
        checks.append(CheckResult("paper_todo_scope", True, f"{len(todos)} allowed TODO marker(s)"))

    missing_limitations = [
        name
        for name, patterns in REQUIRED_LIMITATION_PATTERNS.items()
        if not _text_contains_all(tex, patterns)
    ]
    if missing_limitations:
        checks.append(
            CheckResult(
                "paper_limitations_coverage",
                False,
                "missing: " + ", ".join(missing_limitations),
            )
        )
    else:
        checks.append(
            CheckResult(
                "paper_limitations_coverage",
                True,
                f"{len(REQUIRED_LIMITATION_PATTERNS)} required limitation topic(s)",
            )
        )

    missing_figures = [
        name
        for name, label in REQUIRED_FIGURE_LABELS.items()
        if f"\\label{{{label}}}" not in tex
    ]
    if missing_figures:
        checks.append(
            CheckResult(
                "paper_required_figures",
                False,
                "missing: " + ", ".join(missing_figures),
            )
        )
    else:
        checks.append(
            CheckResult(
                "paper_required_figures",
                True,
                f"{len(REQUIRED_FIGURE_LABELS)} required figure label(s)",
            )
        )
    return checks


def validate_paper_draft(
    paper_dir: Path,
    *,
    main_tex: str = "main.tex",
    min_pages: int = 4,
    max_pages: int = 6,
    timeout: int = 60,
) -> list[CheckResult]:
    paper_dir = paper_dir.resolve()
    tex_path = paper_dir / main_tex
    checks: list[CheckResult] = []
    if not tex_path.exists():
        return [CheckResult("paper_tex_exists", False, f"missing {tex_path}")]
    checks.append(CheckResult("paper_tex_exists", True, str(tex_path)))
    tex = tex_path.read_text(encoding="utf-8")
    checks.extend(_paper_text_checks(tex))
    if not all(check.ok for check in checks):
        return checks

    with tempfile.TemporaryDirectory(prefix="bed_llm_paper_build_") as tmp:
        build_dir = Path(tmp)
        for bib_path in paper_dir.glob("*.bib"):
            (build_dir / bib_path.name).write_text(bib_path.read_text(encoding="utf-8"), encoding="utf-8")
        commands = [
            (
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(build_dir), main_tex],
                paper_dir,
            ),
            (["bibtex", Path(main_tex).stem], build_dir),
            (
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(build_dir), main_tex],
                paper_dir,
            ),
            (
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(build_dir), main_tex],
                paper_dir,
            ),
        ]
        combined_output = ""
        for command, cwd in commands:
            try:
                result = _run(command, cwd=cwd, timeout=timeout)
            except FileNotFoundError as exc:
                checks.append(CheckResult("paper_compile", False, f"{command[0]} not found: {exc}"))
                return checks
            except subprocess.TimeoutExpired:
                checks.append(CheckResult("paper_compile", False, f"{' '.join(command)} timed out"))
                return checks
            combined_output += result.stdout or ""
            if result.returncode != 0:
                checks.append(
                    CheckResult(
                        "paper_compile",
                        False,
                        f"{' '.join(command)} failed with {result.returncode}: {_tail(result.stdout or '')}",
                    )
                )
                return checks

        pdf_path = build_dir / f"{Path(main_tex).stem}.pdf"
        if not pdf_path.exists():
            checks.append(CheckResult("paper_compile", False, f"missing compiled PDF {pdf_path}"))
            return checks
        checks.append(CheckResult("paper_compile", True, str(pdf_path)))

        pages = _pdf_page_count(pdf_path, latex_output=combined_output)
        if pages is None:
            checks.append(CheckResult("paper_page_count", False, "could not determine page count"))
        elif min_pages <= pages <= max_pages:
            checks.append(CheckResult("paper_page_count", True, f"{pages} pages"))
        else:
            checks.append(
                CheckResult(
                    "paper_page_count",
                    False,
                    f"{pages} pages outside target range [{min_pages}, {max_pages}]",
                )
            )
    return checks


def summary_payload(results: list[CheckResult]) -> dict[str, Any]:
    return {
        "ok": all(result.ok for result in results),
        "checks": [
            {"name": result.name, "ok": result.ok, "detail": result.detail}
            for result in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and validate the Path A workshop paper draft.")
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--main-tex", default="main.tex")
    parser.add_argument("--min-pages", type=int, default=4)
    parser.add_argument("--max-pages", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = validate_paper_draft(
        args.paper_dir,
        main_tex=args.main_tex,
        min_pages=args.min_pages,
        max_pages=args.max_pages,
        timeout=args.timeout,
    )
    payload = summary_payload(results)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for result in results:
            status = "ok" if result.ok else "missing"
            print(f"[{status}] {result.name}: {result.detail}")
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
