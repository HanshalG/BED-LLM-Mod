from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


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
