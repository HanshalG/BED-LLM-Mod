from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED_PATHS = (
    "results/ranking_fidelity/PATH_B_GATE0_TASK_LOSS.md",
    "results/ranking_fidelity/PATH_B_GATE0_DIAGNOSIS.md",
    "results/ranking_fidelity/path_b_gate0_task_loss.json",
    "plots/ranking_fidelity/path_b_gate0_task_loss.png",
    "results/path_b/GATE1_NOT_RUN.md",
    "paper/main.tex",
    "output/pdf/path_b_workshop_draft.pdf",
)


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    for relative_path in REQUIRED_PATHS:
        path = root / relative_path
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {relative_path}")

    result_path = root / "results/ranking_fidelity/path_b_gate0_task_loss.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        gate = result.get("gate", {})
        if gate.get("status") != "fail":
            errors.append(f"expected failed Gate 0, found {gate.get('status')!r}")
        value = gate.get("best_spearman_task_vs_realized_risk")
        threshold = gate.get("threshold")
        if value is None or threshold is None or not math.isfinite(float(value)):
            errors.append("Gate 0 correlation or threshold is missing/non-finite")
        elif float(value) >= float(threshold):
            errors.append(f"failed gate is inconsistent: rho={value} threshold={threshold}")
        method = result.get("method", {})
        if method.get("truth_used_by_estimated_scorer") is not False:
            errors.append("estimated scorer must explicitly record truth_used=false")
        if method.get("uses_llm") is not False:
            errors.append("Gate 0 replay must explicitly record uses_llm=false")

    gate1_path = root / "results/path_b/GATE1_NOT_RUN.md"
    if gate1_path.exists():
        gate1 = gate1_path.read_text(encoding="utf-8").lower()
        for phrase in ("no gate 1 slurm jobs", "phase 2 was not", "claims are not made"):
            if phrase not in gate1:
                errors.append(f"Gate 1 stop artifact missing phrase: {phrase!r}")

    paper_path = root / "paper/main.tex"
    if paper_path.exists():
        paper = paper_path.read_text(encoding="utf-8")
        for phrase in (
            "rho=0.231",
            "no LLM calls",
            "do not launch the downstream policy pilot",
            "no performance claim",
        ):
            if phrase not in paper:
                errors.append(f"paper missing required Path B statement: {phrase!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Path B negative-result package.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    errors = validate(args.root.resolve())
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print(f"OK: {len(REQUIRED_PATHS)} required Path B artifacts")
    print("OK: Gate 0 failure is numerically consistent and truth-free")
    print("OK: Gate 1 stop and paper claims are explicit")


if __name__ == "__main__":
    main()
