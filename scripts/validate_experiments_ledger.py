from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_HEADERS = [
    "Date",
    "Run / job",
    "Config",
    "Model",
    "Status",
    "Key metric / purpose",
    "Artifacts",
    "Commit / tag",
]

REQUIRED_EVIDENCE = {
    "ranking_fidelity": [
        "Ranking-fidelity gate",
        "results/ranking_fidelity/PHASE1_26B_A4B_GATE.md",
        "results/ranking_fidelity/REPORT.md",
    ],
    "constrained_oracle": [
        "Constrained oracle",
        "results/constrained_oracle/REPORT.md",
    ],
    "robustness": [
        "branch_decoy_local_robustness",
        "results/constrained_oracle_robustness/branch_decoy_local_robustness_REPORT.md",
        "plots/constrained_oracle_robustness/branch_decoy_local_robustness_heatmap.png",
    ],
    "cost_vs_depth": [
        "path_a_preregistered_cost_vs_depth",
        "results/cost_vs_depth/path_a_preregistered_cost_vs_depth.md",
        "results/cost_vs_depth/path_a_preregistered_cost_vs_depth.json",
    ],
    "paper_validation": [
        "Path A paper draft",
        "scripts/validate_paper_draft.py",
        "paper/main.tex",
    ],
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _table_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    ]


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_table(text: str) -> tuple[list[str], list[dict[str, str]]]:
    lines = _table_lines(text)
    if len(lines) < 2:
        return [], []
    headers = _split_row(lines[0])
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = _split_row(line)
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return headers, rows


def _extract_paths(text: str) -> list[str]:
    paths = re.findall(r"`([^`]+)`", text)
    suffixes = (".md", ".json", ".png", ".py", ".tex")
    return [
        path
        for path in paths
        if " " not in path and ("/" in path or path.endswith(suffixes))
    ]


def validate_experiments_ledger(
    ledger_path: Path,
    *,
    root: Path = Path("."),
) -> list[CheckResult]:
    ledger_path = Path(ledger_path)
    root = root.resolve()
    if not ledger_path.exists():
        return [CheckResult("ledger_exists", False, f"missing {ledger_path}")]

    text = ledger_path.read_text(encoding="utf-8")
    headers, rows = _parse_table(text)
    checks: list[CheckResult] = [CheckResult("ledger_exists", True, str(ledger_path))]
    if headers != EXPECTED_HEADERS:
        checks.append(CheckResult("ledger_headers", False, f"headers={headers}"))
        return checks
    checks.append(CheckResult("ledger_headers", True, ", ".join(headers)))
    if not rows:
        checks.append(CheckResult("ledger_rows", False, "no table rows found"))
        return checks
    checks.append(CheckResult("ledger_rows", True, f"{len(rows)} rows"))

    active_rows = [
        row["Run / job"]
        for row in rows
        if any(word in row["Status"].lower() for word in ("running", "pending"))
    ]
    checks.append(
        CheckResult(
            "no_active_status_rows",
            not active_rows,
            "none" if not active_rows else "; ".join(active_rows),
        )
    )

    empty_commit_rows = [
        row["Run / job"]
        for row in rows
        if row["Commit / tag"].strip() in {"", "TBD", "todo", "unknown"}
    ]
    checks.append(
        CheckResult(
            "commit_or_tag_filled",
            not empty_commit_rows,
            "all rows filled" if not empty_commit_rows else "; ".join(empty_commit_rows),
        )
    )

    ledger_text_lower = text.lower()
    for name, needles in REQUIRED_EVIDENCE.items():
        missing_text = [needle for needle in needles if needle.lower() not in ledger_text_lower]
        if missing_text:
            checks.append(CheckResult(f"required_{name}", False, f"missing ledger text: {missing_text}"))
            continue
        missing_paths = [
            path
            for path in needles
            if ("/" in path or path.endswith(".py") or path.endswith(".tex"))
            and not (root / path).exists()
        ]
        checks.append(
            CheckResult(
                f"required_{name}",
                not missing_paths,
                "ok" if not missing_paths else f"missing artifacts: {missing_paths}",
            )
        )

    artifact_missing: list[str] = []
    for row in rows:
        status = row["Status"].lower()
        if "complete" not in status:
            continue
        for path in _extract_paths(row["Artifacts"]):
            # Cluster-only partial run dirs are not completion evidence.
            if path.startswith("runs/"):
                continue
            if not (root / path).exists():
                artifact_missing.append(f"{row['Run / job']}: {path}")
    checks.append(
        CheckResult(
            "complete_artifacts_exist",
            not artifact_missing,
            "all complete-row artifacts exist" if not artifact_missing else "; ".join(artifact_missing),
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
    parser = argparse.ArgumentParser(description="Validate the Path A experiments ledger.")
    parser.add_argument("--ledger", type=Path, default=Path("EXPERIMENTS.md"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = summary_payload(validate_experiments_ledger(args.ledger, root=args.root))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for check in payload["checks"]:
            status = "ok" if check["ok"] else "fail"
            print(f"[{status}] {check['name']}: {check['detail']}")
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
