from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _contains_all(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return all(needle.lower() in lower for needle in needles)


def _first_existing_or_glob(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _check_report(
    root: Path,
    *,
    name: str,
    patterns: list[str],
    required_phrases: list[str],
) -> CheckResult:
    path = _first_existing_or_glob(root, patterns)
    if path is None:
        return CheckResult(name, False, f"missing one of: {', '.join(patterns)}")
    text = _read_text(path)
    if not _contains_all(text, required_phrases):
        return CheckResult(
            name,
            False,
            f"{path} exists but lacks required phrases: {', '.join(required_phrases)}",
        )
    return CheckResult(name, True, str(path))


def _check_headline_plot(root: Path) -> CheckResult:
    path = _first_existing_or_glob(
        root,
        ["plots/location_depth_sweeps/*_headline_rmse.png"],
    )
    if path is None:
        return CheckResult("headline_rmse_plot", False, "missing plots/location_depth_sweeps/*_headline_rmse.png")
    if path.stat().st_size <= 0:
        return CheckResult("headline_rmse_plot", False, f"{path} is empty")
    return CheckResult("headline_rmse_plot", True, str(path))


def _check_ranking_fidelity_plot(root: Path) -> CheckResult:
    path = _first_existing_or_glob(
        root,
        [
            "plots/ranking_fidelity/*_diagnostics.png",
            "plots/ranking_fidelity/*_plot.png",
            "results/ranking_fidelity/*_diagnostics.png",
            "results/ranking_fidelity/*_plot.png",
        ],
    )
    if path is None:
        return CheckResult(
            "ranking_fidelity_diagnostics_plot",
            False,
            "missing ranking-fidelity diagnostics plot",
        )
    if path.stat().st_size <= 0:
        return CheckResult("ranking_fidelity_diagnostics_plot", False, f"{path} is empty")
    return CheckResult("ranking_fidelity_diagnostics_plot", True, str(path))


def _check_cost_plot(root: Path) -> CheckResult:
    path = _first_existing_or_glob(
        root,
        [
            "plots/cost_vs_depth/*_cost_vs_depth.png",
            "results/cost_vs_depth/*_cost_vs_depth.png",
        ],
    )
    if path is None:
        return CheckResult(
            "cost_vs_depth_plot",
            False,
            "missing cost-vs-depth plot",
        )
    if path.stat().st_size <= 0:
        return CheckResult("cost_vs_depth_plot", False, f"{path} is empty")
    return CheckResult("cost_vs_depth_plot", True, str(path))


def _check_qualitative_examples(root: Path) -> CheckResult:
    report_paths = sorted(root.glob("results/location_qualitative/*_qualitative_examples.md"))
    if not report_paths:
        return CheckResult(
            "qualitative_strategy_examples",
            False,
            "missing results/location_qualitative/*_qualitative_examples.md",
        )
    valid_report_path: Path | None = None
    for report_path in report_paths:
        text = _read_text(report_path)
        if "No StrategyEIG examples were available." in text:
            continue
        if _contains_all(
            text,
            ["Qualitative Location Strategy Examples", "RMSE delta vs EIG", "Root query"],
        ):
            valid_report_path = report_path
            break
    if valid_report_path is None:
        return CheckResult(
            "qualitative_strategy_examples",
            False,
            "qualitative reports exist but contain no complete StrategyEIG examples",
        )

    plot_path = _first_existing_or_glob(
        root,
        ["results/location_qualitative/*_qualitative_example_*.png"],
    )
    if plot_path is None:
        return CheckResult(
            "qualitative_strategy_examples",
            False,
            "missing results/location_qualitative/*_qualitative_example_*.png",
        )
    if plot_path.stat().st_size <= 0:
        return CheckResult("qualitative_strategy_examples", False, f"{plot_path} is empty")
    return CheckResult("qualitative_strategy_examples", True, f"{valid_report_path}; {plot_path}")


def validate_path_a_package(root: Path) -> list[CheckResult]:
    return [
        _check_report(
            root,
            name="ranking_fidelity_gate",
            patterns=[
                "results/ranking_fidelity/REPORT.md",
                "results/ranking_fidelity/PHASE1_26B_A4B_GATE.md",
            ],
            required_phrases=["spearman", "top-1", "snr"],
        ),
        _check_ranking_fidelity_plot(root),
        _check_report(
            root,
            name="constrained_oracle",
            patterns=["results/constrained_oracle/REPORT.md"],
            required_phrases=["planner", "greedy", "rmse"],
        ),
        _check_report(
            root,
            name="depth_sweep_headline_and_control",
            patterns=["results/location_depth_sweeps/*_REPORT.md"],
            required_phrases=[
                "headline constrained depths",
                "strategyeig-d1",
                "strategyeig-d3",
                "strategyeig-d5",
                "strategyeig-myopic",
                "paired final delta vs eig",
            ],
        ),
        _check_headline_plot(root),
        _check_qualitative_examples(root),
        _check_report(
            root,
            name="cost_vs_depth",
            patterns=["results/cost_vs_depth/*_cost_vs_depth.md"],
            required_phrases=[
                "llm cost vs depth",
                "brute-force n-step eig proxy",
                "bf/strategy root-set ratio",
            ],
        ),
        _check_cost_plot(root),
    ]


def summary_payload(results: list[CheckResult]) -> dict[str, Any]:
    return {
        "ok": all(result.ok for result in results),
        "checks": [
            {"name": result.name, "ok": result.ok, "detail": result.detail}
            for result in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Path A minimum publishable evidence package.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root containing results/ and plots/")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    results = validate_path_a_package(args.root)
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
