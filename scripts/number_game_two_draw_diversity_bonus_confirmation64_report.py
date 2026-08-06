#!/usr/bin/env python3
"""Render the verified staged-64 diversity result without selective fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_two_draw_diversity_bonus_audit as audit
from scripts import number_game_two_draw_diversity_bonus_confirmation64_daily_execute as daily
from scripts.number_game_two_draw_diversity_bonus_confirmation64_verify import (
    verify_completed_confirmation,
)


REPORT_PATH = REPO_ROOT / (
    "results/nonmyopic/"
    "NUMBER_GAME_TWO_DRAW_DIVERSITY_BONUS_CONFIRMATION64_RESULT.md"
)
COMPARISON_ORDER = (
    ("crossfit_depth_two", "Cross-fitted dynamic depth two"),
    ("original_depth_three", "Unadjusted dynamic depth three"),
    ("myopic_eig", "Myopic EIG"),
    ("fixed_support_depth_three", "Fixed-support depth three"),
    ("positive_test_strategy", "Positive-test strategy"),
    ("uniform_random_candidate_root", "Uniform random roots"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _status_text(status: str) -> str:
    if status == "passed":
        return (
            "The prospective monotonic-depth confirmation passed every "
            "frozen mechanics and scientific gate."
        )
    if status == "gated_null":
        return (
            "The prospective selector is a gated null. The frozen "
            "coefficient route closes without tuning on this cohort."
        )
    if status == "mechanics_failed":
        return (
            "The experiment is mechanics-failed; no scientific efficacy "
            "claim is made."
        )
    raise ValueError(f"unexpected staged result status: {status}")


def render_report(
    *,
    run_dir: Path,
    verifier: Callable[..., dict[str, Any]] = verify_completed_confirmation,
) -> str:
    verification = verifier(run_dir=run_dir)
    if verification.get("status") != "verified" or not all(
        (verification.get("checks") or {}).values()
    ):
        raise RuntimeError("staged result is not independently verified")
    result = _load(run_dir / "RESULT.json")
    comparisons = result.get("comparisons") or {}
    missing = [name for name, _ in COMPARISON_ORDER if name not in comparisons]
    if missing:
        raise ValueError("verified result omitted comparisons: " + ", ".join(missing))
    gates = result.get("scientific_gates") or {}
    if not gates:
        raise ValueError("verified result omitted scientific gates")

    lines = [
        "# Number Game Diversity-Bonus Confirmation-64 Result",
        "",
        f"Status: **{result['status']}**.",
        "",
        _status_text(str(result["status"])),
        "",
        "## Paired Brier Comparisons",
        "",
        (
            "| Baseline | Bonus d3 mean (SD) | Baseline mean (SD) | "
            "Relative reduction | Paired difference 95% CI | W/T/L | "
            "Changed roots |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in COMPARISON_ORDER:
        item = comparisons[name]
        interval = item["tree_bootstrap_95pct"]
        lines.append(
            "| "
            + label
            + " | "
            + f"{_fmt(item['candidate_mean_brier'])} "
            + f"({_fmt(item['candidate_brier_sample_sd'])})"
            + " | "
            + f"{_fmt(item['baseline_mean_brier'])} "
            + f"({_fmt(item['baseline_brier_sample_sd'])})"
            + " | "
            + f"{100.0 * float(item['relative_brier_reduction']):.2f}%"
            + " | "
            + f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"
            + " | "
            + f"{item['wins']}/{item['ties']}/{item['losses']}"
            + " | "
            + str(item["changed_roots"])
            + " |"
        )

    lines.extend(
        [
            "",
            "The PTS and uniform-random baselines are paired within tree and "
            "average their two frozen roots before comparison. They are "
            "descriptive controls, not scientific gates.",
            "",
            "## Frozen Gates",
            "",
        ]
    )
    for name, passed in gates.items():
        lines.append(f"- `{name}`: **{'pass' if passed else 'fail'}**")

    rank = result.get("rank_metrics") or {}
    lines.extend(
        [
            "",
            "## Ranking Diagnostics",
            "",
            (
                "- Mean candidate-root Spearman: "
                f"original `{_fmt(rank['original_mean_candidate_root_spearman'])}`, "
                f"bonus `{_fmt(rank['bonus_mean_candidate_root_spearman'])}`."
            ),
            (
                "- Mean candidate-set oracle regret: "
                f"original `{_fmt(rank['original_mean_candidate_set_oracle_regret'])}`, "
                f"bonus `{_fmt(rank['bonus_mean_candidate_set_oracle_regret'])}`."
            ),
            "",
            "## Mechanics And Provenance",
            "",
            f"- trees / accepted requests: `64 / {result['usage']['adapter_requests']}`;",
            f"- total measured cost: `${float(result['usage']['run_cost_usd']):.8f}`;",
            "- model calls made by this reporter: `0`;",
            f"- result SHA-256: `{audit.sha256_file(run_dir / 'RESULT.json')}`;",
            f"- verification SHA-256: `{audit.sha256_file(run_dir / 'VERIFICATION.json')}`.",
            "",
            "This report preserves the source result status and does not "
            "reclassify any earlier cohort.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(*, run_dir: Path, output_path: Path) -> str:
    report = render_report(run_dir=run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=daily.RUN_DIR)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = write_report(run_dir=args.run_dir, output_path=args.output)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
