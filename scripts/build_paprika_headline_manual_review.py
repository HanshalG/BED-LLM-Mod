#!/usr/bin/env python3
"""Build the preregistered Paprika headline manual-review packet."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_paprika_step1 import load_arm
except ModuleNotFoundError:
    from analyze_paprika_step1 import load_arm


def _record_map(run_dir: Path, method: str) -> dict[str, dict[str, Any]]:
    arm = load_arm(run_dir, method)
    return {str(record["task_id"]): record for record in arm.records}


def _block(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, indent=2, ensure_ascii=False)
    return f"<pre>{html.escape(value)}</pre>"


def build_review_packet(
    analysis: dict[str, Any],
    headline_run: Path,
    naive_nonthinking_run: Path,
    best_n_run: Path,
) -> str:
    plan = analysis.get("manual_review_plan") or {}
    review_ids = list(plan.get("all_review_task_ids") or [])
    if not review_ids:
        raise ValueError("Analysis contains no manual-review task IDs")
    disagreement_ids = set(plan.get("success_disagreement_task_ids") or [])
    spot_ids = set(plan.get("random_spot_check_task_ids") or [])
    arms = {
        "arbitration": _record_map(headline_run, "NaivePrimaryArbitration"),
        "candidate0": _record_map(headline_run, "NaivePrimaryCandidate0"),
        "naive_thinking": _record_map(headline_run, "naive"),
        "naive_nonthinking": _record_map(naive_nonthinking_run, "naive"),
        "best_n_eig": _record_map(best_n_run, "EIG"),
    }
    missing = {
        arm: sorted(set(review_ids) - set(records))
        for arm, records in arms.items()
        if set(review_ids) - set(records)
    }
    if missing:
        raise ValueError(f"Review tasks missing from arm records: {missing}")

    lines = [
        "# Paprika Headline Manual Review",
        "",
        "Status: **PENDING MANUAL REVIEW**",
        "",
        f"Analyzer status: `{analysis.get('status')}`",
        "",
        f"Review set: {len(disagreement_ids)} success disagreements plus "
        f"{len(spot_ids)} seeded remaining-task checks; {len(review_ids)} unique tasks.",
        "",
        "A correct performed private remedy claimed to fail, or an incorrect remedy "
        "claimed to succeed, invalidates the complete headline.",
        "",
        "- [ ] Every terminal success matches the private remedy.",
        "- [ ] Every attempted exact or near remedy receives a faithful response.",
        "- [ ] No correct performed remedy is reported as failing.",
        "- [ ] No incorrect remedy is reported as succeeding.",
        "",
    ]
    for task_id in review_ids:
        reference = arms["arbitration"][task_id]
        reasons = []
        if task_id in disagreement_ids:
            reasons.append("success disagreement")
        if task_id in spot_ids:
            reasons.append("seeded spot check")
        lines.extend(
            [
                f"## {task_id}",
                "",
                f"Review reason: {', '.join(reasons)}",
                "",
                "### Public Scenario",
                "",
                _block(reference.get("scenario", "")),
                "",
                "### Private Remedy",
                "",
                _block(reference.get("solution", "")),
                "",
            ]
        )
        for arm_name, records in arms.items():
            record = records[task_id]
            lines.extend([f"### {arm_name}", ""])
            turns = list(record.get("turns", []))
            if not turns:
                lines.extend(["No turns recorded.", ""])
                continue
            for index, turn in enumerate(turns, start=1):
                lines.extend(
                    [
                        f"**Turn {index}; terminal={bool(turn.get('goal_reached'))}**",
                        "",
                        "Query:",
                        _block(turn.get("query", "")),
                        "",
                        "Reply:",
                        _block(turn.get("reply", "")),
                        "",
                    ]
                )
        lines.extend(["Task verdict: **PENDING**", "", "---", ""])
    lines.extend(
        [
            "## Final Verdict",
            "",
            "Headline endpoint audit: **PENDING**",
            "",
            "Reviewer notes:",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--headline-run", type=Path, required=True)
    parser.add_argument("--naive-nonthinking-run", type=Path, required=True)
    parser.add_argument("--best-n-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    packet = build_review_packet(
        analysis,
        args.headline_run,
        args.naive_nonthinking_run,
        args.best_n_run,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(packet)
    print(args.output)


if __name__ == "__main__":
    main()
