#!/usr/bin/env python3
"""Analyze a real-model Paprika Step 0 smoke without overstating its evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ERROR_RE = re.compile(r"Traceback|RuntimeError|ValueError|CUDA out of memory|Killed", re.I)


def _load_one(run_dir: Path, name: str) -> tuple[Path, Any]:
    matches = sorted(run_dir.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} under {run_dir}, found {len(matches)}")
    return matches[0], json.loads(matches[0].read_text())


def analyze(run_dir: Path, *, coverage_threshold: float = 0.85) -> dict[str, Any]:
    smoke_path, trials = _load_one(run_dir, "paprika_smoke.json")
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise ValueError(f"Missing {metrics_path}")
    metrics_payload = json.loads(metrics_path.read_text())
    items = metrics_payload.get("items", [])
    if len(items) != 1:
        raise ValueError(f"Expected one smoke metrics item, found {len(items)}")
    metrics = items[0].get("metrics", {})

    turns = [turn for trial in trials for turn in trial.get("turns", [])]
    clean = sum(turn.get("mapped_cleanly") is True for turn in turns)
    coverage = clean / len(turns) if turns else 0.0
    failures_trace = metrics.get("structured_parse_failures", [])
    retries_trace = metrics.get("structured_parse_retries", [])
    terminal_failures = max(failures_trace, default=0.0)
    retries = max(retries_trace, default=0.0)

    log_path = run_dir / "run.log"
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    forced_exits = log_text.count("Forced thinking exit")
    usage_events = max(
        log_text.count('"event": "llm_token_usage"'),
        log_text.count("llm_token_usage"),
    )
    error_lines = sorted({line.strip() for line in log_text.splitlines() if ERROR_RE.search(line)})
    automated_pass = (
        len(trials) == 5
        and bool(turns)
        and coverage >= coverage_threshold
        and terminal_failures == 0
        and not error_lines
    )
    return {
        "status": "automated_pass_manual_review_pending" if automated_pass else "automated_fail",
        "automated_pass": automated_pass,
        "manual_transcript_review_required": True,
        "run_dir": str(run_dir.resolve()),
        "smoke_artifact": str(smoke_path.resolve()),
        "num_tasks": len(trials),
        "num_turns": len(turns),
        "cleanly_mapped_turns": clean,
        "answer_set_coverage": coverage,
        "coverage_threshold": coverage_threshold,
        "structured_parse_retries": retries,
        "structured_parse_failures": terminal_failures,
        "forced_thinking_exits": forced_exits,
        "llm_usage_events": usage_events,
        "forced_exit_rate": forced_exits / usage_events if usage_events else None,
        "runtime_error_lines": error_lines,
        "resolved_tasks": sum(bool(trial.get("final_metrics", {}).get("resolved")) for trial in trials),
        "task_summaries": [
            {
                "task_id": trial.get("task_id"),
                "scenario": trial.get("scenario"),
                "num_turns": len(trial.get("turns", [])),
                "resolved": bool(trial.get("final_metrics", {}).get("resolved")),
                "queries": [turn.get("query") for turn in trial.get("turns", [])],
                "kinds": [turn.get("kind") for turn in trial.get("turns", [])],
                "replies": [turn.get("reply") for turn in trial.get("turns", [])],
            }
            for trial in trials
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--coverage-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.run_dir, coverage_threshold=args.coverage_threshold)
    output = args.output or args.run_dir / "paprika_step0_analysis.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
