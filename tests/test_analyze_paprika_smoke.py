from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_paprika_smoke import analyze


def _write_run(tmp_path: Path, *, clean: int, total: int, failures: float = 0.0) -> Path:
    run_dir = tmp_path / "run"
    item_dir = run_dir / "items" / "000_EIG"
    item_dir.mkdir(parents=True)
    trials = []
    cursor = 0
    for task_index in range(5):
        turns = []
        for turn_index in range(total // 5):
            turns.append({
                "query": f"q{turn_index}",
                "reply": f"a{turn_index}",
                "mapped_cleanly": cursor < clean,
            })
            cursor += 1
        trials.append({
            "task_id": f"task-{task_index}",
            "scenario": f"scenario-{task_index}",
            "turns": turns,
            "final_metrics": {"resolved": float(task_index == 0)},
        })
    (item_dir / "paprika_smoke.json").write_text(json.dumps(trials))
    (run_dir / "metrics.json").write_text(json.dumps({"items": [{"metrics": {
        "structured_parse_retries": [2.0],
        "structured_parse_failures": [failures],
    }}]}))
    (run_dir / "run.log").write_text(
        '\n'.join(['{"event": "llm_token_usage"}'] * 10 + ["Forced thinking exit"] * 2)
    )
    return run_dir


def test_smoke_analysis_passes_coverage_but_requires_manual_review(tmp_path: Path) -> None:
    report = analyze(_write_run(tmp_path, clean=9, total=10))
    assert report["automated_pass"] is True
    assert report["status"] == "automated_pass_manual_review_pending"
    assert report["answer_set_coverage"] == 0.9
    assert report["forced_exit_rate"] == 0.2
    assert report["resolved_tasks"] == 1
    assert report["manual_transcript_review_required"] is True


def test_smoke_analysis_fails_low_coverage_or_terminal_parse_failure(tmp_path: Path) -> None:
    low = analyze(_write_run(tmp_path / "low", clean=8, total=10))
    failed = analyze(_write_run(tmp_path / "failed", clean=10, total=10, failures=1.0))
    assert low["automated_pass"] is False
    assert failed["automated_pass"] is False
