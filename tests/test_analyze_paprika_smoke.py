from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_paprika_smoke import analyze


def _write_run(
    tmp_path: Path,
    *,
    clean: int,
    total: int,
    failures: float = 0.0,
    faithfulness_rate: float = 0.0,
) -> Path:
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
        "simulator_faithfulness_raw_contradiction_rate": [0.1],
        "simulator_faithfulness_final_inconsistency_rate": [faithfulness_rate],
        "simulator_faithfulness_failures": [float(faithfulness_rate > 0.0)],
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
    assert report["simulator_faithfulness_final_inconsistency_rate"] == 0.0


def test_smoke_analysis_fails_low_coverage_or_terminal_parse_failure(tmp_path: Path) -> None:
    low = analyze(_write_run(tmp_path / "low", clean=8, total=10))
    failed = analyze(_write_run(tmp_path / "failed", clean=10, total=10, failures=1.0))
    assert low["automated_pass"] is False
    assert failed["automated_pass"] is False


def test_smoke_analysis_fails_missing_or_nonzero_faithfulness_metric(tmp_path: Path) -> None:
    inconsistent = analyze(
        _write_run(tmp_path / "inconsistent", clean=10, total=10, faithfulness_rate=0.1)
    )
    missing_run = _write_run(tmp_path / "missing", clean=10, total=10)
    payload = json.loads((missing_run / "metrics.json").read_text())
    metrics = payload["items"][0]["metrics"]
    metrics.pop("simulator_faithfulness_final_inconsistency_rate")
    (missing_run / "metrics.json").write_text(json.dumps(payload))
    for trial in json.loads(next(missing_run.rglob("paprika_smoke.json")).read_text()):
        assert "simulator_faithfulness_final_inconsistency_rate" not in trial["final_metrics"]
    assert inconsistent["automated_pass"] is False
    assert analyze(missing_run)["automated_pass"] is False


def test_smoke_analysis_uses_cumulative_artifact_retry_counts(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, clean=10, total=10)
    path = next(run_dir.rglob("paprika_smoke.json"))
    trials = json.loads(path.read_text())
    trials[-1]["final_metrics"]["structured_parse_retries"] = 12.0
    path.write_text(json.dumps(trials))
    assert analyze(run_dir)["structured_parse_retries"] == 12.0
