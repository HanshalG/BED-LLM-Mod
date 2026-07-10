from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_paprika_step1 import analyze


def _write_run(path: Path, methods: dict[str, list[int | None]]) -> None:
    items = []
    for index, (method, resolution_turns) in enumerate(methods.items()):
        item_dir = path / "items" / f"{index:03d}_{method}"
        item_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for task_index, resolution_turn in enumerate(resolution_turns):
            turns = []
            for turn_index in range(2):
                turns.append(
                    {
                        "mapped_cleanly": True,
                        "goal_reached": resolution_turn == turn_index + 1,
                    }
                )
                if resolution_turn == turn_index + 1:
                    break
            records.append(
                {
                    "task_id": f"customer_service:eval:{task_index:04d}",
                    "turns": turns,
                }
            )
        artifact = item_dir / "paprika_smoke.json"
        artifact.write_text(json.dumps(records))
        items.append(
            {
                "method": method,
                "artifacts": {"paprika_smoke": str(artifact.relative_to(path))},
                "metrics": {
                    "structured_parse_failures": [0],
                    "backend_cost_usd": [0.1],
                    "backend_requests": [100],
                    "backend_prompt_tokens": [1000],
                    "backend_completion_tokens": [500],
                    "backend_reasoning_tokens": [0],
                    "backend_forced_exits": [0],
                },
            }
        )
    (path / "metrics.json").write_text(json.dumps({"items": items}))


def test_step1_analyzer_applies_paired_gate_rules(tmp_path: Path) -> None:
    scaffolded = tmp_path / "scaffolded"
    naive = tmp_path / "naive"
    _write_run(
        scaffolded,
        {
            "EIG": [1, 1, 2, 2, None, None, None, None, None, None],
            "Full2StepEIG": [1, 1, 1, 1, 1, 1, None, None, None, None],
        },
    )
    _write_run(naive, {"naive": [2, 2, None, None, None, None, None, None, None, None]})

    result = analyze(scaffolded, naive)

    assert result["status"] == "claims1_and_2_pass"
    assert result["claim1_eig_vs_naive"]["directional_pass"] is True
    assert result["claim2_full2_vs_eig"]["gate_pass"] is True
    assert result["arms"]["Full2StepEIG"]["resolution_at_budget"] == 0.6
    assert result["arms"]["EIG"]["answer_set_coverage"] == 1.0


def test_step1_analyzer_stops_when_eig_does_not_beat_naive(tmp_path: Path) -> None:
    scaffolded = tmp_path / "scaffolded"
    naive = tmp_path / "naive"
    _write_run(
        scaffolded,
        {
            "EIG": [None] * 10,
            "Full2StepEIG": [None] * 10,
        },
    )
    _write_run(naive, {"naive": [1] * 10})

    result = analyze(scaffolded, naive)

    assert result["status"] == "claim1_fail_stop"
    assert result["claim1_eig_vs_naive"]["directional_pass"] is False
