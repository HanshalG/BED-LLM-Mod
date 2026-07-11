from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.combine_paprika_step1_splits import METHODS, combine


def _write_shard(root: Path, index: int, *, completed: bool = True) -> Path:
    run_dir = root / f"run_{index}"
    items = []
    for item_index, method in enumerate(METHODS):
        item_dir = run_dir / "items" / f"{item_index:03d}_{method}"
        item_dir.mkdir(parents=True)
        artifact = item_dir / "paprika_smoke.json"
        artifact.write_text(
            json.dumps(
                [
                    {
                        "task_id": f"customer_service:eval:{index:04d}",
                        "turns": [],
                    }
                ]
            )
        )
        items.append(
            {
                "method": method,
                "artifacts": {"paprika_smoke": str(artifact.relative_to(run_dir))},
                "metrics": {
                    "backend_cost_usd": [0.1],
                    "backend_requests": [index + 1],
                    "structured_parse_failures": [0.0],
                    "simulator_faithfulness_observations": [1.0],
                    "simulator_faithfulness_checks": [1.0],
                    "simulator_faithfulness_raw_contradictions": [0.0],
                    "simulator_faithfulness_repairs": [0.0],
                    "simulator_faithfulness_failures": [0.0],
                    "simulator_terminal_claims": [0.0],
                    "simulator_terminal_checks": [0.0],
                    "simulator_terminal_rejections": [0.0],
                },
            }
        )
    (run_dir / "metadata.json").write_text(
        json.dumps({"status": "completed" if completed else "failed"})
    )
    (run_dir / "metrics.json").write_text(json.dumps({"items": items}))
    return run_dir


def test_combine_paprika_splits_validates_tasks_and_sums_usage(tmp_path: Path) -> None:
    runs = [_write_shard(tmp_path, index) for index in range(10)]
    output = combine(runs, tmp_path / "combined")
    payload = json.loads((output / "metrics.json").read_text())
    assert [item["method"] for item in payload["items"]] == list(METHODS)
    assert payload["items"][0]["metrics"]["backend_requests"] == [55]
    assert payload["items"][1]["metrics"]["backend_cost_usd"] == pytest.approx([1.0])
    records = json.loads(
        (output / payload["items"][0]["artifacts"]["paprika_smoke"]).read_text()
    )
    assert [record["task_id"] for record in records] == [
        f"customer_service:eval:{index:04d}" for index in range(10)
    ]


def test_combine_paprika_splits_rejects_incomplete_shard(tmp_path: Path) -> None:
    runs = [_write_shard(tmp_path, index, completed=index != 4) for index in range(10)]
    with pytest.raises(ValueError, match="not complete"):
        combine(runs, tmp_path / "combined")


def test_combine_paprika_splits_supports_eig_only_rescue(tmp_path: Path) -> None:
    runs = [_write_shard(tmp_path, index) for index in range(10)]
    output = combine(runs, tmp_path / "combined", methods=("EIG",))
    payload = json.loads((output / "metrics.json").read_text())
    assert [item["method"] for item in payload["items"]] == ["EIG"]


def test_combine_paprika_splits_supports_naive_only_recovery(tmp_path: Path) -> None:
    runs = [_write_shard(tmp_path, index) for index in range(10)]
    output = combine(runs, tmp_path / "combined", methods=("naive",))
    item = json.loads((output / "metrics.json").read_text())["items"][0]
    assert item["method"] == "naive"
    assert item["metrics"]["simulator_faithfulness_observations"] == [10.0]
    assert item["metrics"]["simulator_faithfulness_final_inconsistency_rate"] == [0.0]
    assert item["metrics"]["simulator_terminal_rejections"] == [0.0]
