#!/usr/bin/env python3
"""Combine isolated one-task Paprika paired shards for the frozen analyzer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


METHODS = ("naive", "EIG", "Full2StepEIG")
SUM_METRICS = (
    "backend_cost_usd",
    "backend_requests",
    "backend_prompt_tokens",
    "backend_completion_tokens",
    "backend_reasoning_tokens",
    "backend_forced_exits",
    "structured_parse_failures",
    "simulator_faithfulness_observations",
    "simulator_faithfulness_checks",
    "simulator_faithfulness_raw_contradictions",
    "simulator_faithfulness_repairs",
    "simulator_faithfulness_failures",
    "simulator_terminal_claims",
    "simulator_terminal_checks",
    "simulator_terminal_rejections",
)


def _method_item(run_dir: Path, method: str) -> dict[str, Any]:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata.get("status") != "completed":
        raise ValueError(f"Paprika shard is not complete: {run_dir}")
    payload = json.loads((run_dir / "metrics.json").read_text())
    matches = [item for item in payload.get("items", []) if item.get("method") == method]
    if len(matches) != 1:
        raise ValueError(f"Expected one {method} item in {run_dir}, found {len(matches)}")
    return matches[0]


def combine(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    methods: Sequence[str] = METHODS,
) -> Path:
    if len(run_dirs) != 10:
        raise ValueError("Paprika Step 1 split combination requires exactly ten shards")
    methods = tuple(methods)
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be a non-empty subset of {METHODS}")
    records_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    metrics_by_method: dict[str, dict[str, float]] = {
        method: {name: 0.0 for name in SUM_METRICS} for method in methods
    }
    for run_dir in run_dirs:
        shard_task_id: str | None = None
        for method in methods:
            item = _method_item(run_dir, method)
            artifact = run_dir / item["artifacts"]["paprika_smoke"]
            records = json.loads(artifact.read_text())
            if not isinstance(records, list) or len(records) != 1:
                raise ValueError(f"Expected one Paprika record in {artifact}")
            task_id = str(records[0]["task_id"])
            if shard_task_id is not None and task_id != shard_task_id:
                raise ValueError(f"Methods target different tasks in {run_dir}")
            shard_task_id = task_id
            records_by_method[method].append(records[0])
            metrics = item.get("metrics", {})
            for name in SUM_METRICS:
                values = metrics.get(name) or [0]
                metrics_by_method[method][name] += float(values[-1])

    expected_ids = [f"customer_service:eval:{index:04d}" for index in range(10)]
    for method in methods:
        records_by_method[method].sort(key=lambda record: str(record["task_id"]))
        task_ids = [str(record["task_id"]) for record in records_by_method[method]]
        if task_ids != expected_ids:
            raise ValueError(f"{method} shards do not cover the canonical ten tasks: {task_ids}")

    output_dir.mkdir(parents=True, exist_ok=False)
    items: list[dict[str, Any]] = []
    for index, method in enumerate(methods):
        item_dir = output_dir / "items" / f"{index:03d}_{method}"
        item_dir.mkdir(parents=True)
        artifact = item_dir / "paprika_smoke.json"
        artifact.write_text(json.dumps(records_by_method[method], indent=2) + "\n")
        metric_payload: dict[str, list[float | int]] = {}
        for name, value in metrics_by_method[method].items():
            if name in {"backend_requests", "backend_prompt_tokens", "backend_completion_tokens", "backend_reasoning_tokens", "backend_forced_exits"}:
                metric_payload[name] = [int(value)]
            else:
                metric_payload[name] = [value]
        observations = metrics_by_method[method]["simulator_faithfulness_observations"]
        metric_payload["simulator_faithfulness_raw_contradiction_rate"] = [
            metrics_by_method[method]["simulator_faithfulness_raw_contradictions"]
            / observations
            if observations
            else 0.0
        ]
        metric_payload["simulator_faithfulness_final_inconsistency_rate"] = [
            metrics_by_method[method]["simulator_faithfulness_failures"] / observations
            if observations
            else 0.0
        ]
        items.append(
            {
                "method": method,
                "metrics": metric_payload,
                "artifacts": {"paprika_smoke": str(artifact.relative_to(output_dir))},
            }
        )
    (output_dir / "metrics.json").write_text(json.dumps({"items": items}, indent=2) + "\n")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "kind": "paprika_step1_split_combination",
                "source_runs": [str(path) for path in run_dirs],
            },
            indent=2,
        )
        + "\n"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=10, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    print(combine(args.run_dirs, args.output_dir, methods=args.methods))


if __name__ == "__main__":
    main()
