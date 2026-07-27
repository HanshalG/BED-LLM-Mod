#!/usr/bin/env python3
"""Run Hotpot comparative planning with root-row codecs throughout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.hotpot_causal_belief_smoke import GateExecutionError
from scripts.hotpot_directional_unlock_audit import sha256_file
from scripts.hotpot_shared_comparative_v2 import (
    DEVELOPMENT_COST_CAP,
    ROOT_COUNT,
    SERVING_COST_CAP,
    development_tasks,
    run_tasks,
)
from scripts.hotpot_shared_comparative_v3 import (
    EXPOSED_CONFIRMATION_IDS,
    exposed_serving_tasks,
    parse_rank_rows,
    rank_row_messages,
)


INTERFACE_VERSION = "hotpot-shared-comparative-v4-1"
SERVING_TASK_INDEX = 1


def parse_myopic_rank_rows(text: str) -> list[int]:
    if text != text.strip():
        raise ValueError("myopic rank rows must be canonical text")
    lines = text.splitlines()
    if len(lines) != ROOT_COUNT:
        raise ValueError("myopic rank response has wrong line count")
    ranks: dict[int, int] = {}
    for root_index, line in enumerate(lines):
        fields = line.split("|")
        if (
            len(fields) != 2
            or fields[0] != f"R{root_index + 1}"
            or not fields[1].isdigit()
            or not 1 <= int(fields[1]) <= ROOT_COUNT
        ):
            raise ValueError("myopic rank row is invalid")
        ranks[root_index] = int(fields[1])
    if set(ranks.values()) != set(range(1, ROOT_COUNT + 1)):
        raise ValueError("myopic ranks are not a complete permutation")
    return sorted(range(ROOT_COUNT), key=ranks.__getitem__)


def myopic_rank_row_messages(
    initial_hypotheses: Sequence[str],
    root_titles: Sequence[str],
) -> list[dict[str, str]]:
    payload = {
        "unresolved_hypotheses": list(initial_hypotheses),
        "root_titles": [
            {"root_id": f"R{index + 1}", "title": title}
            for index, title in enumerate(root_titles)
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Rank four candidate first articles using only the supplied "
                "current belief state and titles. Prefer the article most likely "
                "to resolve answer uncertainty immediately. Return exactly four "
                "lines in numeric root order using literal vertical-bar "
                "delimiters: R1|k, R2|k, R3|k, R4|k. Here k is the root's unique "
                "rank 1 through 4, with 1 best. Use every rank exactly once. "
                "Return no header or other text."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def v4_serving_tasks(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], int]:
    tasks, cohort_rows = exposed_serving_tasks(paths)
    if str(tasks[0]["row"]["id"]) != EXPOSED_CONFIRMATION_IDS[0]:
        raise ValueError("V3 exposed row does not reproduce")

    from scripts.hotpot_future_uplift_confirmation import (
        metadata_splits,
        qualification,
        selected_rows,
    )
    from scripts.hotpot_shared_comparative_v2 import _build_task

    _metadata, splits = metadata_splits(paths)
    cohort = selected_rows(paths, splits["confirmation"])
    qualified = [
        (row, diagnostic)
        for row in cohort
        if (diagnostic := qualification(row))["qualifies"]
    ]
    observed_ids = tuple(str(row["id"]) for row, _ in qualified[:10])
    if observed_ids != EXPOSED_CONFIRMATION_IDS:
        raise ValueError("previously exposed confirmation rows do not reproduce")
    row, diagnostic = qualified[SERVING_TASK_INDEX]
    return [
        _build_task(row, diagnostic["root_context_indices"])
    ], len(cohort)


def _run(
    config: Config,
    *,
    tasks: Sequence[dict[str, Any]],
    stage: str,
    raw_path: Path,
    cohort_rows_materialized: int,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    return run_tasks(
        config,
        tasks=tasks,
        stage=stage,
        raw_path=raw_path,
        cohort_rows_materialized=cohort_rows_materialized,
        model_adapter=model_adapter,
        myopic_message_builder=myopic_rank_row_messages,
        myopic_response_parser=parse_myopic_rank_rows,
        plan_message_builder=rank_row_messages,
        plan_response_parser=parse_rank_rows,
        interface_version=INTERFACE_VERSION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("serving", "development"), required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--train-shard", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if len(args.train_shard) != 2:
        parser.error("exactly two --train-shard values are required")

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_max_output_tokens = min(
        config.openrouter_max_output_tokens, 2048
    )
    config.openrouter_concurrency = min(config.openrouter_concurrency, 64)
    config.log_path = args.output_dir / "run.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"

    if args.stage == "serving":
        tasks, cohort_rows = v4_serving_tasks(args.train_shard)
        config.openrouter_projected_cost_usd = 0.15
        config.openrouter_run_budget_usd = SERVING_COST_CAP
    else:
        tasks, cohort_rows = development_tasks(args.train_shard)
        config.openrouter_projected_cost_usd = 0.75
        config.openrouter_run_budget_usd = DEVELOPMENT_COST_CAP
    try:
        payload = _run(
            config,
            tasks=tasks,
            stage=args.stage,
            raw_path=raw_path,
            cohort_rows_materialized=cohort_rows,
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        (args.output_dir / f"{args.stage.upper()}_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = (args.output_dir / args.stage.upper()).with_suffix(
        ".json"
    )
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output_path),
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
