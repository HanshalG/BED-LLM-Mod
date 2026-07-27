#!/usr/bin/env python3
"""Run Hotpot shared comparative planning with rank-per-root lines."""

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
from scripts.hotpot_future_uplift_confirmation import (
    metadata_splits,
    qualification,
    selected_rows,
)
from scripts.hotpot_shared_comparative_v2 import (
    DEVELOPMENT_COST_CAP,
    ROOT_COUNT,
    SERVING_COST_CAP,
    _build_task,
    development_tasks,
    run_tasks,
)


INTERFACE_VERSION = "hotpot-shared-comparative-v3-1"
EXPOSED_CONFIRMATION_IDS = (
    "5ae67c685542996d980e7b84",
    "5adf89c05542993344016cdd",
    "5ae0e32b55429924de1b71cd",
    "5a863f995542991e771815da",
    "5a8c46fd554299240d9c2106",
    "5ae3ff125542995dadf242a8",
    "5a8214a6554299676cceb207",
    "5ac2acb155429967731025f5",
    "5a8f6efd5542992414482ac9",
    "5ae77824554299540e5a55b9",
)


def parse_rank_rows(text: str, *, candidate_count: int) -> dict[str, Any]:
    if text != text.strip():
        raise ValueError("rank rows must be canonical text")
    lines = text.splitlines()
    if len(lines) != ROOT_COUNT:
        raise ValueError("rank response has wrong line count")
    followups: dict[int, int] = {}
    ranks: dict[int, int] = {}
    for root_index, line in enumerate(lines):
        fields = line.split("|")
        if (
            len(fields) != 3
            or fields[0] != f"R{root_index + 1}"
            or fields[1]
            not in {
                f"T{candidate_index + 1}"
                for candidate_index in range(candidate_count)
            }
            or not fields[2].isdigit()
            or not 1 <= int(fields[2]) <= ROOT_COUNT
        ):
            raise ValueError("rank row is invalid")
        followups[root_index] = int(fields[1][1:]) - 1
        ranks[root_index] = int(fields[2])
    if set(ranks.values()) != set(range(1, ROOT_COUNT + 1)):
        raise ValueError("rank values are not a complete permutation")
    root_order = sorted(range(ROOT_COUNT), key=ranks.__getitem__)
    return {
        "root_order": root_order,
        "followup_indices": followups,
    }


def rank_row_messages(
    *,
    states: Sequence[Sequence[str]],
    all_titles: Sequence[str],
    root_context_indices: Sequence[int],
) -> list[dict[str, str]]:
    roots = []
    for root_index, (state, context_index) in enumerate(
        zip(states, root_context_indices, strict=True)
    ):
        candidates = [
            title
            for index, title in enumerate(all_titles)
            if index != context_index
        ]
        roots.append(
            {
                "root_id": f"R{root_index + 1}",
                "revealed_root_title": all_titles[context_index],
                "post_reveal_belief_state": list(state),
                "candidate_titles": [
                    {"title_id": f"T{index + 1}", "title": title}
                    for index, title in enumerate(candidates)
                ],
            }
        )
    return [
        {
            "role": "system",
            "content": (
                "Compare four complete two-article retrieval paths. For each "
                "root, use its revealed article title and post-reveal belief "
                "state to choose the candidate next title most likely to resolve "
                "the missing evidence. Rank the complete paths by resulting "
                "ability to answer, accounting for evidence already resolved by "
                "the root as well as its continuation. Do not rank continuation "
                "quality alone. Treat roots independently and do not infer an "
                "original question. Return exactly four lines in numeric root "
                "order using literal vertical-bar delimiters: R1|Tm|k, "
                "R2|Tm|k, R3|Tm|k, R4|Tm|k. Here Tm is that root's chosen title "
                "ID and k is its unique rank 1 through 4, with 1 best. Use every "
                "rank exactly once. Return no header or other text."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"root_paths": roots}, separators=(",", ":")
            ),
        },
    ]


def exposed_serving_tasks(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], int]:
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
    row, diagnostic = qualified[0]
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
        tasks, cohort_rows = exposed_serving_tasks(args.train_shard)
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
