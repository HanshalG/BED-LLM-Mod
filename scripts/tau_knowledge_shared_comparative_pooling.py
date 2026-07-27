#!/usr/bin/env python3
"""Compare pooled tau-Knowledge roots on one shared semantic rank scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.tau_knowledge_first_link_scorer import (
    DOCUMENT_EXCERPT_CHARS,
    pairwise_ranking_points,
)
from scripts.tau_knowledge_retrieval_opportunity import (
    GateExecutionError,
    SCHEMA_VERSION,
    _build_model,
    _checkpoint,
    _retrieved_ids,
    _usage_snapshot,
)


INTERFACE_VERSION = 1
ROOT_COUNT = 10
RANK_REPLICATES = 3
SERVING_TASK_COUNT = 2
DEVELOPMENT_TASK_COUNT = 20
PRESENTATION_SEED = 24419
RANDOM_CONTROL_SEED = 24420
MAX_INPUT_CHARS = 100_000


def _canonical_text(value: str) -> str:
    return " ".join(value.split())


def _load_records(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if payload.get("status") not in {"passed", "gate_failed"} or not isinstance(
        records, list
    ):
        raise ValueError(
            "pooling requires a complete scored tau artifact with records"
        )
    if len(records) != DEVELOPMENT_TASK_COUNT:
        raise ValueError("pooling source must contain exactly 20 tasks")
    return payload, records


def merge_record_pools(
    primary_records: Sequence[dict[str, Any]],
    secondary_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    secondary_by_id = {
        str(record["task_id"]): record for record in secondary_records
    }
    if len(secondary_by_id) != len(secondary_records):
        raise ValueError("secondary artifact has duplicate task IDs")
    merged = []
    for primary in primary_records:
        task_id = str(primary["task_id"])
        if task_id not in secondary_by_id:
            raise ValueError("source artifacts have different task sets")
        secondary = secondary_by_id[task_id]
        if primary["required_documents"] != secondary["required_documents"]:
            raise ValueError("source endpoint labels differ")
        primary_branches = list(primary["first_branches"])
        secondary_branches = list(secondary["first_branches"])
        if len(primary_branches) != 5 or len(secondary_branches) != 5:
            raise ValueError("each source must contribute exactly five roots")
        initial_hypotheses = []
        seen_hypotheses = set()
        for item in [
            *primary["initial_information_need_hypotheses"],
            *secondary["initial_information_need_hypotheses"],
        ]:
            cleaned = _canonical_text(str(item))
            normalized = cleaned.casefold()
            if normalized not in seen_hypotheses:
                seen_hypotheses.add(normalized)
                initial_hypotheses.append(cleaned)
        merged.append(
            {
                "task_id": task_id,
                "opening": primary["opening"],
                "secondary_opening_differs": (
                    _canonical_text(primary["opening"])
                    != _canonical_text(secondary["opening"])
                ),
                "initial_information_need_hypotheses": initial_hypotheses,
                "required_documents": list(primary["required_documents"]),
                "first_branches": [*primary_branches, *secondary_branches],
            }
        )
    if set(secondary_by_id) != {str(row["task_id"]) for row in primary_records}:
        raise ValueError("source artifacts have different task sets")
    return merged


def _clean_excerpt(value: str) -> str:
    return _canonical_text(value)[:DOCUMENT_EXCERPT_CHARS]


def compact_rank_input(
    record: dict[str, Any],
    *,
    include_futures: bool,
    presentation_order: Sequence[int],
) -> dict[str, Any]:
    if sorted(presentation_order) != list(range(ROOT_COUNT)):
        raise ValueError("presentation order must be a root permutation")
    document_refs: dict[str, str] = {}
    catalog: list[dict[str, str]] = []

    def add_results(results: Sequence[dict[str, Any]]) -> list[str]:
        refs = []
        for result in results:
            document_id = str(result["id"])
            if document_id not in document_refs:
                document_ref = f"D{len(document_refs) + 1}"
                document_refs[document_id] = document_ref
                catalog.append(
                    {
                        "ref": document_ref,
                        "title": str(result["title"]),
                        "excerpt": _clean_excerpt(str(result["content"])),
                    }
                )
            refs.append(document_refs[document_id])
        return refs

    roots = []
    for root_index in presentation_order:
        branch = record["first_branches"][root_index]
        root: dict[str, Any] = {
            "root_id": f"R{root_index + 1}",
            "query": branch["query"],
            "first_result_refs": add_results(branch["first_results"]),
        }
        if include_futures:
            root["refreshed_information_needs"] = branch[
                "refreshed_information_need_hypotheses"
            ]
            root["followups"] = [
                {
                    "query": followup["query"],
                    "result_refs": add_results(followup["results"]),
                }
                for followup in branch["followups"]
            ]
        roots.append(root)
    payload = {
        "customer_opening": record["opening"],
        "initial_information_needs": record[
            "initial_information_need_hypotheses"
        ],
        "roots": roots,
        "document_catalog": catalog,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > MAX_INPUT_CHARS:
        raise ValueError("pooled comparative input exceeds character cap")
    return payload


def rank_messages(
    record: dict[str, Any],
    *,
    include_futures: bool,
    presentation_order: Sequence[int],
) -> list[dict[str, str]]:
    payload = compact_rank_input(
        record,
        include_futures=include_futures,
        presentation_order=presentation_order,
    )
    if include_futures:
        instruction = (
            "Rank all ten root searches by the best total distinct useful policy "
            "coverage achievable after that root and exactly one of its displayed "
            "followups. Judge whether the first result makes the followup useful and "
            "whether their union covers the customer's prerequisites, exceptions, "
            "eligibility rules, and procedures. Refreshed needs are fallible clues, "
            "not facts. Do not reward document count, verbosity, duplicated evidence, "
            "or a promising query unsupported by its returned documents."
        )
    else:
        instruction = (
            "Rank all ten root searches using only useful policy evidence explicitly "
            "present in their first results. Judge coverage of the customer's "
            "prerequisites, exceptions, eligibility rules, and procedures. Do not "
            "imagine a followup or reward document count, verbosity, duplicated "
            "evidence, or promising query wording unsupported by returned documents."
        )
    return [
        {
            "role": "system",
            "content": (
                "Compare retrieval plans for an internal banking support task. "
                "Required-document labels and evaluation answers are hidden. Make "
                "one comparative ranking rather than independent numeric ratings. "
                "Return one strict line and no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                instruction
                + " Output all root IDs exactly once from best to worst as "
                "`ORDER|R1|R2|...|R10`; the example IDs show syntax only, not the "
                "desired order. Retrieval data: "
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_rank_order(text: str) -> list[int]:
    if text != text.strip() or "\n" in text.strip():
        raise ValueError("rank response must be one canonical line")
    fields = text.split("|")
    expected = {f"R{index + 1}" for index in range(ROOT_COUNT)}
    if len(fields) != ROOT_COUNT + 1 or fields[0] != "ORDER":
        raise ValueError("rank response has invalid shape")
    if set(fields[1:]) != expected:
        raise ValueError("rank response is not a complete root permutation")
    return [int(field[1:]) - 1 for field in fields[1:]]


def presentation_order(
    *,
    task_index: int,
    replicate_index: int,
    include_futures: bool,
) -> list[int]:
    mode_offset = 1_000_003 if include_futures else 0
    seed = (
        PRESENTATION_SEED
        + mode_offset
        + 10_007 * task_index
        + 101 * replicate_index
    )
    order = list(range(ROOT_COUNT))
    random.Random(seed).shuffle(order)
    return order


def order_scores(order: Sequence[int]) -> list[int]:
    if sorted(order) != list(range(ROOT_COUNT)):
        raise ValueError("rank order must be a permutation")
    scores = [0] * ROOT_COUNT
    for position, root_index in enumerate(order):
        scores[root_index] = ROOT_COUNT - position
    return scores


def aggregate_orders(orders: Sequence[Sequence[int]]) -> list[int]:
    if len(orders) != RANK_REPLICATES:
        raise ValueError("unexpected rank replicate count")
    totals = [0] * ROOT_COUNT
    for order in orders:
        scores = order_scores(order)
        totals = [left + right for left, right in zip(totals, scores, strict=True)]
    return sorted(range(ROOT_COUNT), key=lambda index: (-totals[index], index))


def pairwise_order_agreement(
    left: Sequence[int],
    right: Sequence[int],
) -> float:
    left_positions = {root: index for index, root in enumerate(left)}
    right_positions = {root: index for index, root in enumerate(right)}
    agreements = 0
    comparisons = 0
    for first in range(ROOT_COUNT):
        for second in range(first + 1, ROOT_COUNT):
            comparisons += 1
            agreements += (
                (left_positions[first] < left_positions[second])
                == (right_positions[first] < right_positions[second])
            )
    return agreements / comparisons


def _mean_replicate_agreement(orders: Sequence[Sequence[int]]) -> float:
    values = [
        pairwise_order_agreement(orders[left], orders[right])
        for left in range(len(orders))
        for right in range(left + 1, len(orders))
    ]
    return sum(values) / len(values)


def _root_values(record: dict[str, Any]) -> tuple[list[int], list[int]]:
    required = set(record["required_documents"])
    immediate = []
    total = []
    for branch in record["first_branches"]:
        first_ids = _retrieved_ids(branch["first_results"])
        immediate.append(len(required & first_ids))
        total.append(
            max(
                len(
                    required
                    & (
                        first_ids
                        | _retrieved_ids(followup["results"])
                    )
                )
                for followup in branch["followups"]
            )
        )
    return immediate, total


def summarize(
    records: Sequence[dict[str, Any]],
    myopic_orders: Sequence[Sequence[Sequence[int]]],
    nonmyopic_orders: Sequence[Sequence[Sequence[int]]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    expected_tasks = (
        SERVING_TASK_COUNT if stage == "serving_smoke" else DEVELOPMENT_TASK_COUNT
    )
    rows = []
    myopic_points = 0.0
    nonmyopic_points = 0.0
    comparable = 0
    for task_index, record in enumerate(records):
        immediate_values, total_values = _root_values(record)
        myopic_aggregate = aggregate_orders(myopic_orders[task_index])
        nonmyopic_aggregate = aggregate_orders(nonmyopic_orders[task_index])
        myopic_scores = order_scores(myopic_aggregate)
        nonmyopic_scores = order_scores(nonmyopic_aggregate)
        my_points, my_count = pairwise_ranking_points(
            myopic_scores, total_values
        )
        non_points, non_count = pairwise_ranking_points(
            nonmyopic_scores, total_values
        )
        if my_count != non_count:
            raise ValueError("myopic and non-myopic comparisons do not align")
        myopic_points += my_points
        nonmyopic_points += non_points
        comparable += my_count
        myopic_root = myopic_aggregate[0]
        nonmyopic_root = nonmyopic_aggregate[0]
        random_root = random.Random(
            RANDOM_CONTROL_SEED + task_index
        ).randrange(ROOT_COUNT)
        rows.append(
            {
                "task_id": record["task_id"],
                "immediate_values": immediate_values,
                "oracle_tail_values": total_values,
                "myopic_orders": [list(order) for order in myopic_orders[task_index]],
                "nonmyopic_orders": [
                    list(order) for order in nonmyopic_orders[task_index]
                ],
                "myopic_aggregate_order": myopic_aggregate,
                "nonmyopic_aggregate_order": nonmyopic_aggregate,
                "myopic_selected_root_index": myopic_root,
                "nonmyopic_selected_root_index": nonmyopic_root,
                "random_selected_root_index": random_root,
                "myopic_selected_oracle_tail": total_values[myopic_root],
                "nonmyopic_selected_oracle_tail": total_values[nonmyopic_root],
                "random_selected_oracle_tail": total_values[random_root],
                "nonmyopic_advantage_over_myopic": (
                    total_values[nonmyopic_root] - total_values[myopic_root]
                ),
                "nonmyopic_advantage_over_random": (
                    total_values[nonmyopic_root] - total_values[random_root]
                ),
                "myopic_rank_agreement": _mean_replicate_agreement(
                    myopic_orders[task_index]
                ),
                "nonmyopic_rank_agreement": _mean_replicate_agreement(
                    nonmyopic_orders[task_index]
                ),
                "secondary_opening_differs": record[
                    "secondary_opening_differs"
                ],
            }
        )
    generator = usage["generator"]
    expected_requests = expected_tasks * RANK_REPLICATES * 2
    gates = {
        "all_tasks_complete": len(records) == expected_tasks,
        "exact_logical_request_count": int(usage["physical_requests"])
        == expected_requests,
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "zero_forced_exits": int(generator.get("forced_exits", 0)) == 0,
        "all_rankings_complete": all(
            len(row["myopic_orders"]) == RANK_REPLICATES
            and len(row["nonmyopic_orders"]) == RANK_REPLICATES
            for row in rows
        ),
    }
    myopic_accuracy = myopic_points / comparable if comparable else 0.0
    nonmyopic_accuracy = nonmyopic_points / comparable if comparable else 0.0
    myopic_total = sum(row["myopic_selected_oracle_tail"] for row in rows)
    nonmyopic_total = sum(
        row["nonmyopic_selected_oracle_tail"] for row in rows
    )
    random_total = sum(row["random_selected_oracle_tail"] for row in rows)
    advantages = [row["nonmyopic_advantage_over_myopic"] for row in rows]
    mean_myopic_agreement = sum(
        row["myopic_rank_agreement"] for row in rows
    ) / len(rows)
    mean_nonmyopic_agreement = sum(
        row["nonmyopic_rank_agreement"] for row in rows
    ) / len(rows)
    summary = {
        "num_tasks": len(records),
        "task_diagnostics": rows,
        "root_pairwise_comparable_count": comparable,
        "myopic_root_pairwise_accuracy": myopic_accuracy,
        "nonmyopic_root_pairwise_accuracy": nonmyopic_accuracy,
        "root_pairwise_accuracy_gain": nonmyopic_accuracy - myopic_accuracy,
        "myopic_selected_oracle_tail_total": myopic_total,
        "nonmyopic_selected_oracle_tail_total": nonmyopic_total,
        "random_selected_oracle_tail_total": random_total,
        "nonmyopic_advantage_over_myopic": nonmyopic_total - myopic_total,
        "nonmyopic_advantage_over_random": nonmyopic_total - random_total,
        "nonmyopic_vs_myopic_wins": sum(value > 0 for value in advantages),
        "nonmyopic_vs_myopic_ties": sum(value == 0 for value in advantages),
        "nonmyopic_vs_myopic_losses": sum(value < 0 for value in advantages),
        "selected_root_change_count": sum(
            row["myopic_selected_root_index"]
            != row["nonmyopic_selected_root_index"]
            for row in rows
        ),
        "mean_myopic_rank_agreement": mean_myopic_agreement,
        "mean_nonmyopic_rank_agreement": mean_nonmyopic_agreement,
    }
    if stage == "development":
        gates.update(
            {
                "nonmyopic_pairwise_accuracy_at_least_0_62": (
                    nonmyopic_accuracy >= 0.62
                ),
                "pairwise_accuracy_gain_at_least_0_05": (
                    nonmyopic_accuracy - myopic_accuracy >= 0.05
                ),
                "oracle_tail_gain_over_myopic_at_least_4": (
                    nonmyopic_total - myopic_total >= 4
                ),
                "wins_exceed_losses_by_at_least_2": (
                    sum(value > 0 for value in advantages)
                    - sum(value < 0 for value in advantages)
                    >= 2
                ),
                "selected_root_changes_at_least_4": (
                    summary["selected_root_change_count"] >= 4
                ),
                "nonmyopic_rank_agreement_at_least_0_55": (
                    mean_nonmyopic_agreement >= 0.55
                ),
                "nonmyopic_beats_random_by_at_least_5": (
                    nonmyopic_total - random_total >= 5
                ),
            }
        )
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_gate(
    config: Config,
    *,
    stage: str,
    primary_artifact: str | Path,
    secondary_artifact: str | Path,
    raw_checkpoint_path: Path | None,
) -> dict[str, Any]:
    primary_payload, primary_records = _load_records(primary_artifact)
    secondary_payload, secondary_records = _load_records(secondary_artifact)
    records = merge_record_pools(primary_records, secondary_records)
    if stage == "serving_smoke":
        records = records[:SERVING_TASK_COUNT]
    elif stage != "development":
        raise ValueError("stage must be serving_smoke or development")

    model = _build_model(config)
    raw: dict[str, Any] = {}
    try:
        myopic_keys = [
            (task_index, replicate_index)
            for task_index in range(len(records))
            for replicate_index in range(RANK_REPLICATES)
        ]
        myopic_raw = model.chat_complete_messages_batched(
            [
                rank_messages(
                    records[task_index],
                    include_futures=False,
                    presentation_order=presentation_order(
                        task_index=task_index,
                        replicate_index=replicate_index,
                        include_futures=False,
                    ),
                )
                for task_index, replicate_index in myopic_keys
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["myopic_rankings"] = myopic_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        parsed_myopic = [parse_rank_order(text) for text in myopic_raw]

        nonmyopic_keys = [
            (task_index, replicate_index)
            for task_index in range(len(records))
            for replicate_index in range(RANK_REPLICATES)
        ]
        nonmyopic_raw = model.chat_complete_messages_batched(
            [
                rank_messages(
                    records[task_index],
                    include_futures=True,
                    presentation_order=presentation_order(
                        task_index=task_index,
                        replicate_index=replicate_index,
                        include_futures=True,
                    ),
                )
                for task_index, replicate_index in nonmyopic_keys
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["nonmyopic_rankings"] = nonmyopic_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        parsed_nonmyopic = [parse_rank_order(text) for text in nonmyopic_raw]
        usage = _usage_snapshot(model)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    def reshape(values: Sequence[list[int]]) -> list[list[list[int]]]:
        return [
            list(
                values[
                    task_index * RANK_REPLICATES : (task_index + 1)
                    * RANK_REPLICATES
                ]
            )
            for task_index in range(len(records))
        ]

    myopic_orders = reshape(parsed_myopic)
    nonmyopic_orders = reshape(parsed_nonmyopic)
    summary = summarize(
        records,
        myopic_orders,
        nonmyopic_orders,
        usage,
        stage=stage,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "interface_version": INTERFACE_VERSION,
            "model": config.model_pairs[0].questioner.model,
            "primary_artifact_sha256": hashlib.sha256(
                Path(primary_artifact).read_bytes()
            ).hexdigest(),
            "secondary_artifact_sha256": hashlib.sha256(
                Path(secondary_artifact).read_bytes()
            ).hexdigest(),
            "primary_source_status": primary_payload["status"],
            "secondary_source_status": secondary_payload["status"],
            "task_ids": [record["task_id"] for record in records],
            "roots_per_task": ROOT_COUNT,
            "rank_replicates_per_policy": RANK_REPLICATES,
            "presentation_seed": PRESENTATION_SEED,
            "random_control_seed": RANDOM_CONTROL_SEED,
            "rank_aggregation": "Borda sum with canonical root-index tie break",
            "myopic_and_nonmyopic_logical_calls_matched": True,
            "myopic_sees_first_results_only": True,
            "nonmyopic_sees_complete_two_step_trees": True,
            "oracle_continuation_used_for_first_link_endpoint_only": True,
            "required_documents_hidden_from_model": True,
            "reasoning_disabled": True,
            "raw_responses_private_and_untracked": True,
            "development_tasks_previously_open": stage == "development",
        },
        "summary": summary,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "development"),
        required=True,
    )
    parser.add_argument("--primary-artifact", type=Path, required=True)
    parser.add_argument("--secondary-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.20
        config.openrouter_run_budget_usd = 0.75
    else:
        config.openrouter_projected_cost_usd = 2.50
        config.openrouter_run_budget_usd = 5.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "DEVELOPMENT.json"
    )
    try:
        payload = run_gate(
            config,
            stage=args.stage,
            primary_artifact=args.primary_artifact,
            secondary_artifact=args.secondary_artifact,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        (args.output_dir / f"{args.stage.upper()}_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = args.output_dir / output_name
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output_path),
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
