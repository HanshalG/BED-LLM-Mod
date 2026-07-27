#!/usr/bin/env python3
"""Adjudicate tau-Knowledge myopic and future finalists with balanced duels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.tau_knowledge_first_link_scorer import DOCUMENT_EXCERPT_CHARS
from scripts.tau_knowledge_retrieval_opportunity import (
    GateExecutionError,
    SCHEMA_VERSION,
    _build_model,
    _checkpoint,
    _usage_snapshot,
)
from scripts.tau_knowledge_shared_comparative_pooling import (
    DEVELOPMENT_TASK_COUNT,
    MAX_INPUT_CHARS,
    _load_records,
    _root_values,
    merge_record_pools,
)


INTERFACE_VERSION = 1
SERVING_TASK_COUNT = 2
DUEL_CALLS_PER_CHANGED_TASK = 2


def _canonical_text(value: str) -> str:
    return " ".join(value.split())


def _load_rank_rows(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if payload.get("status") != "gate_failed" or not isinstance(summary, dict):
        raise ValueError("finalist duel requires the frozen failed rank artifact")
    rows = summary.get("task_diagnostics")
    if not isinstance(rows, list) or len(rows) != DEVELOPMENT_TASK_COUNT:
        raise ValueError("rank artifact lacks complete task diagnostics")
    return payload, rows


def load_finalist_records(
    primary_artifact: str | Path,
    secondary_artifact: str | Path,
    rank_artifact: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _primary_payload, primary_records = _load_records(primary_artifact)
    _secondary_payload, secondary_records = _load_records(secondary_artifact)
    merged = merge_record_pools(primary_records, secondary_records)
    rank_payload, rank_rows = _load_rank_rows(rank_artifact)
    rows_by_id = {str(row["task_id"]): row for row in rank_rows}
    if len(rows_by_id) != len(rank_rows):
        raise ValueError("rank artifact has duplicate task IDs")
    result = []
    for record in merged:
        task_id = str(record["task_id"])
        if task_id not in rows_by_id:
            raise ValueError("rank and proposal artifacts have different tasks")
        row = rows_by_id[task_id]
        myopic_root = int(row["myopic_selected_root_index"])
        nonmyopic_root = int(row["nonmyopic_selected_root_index"])
        if not 0 <= myopic_root < len(record["first_branches"]):
            raise ValueError("myopic finalist index is invalid")
        if not 0 <= nonmyopic_root < len(record["first_branches"]):
            raise ValueError("non-myopic finalist index is invalid")
        result.append(
            {
                **record,
                "myopic_finalist_index": myopic_root,
                "nonmyopic_finalist_index": nonmyopic_root,
            }
        )
    if set(rows_by_id) != {str(record["task_id"]) for record in merged}:
        raise ValueError("rank and proposal artifacts have different tasks")
    return result, rank_payload


def _clean_excerpt(value: str) -> str:
    return _canonical_text(value)[:DOCUMENT_EXCERPT_CHARS]


def compact_duel_input(
    record: dict[str, Any],
    *,
    candidate_a_root: int,
    candidate_b_root: int,
) -> dict[str, Any]:
    if candidate_a_root == candidate_b_root:
        raise ValueError("duel candidates must differ")
    document_refs: dict[str, str] = {}
    catalog: list[dict[str, str]] = []

    def add_results(results: Sequence[dict[str, Any]]) -> list[str]:
        refs = []
        for result in results:
            document_id = str(result["id"])
            if document_id not in document_refs:
                ref = f"D{len(document_refs) + 1}"
                document_refs[document_id] = ref
                catalog.append(
                    {
                        "ref": ref,
                        "title": str(result["title"]),
                        "excerpt": _clean_excerpt(str(result["content"])),
                    }
                )
            refs.append(document_refs[document_id])
        return refs

    def compact_candidate(label: str, root_index: int) -> dict[str, Any]:
        branch = record["first_branches"][root_index]
        return {
            "candidate": label,
            "root_query": branch["query"],
            "first_result_refs": add_results(branch["first_results"]),
            "refreshed_information_needs": branch[
                "refreshed_information_need_hypotheses"
            ],
            "followups": [
                {
                    "query": followup["query"],
                    "result_refs": add_results(followup["results"]),
                }
                for followup in branch["followups"]
            ],
        }

    payload = {
        "customer_opening": record["opening"],
        "initial_information_needs": record[
            "initial_information_need_hypotheses"
        ],
        "candidates": [
            compact_candidate("A", candidate_a_root),
            compact_candidate("B", candidate_b_root),
        ],
        "document_catalog": catalog,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > MAX_INPUT_CHARS:
        raise ValueError("finalist duel input exceeds character cap")
    return payload


def duel_messages(
    record: dict[str, Any],
    *,
    candidate_a_root: int,
    candidate_b_root: int,
) -> list[dict[str, str]]:
    payload = compact_duel_input(
        record,
        candidate_a_root=candidate_a_root,
        candidate_b_root=candidate_b_root,
    )
    schema = {
        "candidate_a_coverage": "one short evidence-grounded sentence",
        "candidate_b_coverage": "one short evidence-grounded sentence",
        "winner": "A, B, or T for a genuine tie",
    }
    return [
        {
            "role": "system",
            "content": (
                "Compare two retrieval plans for an internal banking support task. "
                "Required-document labels and evaluation answers are hidden. Assess "
                "each plan's supplied evidence before choosing. Return one strict "
                "JSON object and no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each candidate, inspect the union of its first results and best "
                "one displayed followup. Compare distinct useful policy coverage of "
                "the customer's prerequisites, exceptions, eligibility rules, and "
                "procedures. The refreshed needs are fallible clues, not facts. Do "
                "not reward document count, verbosity, duplicated evidence, or "
                "promising query wording unsupported by returned documents. Use T "
                "only when the candidates are genuinely indistinguishable. Return "
                "exactly these keys: "
                + json.dumps(schema, ensure_ascii=True, separators=(",", ":"))
                + ". Retrieval data: "
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_duel(text: str) -> dict[str, str]:
    payload = _parse_json_object(text)
    expected = {
        "candidate_a_coverage",
        "candidate_b_coverage",
        "winner",
    }
    if set(payload) != expected:
        raise ValueError("duel response has unexpected keys")
    result = {}
    for key in ("candidate_a_coverage", "candidate_b_coverage"):
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("duel coverage must be a nonempty string")
        result[key] = _canonical_text(value)
    winner = payload["winner"]
    if winner not in {"A", "B", "T"}:
        raise ValueError("duel winner must be A, B, or T")
    result["winner"] = winner
    return result


def resolve_balanced_duel(
    *,
    myopic_root: int,
    nonmyopic_root: int,
    forward: dict[str, str],
    reversed_order: dict[str, str],
) -> tuple[int, str]:
    if myopic_root == nonmyopic_root:
        return myopic_root, "same_finalist"

    def canonical_winner(
        response: dict[str, str],
        *,
        candidate_a_root: int,
        candidate_b_root: int,
    ) -> int | None:
        if response["winner"] == "T":
            return None
        return (
            candidate_a_root
            if response["winner"] == "A"
            else candidate_b_root
        )

    forward_winner = canonical_winner(
        forward,
        candidate_a_root=myopic_root,
        candidate_b_root=nonmyopic_root,
    )
    reverse_winner = canonical_winner(
        reversed_order,
        candidate_a_root=nonmyopic_root,
        candidate_b_root=myopic_root,
    )
    if (
        forward_winner is not None
        and forward_winner == reverse_winner
    ):
        if forward_winner == nonmyopic_root:
            return nonmyopic_root, "unanimous_nonmyopic"
        return myopic_root, "unanimous_myopic"
    return myopic_root, "fallback_myopic"


def _selected_vs_all(
    values: Sequence[int],
    selected_root: int,
) -> tuple[int, int]:
    points = 0
    comparable = 0
    for root_index, value in enumerate(values):
        if root_index == selected_root or value == values[selected_root]:
            continue
        comparable += 1
        points += values[selected_root] > value
    return points, comparable


def summarize(
    records: Sequence[dict[str, Any]],
    parsed_duels: dict[str, tuple[dict[str, str], dict[str, str]]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    expected_tasks = (
        SERVING_TASK_COUNT if stage == "serving_smoke" else DEVELOPMENT_TASK_COUNT
    )
    rows = []
    myopic_top_points = 0
    myopic_top_comparable = 0
    selected_top_points = 0
    selected_top_comparable = 0
    for record in records:
        task_id = str(record["task_id"])
        myopic_root = int(record["myopic_finalist_index"])
        nonmyopic_root = int(record["nonmyopic_finalist_index"])
        if myopic_root == nonmyopic_root:
            selected_root, decision = myopic_root, "same_finalist"
            forward = None
            reversed_order = None
        else:
            forward, reversed_order = parsed_duels[task_id]
            selected_root, decision = resolve_balanced_duel(
                myopic_root=myopic_root,
                nonmyopic_root=nonmyopic_root,
                forward=forward,
                reversed_order=reversed_order,
            )
        _immediate, values = _root_values(record)
        my_points, my_count = _selected_vs_all(values, myopic_root)
        selected_points, selected_count = _selected_vs_all(
            values, selected_root
        )
        myopic_top_points += my_points
        myopic_top_comparable += my_count
        selected_top_points += selected_points
        selected_top_comparable += selected_count
        rows.append(
            {
                "task_id": task_id,
                "myopic_finalist_index": myopic_root,
                "nonmyopic_finalist_index": nonmyopic_root,
                "selected_root_index": selected_root,
                "decision": decision,
                "forward": forward,
                "reversed": reversed_order,
                "oracle_tail_values": values,
                "myopic_oracle_tail": values[myopic_root],
                "selected_oracle_tail": values[selected_root],
                "advantage_over_myopic": (
                    values[selected_root] - values[myopic_root]
                ),
            }
        )
    changed_count = sum(
        row["myopic_finalist_index"] != row["nonmyopic_finalist_index"]
        for row in rows
    )
    expected_requests = changed_count * DUEL_CALLS_PER_CHANGED_TASK
    generator = usage["generator"]
    gates = {
        "all_tasks_complete": len(records) == expected_tasks,
        "exact_logical_request_count": int(usage["physical_requests"])
        == expected_requests,
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "zero_forced_exits": int(generator.get("forced_exits", 0)) == 0,
        "all_changed_tasks_have_balanced_duels": (
            len(parsed_duels) == changed_count
        ),
    }
    myopic_total = sum(row["myopic_oracle_tail"] for row in rows)
    selected_total = sum(row["selected_oracle_tail"] for row in rows)
    advantages = [row["advantage_over_myopic"] for row in rows]
    unanimous_count = sum(
        row["decision"] in {"unanimous_nonmyopic", "unanimous_myopic"}
        for row in rows
    )
    fallback_count = sum(
        row["decision"] == "fallback_myopic" for row in rows
    )
    myopic_top_accuracy = (
        myopic_top_points / myopic_top_comparable
        if myopic_top_comparable
        else 0.0
    )
    selected_top_accuracy = (
        selected_top_points / selected_top_comparable
        if selected_top_comparable
        else 0.0
    )
    summary = {
        "num_tasks": len(records),
        "changed_finalist_count": changed_count,
        "unanimous_duel_count": unanimous_count,
        "fallback_count": fallback_count,
        "myopic_selected_oracle_tail_total": myopic_total,
        "duel_selected_oracle_tail_total": selected_total,
        "duel_advantage_over_myopic": selected_total - myopic_total,
        "duel_vs_myopic_wins": sum(value > 0 for value in advantages),
        "duel_vs_myopic_ties": sum(value == 0 for value in advantages),
        "duel_vs_myopic_losses": sum(value < 0 for value in advantages),
        "myopic_selected_vs_all_accuracy": myopic_top_accuracy,
        "duel_selected_vs_all_accuracy": selected_top_accuracy,
        "selected_vs_all_accuracy_gain": (
            selected_top_accuracy - myopic_top_accuracy
        ),
        "task_diagnostics": rows,
    }
    if stage == "development":
        gates.update(
            {
                "unanimous_duels_at_least_8": unanimous_count >= 8,
                "fallbacks_at_most_5": fallback_count <= 5,
                "oracle_tail_gain_over_myopic_at_least_4": (
                    selected_total - myopic_total >= 4
                ),
                "wins_exceed_losses_by_at_least_3": (
                    sum(value > 0 for value in advantages)
                    - sum(value < 0 for value in advantages)
                    >= 3
                ),
                "losses_at_most_1": sum(value < 0 for value in advantages) <= 1,
                "selected_vs_all_accuracy_at_least_0_70": (
                    selected_top_accuracy >= 0.70
                ),
                "selected_vs_all_accuracy_gain_at_least_0_08": (
                    selected_top_accuracy - myopic_top_accuracy >= 0.08
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
    rank_artifact: str | Path,
    raw_checkpoint_path: Path | None,
) -> dict[str, Any]:
    records, rank_payload = load_finalist_records(
        primary_artifact,
        secondary_artifact,
        rank_artifact,
    )
    if stage == "serving_smoke":
        records = records[:SERVING_TASK_COUNT]
    elif stage != "development":
        raise ValueError("stage must be serving_smoke or development")
    changed = [
        record
        for record in records
        if record["myopic_finalist_index"]
        != record["nonmyopic_finalist_index"]
    ]
    model = _build_model(config)
    raw: dict[str, Any] = {}
    try:
        messages = []
        keys = []
        for record in changed:
            myopic_root = int(record["myopic_finalist_index"])
            nonmyopic_root = int(record["nonmyopic_finalist_index"])
            messages.append(
                duel_messages(
                    record,
                    candidate_a_root=myopic_root,
                    candidate_b_root=nonmyopic_root,
                )
            )
            keys.append((str(record["task_id"]), "forward"))
            messages.append(
                duel_messages(
                    record,
                    candidate_a_root=nonmyopic_root,
                    candidate_b_root=myopic_root,
                )
            )
            keys.append((str(record["task_id"]), "reversed"))
        responses = model.chat_complete_messages_batched(
            messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["duels"] = [
            {"task_id": task_id, "order": order, "response": response}
            for (task_id, order), response in zip(keys, responses, strict=True)
        ]
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        parsed = [parse_duel(text) for text in responses]
        parsed_by_task: dict[
            str, dict[str, dict[str, str]]
        ] = {}
        for (task_id, order), response in zip(keys, parsed, strict=True):
            parsed_by_task.setdefault(task_id, {})[order] = response
        parsed_duels = {
            task_id: (values["forward"], values["reversed"])
            for task_id, values in parsed_by_task.items()
        }
        usage = _usage_snapshot(model)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    summary = summarize(
        records,
        parsed_duels,
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
            "rank_artifact_sha256": hashlib.sha256(
                Path(rank_artifact).read_bytes()
            ).hexdigest(),
            "rank_artifact_status": rank_payload["status"],
            "task_ids": [record["task_id"] for record in records],
            "balanced_ab_orders": True,
            "unanimity_required_for_override": True,
            "disagreement_or_tie_fallback": "myopic finalist",
            "complete_future_trees_visible": True,
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
    parser.add_argument("--rank-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_max_output_tokens = min(
        config.openrouter_max_output_tokens, 512
    )
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.10
        config.openrouter_run_budget_usd = 0.40
    else:
        config.openrouter_projected_cost_usd = 0.75
        config.openrouter_run_budget_usd = 1.50
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
            rank_artifact=args.rank_artifact,
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
