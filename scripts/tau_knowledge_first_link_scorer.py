#!/usr/bin/env python3
"""Evaluate a target-blind first-link scorer on tau-Knowledge retrieval trees."""

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
from scripts.tau_knowledge_retrieval_opportunity import (
    BM25Corpus,
    FIRST_QUERY_COUNT,
    FOLLOWUP_QUERY_COUNT,
    FORMAL_IDS,
    GateExecutionError,
    HYPOTHESIS_COUNT,
    SCHEMA_VERSION,
    SEARCH_TOP_K,
    TAU_COMMIT,
    _build_model,
    _checkpoint,
    _git_head,
    _serialize_results,
    _usage_snapshot,
    analyze_record,
    extract_opening,
)
from scripts.tau_knowledge_retrieval_opportunity_v2 import (
    followup_messages,
    initial_messages,
    parse_followup,
    parse_initial,
)


SMOKE_IDS = ("task_002", "task_024")
EXPECTED_REQUESTS = {
    "serving_smoke": len(SMOKE_IDS) * 2,
    "confirmation": len(FORMAL_IDS)
    * (1 + FIRST_QUERY_COUNT + 2),
}
DOCUMENT_EXCERPT_CHARS = 700
MAX_SCORER_INPUT_CHARS = 100_000


def load_confirmation_corpus_and_tasks(
    tau_root: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    root = Path(tau_root)
    if _git_head(root) != TAU_COMMIT:
        raise ValueError("tau-Knowledge checkout commit does not match")
    domain = root / "data" / "tau2" / "domains" / "banking_knowledge"
    document_paths = sorted((domain / "documents").glob("*.json"))
    task_paths = sorted((domain / "tasks").glob("task_*.json"))
    if len(document_paths) != 698 or len(task_paths) != 97:
        raise ValueError("tau-Knowledge corpus shape does not match")
    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in document_paths
    ]
    tasks = [
        json.loads(path.read_text(encoding="utf-8")) for path in task_paths
    ]
    if any(
        set(document) != {"id", "title", "content"} for document in documents
    ):
        raise ValueError("tau-Knowledge document schema does not match")
    by_id = {task["id"]: task for task in tasks}
    selected = [by_id[task_id] for task_id in FORMAL_IDS]
    if [task["id"] for task in selected] != list(FORMAL_IDS):
        raise ValueError("sealed confirmation order does not match")
    if any(not task.get("required_documents") for task in selected):
        raise ValueError("sealed confirmation task lacks endpoint")
    for task in selected:
        extract_opening(task)
    return documents, selected


def load_smoke_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if payload.get("status") != "passed" or not isinstance(records, list):
        raise ValueError("scorer smoke requires a passed opportunity artifact")
    if tuple(record.get("task_id") for record in records) != SMOKE_IDS:
        raise ValueError("scorer smoke artifact task IDs do not match")
    return records


def _clean_excerpt(value: str) -> str:
    return " ".join(value.split())[:DOCUMENT_EXCERPT_CHARS]


def compact_scorer_input(
    record: dict[str, Any],
    *,
    include_followups: bool,
) -> dict[str, Any]:
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

    roots = []
    for root_index, branch in enumerate(record["first_branches"], start=1):
        compact: dict[str, Any] = {
            "root": root_index,
            "query": branch["query"],
            "first_result_refs": add_results(branch["first_results"]),
        }
        if include_followups:
            compact["refreshed_information_needs"] = branch[
                "refreshed_information_need_hypotheses"
            ]
            compact["followups"] = [
                {
                    "followup": followup_index,
                    "query": followup["query"],
                    "result_refs": add_results(followup["results"]),
                }
                for followup_index, followup in enumerate(
                    branch["followups"], start=1
                )
            ]
        roots.append(compact)

    payload = {
        "customer_opening": record["opening"],
        "initial_information_needs": record[
            "initial_information_need_hypotheses"
        ],
        "roots": roots,
        "document_catalog": catalog,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > MAX_SCORER_INPUT_CHARS:
        raise ValueError("scorer input exceeds frozen character cap")
    return payload


def _score_schema(*, include_followups: bool) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for index in range(1, FIRST_QUERY_COUNT + 1):
        schema[f"root_{index}_score"] = "integer 0 through 100"
        if include_followups:
            schema[f"root_{index}_best_followup"] = (
                f"integer 1 through {FOLLOWUP_QUERY_COUNT}"
            )
        schema[f"root_{index}_rationale"] = "one short sentence"
    return schema


def scorer_messages(
    record: dict[str, Any],
    *,
    include_followups: bool,
) -> list[dict[str, str]]:
    tree = compact_scorer_input(
        record,
        include_followups=include_followups,
    )
    if include_followups:
        instruction = (
            "Score each first-search root by the best total coverage achievable "
            "after that root and exactly one of its shown followups. Reward distinct "
            "policy prerequisites, exceptions, eligibility rules, and procedures "
            "that resolve the customer's inferred information needs. Do not reward "
            "document count, verbosity, or duplicate coverage. For each root also "
            "choose its best shown followup."
        )
    else:
        instruction = (
            "Score each first-search root using only its shown first results. Reward "
            "distinct policy prerequisites, exceptions, eligibility rules, and "
            "procedures that resolve the customer's inferred information needs. Do "
            "not imagine a followup, reward document count, verbosity, or duplicate "
            "coverage."
        )
    return [
        {
            "role": "system",
            "content": (
                "Evaluate retrieval coverage for an internal banking support task. "
                "Ground every score in the customer opening and supplied document "
                "text. Required-document labels and evaluation answers are hidden. "
                "Return one flat strict JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                instruction
                + " Use the full 0-100 range when warranted and distinguish roots "
                "whenever their useful coverage differs. Return exactly these keys: "
                + json.dumps(
                    _score_schema(include_followups=include_followups),
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + ". Retrieval data: "
                + json.dumps(tree, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_scores(
    text: str,
    *,
    include_followups: bool,
) -> dict[str, Any]:
    payload = _parse_json_object(text)
    expected = _score_schema(include_followups=include_followups)
    if set(payload) != set(expected):
        raise ValueError("scorer response has unexpected keys")
    scores = []
    best_followups = []
    rationales = []
    for index in range(1, FIRST_QUERY_COUNT + 1):
        score = payload[f"root_{index}_score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not float(score).is_integer()
            or not 0 <= int(score) <= 100
        ):
            raise ValueError("root score must be an integer from 0 through 100")
        scores.append(int(score))
        if include_followups:
            followup = payload[f"root_{index}_best_followup"]
            if (
                isinstance(followup, bool)
                or not isinstance(followup, (int, float))
                or not float(followup).is_integer()
                or not 1 <= int(followup) <= FOLLOWUP_QUERY_COUNT
            ):
                raise ValueError("best followup index is invalid")
            best_followups.append(int(followup) - 1)
        rationale = payload[f"root_{index}_rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("root rationale must be a nonempty string")
        rationales.append(" ".join(rationale.split()))
    return {
        "scores": scores,
        "best_followup_indices": best_followups,
        "rationales": rationales,
    }


def _selected_index(scores: Sequence[int]) -> int:
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def pairwise_ranking_points(
    scores: Sequence[int],
    values: Sequence[int],
) -> tuple[float, int]:
    points = 0.0
    comparable = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] == values[right]:
                continue
            comparable += 1
            score_delta = scores[left] - scores[right]
            value_delta = values[left] - values[right]
            if score_delta == 0:
                points += 0.5
            elif (score_delta > 0) == (value_delta > 0):
                points += 1.0
    return points, comparable


def analyze_scored_record(
    record: dict[str, Any],
    myopic: dict[str, Any],
    nonmyopic: dict[str, Any],
) -> dict[str, Any]:
    endpoint = analyze_record(record)
    root_values = [max(row) for row in endpoint["pair_counts"]]
    myopic_root = _selected_index(myopic["scores"])
    nonmyopic_root = _selected_index(nonmyopic["scores"])
    oracle_value = max(root_values)
    oracle_roots = [
        index for index, value in enumerate(root_values) if value == oracle_value
    ]
    chosen_followup = nonmyopic["best_followup_indices"][nonmyopic_root]
    myopic_points, comparable = pairwise_ranking_points(
        myopic["scores"], root_values
    )
    nonmyopic_points, nonmyopic_comparable = pairwise_ranking_points(
        nonmyopic["scores"], root_values
    )
    if comparable != nonmyopic_comparable:
        raise ValueError("myopic and non-myopic comparisons do not align")
    return {
        "task_id": record["task_id"],
        "root_oracle_tail_values": root_values,
        "oracle_root_indices": oracle_roots,
        "oracle_root_value": oracle_value,
        "oracle_strength_greedy_root_index": endpoint["greedy_first_index"],
        "oracle_strength_greedy_root_value": endpoint[
            "greedy_continuation_required_document_count"
        ],
        "structural_nonmyopic_gap": endpoint[
            "nonmyopic_required_document_gap"
        ],
        "myopic_scores": myopic["scores"],
        "myopic_selected_root_index": myopic_root,
        "myopic_selected_root_value": root_values[myopic_root],
        "nonmyopic_scores": nonmyopic["scores"],
        "nonmyopic_selected_root_index": nonmyopic_root,
        "nonmyopic_selected_root_value": root_values[nonmyopic_root],
        "nonmyopic_selected_followup_index": chosen_followup,
        "nonmyopic_selected_pair_value": endpoint["pair_counts"][
            nonmyopic_root
        ][chosen_followup],
        "nonmyopic_root_is_oracle_optimal": nonmyopic_root in oracle_roots,
        "myopic_root_is_oracle_optimal": myopic_root in oracle_roots,
        "root_policy_advantage": root_values[nonmyopic_root]
        - root_values[myopic_root],
        "root_policy_advantage_vs_oracle_strength_greedy": root_values[
            nonmyopic_root
        ]
        - endpoint["greedy_continuation_required_document_count"],
        "pairwise_comparable_count": comparable,
        "myopic_pairwise_points": myopic_points,
        "nonmyopic_pairwise_points": nonmyopic_points,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    myopic_scores: Sequence[dict[str, Any]],
    nonmyopic_scores: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    diagnostics = [
        analyze_scored_record(record, myopic, nonmyopic)
        for record, myopic, nonmyopic in zip(
            records, myopic_scores, nonmyopic_scores, strict=True
        )
    ]
    myopic_points = sum(row["myopic_pairwise_points"] for row in diagnostics)
    nonmyopic_points = sum(
        row["nonmyopic_pairwise_points"] for row in diagnostics
    )
    comparable = sum(
        row["pairwise_comparable_count"] for row in diagnostics
    )
    myopic_accuracy = myopic_points / comparable if comparable else 0.0
    nonmyopic_accuracy = nonmyopic_points / comparable if comparable else 0.0
    advantages = [row["root_policy_advantage"] for row in diagnostics]
    structural_gaps = [
        row["structural_nonmyopic_gap"] for row in diagnostics
    ]
    opportunity_rows = [
        row for row in diagnostics if row["structural_nonmyopic_gap"] >= 1
    ]
    opportunity_hits = sum(
        row["nonmyopic_root_is_oracle_optimal"] for row in opportunity_rows
    )
    base_gates = {
        "all_cases_complete": len(records)
        == (len(SMOKE_IDS) if stage == "serving_smoke" else len(FORMAL_IDS)),
        "exact_physical_request_count": int(usage["physical_requests"])
        == EXPECTED_REQUESTS[stage],
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_scores_complete": len(myopic_scores)
        == len(nonmyopic_scores)
        == len(records),
    }
    if stage == "serving_smoke":
        gates = {
            **base_gates,
            "myopic_scores_vary_each_case": all(
                len(set(score["scores"])) >= 2 for score in myopic_scores
            ),
            "nonmyopic_scores_vary_each_case": all(
                len(set(score["scores"])) >= 2 for score in nonmyopic_scores
            ),
        }
    else:
        gates = {
            **base_gates,
            "structural_gap_count_at_least_4": len(opportunity_rows) >= 4,
            "mean_structural_gap_at_least_0_20": (
                sum(structural_gaps) / len(structural_gaps) >= 0.20
            ),
            "comparable_root_pairs_at_least_50": comparable >= 50,
            "nonmyopic_pairwise_accuracy_at_least_0_60": (
                nonmyopic_accuracy >= 0.60
            ),
            "pairwise_accuracy_gain_at_least_0_05": (
                nonmyopic_accuracy - myopic_accuracy >= 0.05
            ),
            "selected_roots_differ_at_least_4": sum(
                row["myopic_selected_root_index"]
                != row["nonmyopic_selected_root_index"]
                for row in diagnostics
            )
            >= 4,
            "root_policy_wins_at_least_3": sum(value > 0 for value in advantages)
            >= 3,
            "root_policy_losses_at_most_2": sum(value < 0 for value in advantages)
            <= 2,
            "root_policy_total_advantage_at_least_2": sum(advantages) >= 2,
            "opportunity_oracle_root_hit_rate_at_least_0_50": (
                bool(opportunity_rows)
                and opportunity_hits / len(opportunity_rows) >= 0.50
            ),
        }
    gates["all_pass"] = all(gates.values())
    return {
        "num_cases": len(records),
        "case_diagnostics": diagnostics,
        "myopic_pairwise_accuracy": myopic_accuracy,
        "nonmyopic_pairwise_accuracy": nonmyopic_accuracy,
        "pairwise_accuracy_gain": nonmyopic_accuracy - myopic_accuracy,
        "comparable_root_pairs": comparable,
        "structural_gap_count": len(opportunity_rows),
        "mean_structural_gap": (
            sum(structural_gaps) / len(structural_gaps)
            if structural_gaps
            else 0.0
        ),
        "selected_root_difference_count": sum(
            row["myopic_selected_root_index"]
            != row["nonmyopic_selected_root_index"]
            for row in diagnostics
        ),
        "root_policy_win_count": sum(value > 0 for value in advantages),
        "root_policy_loss_count": sum(value < 0 for value in advantages),
        "root_policy_tie_count": sum(value == 0 for value in advantages),
        "root_policy_total_advantage": sum(advantages),
        "root_policy_mean_advantage": (
            sum(advantages) / len(advantages) if advantages else 0.0
        ),
        "opportunity_oracle_root_hit_count": opportunity_hits,
        "opportunity_oracle_root_hit_rate": (
            opportunity_hits / len(opportunity_rows)
            if opportunity_rows
            else 0.0
        ),
        "gates": gates,
    }


def _generate_confirmation_trees(
    model: Any,
    corpus: BM25Corpus,
    tasks: Sequence[dict[str, Any]],
    config: Config,
    raw: dict[str, Any],
    raw_checkpoint_path: Path | None,
) -> list[dict[str, Any]]:
    openings = [extract_opening(task) for task in tasks]
    initial_raw = model.chat_complete_messages_batched(
        [initial_messages(opening) for opening in openings],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["initial"] = initial_raw
    _checkpoint(raw_checkpoint_path, stage="confirmation", raw=raw)
    initial = [parse_initial(text) for text in initial_raw]
    first_results = [
        [corpus.search(query) for query in queries]
        for _hypotheses, queries in initial
    ]
    followup_keys = [
        (case_index, first_index)
        for case_index in range(len(tasks))
        for first_index in range(FIRST_QUERY_COUNT)
    ]
    followup_raw = model.chat_complete_messages_batched(
        [
            followup_messages(
                openings[case_index],
                initial[case_index][1][first_index],
                first_results[case_index][first_index],
            )
            for case_index, first_index in followup_keys
        ],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["followups"] = followup_raw
    _checkpoint(raw_checkpoint_path, stage="confirmation", raw=raw)
    parsed_followups = [
        parse_followup(
            text,
            first_query=initial[case_index][1][first_index],
        )
        for (case_index, first_index), text in zip(
            followup_keys, followup_raw, strict=True
        )
    ]
    lookup = {
        key: value
        for key, value in zip(
            followup_keys, parsed_followups, strict=True
        )
    }
    records = []
    for case_index, task in enumerate(tasks):
        initial_hypotheses, first_queries = initial[case_index]
        branches = []
        for first_index, first_query in enumerate(first_queries):
            hypotheses, queries = lookup[case_index, first_index]
            branches.append(
                {
                    "query": first_query,
                    "first_results": _serialize_results(
                        first_results[case_index][first_index]
                    ),
                    "refreshed_information_need_hypotheses": hypotheses,
                    "followups": [
                        {
                            "query": query,
                            "results": _serialize_results(
                                corpus.search(query)
                            ),
                        }
                        for query in queries
                    ],
                }
            )
        records.append(
            {
                "task_id": task["id"],
                "opening": openings[case_index],
                "required_documents": list(task["required_documents"]),
                "initial_information_need_hypotheses": initial_hypotheses,
                "first_branches": branches,
            }
        )
    return records


def run_gate(
    config: Config,
    *,
    stage: str,
    tau_root: str | Path,
    smoke_artifact: str | Path | None = None,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    model = _build_model(config)
    raw: dict[str, Any] = {}
    try:
        if stage == "serving_smoke":
            if smoke_artifact is None:
                raise ValueError("serving smoke requires --smoke-artifact")
            records = load_smoke_records(smoke_artifact)
        elif stage == "confirmation":
            documents, tasks = load_confirmation_corpus_and_tasks(tau_root)
            records = _generate_confirmation_trees(
                model,
                BM25Corpus(documents),
                tasks,
                config,
                raw,
                raw_checkpoint_path,
            )
        else:
            raise ValueError("stage must be serving_smoke or confirmation")

        myopic_raw = model.chat_complete_messages_batched(
            [
                scorer_messages(record, include_followups=False)
                for record in records
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["myopic_scores"] = myopic_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        myopic_scores = [
            parse_scores(text, include_followups=False) for text in myopic_raw
        ]

        nonmyopic_raw = model.chat_complete_messages_batched(
            [
                scorer_messages(record, include_followups=True)
                for record in records
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["nonmyopic_scores"] = nonmyopic_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        nonmyopic_scores = [
            parse_scores(text, include_followups=True)
            for text in nonmyopic_raw
        ]
        usage = _usage_snapshot(model)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    summary = summarize(
        records,
        myopic_scores,
        nonmyopic_scores,
        usage,
        stage=stage,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "source_repository": "https://github.com/sierra-research/tau2-bench",
            "source_commit": TAU_COMMIT,
            "task_ids": (
                list(SMOKE_IDS) if stage == "serving_smoke" else list(FORMAL_IDS)
            ),
            "first_query_count": FIRST_QUERY_COUNT,
            "followup_query_count": FOLLOWUP_QUERY_COUNT,
            "search_top_k": SEARCH_TOP_K,
            "hypothesis_count": HYPOTHESIS_COUNT,
            "document_excerpt_chars": DOCUMENT_EXCERPT_CHARS,
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "myopic_scorer_sees_followups": False,
            "nonmyopic_scorer_sees_full_tree": True,
            "primary_endpoint_uses_oracle_continuation_under_selected_root": True,
            "required_documents_hidden_from_model": True,
            "full_user_scripts_hidden_from_model": True,
            "reasoning_disabled": True,
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "records": records,
        "myopic_scores": myopic_scores,
        "nonmyopic_scores": nonmyopic_scores,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "confirmation"),
        required=True,
    )
    parser.add_argument("--smoke-artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.20
        config.openrouter_run_budget_usd = 1.00
    else:
        config.openrouter_projected_cost_usd = 3.00
        config.openrouter_run_budget_usd = 8.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "CONFIRMATION.json"
    )
    try:
        payload = run_gate(
            config,
            stage=args.stage,
            tau_root=args.tau_root,
            smoke_artifact=args.smoke_artifact,
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
