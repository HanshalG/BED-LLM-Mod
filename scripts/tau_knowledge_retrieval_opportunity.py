#!/usr/bin/env python3
"""Gate non-myopic retrieval opportunity on official tau-Knowledge tasks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.movielens_profile_dynamics_gate import _parse_json_object


SCHEMA_VERSION = 1
TAU_COMMIT = "1d244f5dca42944b67a379b44bfeb9f5748f189d"
SELECTION_SEED = 24334
FIRST_QUERY_COUNT = 5
FOLLOWUP_QUERY_COUNT = 4
SEARCH_TOP_K = 3
HYPOTHESIS_COUNT = 8
SMOKE_IDS = ("task_055", "task_056")
OPPORTUNITY_IDS = (
    "task_064",
    "task_073",
    "task_078",
    "task_070",
    "task_075",
    "task_065",
)
FORMAL_IDS = (
    "task_039",
    "task_054",
    "task_074",
    "task_080",
    "task_058",
    "task_076",
    "task_038",
    "task_063",
    "task_035",
    "task_069",
    "task_061",
    "task_040",
    "task_066",
    "task_057",
    "task_077",
    "task_053",
    "task_041",
    "task_059",
    "task_072",
    "task_079",
)
PREVIOUSLY_AUDITED_IDS = (
    "task_001",
    "task_026",
    "task_050",
    "task_060",
    "task_067",
    "task_068",
    "task_071",
    "task_088",
    "task_092",
)
OPENING_PATTERN = re.compile(
    r'^\s*1\.\s*\*\*Opening[^:]*:\*\*\s*["“](.+?)["”]\s*$',
    flags=re.IGNORECASE | re.MULTILINE,
)
OLD_OPENING_PATTERNS = (
    re.compile(
        r'^\s*1\.\s*\*\*Opening:\*\*\s*["“](.+?)["”]\s*$',
        flags=re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r'^\s*\*\*Opening[^:]*:\*\*\s*["“](.+?)["”]\s*$',
        flags=re.IGNORECASE | re.MULTILINE,
    ),
)
EXPECTED_REQUESTS = {
    "serving_smoke": len(SMOKE_IDS) * (1 + FIRST_QUERY_COUNT),
    "opportunity": len(OPPORTUNITY_IDS) * (1 + FIRST_QUERY_COUNT),
}


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def selected_ids(stage: str) -> tuple[str, ...]:
    if stage == "serving_smoke":
        return SMOKE_IDS
    if stage == "opportunity":
        return OPPORTUNITY_IDS
    raise ValueError("stage must be serving_smoke or opportunity")


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def extract_opening(task: dict[str, Any]) -> str:
    instructions = task["user_scenario"]["instructions"]
    match = OPENING_PATTERN.search(instructions)
    if match is None:
        raise ValueError(f"task {task.get('id')} lacks a scripted opening")
    return " ".join(match.group(1).split())


def load_corpus_and_tasks(
    tau_root: str | Path,
    stage: str,
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

    old_audited = {
        task["id"]
        for task in tasks
        if any(
            pattern.search(task["user_scenario"]["instructions"])
            for pattern in OLD_OPENING_PATTERNS
        )
    }
    eligible = sorted(
        task["id"]
        for task in tasks
        if OPENING_PATTERN.search(task["user_scenario"]["instructions"])
        and task["id"] not in old_audited
        and task["id"] not in PREVIOUSLY_AUDITED_IDS
    )
    random.Random(SELECTION_SEED).shuffle(eligible)
    frozen = list(SMOKE_IDS + OPPORTUNITY_IDS + FORMAL_IDS)
    if eligible != frozen:
        raise ValueError("frozen fresh tau-Knowledge split does not reproduce")
    wanted = selected_ids(stage)
    selected = [by_id[task_id] for task_id in wanted]
    if any(not task.get("required_documents") for task in selected):
        raise ValueError("selected task lacks required-document endpoints")
    return documents, selected


class BM25Corpus:
    def __init__(self, documents: Sequence[dict[str, str]]) -> None:
        self.documents = list(documents)
        self.by_id = {document["id"]: document for document in documents}
        tokenized = [
            (document["title"] + " " + document["content"]).lower().split()
            for document in documents
        ]
        self.index = BM25Okapi(tokenized)

    def search(self, query: str) -> list[dict[str, Any]]:
        scores = self.index.get_scores(query.lower().split())
        indices = sorted(
            range(len(scores)),
            key=lambda index: (float(scores[index]), -index),
            reverse=True,
        )[:SEARCH_TOP_K]
        return [
            {
                **self.documents[index],
                "bm25_score": float(scores[index]),
            }
            for index in indices
        ]


def initial_messages(opening: str) -> list[dict[str, str]]:
    schema = {
        "information_need_hypotheses": [
            f"distinct policy or factual need {index + 1}"
            for index in range(HYPOTHESIS_COUNT)
        ],
        "direct_queries": [
            f"specific direct search query {index + 1}" for index in range(2)
        ],
        "enabling_queries": [
            f"broader prerequisite-finding query {index + 1}"
            for index in range(3)
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Plan searches over a large internal banking policy knowledge base. "
                "The relevant documents are hidden. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"From only the customer's first utterance, list exactly "
                f"{HYPOTHESIS_COUNT} distinct concrete policy or factual information "
                "needs that might be required. Then propose exactly two diverse "
                "specific direct searches and three diverse enabling searches aimed "
                "at finding prerequisites, procedures, exceptions, or terminology "
                "that would make a later search more precise. Queries must be short "
                "plain search strings, not questions to the customer. Do not assume "
                "you know the hidden required documents. Return exactly this schema: "
                + json.dumps(schema, ensure_ascii=True, separators=(",", ":"))
                + ". Customer opening: "
                + json.dumps(opening, ensure_ascii=True)
            ),
        },
    ]


def followup_messages(
    opening: str,
    first_query: str,
    first_results: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    schema = {
        "information_need_hypotheses": [
            f"updated policy or factual need {index + 1}"
            for index in range(HYPOTHESIS_COUNT)
        ],
        "followup_queries": [
            f"document-conditioned search query {index + 1}"
            for index in range(FOLLOWUP_QUERY_COUNT)
        ],
    }
    visible_results = [
        {
            "document_id": result["id"],
            "title": result["title"],
            "content": result["content"],
        }
        for result in first_results
    ]
    return [
        {
            "role": "system",
            "content": (
                "Continue a search over an internal banking policy knowledge base. "
                "Use retrieved text to identify missing prerequisites and linked "
                "policies. Relevant document IDs remain hidden. Return strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Refresh exactly {HYPOTHESIS_COUNT} distinct concrete information "
                f"needs, then propose exactly {FOLLOWUP_QUERY_COUNT} distinct short "
                "search strings that exploit terminology or dependencies in the "
                "retrieved documents to find still-missing policy. Do not repeat the "
                "first query and do not merely search a retrieved document title. "
                "Return exactly this schema: "
                + json.dumps(schema, ensure_ascii=True, separators=(",", ":"))
                + ". Data: "
                + json.dumps(
                    {
                        "customer_opening": opening,
                        "first_query": first_query,
                        "first_results": visible_results,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def _parse_string_list(
    value: Any,
    *,
    count: int,
    field: str,
) -> list[str]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{field} must contain exactly {count} strings")
    cleaned = [
        " ".join(item.split()) if isinstance(item, str) else ""
        for item in value
    ]
    normalized = [_normalized(item) for item in cleaned]
    if any(not item for item in normalized) or len(set(normalized)) != count:
        raise ValueError(f"{field} must contain distinct nonempty strings")
    return cleaned


def parse_initial(text: str) -> tuple[list[str], list[str]]:
    payload = _parse_json_object(text)
    expected = {
        "information_need_hypotheses",
        "direct_queries",
        "enabling_queries",
    }
    if set(payload) != expected:
        raise ValueError("initial response has unexpected fields")
    hypotheses = _parse_string_list(
        payload["information_need_hypotheses"],
        count=HYPOTHESIS_COUNT,
        field="information_need_hypotheses",
    )
    direct = _parse_string_list(
        payload["direct_queries"],
        count=2,
        field="direct_queries",
    )
    enabling = _parse_string_list(
        payload["enabling_queries"],
        count=3,
        field="enabling_queries",
    )
    queries = [*direct, *enabling]
    if len({_normalized(query) for query in queries}) != FIRST_QUERY_COUNT:
        raise ValueError("first queries must be globally distinct")
    return hypotheses, queries


def parse_followup(text: str, *, first_query: str) -> tuple[list[str], list[str]]:
    payload = _parse_json_object(text)
    expected = {"information_need_hypotheses", "followup_queries"}
    if set(payload) != expected:
        raise ValueError("followup response has unexpected fields")
    hypotheses = _parse_string_list(
        payload["information_need_hypotheses"],
        count=HYPOTHESIS_COUNT,
        field="information_need_hypotheses",
    )
    queries = _parse_string_list(
        payload["followup_queries"],
        count=FOLLOWUP_QUERY_COUNT,
        field="followup_queries",
    )
    if _normalized(first_query) in {_normalized(query) for query in queries}:
        raise ValueError("followup repeats its first query")
    return hypotheses, queries


def _retrieved_ids(results: Sequence[dict[str, Any]]) -> set[str]:
    return {str(result["id"]) for result in results}


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    required = set(record["required_documents"])
    one_step_counts = [
        len(required & _retrieved_ids(branch["first_results"]))
        for branch in record["first_branches"]
    ]
    pair_counts = [
        [
            len(
                required
                & (
                    _retrieved_ids(branch["first_results"])
                    | _retrieved_ids(followup["results"])
                )
            )
            for followup in branch["followups"]
        ]
        for branch in record["first_branches"]
    ]
    greedy_first = max(
        range(FIRST_QUERY_COUNT),
        key=lambda index: (one_step_counts[index], -index),
    )
    oracle_first, oracle_second = max(
        (
            (first_index, second_index)
            for first_index in range(FIRST_QUERY_COUNT)
            for second_index in range(FOLLOWUP_QUERY_COUNT)
        ),
        key=lambda pair: (
            pair_counts[pair[0]][pair[1]],
            -pair[0],
            -pair[1],
        ),
    )
    greedy_continuation = max(pair_counts[greedy_first])
    oracle_pair = pair_counts[oracle_first][oracle_second]
    distinct_top1 = len(
        {
            branch["first_results"][0]["id"]
            for branch in record["first_branches"]
        }
    )
    return {
        "required_document_count": len(required),
        "distinct_first_top1_count": distinct_top1,
        "best_one_step_required_document_count": max(one_step_counts),
        "greedy_first_index": greedy_first,
        "oracle_first_index": oracle_first,
        "oracle_second_index": oracle_second,
        "oracle_first_differs_from_greedy": oracle_first != greedy_first,
        "oracle_pair_required_document_count": oracle_pair,
        "greedy_continuation_required_document_count": greedy_continuation,
        "pair_gain_over_best_one_step": oracle_pair - max(one_step_counts),
        "nonmyopic_required_document_gap": oracle_pair - greedy_continuation,
        "one_step_counts": one_step_counts,
        "pair_counts": pair_counts,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
    expected_case_count: int | None = None,
    expected_request_count: int | None = None,
) -> dict[str, Any]:
    diagnostics = [analyze_record(record) for record in records]
    if expected_case_count is None:
        expected_case_count = len(selected_ids(stage))
    if expected_request_count is None:
        expected_request_count = EXPECTED_REQUESTS[stage]
    base_gates = {
        "all_cases_complete": len(records) == expected_case_count,
        "exact_physical_request_count": int(usage["physical_requests"])
        == expected_request_count,
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_query_trees_complete": all(
            len(record["first_branches"]) == FIRST_QUERY_COUNT
            and all(
                len(branch["followups"]) == FOLLOWUP_QUERY_COUNT
                for branch in record["first_branches"]
            )
            for record in records
        ),
    }
    summary: dict[str, Any] = {
        "num_cases": len(records),
        "case_diagnostics": [
            {"task_id": record["task_id"], **diagnostic}
            for record, diagnostic in zip(records, diagnostics, strict=True)
        ],
    }
    if stage == "serving_smoke":
        gates = {
            **base_gates,
            "at_least_3_distinct_first_top1_each": all(
                row["distinct_first_top1_count"] >= 3
                for row in diagnostics
            ),
        }
    else:
        mean_distinct = float(
            np.mean([row["distinct_first_top1_count"] for row in diagnostics])
        )
        pair_gain_count = sum(
            row["pair_gain_over_best_one_step"] >= 1 for row in diagnostics
        )
        first_change_count = sum(
            row["oracle_first_differs_from_greedy"] for row in diagnostics
        )
        gap_count = sum(
            row["nonmyopic_required_document_gap"] >= 1
            for row in diagnostics
        )
        pair_any_count = sum(
            row["oracle_pair_required_document_count"] >= 1
            for row in diagnostics
        )
        mean_pair_gain = float(
            np.mean([row["pair_gain_over_best_one_step"] for row in diagnostics])
        )
        mean_gap = float(
            np.mean(
                [
                    row["nonmyopic_required_document_gap"]
                    for row in diagnostics
                ]
            )
        )
        summary.update(
            {
                "mean_distinct_first_top1_count": mean_distinct,
                "oracle_pair_any_required_document_count": pair_any_count,
                "pair_gain_at_least_1_count": pair_gain_count,
                "oracle_first_differs_from_greedy_count": first_change_count,
                "nonmyopic_gap_at_least_1_count": gap_count,
                "mean_pair_gain_over_best_one_step": mean_pair_gain,
                "mean_nonmyopic_required_document_gap": mean_gap,
            }
        )
        gates = {
            **base_gates,
            "mean_distinct_first_top1_at_least_3": mean_distinct >= 3.0,
            "oracle_pair_any_required_document_count_at_least_4": (
                pair_any_count >= 4
            ),
            "pair_gain_count_at_least_3": pair_gain_count >= 3,
            "oracle_first_differs_count_at_least_2": first_change_count >= 2,
            "nonmyopic_gap_count_at_least_2": gap_count >= 2,
            "mean_pair_gain_at_least_0_50": mean_pair_gain >= 0.50,
            "mean_nonmyopic_gap_at_least_0_33": mean_gap >= 0.33,
        }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def _build_model(config: Config) -> Any:
    if len(config.model_pairs) != 1:
        raise ValueError("tau-Knowledge gate requires one model pair")
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    return build_model_adapter(spec, config)


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "generator": snapshot,
    }


def _checkpoint(path: Path | None, *, stage: str, raw: dict[str, Any]) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "stage": stage, **raw},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _serialize_results(
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": result["id"],
            "title": result["title"],
            "content": result["content"],
            "bm25_score": result["bm25_score"],
        }
        for result in results
    ]


def run_gate(
    config: Config,
    *,
    tau_root: str | Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    documents, tasks = load_corpus_and_tasks(tau_root, stage)
    corpus = BM25Corpus(documents)
    model = _build_model(config)
    raw: dict[str, Any] = {}
    try:
        openings = [extract_opening(task) for task in tasks]
        initial_raw = model.chat_complete_messages_batched(
            [initial_messages(opening) for opening in openings],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial"] = initial_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
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
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        parsed_followups = [
            parse_followup(
                text,
                first_query=initial[case_index][1][first_index],
            )
            for (case_index, first_index), text in zip(
                followup_keys, followup_raw, strict=True
            )
        ]
        followup_lookup = {
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
                hypotheses, queries = followup_lookup[
                    case_index, first_index
                ]
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
        usage = _usage_snapshot(model)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    summary = summarize(records, usage, stage=stage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "source_repository": "https://github.com/sierra-research/tau2-bench",
            "source_commit": TAU_COMMIT,
            "selection_seed": SELECTION_SEED,
            "task_ids": list(selected_ids(stage)),
            "sealed_confirmation_ids": list(FORMAL_IDS),
            "first_query_count": FIRST_QUERY_COUNT,
            "followup_query_count": FOLLOWUP_QUERY_COUNT,
            "search_top_k": SEARCH_TOP_K,
            "hypothesis_count": HYPOTHESIS_COUNT,
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "official_bm25_implementation": "rank-bm25 0.2.2 BM25Okapi",
            "required_documents_hidden_from_model": True,
            "reasoning_disabled": True,
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "opportunity"),
        required=True,
    )
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
        config.openrouter_projected_cost_usd = 0.60
        config.openrouter_run_budget_usd = 2.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "OPPORTUNITY.json"
    )
    try:
        payload = run_gate(
            config,
            tau_root=args.tau_root,
            stage=args.stage,
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
