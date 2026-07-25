#!/usr/bin/env python3
"""Run flat-schema tau-Knowledge retrieval opportunity on fresh tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

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
    OPENING_PATTERN,
    SCHEMA_VERSION,
    SEARCH_TOP_K,
    TAU_COMMIT,
    _build_model,
    _checkpoint,
    _git_head,
    _normalized,
    _serialize_results,
    _usage_snapshot,
    summarize,
)


SELECTION_SEED = 24335
SMOKE_IDS = ("task_002", "task_024")
OPPORTUNITY_IDS = (
    "task_018",
    "task_008",
    "task_014",
    "task_010",
    "task_006",
    "task_003",
)
PREVIOUSLY_INSPECTED_IDS = ("task_001", "task_026")
EXPECTED_REQUESTS = {
    "serving_smoke": len(SMOKE_IDS) * (1 + 1 + FIRST_QUERY_COUNT),
    "opportunity": len(OPPORTUNITY_IDS) * (1 + 1 + FIRST_QUERY_COUNT),
}


def selected_ids(stage: str) -> tuple[str, ...]:
    if stage == "serving_smoke":
        return SMOKE_IDS
    if stage == "opportunity":
        return OPPORTUNITY_IDS
    raise ValueError("stage must be serving_smoke or opportunity")


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
    by_id = {task["id"]: task for task in tasks}
    eligible = sorted(
        task["id"]
        for task in tasks
        if not OPENING_PATTERN.search(task["user_scenario"]["instructions"])
        and task["id"] not in PREVIOUSLY_INSPECTED_IDS
    )
    random.Random(SELECTION_SEED).shuffle(eligible)
    if eligible[: len(SMOKE_IDS + OPPORTUNITY_IDS)] != list(
        SMOKE_IDS + OPPORTUNITY_IDS
    ):
        raise ValueError("frozen V2 tau-Knowledge split does not reproduce")
    selected = [by_id[task_id] for task_id in selected_ids(stage)]
    if any(not task.get("required_documents") for task in selected):
        raise ValueError("selected V2 task lacks required-document endpoints")
    return documents, selected


def opening_messages(instructions: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Act only as a customer-simulator preprocessor. Convert a private "
                "customer script into the single first utterance the customer would "
                "naturally say before the agent asks anything. Return strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write one concise opening utterance. Include the immediate request "
                "and only details the script says or implies should be volunteered "
                "initially. Do not reveal verification data, later conversation "
                "steps, hidden tool actions, policy solutions, or evaluation criteria. "
                'Return exactly {"opening":"..."}. Private script: '
                + json.dumps(instructions, ensure_ascii=True)
            ),
        },
    ]


def parse_opening(text: str) -> str:
    payload = _parse_json_object(text)
    if set(payload) != {"opening"} or not isinstance(
        payload["opening"], str
    ):
        raise ValueError("opening response must contain only one string")
    opening = " ".join(payload["opening"].split())
    if not opening or len(opening) > 1200:
        raise ValueError("generated opening has invalid length")
    return opening


def _flat_schema(
    *,
    query_prefixes: tuple[tuple[str, int], ...],
) -> dict[str, str]:
    schema = {
        f"need_{index + 1}": f"distinct policy or factual need {index + 1}"
        for index in range(HYPOTHESIS_COUNT)
    }
    for prefix, count in query_prefixes:
        schema.update(
            {
                f"{prefix}_{index + 1}": f"short search string {index + 1}"
                for index in range(count)
            }
        )
    return schema


def initial_messages(opening: str) -> list[dict[str, str]]:
    schema = _flat_schema(
        query_prefixes=(("direct_query", 2), ("enabling_query", 3))
    )
    return [
        {
            "role": "system",
            "content": (
                "Plan searches over a large internal banking policy knowledge base. "
                "Relevant documents are hidden. Return one flat strict JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"From only the customer's first utterance, fill exactly "
                f"{HYPOTHESIS_COUNT} distinct concrete information needs, two "
                "specific direct searches, and three diverse enabling searches for "
                "prerequisites, procedures, exceptions, or terminology. Queries are "
                "short search strings, not customer questions. Every value must be a "
                "nonempty string. Return exactly these flat keys and no arrays: "
                + json.dumps(schema, ensure_ascii=True, separators=(",", ":"))
                + ". Customer opening: "
                + json.dumps(opening, ensure_ascii=True)
            ),
        },
    ]


def followup_messages(
    opening: str,
    first_query: str,
    first_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    schema = _flat_schema(
        query_prefixes=(("followup_query", FOLLOWUP_QUERY_COUNT),)
    )
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
                "Continue an internal banking policy search. Use retrieved text to "
                "identify missing prerequisites and linked policies. Relevant "
                "document IDs are hidden. Return one flat strict JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Refresh exactly {HYPOTHESIS_COUNT} distinct information needs and "
                f"write exactly {FOLLOWUP_QUERY_COUNT} distinct short searches that "
                "exploit retrieved terminology or dependencies. Do not repeat the "
                "first query or merely search a retrieved title. Every value must be "
                "a nonempty string. Return exactly these flat keys and no arrays: "
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


def _parse_flat_values(
    text: str,
    *,
    query_prefixes: tuple[tuple[str, int], ...],
) -> tuple[list[str], list[str]]:
    payload = _parse_json_object(text)
    schema = _flat_schema(query_prefixes=query_prefixes)
    if set(payload) != set(schema):
        raise ValueError("flat query response has unexpected keys")
    hypotheses = [
        payload[f"need_{index + 1}"] for index in range(HYPOTHESIS_COUNT)
    ]
    queries = [
        payload[f"{prefix}_{index + 1}"]
        for prefix, count in query_prefixes
        for index in range(count)
    ]
    if any(not isinstance(value, str) for value in [*hypotheses, *queries]):
        raise ValueError("flat query response values must all be strings")
    hypotheses = [" ".join(value.split()) for value in hypotheses]
    queries = [" ".join(value.split()) for value in queries]
    if (
        any(not _normalized(value) for value in hypotheses)
        or len({_normalized(value) for value in hypotheses})
        != HYPOTHESIS_COUNT
        or any(not _normalized(value) for value in queries)
        or len({_normalized(value) for value in queries}) != len(queries)
    ):
        raise ValueError("flat hypotheses and queries must be distinct and nonempty")
    return hypotheses, queries


def parse_initial(text: str) -> tuple[list[str], list[str]]:
    return _parse_flat_values(
        text,
        query_prefixes=(("direct_query", 2), ("enabling_query", 3)),
    )


def parse_followup(text: str, *, first_query: str) -> tuple[list[str], list[str]]:
    hypotheses, queries = _parse_flat_values(
        text,
        query_prefixes=(("followup_query", FOLLOWUP_QUERY_COUNT),),
    )
    if _normalized(first_query) in {_normalized(query) for query in queries}:
        raise ValueError("followup repeats its first query")
    return hypotheses, queries


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
        opening_raw = model.chat_complete_messages_batched(
            [
                opening_messages(task["user_scenario"]["instructions"])
                for task in tasks
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["openings"] = opening_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        openings = [parse_opening(text) for text in opening_raw]

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
        usage = _usage_snapshot(model)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    summary = summarize(
        records,
        usage,
        stage=stage,
        expected_case_count=len(selected_ids(stage)),
        expected_request_count=EXPECTED_REQUESTS[stage],
    )
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
            "flat_numbered_schema": True,
            "opening_generated_once_from_private_user_script": True,
            "policy_model_sees_only_opening": True,
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
        config.openrouter_projected_cost_usd = 0.70
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
