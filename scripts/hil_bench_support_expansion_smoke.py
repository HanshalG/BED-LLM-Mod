#!/usr/bin/env python3
"""Test whether HiL-Bench observations expand LLM-generated blocker support."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol, Sequence

from rank_bm25 import BM25Okapi

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.hil_bench_progressive_opportunity import (
    DEVELOPMENT_IDS,
    HIL_COMMIT,
    _git_head,
    extract_problem,
)
from scripts.movielens_profile_dynamics_gate import _parse_json_object


INTERFACE_VERSION = "hil-bench-support-expansion-smoke-2"
MODEL_ID = "openai/gpt-5.4"
TASK_IDS = DEVELOPMENT_IDS[2:4]
SEED = 24_394
SUPPORT_SIZE = 4
QUERY_COUNT = 2
SEARCH_TOP_K = 5
EXPECTED_REQUESTS = 10
MAX_COST_USD = 0.75


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class SmokeExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _strict_question(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("question must be a string")
    question = " ".join(value.split())
    if not question or len(question) > 240 or question[-1] not in {"?", "？"}:
        raise ValueError("question must be concise and end with a question mark")
    if question.count("?") + question.count("？") != 1:
        raise ValueError("question must contain exactly one question mark")
    return question


def _parse_question_support(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != SUPPORT_SIZE:
        raise ValueError(
            f"blocker_questions must contain exactly {SUPPORT_SIZE} strings"
        )
    rows = [
        {
            "hypothesis": _strict_question(question).rstrip("?？"),
            "question": _strict_question(question),
        }
        for question in value
    ]
    if len({_normalize(row["question"]) for row in rows}) != SUPPORT_SIZE:
        raise ValueError("support questions must be distinct")
    return rows


def parse_initial(response: str) -> dict[str, Any]:
    payload = _parse_json_object(response)
    if set(payload) != {"blocker_questions", "business_search_queries"}:
        raise ValueError("initial response has unexpected keys")
    rows = _parse_question_support(payload["blocker_questions"])
    queries = payload["business_search_queries"]
    if not isinstance(queries, list) or len(queries) != QUERY_COUNT:
        raise ValueError(f"must return exactly {QUERY_COUNT} search queries")
    cleaned = [" ".join(str(query).split()) for query in queries]
    if any(not 3 <= len(query) <= 120 for query in cleaned):
        raise ValueError("search query has invalid length")
    if len({_normalize(query) for query in cleaned}) != QUERY_COUNT:
        raise ValueError("search queries must be distinct")
    return {"hypotheses": rows, "business_search_queries": cleaned}


def parse_refresh(response: str) -> list[dict[str, str]]:
    payload = _parse_json_object(response)
    if set(payload) != {"blocker_questions"}:
        raise ValueError("refresh response has unexpected keys")
    return _parse_question_support(payload["blocker_questions"])


def parse_judgment(
    response: str,
    candidate_ids: set[str],
    blocker_ids: set[str],
) -> dict[str, str | None]:
    payload = _parse_json_object(response)
    if set(payload) != {"matches"} or not isinstance(payload["matches"], list):
        raise ValueError("judge response must contain matches")
    matches: dict[str, str | None] = {}
    for row in payload["matches"]:
        if not isinstance(row, dict) or set(row) != {"candidate_id", "blocker_id"}:
            raise ValueError("judge row has unexpected keys")
        candidate_id = row["candidate_id"]
        blocker_id = row["blocker_id"]
        if candidate_id not in candidate_ids or candidate_id in matches:
            raise ValueError("judge returned invalid or duplicate candidate")
        if blocker_id is not None and blocker_id not in blocker_ids:
            raise ValueError("judge returned invalid blocker")
        matches[candidate_id] = blocker_id
    if set(matches) != candidate_ids:
        raise ValueError("judge did not classify every candidate")
    return matches


def _visible_task(hil_root: Path, task_id: str) -> dict[str, Any]:
    root = hil_root / "harbor_sql" / task_id
    instruction = (root / "ask_human" / "instruction.md").read_text(
        encoding="utf-8"
    )
    business = json.loads(
        (root / "shared" / "data" / "business_info.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "task_id": task_id,
        "problem": extract_problem(instruction),
        "business_info": business["business_info"],
    }


def _hidden_registry(hil_root: Path, task_id: str) -> list[dict[str, Any]]:
    path = (
        hil_root
        / "harbor_sql"
        / task_id
        / "shared"
        / "ask-human-data"
        / "blocker_registry.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["blockers"]


def search_business_info(
    documents: Sequence[str],
    query: str,
) -> list[dict[str, Any]]:
    tokenized = [_normalize(document).split() for document in documents]
    index = BM25Okapi(tokenized)
    scores = index.get_scores(_normalize(query).split())
    indices = sorted(
        range(len(documents)),
        key=lambda index_value: (float(scores[index_value]), -index_value),
        reverse=True,
    )[:SEARCH_TOP_K]
    return [
        {"document_index": index_value, "text": documents[index_value]}
        for index_value in indices
    ]


def _support_schema() -> dict[str, Any]:
    return {
        "blocker_questions": [
            f"single targeted clarification question {index + 1}?"
            for index in range(SUPPORT_SIZE)
        ]
    }


def initial_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    schema = {
        **_support_schema(),
        "business_search_queries": [
            f"short business-document search {index + 1}"
            for index in range(QUERY_COUNT)
        ],
    }
    request = {
        "database_question": task["problem"],
        "required_schema": schema,
    }
    return [
        {
            "role": "system",
            "content": (
                "You identify missing or ambiguous requirements in a SQL analytics "
                "request. Hidden task blockers exist but are not available to you. "
                "Return strict JSON only. Questions must each target one ambiguity, "
                "show useful analysis rather than merely repeat a term, contain one "
                "question mark, and not combine topics."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate exactly {SUPPORT_SIZE} distinct clarification questions "
                "representing possible hidden blockers, plus exactly "
                f"{QUERY_COUNT} diverse searches for potentially relevant business "
                "documentation. Use only the database question. "
                + json.dumps(request, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def refresh_messages(
    task: dict[str, Any],
    initial: dict[str, Any],
    *,
    evidence: dict[str, Any] | None,
) -> list[dict[str, str]]:
    request: dict[str, Any] = {
        "database_question": task["problem"],
        "initial_support": initial["hypotheses"],
        "required_schema": _support_schema(),
    }
    if evidence is None:
        instruction = (
            "Regenerate a fresh alternative support using only the database "
            "question. Do not assume any external observation."
        )
        request["external_observation"] = None
    else:
        instruction = (
            "Regenerate the support after interpreting the retrieved business "
            "documents. Prioritize ambiguities made concrete by this observation."
        )
        request["external_observation"] = evidence
    return [
        {
            "role": "system",
            "content": (
                "You update a generated belief support over missing or ambiguous "
                "requirements. Return strict JSON only. Questions must each target "
                "one ambiguity, show useful analysis rather than merely repeat a "
                "term, contain one question mark, and not combine topics."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{instruction} Return exactly {SUPPORT_SIZE} distinct blocker "
                "questions. "
                + json.dumps(request, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def judge_messages(
    task: dict[str, Any],
    registry: list[dict[str, Any]],
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    blockers = [
        {
            "blocker_id": blocker["id"],
            "description": blocker["description"],
            "example_questions": blocker.get("example_questions", []),
            "type": blocker.get("type"),
        }
        for blocker in registry
    ]
    required = {
        "matches": [
            {"candidate_id": candidate["candidate_id"], "blocker_id": None}
            for candidate in candidates
        ]
    }
    request = {
        "database_question": task["problem"],
        "blockers": blockers,
        "candidate_questions": candidates,
        "required_schema": required,
    }
    return [
        {
            "role": "system",
            "content": (
                "Act as a strict external blocker-question matcher. Match a "
                "candidate only when it is a concise, single-topic question with "
                "clear intent to solve the exact blocker. Reject broad questions, "
                "simple term repetition, assumptions asking for confirmation, or "
                "partial/tangential matches. Each candidate can match at most one "
                "blocker. Return strict JSON only and classify every candidate."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    candidates = []
    groups = [
        ("initial", record["initial"]["hypotheses"]),
        ("control", record["control"]),
    ]
    groups.extend(
        (f"branch_{index}", branch["hypotheses"])
        for index, branch in enumerate(record["branches"])
    )
    for group, rows in groups:
        for index, row in enumerate(rows):
            candidates.append(
                {
                    "candidate_id": f"{group}_{index}",
                    "question": row["question"],
                }
            )
    return candidates


def _usage(model: ChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "adapter_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "model": snapshot,
    }


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _matched_by_group(matches: dict[str, str | None], group: str) -> set[str]:
    return {
        blocker_id
        for candidate_id, blocker_id in matches.items()
        if candidate_id.startswith(group + "_") and blocker_id is not None
    }


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    task_metrics = []
    total_initial: set[tuple[str, str]] = set()
    total_control: set[tuple[str, str]] = set()
    total_best_evidence: set[tuple[str, str]] = set()
    recovered_business = 0
    for record in records:
        matches = record["matches"]
        initial = _matched_by_group(matches, "initial")
        control = _matched_by_group(matches, "control")
        branches = [
            _matched_by_group(matches, f"branch_{index}")
            for index in range(QUERY_COUNT)
        ]
        best_index = max(
            range(len(branches)),
            key=lambda index: (len(branches[index]), -index),
        )
        best = branches[best_index]
        baseline = initial | control
        new = best - baseline
        blocker_types = {
            blocker["id"]: str(blocker["type"]).strip().casefold()
            for blocker in record["registry"]
        }
        business_new = {
            blocker_id
            for blocker_id in new
            if blocker_types[blocker_id] == "business info"
        }
        recovered_business += len(business_new)
        total_initial.update((record["task_id"], value) for value in initial)
        total_control.update((record["task_id"], value) for value in control)
        total_best_evidence.update((record["task_id"], value) for value in best)
        task_metrics.append(
            {
                "task_id": record["task_id"],
                "blocker_count": len(record["registry"]),
                "initial_matched_blockers": len(initial),
                "control_matched_blockers": len(control),
                "branch_matched_blockers": [len(values) for values in branches],
                "best_evidence_branch": best_index,
                "best_evidence_matched_blockers": len(best),
                "new_over_initial_and_control": len(new),
                "new_business_blockers": len(business_new),
            }
        )
    efficacy = {
        "tasks_with_evidence_gain_over_initial_and_control": sum(
            row["new_over_initial_and_control"] > 0 for row in task_metrics
        ),
        "recovered_business_blockers": recovered_business,
        "initial_matched_blockers_total": len(total_initial),
        "control_matched_blockers_total": len(total_control),
        "best_evidence_matched_blockers_total": len(total_best_evidence),
    }
    return {"task_metrics": task_metrics, "efficacy": efficacy}


def run_smoke(
    config: Config,
    *,
    hil_root: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    if _git_head(hil_root) != HIL_COMMIT:
        raise ValueError("HiL-Bench checkout commit does not match")
    visible = [_visible_task(hil_root, task_id) for task_id in TASK_IDS]
    raw: dict[str, Any] = {
        "task_ids": list(TASK_IDS),
        "hidden_registry_loaded": False,
        "visible": visible,
    }
    try:
        initial_responses = model.chat_complete_messages_batched(
            [initial_messages(task) for task in visible],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=1600,
        )
        initial = [parse_initial(response) for response in initial_responses]
        raw["initial_responses"] = initial_responses
        raw["initial"] = initial
        _checkpoint(raw_path, raw)

        refresh_requests = []
        refresh_layout: list[tuple[int, str, int | None, dict[str, Any] | None]] = []
        evidence_by_task: list[list[dict[str, Any]]] = []
        for task_index, (task, support) in enumerate(zip(visible, initial)):
            task_evidence = []
            refresh_requests.append(
                refresh_messages(task, support, evidence=None)
            )
            refresh_layout.append((task_index, "control", None, None))
            for query_index, query in enumerate(support["business_search_queries"]):
                evidence = {
                    "search_query": query,
                    "retrieved_documents": search_business_info(
                        task["business_info"], query
                    ),
                }
                task_evidence.append(evidence)
                refresh_requests.append(
                    refresh_messages(task, support, evidence=evidence)
                )
                refresh_layout.append(
                    (task_index, "branch", query_index, evidence)
                )
            evidence_by_task.append(task_evidence)
        refresh_responses = model.chat_complete_messages_batched(
            refresh_requests,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=1400,
        )
        raw["refresh_responses"] = refresh_responses
        _checkpoint(raw_path, raw)
        parsed_refreshes = [parse_refresh(response) for response in refresh_responses]
        records = [
            {
                "task_id": task["task_id"],
                "problem": task["problem"],
                "initial": support,
                "control": None,
                "branches": [None] * QUERY_COUNT,
                "evidence": evidence_by_task[index],
            }
            for index, (task, support) in enumerate(zip(visible, initial))
        ]
        for layout, parsed in zip(refresh_layout, parsed_refreshes):
            task_index, kind, query_index, _ = layout
            if kind == "control":
                records[task_index]["control"] = parsed
            else:
                records[task_index]["branches"][int(query_index)] = {
                    "hypotheses": parsed
                }
        raw["records_before_endpoint"] = records
        _checkpoint(raw_path, raw)

        registries = [_hidden_registry(hil_root, task_id) for task_id in TASK_IDS]
        raw["hidden_registry_loaded"] = True
        judge_requests = []
        candidate_lists = []
        for task, record, registry in zip(visible, records, registries):
            candidates = _candidates(record)
            candidate_lists.append(candidates)
            judge_requests.append(judge_messages(task, registry, candidates))
        judge_responses = model.chat_complete_messages_batched(
            judge_requests,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=1200,
        )
        for record, registry, candidates, response in zip(
            records, registries, candidate_lists, judge_responses
        ):
            record["registry"] = registry
            record["matches"] = parse_judgment(
                response,
                {candidate["candidate_id"] for candidate in candidates},
                {blocker["id"] for blocker in registry},
            )
        raw["judge_responses"] = judge_responses
        raw["records_with_endpoint"] = records
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    analysis = analyze(records)
    distinct_observations = all(
        record["evidence"][0]["retrieved_documents"]
        != record["evidence"][1]["retrieved_documents"]
        for record in records
    )
    distinct_refreshes = all(
        len(
            {
                tuple(
                    _normalize(row["question"])
                    for row in support
                )
                for support in (
                    [record["initial"]["hypotheses"], record["control"]]
                    + [
                        branch["hypotheses"]
                        for branch in record["branches"]
                    ]
                )
            }
        )
        == 2 + QUERY_COUNT
        for record in records
    )
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_policy_and_judge_objects_parse": True,
        "hidden_registry_loaded_only_after_policy": raw["hidden_registry_loaded"],
        "branch_observations_distinct": distinct_observations,
        "all_generated_supports_distinct": distinct_refreshes,
        "evidence_gain_on_at_least_one_task": (
            analysis["efficacy"]["tasks_with_evidence_gain_over_initial_and_control"]
            >= 1
        ),
        "at_least_one_new_business_blocker": (
            analysis["efficacy"]["recovered_business_blockers"] >= 1
        ),
        "best_evidence_total_beats_each_control": (
            analysis["efficacy"]["best_evidence_matched_blockers_total"]
            > analysis["efficacy"]["initial_matched_blockers_total"]
            and analysis["efficacy"]["best_evidence_matched_blockers_total"]
            > analysis["efficacy"]["control_matched_blockers_total"]
        ),
        "cost_at_most_0_75": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    public_tasks = []
    for record in records:
        public_tasks.append(
            {
                "task_id": record["task_id"],
                "search_queries": record["initial"]["business_search_queries"],
                "retrieved_document_indices": [
                    [
                        item["document_index"]
                        for item in evidence["retrieved_documents"]
                    ]
                    for evidence in record["evidence"]
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_repository": "https://github.com/hilbenchauthors/hil-bench",
            "source_commit": HIL_COMMIT,
            "task_ids": list(TASK_IDS),
            "seed": SEED,
            "model": MODEL_ID,
            "support_size": SUPPORT_SIZE,
            "query_count": QUERY_COUNT,
            "search": "rank-bm25 0.2.2 BM25Okapi, top 5",
            "expected_requests": EXPECTED_REQUESTS,
            "policy_requests": 8,
            "post_freeze_judge_requests": 2,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "hidden_registry_available_to_policy": False,
        },
        **analysis,
        "public_tasks": public_tasks,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            request = _parse_json_object(messages[-1]["content"])
            if "candidate_questions" in request:
                business_blocker = next(
                    blocker["blocker_id"]
                    for blocker in request["blockers"]
                    if str(blocker.get("type", "")).casefold() == "business info"
                )
                responses.append(
                    json.dumps(
                        {
                            "matches": [
                                {
                                    "candidate_id": candidate["candidate_id"],
                                    "blocker_id": (
                                        business_blocker
                                        if candidate["candidate_id"].startswith(
                                            "branch_0_"
                                        )
                                        and candidate["candidate_id"].endswith("_0")
                                        else None
                                    ),
                                }
                                for candidate in request["candidate_questions"]
                            ]
                        }
                    )
                )
                continue
            suffix = self.requests + len(responses)
            support = [
                f"Which fixture mapping {suffix} {index} should be used?"
                for index in range(SUPPORT_SIZE)
            ]
            payload: dict[str, Any] = {"blocker_questions": support}
            if "business_search_queries" in request["required_schema"]:
                problem_words = re.findall(
                    r"[a-z0-9]+", request["database_question"].casefold()
                )
                payload["business_search_queries"] = [
                    " ".join(problem_words[:5]),
                    " ".join(problem_words[-5:]),
                ]
            responses.append(json.dumps(payload))
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
        }


def _nonthinking_spec(spec: Any) -> Any:
    return replace(
        spec,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )


def _build_model(config: Config) -> ChatModel:
    spec = _nonthinking_spec(config.model_pairs[0].questioner)
    if spec.model != MODEL_ID:
        raise ValueError("HiL-Bench smoke config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hil-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.30
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = EXPECTED_REQUESTS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = DeterministicFixtureModel() if args.dry_run else _build_model(config)
    try:
        payload = run_smoke(
            config,
            hil_root=args.hil_root,
            raw_path=raw_path,
            model=model,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, SmokeExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "SMOKE_FAILURE.json", failure)
        raise
    _checkpoint(args.output_dir / "SMOKE.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "efficacy": payload["efficacy"],
                "task_metrics": payload["task_metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
