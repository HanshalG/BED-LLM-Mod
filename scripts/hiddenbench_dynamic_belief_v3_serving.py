#!/usr/bin/env python3
"""Produce the ten label-free HiddenBench dynamic-belief V3 responses."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_dark_matter_semantic_smoke import NonReasoningOpenRouterAdapter
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.hiddenbench_dynamic_belief_v3_codec import (
    parse_refresh,
    parse_root,
    parse_routing,
    refresh_response_format,
    root_response_format,
    routing_response_format,
)
from scripts.hiddenbench_dynamic_belief_v3_math import (
    CHANNEL_IDS,
    QUERY_IDS,
    best_query,
    branch_diagnostics,
    dynamic_depth_two,
    fixed_depth_two,
    tv,
)


INTERFACE_VERSION = "hiddenbench-dynamic-belief-v3-serving-v1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
PROTOCOL_SHA256 = "cab055aed50b22abfc1b90e720523882d8d6320616f0ab5f474a9340bf428bba"
MODEL_SEEDS = tuple(range(202608135000, 202608135010))
EXPECTED_REQUESTS = 10
MAX_TOKENS = 8192
CONCURRENCY = 4
MAX_REQUEST_COST_USD = 0.004
RUN_CAP_USD = 0.040
MIN_OPTION_TV = 0.10
SOURCE_PATH = Path("/tmp/bed-source-audits/hiddenbench/data/benchmark.json")


class StructuredAdapter(Protocol):
    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_formats: Sequence[dict[str, Any]],
        max_new_tokens: int,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def load_views(source_path: Path) -> dict[str, Any]:
    custodian = Path(__file__).with_name(
        "hiddenbench_dynamic_belief_v3_serving_custodian.py"
    )
    completed = subprocess.run(
        [sys.executable, str(custodian), "--source", str(source_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if (
        not isinstance(value, dict)
        or set(value) != {"planner", "router"}
        or len(value["planner"]) != 4
        or len(value["router"]) != 4
    ):
        raise RuntimeError("V3 serving custodian returned invalid views")
    return value


def root_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "description": task["description"],
        "shared_facts": task["shared_facts"],
        "options": task["options"],
    }
    return [
        {
            "role": "system",
            "content": (
                "Design a finite semantic Bayesian experiment. Return only the required JSON. "
                "Give a calibrated prior, four distinct requests for one private evidence dimension, "
                "three exhaustive qualitative response channels per request, and P(channel|option,query). "
                "Never ask which option is correct and never state a correct option."
            ),
        },
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]


def public_root(root: dict[str, Any], option_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "prior": [
            {"option_id": option_id, "probability": probability}
            for option_id, probability in zip(option_ids, root["prior"], strict=True)
        ],
        "queries": [
            {
                "query_id": query_id,
                "request": root["queries"][query_id]["request"],
                "target_dimension": root["queries"][query_id]["target_dimension"],
                "channels": [
                    {
                        "channel_id": channel_id,
                        "description": root["queries"][query_id]["channels"][channel_id],
                    }
                    for channel_id in CHANNEL_IDS
                ],
                "likelihoods": [
                    {
                        "option_id": option_id,
                        "channels": [
                            {"channel_id": channel_id, "probability": probability}
                            for channel_id, probability in zip(
                                CHANNEL_IDS,
                                root["queries"][query_id]["likelihoods"][option_id],
                                strict=True,
                            )
                        ],
                    }
                    for option_id in option_ids
                ],
            }
            for query_id in QUERY_IDS
        ],
    }


def refresh_messages(
    task: dict[str, Any], root: dict[str, Any], option_ids: Sequence[str]
) -> list[dict[str, str]]:
    payload = {
        "task": {
            "description": task["description"],
            "shared_facts": task["shared_facts"],
            "options": task["options"],
        },
        "root_model": public_root(root, option_ids),
    }
    return [
        {
            "role": "system",
            "content": (
                "Regenerate the answer-option belief independently for every hypothetical "
                "query/response-channel branch. Return only the required JSON. Condition on the "
                "semantic meaning of the request and channel. Do not merely copy the numeric Bayes "
                "posterior, do not name a correct answer, and cover all twelve branches exactly once."
            ),
        },
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]


def routing_payload(
    router_tasks: Sequence[dict[str, Any]],
    planner_tasks: Sequence[dict[str, Any]],
    roots: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    tasks = []
    for router, planner, root in zip(router_tasks, planner_tasks, roots, strict=True):
        tasks.append(
            {
                "slot": router["slot"],
                "description": router["description"],
                "private_facts": router["private_facts"],
                "queries": [
                    {
                        "query_id": query_id,
                        "request": root["queries"][query_id]["request"],
                        "target_dimension": root["queries"][query_id]["target_dimension"],
                        "channels": [
                            {
                                "channel_id": channel_id,
                                "description": root["queries"][query_id]["channels"][channel_id],
                            }
                            for channel_id in CHANNEL_IDS
                        ],
                    }
                    for query_id in QUERY_IDS
                ],
            }
        )
    return {"tasks": tasks}


def routing_messages(payload: dict[str, Any], *, auditor: bool) -> list[dict[str, str]]:
    role = "Independently audit" if auditor else "Route"
    return [
        {
            "role": "system",
            "content": (
                f"{role} each evidence request against the private facts. Return only the required JSON. "
                "For every task/query choose the one fact that best directly addresses the request and "
                "the one supplied response channel that best describes that exact fact. Never add text, "
                "never infer a correct answer, and cover all sixteen mappings exactly once."
            ),
        },
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]


class PerRequestSeedAdapter(NonReasoningOpenRouterAdapter):
    def __init__(self, spec: ModelSpec, config: Config) -> None:
        super().__init__(spec, config)
        self._request_seed = threading.local()

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=True,
            response_format=response_format,
        )
        payload["provider"] = {"require_parameters": True}
        seed = getattr(self._request_seed, "value", None)
        if seed is None:
            raise RuntimeError("request seed was not bound")
        payload["seed"] = int(seed)
        return payload

    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_formats: Sequence[dict[str, Any]],
        max_new_tokens: int,
    ) -> list[str]:
        if not (len(batch_messages) == len(seeds) == len(response_formats)):
            raise ValueError("messages, seeds, and schemas differ in length")

        def request(item: tuple[list[dict[str, str]], int, dict[str, Any]]) -> str:
            messages, seed, response_format = item
            self._request_seed.value = int(seed)
            try:
                return self._complete_request(
                    messages,
                    temperature,
                    1,
                    max_new_tokens,
                    allow_forced_final=False,
                    disable_reasoning=True,
                    response_format=response_format,
                )[0]
            finally:
                del self._request_seed.value

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(
                executor.map(
                    request,
                    zip(batch_messages, seeds, response_formats, strict=True),
                )
            )


def build_adapter(run_id: str, output_dir: Path) -> PerRequestSeedAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=245.0,
        openrouter_run_budget_usd=RUN_CAP_USD,
        openrouter_projected_cost_usd=RUN_CAP_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return PerRequestSeedAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65536),
        config,
    )


def matrices(root: dict[str, Any], option_ids: Sequence[str]) -> dict[str, list[list[float]]]:
    return {
        query_id: [
            root["queries"][query_id]["likelihoods"][option_id]
            for option_id in option_ids
        ]
        for query_id in QUERY_IDS
    }


def semantic_summary(
    planner_tasks: Sequence[dict[str, Any]],
    roots: Sequence[dict[str, Any]],
    refreshes: Sequence[dict[str, dict[str, list[float]]]],
    router: dict[str, dict[str, dict[str, str]]],
    auditor: dict[str, dict[str, dict[str, str]]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    task_summaries = []
    for task, root, refreshed in zip(planner_tasks, roots, refreshes, strict=True):
        option_ids = tuple(item["id"] for item in task["options"])
        likelihoods = matrices(root, option_ids)
        one_step = {
            query_id: best_query(root["prior"], {query_id: likelihoods[query_id]})[1]
            for query_id in QUERY_IDS
        }
        ordered = sorted(QUERY_IDS, key=lambda query_id: (-one_step[query_id], query_id))
        option_sensitive = sum(
            max(
                tv(likelihoods[query_id][left], likelihoods[query_id][right])
                for left in range(len(option_ids))
                for right in range(left + 1, len(option_ids))
            )
            >= MIN_OPTION_TV
            for query_id in QUERY_IDS
        )
        dynamic = dynamic_depth_two(root["prior"], likelihoods, refreshed)
        fixed = fixed_depth_two(root["prior"], likelihoods)
        diagnostics = branch_diagnostics(root["prior"], likelihoods, refreshed)
        task_summaries.append(
            {
                "slot": task["slot"],
                "option_count": len(option_ids),
                "option_sensitive_queries": option_sensitive,
                "max_one_step_eig": one_step[ordered[0]],
                "one_step_eig_range": max(one_step.values()) - min(one_step.values()),
                "myopic_first_query_id": ordered[0],
                "myopic_margin": one_step[ordered[0]] - one_step[ordered[1]],
                "dynamic": dynamic,
                "fixed": fixed,
                "refresh": diagnostics,
            }
        )
    exact_mapping_agreement = all(router == auditor for _ in (0,))
    distinct_fact_counts = {
        slot: len({mapping["fact_id"] for mapping in mappings.values()})
        for slot, mappings in router.items()
    }
    gates = {
        "option_sensitive_likelihoods": all(
            task["option_sensitive_queries"] >= 2 for task in task_summaries
        ),
        "nondegenerate_root_eig": all(
            task["max_one_step_eig"] >= 0.005
            and task["one_step_eig_range"] >= 0.001
            and task["myopic_margin"] >= 0.0001
            for task in task_summaries
        ),
        "refresh_answer_obedience": all(
            task["refresh"]["obedient_branch_count"] >= 10
            and task["refresh"]["mean_compatibility_increase"] >= 0.01
            for task in task_summaries
        ),
        "refresh_calibrated_irreducibility": all(
            0.01 <= task["refresh"]["mean_exact_bayes_tv"] <= 0.20
            and task["refresh"]["branches_exact_bayes_tv_at_least_001"] >= 6
            and task["refresh"]["mean_within_query_pairwise_tv"] >= 0.10
            for task in task_summaries
        ),
        "dynamic_planning_nondegenerate": all(
            task["dynamic"]["margin"] >= 0.0001
            and task["dynamic"]["response_contingent_first_queries"] >= 2
            for task in task_summaries
        ),
        "dynamic_changes_first_query": sum(
            task["dynamic"]["first_query_id"] != task["myopic_first_query_id"]
            for task in task_summaries
        )
        >= 2,
        "independent_routing_consensus": exact_mapping_agreement
        and all(value >= 3 for value in distinct_fact_counts.values()),
    }
    return {
        "tasks": task_summaries,
        "changed_first_query_tasks": sum(
            task["dynamic"]["first_query_id"] != task["myopic_first_query_id"]
            for task in task_summaries
        ),
        "distinct_fact_counts": distinct_fact_counts,
        "exact_router_auditor_agreement": exact_mapping_agreement,
    }, gates


def usage_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retries": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "forced_final_requests": int(snapshot.get("forced_final_requests", 0)),
        "cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
    }


def run_serving(
    *, source_path: Path, output_dir: Path, adapter: StructuredAdapter
) -> dict[str, Any]:
    protocol = REPO_ROOT / "results/nonmyopic/HIDDENBENCH_DYNAMIC_BELIEF_V3_MECHANICS_PROTOCOL_20260813.md"
    if file_digest(protocol) != PROTOCOL_SHA256:
        raise RuntimeError("V3 mechanics protocol changed")
    views = load_views(source_path)
    planner_tasks = views["planner"]
    router_tasks = views["router"]
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"roots": [], "refreshes": [], "router": [], "auditor": []}

    root_messages_batch = [root_messages(task) for task in planner_tasks]
    option_ids_by_task = [tuple(option["id"] for option in task["options"]) for task in planner_tasks]
    root_responses = adapter.chat_complete_seeded_messages_batched_structured(
        root_messages_batch,
        MODEL_SEEDS[:4],
        temperature=0.0,
        response_formats=[root_response_format(option_ids) for option_ids in option_ids_by_task],
        max_new_tokens=MAX_TOKENS,
    )
    raw["roots"] = root_responses
    checkpoint(raw_path, raw)
    roots = [
        parse_root(response, option_ids)
        for response, option_ids in zip(root_responses, option_ids_by_task, strict=True)
    ]

    refresh_responses = adapter.chat_complete_seeded_messages_batched_structured(
        [refresh_messages(task, root, option_ids) for task, root, option_ids in zip(planner_tasks, roots, option_ids_by_task, strict=True)],
        MODEL_SEEDS[4:8],
        temperature=0.0,
        response_formats=[refresh_response_format(option_ids) for option_ids in option_ids_by_task],
        max_new_tokens=MAX_TOKENS,
    )
    raw["refreshes"] = refresh_responses
    checkpoint(raw_path, raw)
    refreshes = [
        parse_refresh(response, option_ids)
        for response, option_ids in zip(refresh_responses, option_ids_by_task, strict=True)
    ]

    route_payload = routing_payload(router_tasks, planner_tasks, roots)
    routing_responses = adapter.chat_complete_seeded_messages_batched_structured(
        [routing_messages(route_payload, auditor=False), routing_messages(route_payload, auditor=True)],
        MODEL_SEEDS[8:10],
        temperature=0.0,
        response_formats=[routing_response_format(), routing_response_format()],
        max_new_tokens=MAX_TOKENS,
    )
    raw["router"] = [routing_responses[0]]
    raw["auditor"] = [routing_responses[1]]
    checkpoint(raw_path, raw)
    fact_ids_by_slot = {
        task["slot"]: tuple(fact["id"] for fact in task["private_facts"])
        for task in router_tasks
    }
    router = parse_routing(routing_responses[0], fact_ids_by_slot)
    auditor = parse_routing(routing_responses[1], fact_ids_by_slot)
    semantic, semantic_gates = semantic_summary(
        planner_tasks, roots, refreshes, router, auditor
    )
    usage = usage_summary(adapter.usage_snapshot())
    transport_gates = {
        "exact_transport": usage["accepted_requests"] == usage["http_attempts"] == 10
        and usage["retries"] == usage["reasoning_tokens"] == usage["forced_exits"] == usage["forced_final_requests"] == 0,
        "strict_complete_parse": all(len(raw[key]) == expected for key, expected in {"roots": 4, "refreshes": 4, "router": 1, "auditor": 1}.items()),
        "within_run_cap": usage["cost_usd"] <= RUN_CAP_USD + 1e-12,
    }
    gates = {**transport_gates, **semantic_gates}
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "serving_pass" if passed else "serving_failed_closed",
        "decision": "endpoint_authorized" if passed else "close_exact_v3_interface",
        "authorizes": "endpoint_only" if passed else "nothing",
        "protocol_sha256": PROTOCOL_SHA256,
        "raw_response_sha256": file_digest(raw_path),
        "semantic": semantic,
        "consensus_routing": router if router == auditor else {},
        "gates": gates,
        "usage": usage,
        "registered_answers_opened": False,
        "endpoint_scores_opened": False,
    }
    checkpoint(output_dir / "LABEL_FREE_RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_serving(
        source_path=args.source.resolve(),
        output_dir=args.output_dir.resolve(),
        adapter=build_adapter(args.run_id, args.output_dir.resolve()),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "serving_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
