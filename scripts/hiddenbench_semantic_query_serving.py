#!/usr/bin/env python3
"""Run the frozen ten-call HiddenBench semantic-query serving gate."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_dark_matter_semantic_smoke import (
    NonReasoningOpenRouterAdapter,
)
from scripts.discoverphysics_dark_matter_structured_replication_v3 import (
    use_default_structured_routing,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


INTERFACE_VERSION = "hiddenbench-semantic-query-serving-v1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
PROTOCOL_SHA256 = "ccc2bf5e9160daa55f63b0b9247d863578a4a82152604459aa21a1d54c0eac53"
MODEL_SEEDS = tuple(range(202608133000, 202608133010))
EXPECTED_REQUESTS = 10
MAX_TOKENS = 4096
RUN_BUDGET_USD = 0.015
MAX_REQUEST_COST_USD = 0.0015
QUERY_IDS = ("Q1", "Q2", "Q3", "Q4")
CHANNEL_IDS = ("C1", "C2", "C3")
DIRECT_ANSWER_PHRASES = ("correct answer", "which option", "choose the answer")
MIN_TV = 0.10
MIN_MAX_EIG = 0.005
MIN_EIG_RANGE = 0.001
MIN_WIN_MARGIN = 0.0001


class StructuredAdapter(Protocol):
    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def normalized_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def strict_object(response: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def probability_row(value: Any, *, ids: Sequence[str], label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != len(ids):
        raise ValueError(f"{label} has the wrong length")
    found: dict[str, float] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "probability"}:
            raise ValueError(f"{label} has the wrong fields")
        item_id = item["id"]
        probability = item["probability"]
        if item_id not in ids or item_id in found:
            raise ValueError(f"{label} has invalid or duplicate IDs")
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or probability < 0:
            raise ValueError(f"{label} has an invalid probability")
        found[item_id] = float(probability)
    if set(found) != set(ids) or abs(sum(found.values()) - 1.0) > 1e-6:
        raise ValueError(f"{label} is not normalized")
    return [found[item_id] for item_id in ids]


def root_response_format(option_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hiddenbench_semantic_root",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["prior", "queries"],
                "properties": {
                    "prior": {
                        "type": "array", "minItems": len(option_ids), "maxItems": len(option_ids),
                        "items": {"type": "object", "additionalProperties": False, "required": ["id", "probability"], "properties": {"id": {"type": "string", "enum": list(option_ids)}, "probability": {"type": "number", "minimum": 0, "maximum": 1}}},
                    },
                    "queries": {
                        "type": "array", "minItems": 4, "maxItems": 4,
                        "items": {"type": "object", "additionalProperties": False, "required": ["id", "request", "target_dimension"], "properties": {"id": {"type": "string", "enum": list(QUERY_IDS)}, "request": {"type": "string", "minLength": 1}, "target_dimension": {"type": "string", "minLength": 1}}},
                    },
                },
            },
        },
    }


def world_response_format(option_ids: Sequence[str]) -> dict[str, Any]:
    likelihood_item = {
        "type": "object", "additionalProperties": False, "required": ["option_id", "channels"],
        "properties": {"option_id": {"type": "string", "enum": list(option_ids)}, "channels": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "additionalProperties": False, "required": ["id", "probability"], "properties": {"id": {"type": "string", "enum": list(CHANNEL_IDS)}, "probability": {"type": "number", "minimum": 0, "maximum": 1}}}}},
    }
    query_item = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "channels", "likelihoods", "followups"],
        "properties": {
            "id": {"type": "string", "enum": list(QUERY_IDS)},
            "channels": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "additionalProperties": False, "required": ["id", "description"], "properties": {"id": {"type": "string", "enum": list(CHANNEL_IDS)}, "description": {"type": "string", "minLength": 1}}}},
            "likelihoods": {"type": "array", "minItems": len(option_ids), "maxItems": len(option_ids), "items": likelihood_item},
            "followups": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "additionalProperties": False, "required": ["channel_id", "query_id"], "properties": {"channel_id": {"type": "string", "enum": list(CHANNEL_IDS)}, "query_id": {"type": "string", "enum": list(QUERY_IDS)}}}},
        },
    }
    return {"type": "json_schema", "json_schema": {"name": "hiddenbench_semantic_world", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["query_models"], "properties": {"query_models": {"type": "array", "minItems": 4, "maxItems": 4, "items": query_item}}}}}


def router_response_format(fact_ids: Sequence[str]) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "hiddenbench_fact_router", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["addressed", "fact_id"], "properties": {"addressed": {"type": "boolean"}, "fact_id": {"type": "string", "enum": list(fact_ids)}}}}}


def parse_root(response: str, option_ids: Sequence[str]) -> dict[str, Any]:
    value = strict_object(response, label="root response")
    if set(value) != {"prior", "queries"}:
        raise ValueError("root response has the wrong fields")
    prior = probability_row(value["prior"], ids=option_ids, label="prior")
    queries = value["queries"]
    if not isinstance(queries, list) or len(queries) != 4:
        raise ValueError("root response must have four queries")
    parsed: dict[str, dict[str, str]] = {}
    requests: set[str] = set()
    dimensions: set[str] = set()
    for item in queries:
        if not isinstance(item, dict) or set(item) != {"id", "request", "target_dimension"}:
            raise ValueError("query has the wrong fields")
        query_id = item["id"]
        request = item["request"]
        dimension = item["target_dimension"]
        if query_id not in QUERY_IDS or query_id in parsed or not isinstance(request, str) or not request.strip() or not isinstance(dimension, str) or not dimension.strip():
            raise ValueError("query is invalid")
        normalized_request = normalized_text(request)
        normalized_dimension = normalized_text(dimension)
        if normalized_request in requests or normalized_dimension in dimensions or any(phrase in normalized_request for phrase in DIRECT_ANSWER_PHRASES):
            raise ValueError("query is duplicate or asks for the answer directly")
        requests.add(normalized_request)
        dimensions.add(normalized_dimension)
        parsed[query_id] = {"request": request.strip(), "target_dimension": dimension.strip()}
    if set(parsed) != set(QUERY_IDS):
        raise ValueError("query IDs are incomplete")
    return {"prior": prior, "queries": parsed}


def parse_world(response: str, option_ids: Sequence[str]) -> dict[str, Any]:
    value = strict_object(response, label="world response")
    if set(value) != {"query_models"} or not isinstance(value["query_models"], list) or len(value["query_models"]) != 4:
        raise ValueError("world response has the wrong fields or length")
    models: dict[str, Any] = {}
    for item in value["query_models"]:
        if not isinstance(item, dict) or set(item) != {"id", "channels", "likelihoods", "followups"}:
            raise ValueError("query model has the wrong fields")
        query_id = item["id"]
        if query_id not in QUERY_IDS or query_id in models:
            raise ValueError("query model ID is invalid or duplicated")
        channels = item["channels"]
        if not isinstance(channels, list) or len(channels) != 3:
            raise ValueError("query model needs three channels")
        descriptions: dict[str, str] = {}
        normalized_descriptions: set[str] = set()
        for channel in channels:
            if not isinstance(channel, dict) or set(channel) != {"id", "description"} or channel["id"] not in CHANNEL_IDS or channel["id"] in descriptions or not isinstance(channel["description"], str) or not channel["description"].strip():
                raise ValueError("channel is invalid")
            description = channel["description"].strip()
            normalized = normalized_text(description)
            if normalized in normalized_descriptions:
                raise ValueError("channel descriptions are duplicated")
            descriptions[channel["id"]] = description
            normalized_descriptions.add(normalized)
        if set(descriptions) != set(CHANNEL_IDS):
            raise ValueError("channel IDs are incomplete")
        rows: dict[str, list[float]] = {}
        likelihoods = item["likelihoods"]
        if not isinstance(likelihoods, list) or len(likelihoods) != len(option_ids):
            raise ValueError("likelihood option coverage is wrong")
        for row in likelihoods:
            if not isinstance(row, dict) or set(row) != {"option_id", "channels"} or row["option_id"] not in option_ids or row["option_id"] in rows:
                raise ValueError("likelihood row is invalid")
            rows[row["option_id"]] = probability_row(row["channels"], ids=CHANNEL_IDS, label=f"{query_id}.{row['option_id']}")
        if set(rows) != set(option_ids):
            raise ValueError("likelihood rows are incomplete")
        followups: dict[str, str] = {}
        if not isinstance(item["followups"], list) or len(item["followups"]) != 3:
            raise ValueError("followup coverage is wrong")
        for followup in item["followups"]:
            if not isinstance(followup, dict) or set(followup) != {"channel_id", "query_id"} or followup["channel_id"] not in CHANNEL_IDS or followup["channel_id"] in followups or followup["query_id"] not in QUERY_IDS or followup["query_id"] == query_id:
                raise ValueError("followup is invalid")
            followups[followup["channel_id"]] = followup["query_id"]
        if set(followups) != set(CHANNEL_IDS):
            raise ValueError("followup channels are incomplete")
        models[query_id] = {"channels": descriptions, "likelihoods": rows, "followups": followups}
    if set(models) != set(QUERY_IDS):
        raise ValueError("query models are incomplete")
    return models


def parse_router(response: str, fact_ids: Sequence[str]) -> dict[str, Any]:
    value = strict_object(response, label="router response")
    if set(value) != {"addressed", "fact_id"} or not isinstance(value["addressed"], bool) or value["fact_id"] not in fact_ids:
        raise ValueError("router response is invalid")
    return {"addressed": value["addressed"], "fact_id": value["fact_id"]}


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def posterior(prior: Sequence[float], likelihoods: Sequence[Sequence[float]], channel: int) -> tuple[float, list[float]]:
    joint = [prior[i] * likelihoods[i][channel] for i in range(len(prior))]
    evidence = sum(joint)
    if evidence <= 0:
        return 0.0, list(prior)
    return evidence, [value / evidence for value in joint]


def query_eig(prior: Sequence[float], likelihoods: Sequence[Sequence[float]]) -> float:
    return entropy(prior) - sum(p * entropy(post) for p, post in (posterior(prior, likelihoods, channel) for channel in range(3)))


def score_model(prior: Sequence[float], world: dict[str, Any], option_ids: Sequence[str]) -> dict[str, Any]:
    matrices = {query_id: [world[query_id]["likelihoods"][option_id] for option_id in option_ids] for query_id in QUERY_IDS}
    one_step = {query_id: query_eig(prior, matrices[query_id]) for query_id in QUERY_IDS}
    depth_two: dict[str, float] = {}
    for first_id in QUERY_IDS:
        future = 0.0
        for channel in range(3):
            probability, updated = posterior(prior, matrices[first_id], channel)
            if probability > 0:
                future += probability * max(query_eig(updated, matrices[second_id]) for second_id in QUERY_IDS if second_id != first_id)
        depth_two[first_id] = one_step[first_id] + future
    greedy_order = sorted(QUERY_IDS, key=lambda query_id: (-one_step[query_id], query_id))
    depth_order = sorted(QUERY_IDS, key=lambda query_id: (-depth_two[query_id], query_id))
    return {
        "one_step": one_step,
        "depth_two": depth_two,
        "greedy_query_id": greedy_order[0],
        "depth_two_query_id": depth_order[0],
        "max_one_step_eig": one_step[greedy_order[0]],
        "one_step_range": max(one_step.values()) - min(one_step.values()),
        "greedy_margin": one_step[greedy_order[0]] - one_step[greedy_order[1]],
        "depth_two_margin": depth_two[depth_order[0]] - depth_two[depth_order[1]],
    }


def root_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    prompt = {"description": task["description"], "shared_facts": task["shared_facts"], "options": task["options"]}
    return [{"role": "system", "content": "You design Bayesian evidence requests. Return only the required JSON. Requests must ask for one evidence dimension, never ask which option is correct, and must be answerable by one additional private fact."}, {"role": "user", "content": json.dumps(prompt, sort_keys=True)}]


def world_messages(task: dict[str, Any], root: dict[str, Any]) -> list[dict[str, str]]:
    prompt = {"description": task["description"], "shared_facts": task["shared_facts"], "options": task["options"], "prior": [{"id": option["id"], "probability": probability} for option, probability in zip(task["options"], root["prior"], strict=True)], "queries": [{"id": query_id, **root["queries"][query_id]} for query_id in QUERY_IDS]}
    return [{"role": "system", "content": "You build a finite semantic observation model for Bayesian experimental design. For each evidence request, define the same three response channels under every option, estimate P(channel|option,query), and choose a different modeled query after each channel. Return only the required JSON. Do not infer or state which option is correct."}, {"role": "user", "content": json.dumps(prompt, sort_keys=True)}]


def router_messages(task: dict[str, Any], query_id: str, request: str) -> list[dict[str, str]]:
    prompt = {"description": task["description"], "query": {"id": query_id, "request": request}, "private_facts": task["private_facts"]}
    return [{"role": "system", "content": "Route the evidence request to the one private fact that best and directly addresses it. Return only its fact ID and whether it is addressed. Never add observation text."}, {"role": "user", "content": json.dumps(prompt, sort_keys=True)}]


def load_projected_views(source_path: Path, *, custodian_path: Path | None = None) -> dict[str, Any]:
    custodian = custodian_path or Path(__file__).with_name("hiddenbench_semantic_query_custodian.py")
    completed = subprocess.run([sys.executable, str(custodian), "--source", str(source_path)], check=True, capture_output=True, text=True)
    views = json.loads(completed.stdout)
    if not isinstance(views, dict) or set(views) != {"planner", "router"} or len(views["planner"]) != 4 or len(views["router"]) != 4:
        raise RuntimeError("custodian returned invalid views")
    return views


class PerRequestSeedAdapter(NonReasoningOpenRouterAdapter):
    def __init__(self, spec: ModelSpec, config: Config) -> None:
        super().__init__(spec, config)
        self._request_seed = threading.local()

    def _payload(self, messages: list[dict[str, Any]], temperature: float, n: int, max_tokens: int | None = None, *, disable_reasoning: bool = False, response_format: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = super()._payload(messages, temperature, n, max_tokens, disable_reasoning=True, response_format=response_format)
        payload = use_default_structured_routing(payload)
        payload["provider"] = {"require_parameters": True}
        seed = getattr(self._request_seed, "value", None)
        if seed is None:
            raise RuntimeError("request seed was not bound")
        payload["seed"] = int(seed)
        return payload

    def chat_complete_seeded_messages_batched_structured(self, batch_messages: Sequence[list[dict[str, str]]], seeds: Sequence[int], *, temperature: float, response_format: dict[str, Any], max_new_tokens: int | None = None) -> list[str]:
        if len(batch_messages) != len(seeds):
            raise ValueError("message and seed counts differ")
        def request(item: tuple[list[dict[str, str]], int]) -> str:
            messages, seed = item
            self._request_seed.value = int(seed)
            try:
                return self._complete_request(messages, temperature, 1, max_new_tokens, allow_forced_final=False, disable_reasoning=True, response_format=response_format)[0]
            finally:
                del self._request_seed.value
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(request, zip(batch_messages, seeds, strict=True)))


def build_adapter(run_id: str, output_dir: Path) -> PerRequestSeedAdapter:
    config = Config(task="animals", run_id=run_id, log_path=output_dir / "run.log", openrouter_budget_usd=245.0, openrouter_run_budget_usd=RUN_BUDGET_USD, openrouter_projected_cost_usd=RUN_BUDGET_USD, openrouter_concurrency=10, openrouter_max_retries=0, openrouter_backoff_seconds=0.0, openrouter_request_timeout_seconds=300.0, openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD, openrouter_max_output_tokens=MAX_TOKENS, openrouter_spend_path="results/path_e/openrouter_spend.json")
    return PerRequestSeedAdapter(ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65536), config)


def run_serving(*, source_path: Path, output_dir: Path, adapter: StructuredAdapter) -> dict[str, Any]:
    protocol_path = REPO_ROOT / "results/nonmyopic/HIDDENBENCH_SEMANTIC_QUERY_SERVING_PROTOCOL_20260813.md"
    if file_digest(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("HiddenBench serving protocol changed")
    views = load_projected_views(source_path)
    planner = views["planner"]
    router = views["router"]
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    roots: list[dict[str, Any]] = []
    worlds: list[dict[str, Any]] = []
    raw: dict[str, Any] = {"root": [], "world": [], "router": []}
    seed_offset = 0
    for task in planner:
        option_ids = [item["id"] for item in task["options"]]
        response = adapter.chat_complete_seeded_messages_batched_structured([root_messages(task)], [MODEL_SEEDS[seed_offset]], temperature=0.0, response_format=root_response_format(option_ids), max_new_tokens=MAX_TOKENS)[0]
        seed_offset += 1
        raw["root"].append(response)
        checkpoint(private_dir / "RAW_RESPONSES.json", raw)
        roots.append(parse_root(response, option_ids))
    for task, root in zip(planner, roots, strict=True):
        option_ids = [item["id"] for item in task["options"]]
        response = adapter.chat_complete_seeded_messages_batched_structured([world_messages(task, root)], [MODEL_SEEDS[seed_offset]], temperature=0.0, response_format=world_response_format(option_ids), max_new_tokens=MAX_TOKENS)[0]
        seed_offset += 1
        raw["world"].append(response)
        checkpoint(private_dir / "RAW_RESPONSES.json", raw)
        worlds.append(parse_world(response, option_ids))
    router_results = []
    fact_ids = [item["id"] for item in router[0]["private_facts"]]
    for query_id in QUERY_IDS[:2]:
        response = adapter.chat_complete_seeded_messages_batched_structured([router_messages(router[0], query_id, roots[0]["queries"][query_id]["request"])], [MODEL_SEEDS[seed_offset]], temperature=0.0, response_format=router_response_format(fact_ids), max_new_tokens=MAX_TOKENS)[0]
        seed_offset += 1
        raw["router"].append(response)
        checkpoint(private_dir / "RAW_RESPONSES.json", raw)
        parsed = parse_router(response, fact_ids)
        parsed["observation"] = next(item["text"] for item in router[0]["private_facts"] if item["id"] == parsed["fact_id"])
        router_results.append(parsed)
    scores = [score_model(root["prior"], world, [item["id"] for item in task["options"]]) for task, root, world in zip(planner, roots, worlds, strict=True)]
    tv_counts = []
    followup_counts = []
    for task, world in zip(planner, worlds, strict=True):
        option_ids = [item["id"] for item in task["options"]]
        tv_counts.append(sum(max(0.5 * sum(abs(a - b) for a, b in zip(world[q]["likelihoods"][left], world[q]["likelihoods"][right], strict=True)) for i, left in enumerate(option_ids) for right in option_ids[i + 1:]) >= MIN_TV for q in QUERY_IDS))
        followup_counts.append(sum(len(set(world[q]["followups"].values())) >= 2 for q in QUERY_IDS))
    snapshot = adapter.usage_snapshot()
    usage = {"accepted_requests": int(snapshot.get("adapter_requests", 0)), "http_attempts": int(snapshot.get("http_attempts", 0)), "retries": int(snapshot.get("retry_count", 0)), "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)), "forced_exits": int(snapshot.get("forced_exits", 0)), "forced_final_requests": int(snapshot.get("forced_final_requests", 0)), "cost_usd": float(snapshot.get("adapter_cost_usd", 0.0))}
    gates = {
        "exact_transport": usage["accepted_requests"] == usage["http_attempts"] == EXPECTED_REQUESTS and usage["retries"] == usage["reasoning_tokens"] == usage["forced_exits"] == usage["forced_final_requests"] == 0,
        "strict_schema_all_ten": seed_offset == EXPECTED_REQUESTS,
        "option_sensitive_worlds": all(value >= 2 for value in tv_counts),
        "response_dependent_followups": all(value >= 2 for value in followup_counts),
        "nondegenerate_planner": all(score["max_one_step_eig"] >= MIN_MAX_EIG and score["one_step_range"] >= MIN_EIG_RANGE and score["greedy_margin"] >= MIN_WIN_MARGIN and score["depth_two_margin"] >= MIN_WIN_MARGIN for score in scores),
        "depth_two_changed_first_request": sum(score["greedy_query_id"] != score["depth_two_query_id"] for score in scores) >= 2,
        "exact_distinct_fact_routing": len(router_results) == 2 and all(item["addressed"] for item in router_results) and len({item["fact_id"] for item in router_results}) == 2,
        "within_run_budget": usage["cost_usd"] <= RUN_BUDGET_USD + 1e-12,
    }
    passed = all(gates.values())
    result = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "serving_pass" if passed else "serving_failed_closed", "decision": "opportunity_mechanics_protocol_authorized" if passed else "close_exact_hiddenbench_semantic_query_interface", "protocol_sha256": PROTOCOL_SHA256, "aggregate": {"tasks": 4, "parsed_responses": seed_offset, "option_sensitive_query_counts": tv_counts, "response_dependent_query_counts": followup_counts, "changed_first_request_tasks": sum(score["greedy_query_id"] != score["depth_two_query_id"] for score in scores)}, "scores": scores, "gates": gates, "usage": usage, "authorizes": "separately_frozen_opportunity_mechanics_only" if passed else "nothing"}
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_serving(source_path=args.source.resolve(), output_dir=args.output_dir.resolve(), adapter=build_adapter(args.run_id, args.output_dir.resolve()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "serving_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
