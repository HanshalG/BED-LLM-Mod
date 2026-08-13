#!/usr/bin/env python3
"""Produce the fresh Number Game bitmask semantic-support gate."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_dark_matter_semantic_smoke import NonReasoningOpenRouterAdapter
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_bitmask_semantic_codec import (
    AUDIT_SEEDS, HISTORIES, INTERFACE_VERSION, PROPOSAL_SEEDS,
    audit_messages, audit_response_format, diagnostics, parse_audit,
    parse_proposal, proposal_messages, proposal_response_format,
)


PROPOSAL_MODEL_ID = "qwen/qwen3.7-plus"
AUDIT_MODEL_ID = "openai/gpt-5.6-luna"
PROPOSAL_MAX_TOKENS = 3000
AUDIT_MAX_TOKENS = 1200
STAGE_CAP_USD = 0.08
PROTOCOL = REPO_ROOT / "results/nonmyopic/NUMBER_GAME_BITMASK_SEMANTIC_SUPPORT_GATE_PROTOCOL_20260813.md"
PROTOCOL_SHA256 = "15b6d1ff6da5d62013c2717ebb88ab2740d8893a0cf0c429f91113a306a8f200"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AdapterProtocol(Protocol):
    def complete(self, messages: Sequence[list[dict[str, str]]], seeds: Sequence[int], *, response_format: dict[str, Any], max_tokens: int) -> list[str]: ...
    def usage_snapshot(self) -> dict[str, Any]: ...
    def records(self) -> list[dict[str, Any]]: ...


class SeededAdapter(NonReasoningOpenRouterAdapter):
    def __init__(self, spec: ModelSpec, config: Config, *, authorize: Callable[[], None] | None = None) -> None:
        super().__init__(spec, config)
        self._seed = threading.local()
        self._authorize = authorize
        self._records: list[dict[str, Any]] = []
        self._record_lock = threading.Lock()

    def _payload(self, messages, temperature, n, max_tokens=None, *, disable_reasoning=False, response_format=None):
        payload = super()._payload(messages, temperature, n, max_tokens, disable_reasoning=True, response_format=response_format)
        payload["provider"] = {"require_parameters": True}
        seed = getattr(self._seed, "value", None)
        if seed is None:
            raise RuntimeError("request seed is unbound")
        payload["seed"] = int(seed)
        return payload

    def _post(self, payload):
        if self._authorize is not None:
            self._authorize()
        data = super()._post(payload)
        choices = data.get("choices") or []
        with self._record_lock:
            self._records.append({
                "seed": payload["seed"], "model_requested": payload["model"],
                "model_returned": data.get("model"),
                "prompt_sha256": hashlib.sha256(canonical(payload["messages"])).hexdigest(),
                "payload_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
                "finish_reasons": [row.get("finish_reason") for row in choices if isinstance(row, dict)],
            })
        return data

    def complete(self, messages, seeds, *, response_format, max_tokens):
        if len(messages) != len(seeds):
            raise ValueError("message and seed counts differ")
        def one(item):
            prompt, seed = item
            self._seed.value = int(seed)
            try:
                return self._complete_request(prompt, 0.7, 1, max_tokens, allow_forced_final=False, disable_reasoning=True, response_format=response_format)[0]
            finally:
                del self._seed.value
        with ThreadPoolExecutor(max_workers=10) as pool:
            return list(pool.map(one, zip(messages, seeds, strict=True)))

    def records(self):
        return sorted((dict(row) for row in self._records), key=lambda row: row["seed"])


def build_adapter(*, model: str, run_id: str, output_dir: Path, phase_exposure: float, request_cap: float, max_tokens: int, authorize: Callable[[], None] | None = None) -> SeededAdapter:
    config = Config(task="animals", run_id=run_id, log_path=output_dir / "run.log", openrouter_budget_usd=245.0, openrouter_run_budget_usd=STAGE_CAP_USD, openrouter_projected_cost_usd=phase_exposure, openrouter_concurrency=10, openrouter_max_retries=0, openrouter_backoff_seconds=1.0, openrouter_request_timeout_seconds=300.0, openrouter_max_request_cost_usd=request_cap, openrouter_max_output_tokens=max_tokens, openrouter_spend_path="results/path_e/openrouter_spend.json")
    return SeededAdapter(ModelSpec(model=model, backend="openrouter", max_model_len=65536), config, authorize=authorize)


def usage(adapter: AdapterProtocol) -> dict[str, Any]:
    value = adapter.usage_snapshot()
    return {"accepted_requests": int(value.get("adapter_requests", 0)), "http_attempts": int(value.get("http_attempts", 0)), "retries": int(value.get("retry_count", 0)), "reasoning_tokens": int(value.get("adapter_reasoning_tokens", 0)), "forced_exits": int(value.get("forced_exits", 0)), "forced_final_requests": int(value.get("forced_final_requests", 0)), "cost_usd": float(value.get("adapter_cost_usd", 0.0))}


def run_gate(*, output_dir: Path, proposal_adapter: AdapterProtocol, audit_factory: Callable[[Sequence[list[dict[str, str]]]], AdapterProtocol]) -> dict[str, Any]:
    if file_digest(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("bitmask protocol changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"; private.mkdir(parents=True, exist_ok=True)
    raw_path = private / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"proposal": [], "audit": []}
    proposal_raw = proposal_adapter.complete([proposal_messages(HISTORIES[index // 2]) for index in range(10)], PROPOSAL_SEEDS, response_format=proposal_response_format(), max_tokens=PROPOSAL_MAX_TOKENS)
    raw["proposal"] = proposal_raw; checkpoint(raw_path, raw)
    proposals = [parse_proposal(response, HISTORIES[index // 2]) for index, response in enumerate(proposal_raw)]
    audit_messages_batch = [audit_messages(proposals[index], PROPOSAL_SEEDS[index], HISTORIES[index // 2]) for index in range(10)]
    audit_adapter = audit_factory(audit_messages_batch)
    audit_raw = audit_adapter.complete(audit_messages_batch, AUDIT_SEEDS, response_format=audit_response_format(), max_tokens=AUDIT_MAX_TOKENS)
    raw["audit"] = audit_raw; checkpoint(raw_path, raw)
    audits = [parse_audit(response) for response in audit_raw]
    semantic, semantic_gates = diagnostics(proposals, audits)
    adapters = (proposal_adapter, audit_adapter); phase_usage = [usage(adapter) for adapter in adapters]
    totals = {key: sum(row[key] for row in phase_usage) for key in phase_usage[0]}
    records = [record for adapter in adapters for record in adapter.records()]
    transport_gates = {
        "exact_20_accepted_and_http": totals["accepted_requests"] == totals["http_attempts"] == 20,
        "zero_retries_reasoning_forced": totals["retries"] == totals["reasoning_tokens"] == totals["forced_exits"] == totals["forced_final_requests"] == 0,
        "all_clean_stops": len(records) == 20 and all(row["finish_reasons"] == ["stop"] for row in records),
        "exact_models_and_seeds": {row["seed"] for row in records} == set(PROPOSAL_SEEDS + AUDIT_SEEDS) and all(row["model_requested"] == row["model_returned"] == (PROPOSAL_MODEL_ID if row["seed"] in PROPOSAL_SEEDS else AUDIT_MODEL_ID) for row in records),
        "within_stage_cap": totals["cost_usd"] <= STAGE_CAP_USD + 1e-12,
    }
    gates = {**transport_gates, **semantic_gates}
    passed = all(gates.values())
    result = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "mechanics_pass" if passed else "mechanics_failed_closed", "decision": "freeze_fresh_scientific_protocol" if passed else "close_exact_bitmask_interface", "authorizes": "separate_protocol_only" if passed else "nothing", "protocol_sha256": PROTOCOL_SHA256, "raw_response_sha256": file_digest(raw_path), "semantic": semantic, "transport": {"phases": phase_usage, "totals": totals, "requests": records}, "gates": gates, "number_game_targets_opened": False, "policy_endpoints_opened": False}
    checkpoint(output_dir / "LABEL_FREE_RESULT.json", result)
    return result
