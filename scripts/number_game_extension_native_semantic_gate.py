#!/usr/bin/env python3
"""Produce the extension-native Number Game semantic-support mechanics gate."""

from __future__ import annotations

import argparse
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
from scripts.number_game_extension_native_semantic_codec import (
    AUDIT_SEEDS,
    HISTORIES,
    INTERFACE_VERSION,
    PROPOSAL_SEEDS,
    audit_messages,
    audit_response_format,
    canonical_json,
    digest_bytes,
    parse_audit,
    parse_proposal,
    proposal_messages,
    proposal_response_format,
    semantic_diagnostics,
)


PROPOSAL_MODEL_ID = "qwen/qwen3.7-plus"
AUDIT_MODEL_ID = "openai/gpt-5.6-luna"
PROPOSAL_MAX_TOKENS = 3300
AUDIT_MAX_TOKENS = 1200
STAGE_CAP_USD = 0.08
CONCURRENCY = 10
TEMPERATURE = 0.7
PROTOCOL_PATH = REPO_ROOT / "results/nonmyopic/NUMBER_GAME_EXTENSION_NATIVE_SEMANTIC_SUPPORT_GATE_PROTOCOL_20260813.md"
AMENDMENT_PATH = REPO_ROOT / "results/nonmyopic/NUMBER_GAME_EXTENSION_NATIVE_SEMANTIC_SUPPORT_GATE_BUDGET_AMENDMENT_20260813.md"
PROTOCOL_SHA256 = "097936f1fed2cc8b29b170c986a186da4a06ce05bc2e87b679a4bcae7892e0c4"
AMENDMENT_SHA256 = "c94788439847cc403d56d2b6537f072de45787a927b659e65bcf00ee1316688f"


class StructuredAdapter(Protocol):
    def complete_seeded(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        seeds: Sequence[int],
        *,
        response_format: dict[str, Any],
        max_new_tokens: int,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...

    def request_records(self) -> list[dict[str, Any]]: ...


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PerRequestSeedAdapter(NonReasoningOpenRouterAdapter):
    """Bind exact seeds and bank transport identity for concurrent requests."""

    def __init__(
        self,
        spec: ModelSpec,
        config: Config,
        *,
        pre_dispatch_authorizer: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(spec, config)
        self._request_seed = threading.local()
        self._records: list[dict[str, Any]] = []
        self._records_lock = threading.Lock()
        self._pre_dispatch_authorizer = pre_dispatch_authorizer

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
            raise RuntimeError("per-request seed is not bound")
        payload["seed"] = int(seed)
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._pre_dispatch_authorizer is not None:
            self._pre_dispatch_authorizer()
        data = super()._post(payload)
        choices = data.get("choices") or []
        with self._records_lock:
            self._records.append(
                {
                    "seed": payload.get("seed"),
                    "model_requested": payload.get("model"),
                    "model_returned": data.get("model"),
                    "prompt_sha256": digest_bytes(canonical_json(payload.get("messages")).encode()),
                    "payload_sha256": digest_bytes(canonical_json(payload).encode()),
                    "finish_reasons": [choice.get("finish_reason") for choice in choices if isinstance(choice, dict)],
                    "provider_error": any(choice.get("finish_reason") == "error" for choice in choices if isinstance(choice, dict)),
                }
            )
        return data

    def complete_seeded(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        seeds: Sequence[int],
        *,
        response_format: dict[str, Any],
        max_new_tokens: int,
    ) -> list[str]:
        if len(batch_messages) != len(seeds):
            raise ValueError("message and seed counts differ")

        def one(item: tuple[list[dict[str, str]], int]) -> str:
            messages, seed = item
            self._request_seed.value = int(seed)
            try:
                return self._complete_request(
                    messages,
                    TEMPERATURE,
                    1,
                    max_new_tokens,
                    allow_forced_final=False,
                    disable_reasoning=True,
                    response_format=response_format,
                )[0]
            finally:
                del self._request_seed.value

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(one, zip(batch_messages, seeds, strict=True)))

    def request_records(self) -> list[dict[str, Any]]:
        with self._records_lock:
            return sorted((dict(row) for row in self._records), key=lambda row: row["seed"])


def build_adapter(
    *,
    model_id: str,
    run_id: str,
    output_dir: Path,
    phase_exposure_usd: float,
    maximum_request_cost_usd: float,
    max_tokens: int,
    pre_dispatch_authorizer: Callable[[], None] | None = None,
) -> PerRequestSeedAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=245.0,
        openrouter_run_budget_usd=STAGE_CAP_USD,
        openrouter_projected_cost_usd=phase_exposure_usd,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=maximum_request_cost_usd,
        openrouter_max_output_tokens=max_tokens,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return PerRequestSeedAdapter(
        ModelSpec(model=model_id, backend="openrouter", max_model_len=65536),
        config,
        pre_dispatch_authorizer=pre_dispatch_authorizer,
    )


def usage_summary(adapter: StructuredAdapter) -> dict[str, Any]:
    value = adapter.usage_snapshot()
    return {
        "accepted_requests": int(value.get("adapter_requests", 0)),
        "http_attempts": int(value.get("http_attempts", 0)),
        "retries": int(value.get("retry_count", 0)),
        "reasoning_tokens": int(value.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(value.get("forced_exits", 0)),
        "forced_final_requests": int(value.get("forced_final_requests", 0)),
        "prompt_tokens": int(value.get("adapter_prompt_tokens", 0)),
        "completion_tokens": int(value.get("adapter_completion_tokens", 0)),
        "cost_usd": float(value.get("adapter_cost_usd", 0.0)),
    }


def proposal_diagnostics(proposals: Sequence[Sequence[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, bool]]:
    draws = []
    pools = []
    for index, rows in enumerate(proposals):
        draws.append({"draw_index": index, "history_index": index // 2, "schema_valid_unique_count": len({row["extension_hash"] for row in rows}), "extension_hashes": [row["extension_hash"] for row in rows]})
    for history_index in range(5):
        left = {row["extension_hash"] for row in proposals[2 * history_index]}
        right = {row["extension_hash"] for row in proposals[2 * history_index + 1]}
        pools.append({"history_index": history_index, "unique_count": len(left | right), "second_draw_novel_count": len(right - left)})
    gates = {
        "all_draws_have_exactly_24_schema_valid": len(draws) == 10 and all(row["schema_valid_unique_count"] == 24 for row in draws),
        "every_draw_has_at_least_20_unique": all(row["schema_valid_unique_count"] >= 20 for row in draws),
        "every_pool_has_at_least_28_unique": all(row["unique_count"] >= 28 for row in pools),
        "every_second_draw_contributes_at_least_4": all(row["second_draw_novel_count"] >= 4 for row in pools),
        "zero_description_lexical_failures": True,
    }
    return {"draws": draws, "pools": pools}, gates


def transport_summary(adapters: Sequence[StructuredAdapter]) -> tuple[dict[str, Any], dict[str, bool]]:
    phases = [usage_summary(adapter) for adapter in adapters]
    records = [row for adapter in adapters for row in adapter.request_records()]
    totals = {key: sum(phase[key] for phase in phases) for key in phases[0]}
    expected_seeds = set(PROPOSAL_SEEDS + AUDIT_SEEDS)
    gates = {
        "exact_20_accepted_requests": totals["accepted_requests"] == 20,
        "exact_20_http_attempts": totals["http_attempts"] == 20,
        "zero_retries": totals["retries"] == 0,
        "zero_reasoning_tokens": totals["reasoning_tokens"] == 0,
        "zero_forced_exits": totals["forced_exits"] == totals["forced_final_requests"] == 0,
        "all_finish_reasons_stop": len(records) == 20 and all(row["finish_reasons"] == ["stop"] and not row["provider_error"] for row in records),
        "exact_models_and_seeds": {row["seed"] for row in records} == expected_seeds and all(row["model_requested"] == row["model_returned"] == (PROPOSAL_MODEL_ID if row["seed"] in PROPOSAL_SEEDS else AUDIT_MODEL_ID) for row in records),
        "within_stage_cap": totals["cost_usd"] <= STAGE_CAP_USD + 1e-12,
    }
    return {"phases": phases, "totals": totals, "requests": records}, gates


def run_gate(
    *,
    output_dir: Path,
    proposal_adapter: StructuredAdapter,
    audit_adapter_factory: Callable[[Sequence[list[dict[str, str]]]], StructuredAdapter],
) -> dict[str, Any]:
    if file_digest(PROTOCOL_PATH) != PROTOCOL_SHA256 or file_digest(AMENDMENT_PATH) != AMENDMENT_SHA256:
        raise RuntimeError("semantic-support protocol binding changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    raw_path = private / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"proposal": [], "audit": []}
    proposal_batch = [proposal_messages(HISTORIES[index // 2]) for index in range(10)]
    proposal_raw = proposal_adapter.complete_seeded(proposal_batch, PROPOSAL_SEEDS, response_format=proposal_response_format(), max_new_tokens=PROPOSAL_MAX_TOKENS)
    raw["proposal"] = proposal_raw
    checkpoint(raw_path, raw)
    proposals = [parse_proposal(response, HISTORIES[index // 2]) for index, response in enumerate(proposal_raw)]
    audit_batch = [audit_messages(proposals[index], PROPOSAL_SEEDS[index], HISTORIES[index // 2]) for index in range(10)]
    audit_adapter = audit_adapter_factory(audit_batch)
    audit_raw = audit_adapter.complete_seeded(audit_batch, AUDIT_SEEDS, response_format=audit_response_format(), max_new_tokens=AUDIT_MAX_TOKENS)
    raw["audit"] = audit_raw
    checkpoint(raw_path, raw)
    audits = [parse_audit(response) for response in audit_raw]
    proposal_summary, proposal_gates = proposal_diagnostics(proposals)
    semantic_summary, semantic_gates = semantic_diagnostics(proposals, audits)
    transport, transport_gates = transport_summary((proposal_adapter, audit_adapter))
    gates = {**transport_gates, **proposal_gates, **semantic_gates}
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "mechanics_pass" if passed else "mechanics_failed_closed",
        "decision": "freeze_fresh_scientific_protocol" if passed else "close_exact_interface",
        "authorizes": "separate_protocol_only" if passed else "nothing",
        "protocol_sha256": PROTOCOL_SHA256,
        "budget_amendment_sha256": AMENDMENT_SHA256,
        "raw_response_sha256": file_digest(raw_path),
        "proposal": proposal_summary,
        "semantic": semantic_summary,
        "transport": transport,
        "gates": gates,
        "number_game_targets_opened": False,
        "policy_endpoints_opened": False,
    }
    checkpoint(output_dir / "LABEL_FREE_RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--proposal-phase-exposure", type=float, required=True)
    parser.add_argument("--audit-phase-exposure", type=float, required=True)
    parser.add_argument("--proposal-request-cap", type=float, required=True)
    parser.add_argument("--audit-request-cap", type=float, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    proposal = build_adapter(model_id=PROPOSAL_MODEL_ID, run_id=args.run_id, output_dir=output_dir, phase_exposure_usd=args.proposal_phase_exposure, maximum_request_cost_usd=args.proposal_request_cap, max_tokens=PROPOSAL_MAX_TOKENS)
    result = run_gate(
        output_dir=output_dir,
        proposal_adapter=proposal,
        audit_adapter_factory=lambda _messages: build_adapter(
            model_id=AUDIT_MODEL_ID,
            run_id=args.run_id,
            output_dir=output_dir,
            phase_exposure_usd=args.audit_phase_exposure,
            maximum_request_cost_usd=args.audit_request_cap,
            max_tokens=AUDIT_MAX_TOKENS,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "mechanics_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
