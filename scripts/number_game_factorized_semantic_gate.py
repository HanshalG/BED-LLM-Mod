#!/usr/bin/env python3
"""Produce factorized Number Game semantic-support mechanics."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from typing import Any, Callable, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_bitmask_semantic_gate import SeededAdapter
from scripts.number_game_factorized_semantic_codec import (
    AUDIT_SEEDS,
    HISTORIES,
    PROPOSAL_SEEDS,
    TRANSLATION_SEEDS,
    assert_audit_blind,
    assert_translation_blind,
    audit_messages,
    audit_response_format,
    diagnostics,
    merge_draw,
    parse_audit,
    parse_proposal,
    parse_translation,
    proposal_messages,
    proposal_response_format,
    translation_messages,
    translation_response_format,
)


INTERFACE_VERSION = "number-game-factorized-semantic-support-1"
PROPOSAL_MODEL_ID = "qwen/qwen3.7-plus"
TRANSLATION_MODEL_ID = "deepseek/deepseek-v4-flash-0731"
AUDIT_MODEL_ID = "openai/gpt-5.6-luna"
PROPOSAL_MAX_TOKENS = 1100
TRANSLATION_MAX_TOKENS = 1200
AUDIT_MAX_TOKENS = 1200
STAGE_CAP_USD = 0.08
PROTOCOL = REPO_ROOT / "results/nonmyopic/NUMBER_GAME_FACTORIZED_SEMANTIC_SUPPORT_GATE_PROTOCOL_20260813.md"
PROTOCOL_SHA256 = "b8f63ecf4aaaf981655c2436471aa7dbe2c29cff70537fb1c90a8b05c7d30473"


class Adapter(Protocol):
    def complete(self, messages, seeds, *, response_format, max_tokens): ...
    def usage_snapshot(self): ...
    def records(self): ...


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PhaseAdapter(SeededAdapter):
    def __init__(self, *args, phase_temperature: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.phase_temperature = phase_temperature

    def complete(self, messages, seeds, *, response_format, max_tokens):
        if len(messages) != len(seeds):
            raise ValueError("message and seed counts differ")
        def one(item):
            prompt, seed = item
            self._seed.value = int(seed)
            try:
                return self._complete_request(prompt, self.phase_temperature, 1, max_tokens, allow_forced_final=False, disable_reasoning=True, response_format=response_format)[0]
            finally:
                del self._seed.value
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(one, zip(messages, seeds, strict=True)))


def build_adapter(*, model, run_id, output_dir, phase_exposure, request_cap, max_tokens, temperature, authorize=None) -> PhaseAdapter:
    from helpers import Config, ModelSpec
    config = Config(task="animals", run_id=run_id, log_path=output_dir / "run.log", openrouter_budget_usd=245.0, openrouter_run_budget_usd=STAGE_CAP_USD, openrouter_projected_cost_usd=phase_exposure, openrouter_concurrency=30, openrouter_max_retries=0, openrouter_backoff_seconds=1.0, openrouter_request_timeout_seconds=300.0, openrouter_max_request_cost_usd=request_cap, openrouter_max_output_tokens=max_tokens, openrouter_spend_path="results/path_e/openrouter_spend.json")
    return PhaseAdapter(ModelSpec(model=model, backend="openrouter", max_model_len=65536), config, authorize=authorize, phase_temperature=temperature)


def usage(adapter: Adapter) -> dict[str, Any]:
    value = adapter.usage_snapshot()
    return {
        "accepted_requests": int(value.get("adapter_requests", 0)),
        "http_attempts": int(value.get("http_attempts", 0)),
        "retries": int(value.get("retry_count", 0)),
        "reasoning_tokens": int(value.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(value.get("forced_exits", 0)),
        "forced_final_requests": int(value.get("forced_final_requests", 0)),
        "cost_usd": float(value.get("adapter_cost_usd", 0.0)),
    }


def public_semantic(summary: dict[str, Any]) -> dict[str, Any]:
    allowed = {"total_judgments", "agreement_count", "agreement_rate", "draws", "pools", "observed_history_novel_counts"}
    if set(summary) != allowed:
        raise RuntimeError("semantic summary shape changed")
    forbidden = {"name", "description", "mask", "membership_mask", "observations", "answer"}
    def walk(value):
        if isinstance(value, dict):
            if set(value) & forbidden:
                raise RuntimeError("public semantic summary leaks private content")
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(summary)
    return summary


def run_gate(
    *,
    output_dir: Path,
    proposal_adapter: Adapter,
    translation_factory: Callable[[Sequence[list[dict[str, str]]]], Adapter],
    audit_factory: Callable[[Sequence[list[dict[str, str]]]], Adapter],
) -> dict[str, Any]:
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("factorized protocol changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    raw_path = private / "RAW_RESPONSES.json"
    raw = {"proposal": [], "translation": [], "audit": []}

    proposal_batch = [proposal_messages(HISTORIES[draw // 2], shard) for draw in range(10) for shard in range(3)]
    proposal_raw = proposal_adapter.complete(proposal_batch, PROPOSAL_SEEDS, response_format=proposal_response_format(), max_tokens=PROPOSAL_MAX_TOKENS)
    raw["proposal"] = proposal_raw
    checkpoint(raw_path, raw)
    proposals = [parse_proposal(response) for response in proposal_raw]

    translation_batch = [translation_messages(rows) for rows in proposals]
    for messages in translation_batch:
        assert_translation_blind(messages)
    translation_adapter = translation_factory(translation_batch)
    translation_raw = translation_adapter.complete(translation_batch, TRANSLATION_SEEDS, response_format=translation_response_format(), max_tokens=TRANSLATION_MAX_TOKENS)
    raw["translation"] = translation_raw
    checkpoint(raw_path, raw)
    translated = [parse_translation(response, proposal) for response, proposal in zip(translation_raw, proposals, strict=True)]
    draws = [merge_draw(translated[3 * draw : 3 * draw + 3], HISTORIES[draw // 2])[0] for draw in range(10)]

    audit_batch = [audit_messages(draws[draw], draw, HISTORIES[draw // 2]) for draw in range(10)]
    for messages in audit_batch:
        assert_audit_blind(messages)
    audit_adapter = audit_factory(audit_batch)
    audit_raw = audit_adapter.complete(audit_batch, AUDIT_SEEDS, response_format=audit_response_format(), max_tokens=AUDIT_MAX_TOKENS)
    raw["audit"] = audit_raw
    checkpoint(raw_path, raw)
    semantic, semantic_gates = diagnostics(draws, [parse_audit(response) for response in audit_raw])

    adapters = (proposal_adapter, translation_adapter, audit_adapter)
    phases = [usage(adapter) for adapter in adapters]
    totals = {key: sum(row[key] for row in phases) for key in phases[0]}
    records = [record for adapter in adapters for record in adapter.records()]
    transport_gates = {
        "exact_70_accepted_and_http": totals["accepted_requests"] == totals["http_attempts"] == 70,
        "zero_retries_reasoning_forced": totals["retries"] == totals["reasoning_tokens"] == totals["forced_exits"] == totals["forced_final_requests"] == 0,
        "all_clean_stops": len(records) == 70 and all(row["finish_reasons"] == ["stop"] for row in records),
        "exact_models_and_seeds": {row["seed"] for row in records} == set(PROPOSAL_SEEDS + TRANSLATION_SEEDS + AUDIT_SEEDS) and all(row["model_requested"] == row["model_returned"] == (PROPOSAL_MODEL_ID if row["seed"] in PROPOSAL_SEEDS else TRANSLATION_MODEL_ID if row["seed"] in TRANSLATION_SEEDS else AUDIT_MODEL_ID) for row in records),
        "within_stage_cap": totals["cost_usd"] <= STAGE_CAP_USD + 1e-12,
    }
    gates = {**transport_gates, **semantic_gates}
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "mechanics_pass" if passed else "mechanics_failed_closed",
        "decision": "freeze_fresh_scientific_protocol" if passed else "close_exact_factorized_interface",
        "authorizes": "separate_protocol_only" if passed else "nothing",
        "protocol_sha256": PROTOCOL_SHA256,
        "raw_response_sha256": digest(raw_path),
        "semantic": public_semantic(semantic),
        "transport": {"phases": phases, "totals": totals, "requests": records},
        "gates": gates,
        "targets_opened": False,
        "endpoints_opened": False,
    }
    checkpoint(output_dir / "LABEL_FREE_RESULT.json", result)
    return result
