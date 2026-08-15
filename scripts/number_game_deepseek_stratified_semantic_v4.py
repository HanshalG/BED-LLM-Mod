#!/usr/bin/env python3
"""Produce the frozen DeepSeek stratified semantic V4 response bank."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_stratified_atomic_particle_v3_codec import (
    parse_stratified,
    particle_signature,
    request_identity,
    response_format,
    signature_text,
)


INTERFACE_VERSION = "number-game-deepseek-stratified-semantic-v4-1"
PROTOCOL = REPO_ROOT / "results/nonmyopic/NUMBER_GAME_DEEPSEEK_STRATIFIED_SEMANTIC_V4_PROTOCOL_20260815.md"
PROTOCOL_SHA256 = "69de1f6c02b052d7081e8bffa191eaaee070c3d9e879d90250c885033ff3dd31"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
MODEL_SEEDS = tuple(range(202608210000, 202608210192))
MAX_TOKENS = 220
TEMPERATURE = 0.6
GROUPS = (
    ("initial", ()),
    ("one_step", ((2, True),)),
    ("two_step", ((2, True), (3, False))),
)
GROUP_SIZE = 64
STAGE_CAP_USD = 0.20


class Adapter(Protocol):
    def complete(self, messages, seeds, *, response_format, max_tokens): ...
    def usage_snapshot(self): ...
    def records(self): ...


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def group_requests(group_index: int) -> list[dict[str, Any]]:
    stage, history = GROUPS[group_index]
    protected = tuple(number for number, _ in history)
    offset = group_index * GROUP_SIZE
    rows = []
    for slot in range(GROUP_SIZE):
        anchors, signature, messages = request_identity(
            history,
            protected=protected,
            slot=slot,
            count=GROUP_SIZE,
        )
        rows.append({
            "stage": stage,
            "group_index": group_index,
            "slot": slot,
            "seed": MODEL_SEEDS[offset + slot],
            "observations": [[number, answer] for number, answer in history],
            "protected": list(protected),
            "anchors": list(anchors),
            "signature": signature_text(signature),
            "messages": messages,
        })
    return rows


def parse_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != GROUP_SIZE:
        raise ValueError("semantic group has wrong size")
    parsed = []
    for row in rows:
        history = tuple((int(number), bool(answer)) for number, answer in row["observations"])
        anchors = tuple(int(value) for value in row["anchors"])
        signature = tuple(bit == "1" for bit in row["signature"])
        parsed.append(parse_stratified(row["response"], history, anchors, signature))
    valid = [item.hypothesis for item in parsed if item.hypothesis is not None]
    rejected = Counter(item.rejection for item in parsed if item.rejection)
    extensions = Counter(item.extension for item in valid)
    signatures = Counter(
        particle_signature(item, tuple(int(value) for value in rows[0]["anchors"]))
        for item in valid
    )
    halves = []
    for parity in (0, 1):
        half = [parsed[index].hypothesis for index in range(parity, GROUP_SIZE, 2)]
        half_valid = [item for item in half if item is not None]
        halves.append({
            "parity": parity,
            "valid_particles": len(half_valid),
            "unique_extensions": len({item.extension for item in half_valid}),
            "unique_signatures": len({
                particle_signature(item, tuple(int(value) for value in rows[0]["anchors"]))
                for item in half_valid
            }),
        })
    diagnostic = {
        "stage": rows[0]["stage"],
        "slots": len(rows),
        "valid_particles": len(valid),
        "unique_extensions": len(extensions),
        "unique_signatures": len(signatures),
        "maximum_extension_multiplicity": max(extensions.values(), default=0),
        "rejections": dict(sorted(rejected.items())),
        "signature_counts": {
            signature_text(signature): count for signature, count in sorted(signatures.items())
        },
        "halves": halves,
    }
    group_gates = {
        "valid_at_least_48": diagnostic["valid_particles"] >= 48,
        "signatures_at_least_28": diagnostic["unique_signatures"] >= 28,
        "extensions_at_least_24": diagnostic["unique_extensions"] >= 24,
        "each_half_valid_at_least_24": all(row["valid_particles"] >= 24 for row in halves),
        "each_half_signatures_at_least_24": all(row["unique_signatures"] >= 24 for row in halves),
        "each_half_extensions_at_least_20": all(row["unique_extensions"] >= 20 for row in halves),
        "maximum_multiplicity_at_most_4": diagnostic["maximum_extension_multiplicity"] <= 4,
    }
    diagnostic["gates"] = group_gates
    diagnostic["passed"] = all(group_gates.values())
    return diagnostic


def transport(adapter: Adapter, accepted: int) -> dict[str, Any]:
    snapshot = adapter.usage_snapshot()
    records = list(adapter.records())
    totals = {
        "accepted_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retries": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "forced_final_requests": int(snapshot.get("forced_final_requests", 0)),
        "cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
    }
    gates = {
        "accepted_and_http_match_prefix": totals["accepted_requests"] == totals["http_attempts"] == accepted,
        "zero_retries_reasoning_forced": (
            totals["retries"]
            == totals["reasoning_tokens"]
            == totals["forced_exits"]
            == totals["forced_final_requests"]
            == 0
        ),
        "all_clean_stops": len(records) == accepted and all(row.get("finish_reasons") == ["stop"] for row in records),
        "exact_model_and_seed_prefix": (
            len(records) == accepted
            and {int(row.get("seed", -1)) for row in records} == set(MODEL_SEEDS[:accepted])
            and all(row.get("model_requested") == row.get("model_returned") == MODEL_ID for row in records)
        ),
        "within_stage_cap": totals["cost_usd"] <= STAGE_CAP_USD + 1e-12,
    }
    return {"totals": totals, "records": records, "gates": gates}


def produce(
    *,
    output_dir: Path,
    adapter: Adapter,
    block_authorizer: Callable[[Sequence[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("DeepSeek stratified semantic protocol changed")
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    raw_path = private / "RAW_RESPONSES.json"
    result_path = output_dir / "LABEL_FREE_RESULT.json"
    raw: dict[str, Any] = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "responses": [],
    }
    diagnostics = []
    for group_index in range(len(GROUPS)):
        requests = group_requests(group_index)
        if block_authorizer is not None:
            block_authorizer(requests)
        values = adapter.complete(
            [row["messages"] for row in requests],
            [row["seed"] for row in requests],
            response_format=response_format(),
            max_tokens=MAX_TOKENS,
        )
        if len(values) != GROUP_SIZE:
            raise RuntimeError("adapter omitted semantic responses")
        for request, response in zip(requests, values, strict=True):
            raw["responses"].append({
                **{key: value for key, value in request.items() if key != "messages"},
                "prompt_sha256": hashlib.sha256(canonical_json(request["messages"]).encode()).hexdigest(),
                "response": response,
            })
        checkpoint(raw_path, raw)
        diagnostic = parse_group(raw["responses"][-GROUP_SIZE:])
        diagnostics.append(diagnostic)
        if not diagnostic["passed"]:
            break
    accepted = len(raw["responses"])
    transport_result = transport(adapter, accepted)
    complete = accepted == len(MODEL_SEEDS)
    semantic_pass = complete and all(row["passed"] for row in diagnostics)
    transport_pass = all(transport_result["gates"].values()) and (
        accepted == len(MODEL_SEEDS) if semantic_pass else True
    )
    passed = semantic_pass and transport_pass
    result = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "semantic_pass" if passed else "semantic_null",
        "authorizes": "fresh_mechanics_protocol_only" if passed else "nothing",
        "protocol_sha256": PROTOCOL_SHA256,
        "raw_response_sha256": digest(raw_path),
        "accepted_requests": accepted,
        "complete_bank": complete,
        "groups": diagnostics,
        "transport": transport_result,
        "gates": {
            "all_three_groups_complete": complete,
            "all_group_semantic_gates_pass": semantic_pass,
            "transport_prefix_or_complete_valid": transport_pass,
            "canonical_targets_closed": True,
            "policy_endpoints_closed": True,
        },
        "canonical_targets_opened": False,
        "policy_endpoints_opened": False,
        "development_confirmation_opened": False,
    }
    checkpoint(result_path, result)
    return result
