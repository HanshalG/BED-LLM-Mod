#!/usr/bin/env python3
"""Smoke-test prompt-only LongVid belief and contrastive-rank transport."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.longvid_contrastive_path_belief_mechanics import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    SUPPORT_SIZE,
    _support_for_prompt,
    initial_messages,
    parse_rank,
    parse_support,
    refresh_messages,
    support_signature,
    valid_anchor_count,
)


INTERFACE_VERSION = "longvid-contrastive-prompt-serving-smoke-1"
EXPECTED_REQUESTS = 10
MAX_COST_USD = 0.20
PROJECTED_COST_USD = 0.08
QUESTION = (
    "How did the venue's early funding dispute eventually change the final "
    "community performance?"
)
OBSERVATIONS = (
    "The archivist describes a rejected Meridian grant and a delayed permit.",
    "A producer says the Lantern committee replaced the original sponsor.",
    "The rehearsal moved to Dockside after a transformer failure.",
    "A volunteer translated the Orpheus script for neighborhood performers.",
    "The director cut the Cascade scene after an accessibility review.",
    "Local students built the Juniper set from recycled material.",
    "Ticket income funded a Solstice matinee for displaced residents.",
    "The final Harborlight performance added a public bilingual chorus.",
)


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


def support_template() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for index in range(1, SUPPORT_SIZE + 1):
        payload[f"hypothesis_{index}"] = "distinct evidence-chain hypothesis"
        payload[f"weight_{index}"] = 10 + index
        payload[f"anchor_{index}"] = "QUESTION or one observation token"
        payload[f"query_{index}"] = "concise next search containing anchor"
    return payload


def rank_template() -> dict[str, Any]:
    return {
        "choice": "A or B",
        "confidence": 75,
        "unresolved_need": "short description of the decisive missing link",
    }


def prompt_only_support_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    result[-1]["content"] += (
        "\n\nReturn exactly one valid JSON object with every field shown here, "
        "no markdown or extra fields:\n"
        + json.dumps(support_template(), separators=(",", ":"))
    )
    return result


def prompt_only_rank_schema_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    result[-1]["content"] += (
        "\nReturn exactly one valid JSON object with every field shown here, "
        "no markdown or extra fields:\n"
        + json.dumps(rank_template(), separators=(",", ":"))
    )
    return result


def prompt_only_rank_messages(
    supports: list[list[dict[str, Any]]],
) -> list[dict[str, str]]:
    trajectories = []
    for candidate, branch_supports in zip(("A", "B"), supports, strict=True):
        trajectories.append(
            {
                "candidate": candidate,
                "queries": [
                    f"synthetic branch {candidate} query {index}"
                    for index in range(1, 5)
                ],
                "belief_supports": [
                    _support_for_prompt(support)
                    for support in branch_supports
                ],
            }
        )
    return prompt_only_rank_schema_messages(
        [
        {
            "role": "system",
            "content": (
                "Compare two blinded retrieval trajectories from their evolving "
                "semantic belief supports and queries. Choose the one more likely "
                "to expose a complete four-link evidence chain. Do not answer the "
                "question or reason aloud."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION={QUESTION}\n"
                "TRAJECTORIES="
                + json.dumps(trajectories, separators=(",", ":"))
            ),
        },
        ]
    )


def _usage(model: ChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "prompt_tokens": int(snapshot.get("adapter_prompt_tokens", 0)),
        "completion_tokens": int(snapshot.get("adapter_completion_tokens", 0)),
        "adapter_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "model": snapshot,
    }


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
        for offset, messages in enumerate(batch_messages):
            user = messages[-1]["content"]
            if "TRAJECTORIES=" in user:
                responses.append(
                    json.dumps(
                        {
                            "choice": "B",
                            "confidence": 80,
                            "unresolved_need": "fixture causal bridge",
                        },
                        separators=(",", ":"),
                    )
                )
                continue
            observation_match = re.search(
                r"OBSERVATION:\n(.+?)\nPREVIOUS_SUPPORT:",
                user,
                flags=re.DOTALL,
            )
            if observation_match:
                excluded = set(re.findall(r"[a-z0-9]+", QUESTION.casefold()))
                excluded.update(
                    re.findall(
                        r"[a-z0-9]+",
                        "synthetic initial venue funding query",
                    )
                )
                candidates = [
                    token
                    for token in re.findall(
                        r"[A-Za-z0-9]+", observation_match.group(1)
                    )
                    if token.casefold() not in excluded
                ]
                anchor = candidates[0]
            else:
                anchor = "QUESTION"
            suffix = self.requests + offset
            payload = {}
            for index in range(1, SUPPORT_SIZE + 1):
                payload[f"hypothesis_{index}"] = (
                    f"Fixture chain {suffix} hypothesis number {index}"
                )
                payload[f"weight_{index}"] = 10 + index + (suffix % 3)
                payload[f"anchor_{index}"] = anchor
                payload[f"query_{index}"] = (
                    f"{anchor} fixture next search {suffix} {index}"
                )
            responses.append(json.dumps(payload, separators=(",", ":")))
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def run_smoke(
    config: Config,
    *,
    raw_path: Path,
    model: ChatModel,
    support_message_builder: Callable[
        [list[dict[str, str]]], list[dict[str, str]]
    ] = prompt_only_support_messages,
    rank_message_builder: Callable[
        [list[list[dict[str, Any]]]], list[dict[str, str]]
    ] = prompt_only_rank_messages,
    support_parser: Callable[[str], list[dict[str, Any]]] = parse_support,
    rank_parser: Callable[[str], dict[str, Any]] = parse_rank,
    interface_version: str = INTERFACE_VERSION,
    response_format_name: str = "prompt_only_flat_json",
    max_transport_retries: int = 0,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "interface_version": interface_version,
        "synthetic_fixture_only": True,
        "responses": {"initial": None, "refreshes": [], "rank": None},
    }
    try:
        initial_response = model.chat_complete_messages_batched(
            [support_message_builder(initial_messages(QUESTION))],
            temperature=0.0,
            block_size=1,
            max_new_tokens=MAX_NEW_TOKENS,
        )[0]
        raw["responses"]["initial"] = initial_response
        _checkpoint(raw_path, raw)
        initial = support_parser(initial_response)
        if {row["anchor"] for row in initial} != {"QUESTION"}:
            raise ValueError("initial anchors are not exactly QUESTION")

        requests = [
            support_message_builder(
                refresh_messages(
                    QUESTION,
                    "synthetic initial venue funding query",
                    observation,
                    initial,
                )
            )
            for observation in OBSERVATIONS
        ]
        refresh_responses = model.chat_complete_messages_batched(
            requests,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        raw["responses"]["refreshes"] = refresh_responses
        _checkpoint(raw_path, raw)
        refreshes = [
            support_parser(response) for response in refresh_responses
        ]
        valid_anchor_counts = [
            valid_anchor_count(
                support,
                question=QUESTION,
                previous_query="synthetic initial venue funding query",
                observation=observation,
            )
            for support, observation in zip(
                refreshes, OBSERVATIONS, strict=True
            )
        ]

        rank_response = model.chat_complete_messages_batched(
            [
                rank_message_builder(
                    [refreshes[:4], refreshes[4:]]
                )
            ],
            temperature=0.0,
            block_size=1,
            max_new_tokens=MAX_NEW_TOKENS,
        )[0]
        raw["responses"]["rank"] = rank_response
        _checkpoint(raw_path, raw)
        rank = rank_parser(rank_response)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    initial_signature = support_signature(initial)
    refresh_signatures = [support_signature(support) for support in refreshes]
    changed = sum(
        signature != initial_signature for signature in refresh_signatures
    )
    unique = len(set(refresh_signatures))
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "http_attempts_within_cap": (
            EXPECTED_REQUESTS
            <= usage["http_attempts"]
            <= EXPECTED_REQUESTS + max_transport_retries
        ),
        "transport_retries_within_cap": (
            usage["retry_count"] <= max_transport_retries
        ),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_ten_outputs_parse": True,
        "all_eight_refreshes_change": changed == 8,
        "all_eight_refreshes_unique": unique == 8,
        "all_refreshes_have_four_grounded_anchors": all(
            count >= 4 for count in valid_anchor_counts
        ),
        "rank_choice_and_confidence_valid": (
            rank["choice"] in {"A", "B"}
            and 51 <= rank["confidence"] <= 100
        ),
        "cost_at_most_0_20": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": interface_version,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "synthetic_fixture_only": True,
            "expected_requests": EXPECTED_REQUESTS,
            "response_format": response_format_name,
            "semantic_repairs_or_reissues": 0,
            "max_transport_retries": max_transport_retries,
            "max_cost_usd": MAX_COST_USD,
        },
        "diagnostics": {
            "changed_refresh_count": changed,
            "unique_refresh_count": unique,
            "valid_anchor_counts": valid_anchor_counts,
            "rank": rank,
        },
        "gates": gates,
        "usage": usage,
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
        raise ValueError("LongVid serving config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 8
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = MAX_NEW_TOKENS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel | None = None
    try:
        model = DeterministicFixtureModel() if args.dry_run else _build_model(config)
        payload = run_smoke(config, raw_path=raw_path, model=model)
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
        elif model is not None:
            failure["usage"] = _usage(model)
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
                "diagnostics": payload["diagnostics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
