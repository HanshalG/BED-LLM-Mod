#!/usr/bin/env python3
"""Smoke-test path-dependent semantic support for LongVid four-hop search."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.bright_biology_unlock_audit import tokenize


INTERFACE_VERSION = "longvid-four-hop-semantic-support-smoke-1"
MODEL_ID = "openai/gpt-5.4"
SUPPORT_SIZE = 6
BRANCH_COUNT = 8
EXPECTED_REQUESTS = 10
MAX_COST_USD = 0.20
PROJECTED_COST_USD = 0.08
MIN_VALID_ANCHORS_PER_BRANCH = 4
MIN_UNIQUE_BRANCH_SUPPORTS = 6

FIXTURE = {
    "question": (
        "Which artifact ultimately let Mira signal the stranded survey team, "
        "and how did she obtain it?"
    ),
    "branches": [
        {
            "root_query": "Mira artifact signal survey team",
            "observation": (
                "Inside the flooded archive, Mira found an astrolabe wrapped "
                "in oilcloth beside a ledger naming the hill observatory."
            ),
        },
        {
            "root_query": "artifact used to signal stranded team",
            "observation": (
                "A ferryman traded Mira a cinnabar token after she repaired "
                "the cracked winch on his barge."
            ),
        },
        {
            "root_query": "how Mira obtained signaling artifact",
            "observation": (
                "The abandoned workshop contained a heliograph mount, but its "
                "silvered mirror had been removed by the cartographer."
            ),
        },
        {
            "root_query": "Mira stranded survey rescue object",
            "observation": (
                "A weather journal described alpenglow flashing from the "
                "northern ridge whenever the old beacon was aligned."
            ),
        },
        {
            "root_query": "survey team communication artifact",
            "observation": (
                "At the quarry, a mason said the brass prism was exchanged for "
                "a spool of climbing cord at the eastern depot."
            ),
        },
        {
            "root_query": "Mira signal artifact origin",
            "observation": (
                "The depot inventory listed a semaphore lens delivered by a "
                "botanist who had crossed the cedar ravine."
            ),
        },
        {
            "root_query": "object rescued stranded survey team",
            "observation": (
                "Near the ravine, Mira discovered a caliper engraved with the "
                "same crest shown on the observatory ledger."
            ),
        },
        {
            "root_query": "Mira team signaling chain",
            "observation": (
                "A shepherd recalled that a phosphor flare was stored beneath "
                "the ruined viaduct after the midsummer expedition."
            ),
        },
    ],
}


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


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def fixture_sha256() -> str:
    return hashlib.sha256(_canonical(FIXTURE)).hexdigest()


def parse_support(response: str) -> list[dict[str, Any]]:
    lines = response.splitlines()
    if len(lines) != SUPPORT_SIZE:
        raise ValueError(f"response must contain exactly {SUPPORT_SIZE} lines")
    records = []
    for index, line in enumerate(lines, start=1):
        fields = line.split("|")
        if len(fields) != 5:
            raise ValueError(f"H{index} must contain exactly five pipe fields")
        label, raw_weight, anchor, query, hypothesis = fields
        if label != f"H{index}":
            raise ValueError(f"expected H{index}, got {label!r}")
        if not raw_weight.isdigit() or not 1 <= int(raw_weight) <= 100:
            raise ValueError(f"H{index} weight must be an integer from 1 to 100")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", anchor):
            raise ValueError(f"H{index} anchor must be one token")
        query = " ".join(query.split())
        hypothesis = " ".join(hypothesis.split())
        if not 3 <= len(query) <= 180:
            raise ValueError(f"H{index} search query has invalid length")
        if not 10 <= len(hypothesis) <= 280:
            raise ValueError(f"H{index} hypothesis has invalid length")
        records.append(
            {
                "label": label,
                "weight": int(raw_weight),
                "anchor": anchor,
                "search_query": query,
                "hypothesis": hypothesis,
            }
        )
    if len({_normalize(row["search_query"]) for row in records}) != SUPPORT_SIZE:
        raise ValueError("search queries must be distinct")
    if len({_normalize(row["hypothesis"]) for row in records}) != SUPPORT_SIZE:
        raise ValueError("hypotheses must be distinct")
    if len({row["weight"] for row in records}) < 2:
        raise ValueError("support must use at least two distinct weights")
    return records


def _format_support(support: list[dict[str, Any]]) -> str:
    return "\n".join(
        (
            f"{row['label']}|{row['weight']}|{row['anchor']}|"
            f"{row['search_query']}|{row['hypothesis']}"
        )
        for row in support
    )


def initial_messages(*, preflight: bool = False) -> list[dict[str, str]]:
    question = (
        "Which key opens the observatory, and where was it found?"
        if preflight
        else FIXTURE["question"]
    )
    return [
        {
            "role": "system",
            "content": (
                "You maintain a semantic belief support over possible multi-clip "
                "evidence chains for video retrieval. Do not reason aloud. Return "
                "exactly six lines and no other text. Each line must be "
                "Hn|weight|anchor|search_query|hypothesis. Labels are H1 through "
                "H6 in order. weight is an integer 1..100 and at least two weights "
                "must differ. anchor must be QUESTION. search_query is concise, "
                "hypothesis states a distinct plausible chain of unseen evidence. "
                "Do not use the pipe character inside fields."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate six distinct weighted evidence-chain hypotheses and six "
                f"distinct first searches for this question:\n{question}"
            ),
        },
    ]


def refresh_messages(
    initial: list[dict[str, Any]],
    branch: dict[str, str],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You update a semantic belief support after one retrieved video "
                "caption. Do not reason aloud. Return exactly six lines and no "
                "other text. Each line must be "
                "Hn|weight|anchor|search_query|hypothesis. Labels are H1 through "
                "H6 in order. weight is an integer 1..100 and at least two weights "
                "must differ. For every line, anchor must be one exact alphanumeric "
                "token copied from OBSERVATION, absent from QUESTION and "
                "PREVIOUS_QUERY. Each search_query must use its anchor and seek "
                "the next missing evidence link. Hypotheses must be distinct and "
                "updated from the observation. Do not use pipes inside fields."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{FIXTURE['question']}\n"
                f"PREVIOUS_QUERY:\n{branch['root_query']}\n"
                f"OBSERVATION:\n{branch['observation']}\n"
                f"INITIAL_SUPPORT:\n{_format_support(initial)}"
            ),
        },
    ]


def valid_anchor_count(
    support: list[dict[str, Any]],
    branch: dict[str, str],
) -> int:
    observation = set(tokenize(branch["observation"]))
    excluded = set(tokenize(FIXTURE["question"])).union(
        tokenize(branch["root_query"])
    )
    return sum(
        _normalize(row["anchor"]) in observation
        and _normalize(row["anchor"]) not in excluded
        and _normalize(row["anchor"]) in set(tokenize(row["search_query"]))
        for row in support
    )


def support_signature(support: list[dict[str, Any]]) -> str:
    payload = [
        {
            "weight": row["weight"],
            "query": _normalize(row["search_query"]),
            "hypothesis": _normalize(row["hypothesis"]),
        }
        for row in support
    ]
    return hashlib.sha256(_canonical(payload)).hexdigest()


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


def run_smoke(
    config: Config,
    *,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "fixture": FIXTURE,
        "fixture_sha256": fixture_sha256(),
        "preflight_discarded": False,
    }
    try:
        preflight_responses = model.chat_complete_messages_batched(
            [initial_messages(preflight=True)],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=900,
        )
        raw["preflight_response"] = preflight_responses[0]
        _checkpoint(raw_path, raw)
        parse_support(preflight_responses[0])
        raw["preflight_discarded"] = True
        _checkpoint(raw_path, raw)

        initial_responses = model.chat_complete_messages_batched(
            [initial_messages()],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=900,
        )
        raw["initial_response"] = initial_responses[0]
        _checkpoint(raw_path, raw)
        initial = parse_support(initial_responses[0])

        refresh_responses = model.chat_complete_messages_batched(
            [
                refresh_messages(initial, branch)
                for branch in FIXTURE["branches"]
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=900,
        )
        raw["refresh_responses"] = refresh_responses
        _checkpoint(raw_path, raw)
        refreshes = [parse_support(response) for response in refresh_responses]
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    initial_signature = support_signature(initial)
    refresh_signatures = [support_signature(value) for value in refreshes]
    valid_anchors = [
        valid_anchor_count(support, branch)
        for support, branch in zip(refreshes, FIXTURE["branches"])
    ]
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_ten_supports_parse": True,
        "preflight_discarded": raw["preflight_discarded"],
        "initial_support_has_six_distinct_queries_and_hypotheses": True,
        "every_branch_has_four_valid_observation_anchors": all(
            count >= MIN_VALID_ANCHORS_PER_BRANCH for count in valid_anchors
        ),
        "all_branch_supports_change_from_initial": all(
            signature != initial_signature for signature in refresh_signatures
        ),
        "at_least_six_unique_branch_supports": (
            len(set(refresh_signatures)) >= MIN_UNIQUE_BRANCH_SUPPORTS
        ),
        "cost_at_most_0_20": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "support_size": SUPPORT_SIZE,
            "branch_count": BRANCH_COUNT,
            "expected_requests": EXPECTED_REQUESTS,
            "fixture_sha256": fixture_sha256(),
            "repairs_or_reissues": 0,
            "scientific_longvid_task_accessed": False,
            "max_cost_usd": MAX_COST_USD,
        },
        "diagnostics": {
            "valid_anchor_counts": valid_anchors,
            "initial_support_signature": initial_signature,
            "unique_branch_support_count": len(set(refresh_signatures)),
            "branch_support_signatures": refresh_signatures,
        },
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
            user = messages[-1]["content"]
            observation_match = re.search(
                r"OBSERVATION:\n(.+?)\nINITIAL_SUPPORT:",
                user,
                flags=re.DOTALL,
            )
            if observation_match:
                terms = [
                    token
                    for token in tokenize(observation_match.group(1))
                    if token not in tokenize(FIXTURE["question"])
                ]
                anchor = terms[0]
                suffix = self.requests + len(responses)
            else:
                anchor = "QUESTION"
                suffix = self.requests + len(responses)
            lines = [
                (
                    f"H{index}|{10 + index}|{anchor}|"
                    f"{anchor} fixture search {suffix} {index}|"
                    f"Distinct fixture evidence chain {suffix} branch {index}"
                )
                for index in range(1, SUPPORT_SIZE + 1)
            ]
            responses.append("\n".join(lines))
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
        raise ValueError("LongVid support smoke config selects the wrong model")
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
    config.openrouter_concurrency = EXPECTED_REQUESTS
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = 900
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
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

