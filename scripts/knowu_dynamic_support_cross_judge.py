#!/usr/bin/env python3
"""Independently rescore frozen KnowU support-expansion mechanics."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import knowu_dynamic_support_mechanics as mechanics


INTERFACE_VERSION = "knowu-dynamic-support-cross-judge-1"
MODEL_ID = "google/gemma-4-26b-a4b-it"
SOURCE_PUBLIC_SHA256 = (
    "bf433f1ff1b6f9ec2acbf359f384922380dfa21b7a06ff4be65c60ec82e76039"
)
SOURCE_PRIVATE_SHA256 = (
    "0f12315cb22ca54cf923c9075b7397dc9cd38a814c7c5762120ec0928ef5cf97"
)
EXPECTED_SERVING_REQUESTS = 10
EXPECTED_CONFIRMATION_REQUESTS = 6
SERVING_MAX_COST_USD = 0.05
CONFIRMATION_MAX_COST_USD = 0.15
MINIMUM_PRESENCE_AGREEMENT = 0.80
MINIMUM_TARGET_GAIN = 8
TARGET_WORLD_ID = "T1W3"
TARGET_ROOTS = (0, 2)


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class CrossJudgeExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _usage(model: ChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(
            snapshot.get("adapter_reasoning_tokens", 0)
        ),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "adapter_cost_usd": float(
            snapshot.get("adapter_cost_usd", 0.0)
        ),
        "model": snapshot,
    }


def load_frozen_records(
    public_path: Path,
    private_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _sha256_file(public_path) != SOURCE_PUBLIC_SHA256:
        raise ValueError("KnowU public mechanics artifact hash changed")
    if _sha256_file(private_path) != SOURCE_PRIVATE_SHA256:
        raise ValueError("KnowU private mechanics artifact hash changed")
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    if public.get("status") != "passed":
        raise ValueError("KnowU source mechanics did not pass")
    fixtures = private["private_fixtures"]
    if len(fixtures) != EXPECTED_CONFIRMATION_REQUESTS:
        raise ValueError("KnowU source world count changed")

    records = []
    for world_index, (fixture, metric) in enumerate(
        zip(fixtures, public["world_metrics"])
    ):
        if fixture["world_id"] != metric["world_id"]:
            raise ValueError("KnowU public/private world order changed")
        initial = mechanics.parse_initial(
            private["initial_responses"][world_index]
        )
        refreshed = [
            mechanics.parse_refresh(
                private["refresh_responses"][
                    world_index * mechanics.QUESTION_COUNT + root_index
                ]
            )
            for root_index in range(mechanics.QUESTION_COUNT)
        ]
        records.append(
            {
                "world_id": fixture["world_id"],
                "task_id": fixture["task_id"],
                "task_kind": (
                    "buy_computer"
                    if "BuyComputer" in fixture["task_id"]
                    else "leave_notice"
                ),
                "goal_request": (
                    "帮我选择一台我喜欢的电脑，并帮我在我常用的购物平台下单购买。"
                    if "BuyComputer" in fixture["task_id"]
                    else "我今天身体不舒服，帮我跟组里说一声请个假。"
                ),
                "truth_packet": fixture["truth_packet"],
                "supports": [initial.hypotheses, *refreshed],
                "original_scores": tuple(metric["truth_support_scores"]),
                "original_presence": (
                    bool(metric["initial_truth_present"]),
                    *tuple(bool(value) for value in metric["branch_truth_present"]),
                ),
            }
        )
    return records, public


def _complete(
    model: ChatModel,
    messages: list[list[dict[str, str]]],
) -> list[str]:
    return model.chat_complete_messages_batched(
        messages,
        temperature=0.0,
        block_size=len(messages),
        max_new_tokens=1_100,
    )


def _synthetic_judge_messages(index: int) -> list[dict[str, str]]:
    truth = {
        "shopping": {
            "computer": "portable Linux laptop",
            "platform": "ExampleShop",
        }
    }
    supports = [
        (
            f"Synthetic state {index}-{support_index}-1 uses ExampleShop "
            "for a portable Linux laptop.",
            f"Synthetic state {index}-{support_index}-2 uses OtherShop.",
            f"Synthetic state {index}-{support_index}-3 prefers a desktop.",
            f"Synthetic state {index}-{support_index}-4 is unspecified.",
        )
        for support_index in range(5)
    ]
    return mechanics.judgment_messages(
        "Choose a synthetic computer.",
        "buy_computer",
        truth,
        supports,
    )


def run_serving_gate(
    model: ChatModel,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    raw: dict[str, Any] = {"interface_version": INTERFACE_VERSION}
    try:
        responses = _complete(
            model,
            [
                _synthetic_judge_messages(index)
                for index in range(EXPECTED_SERVING_REQUESTS)
            ],
        )
        raw["responses"] = responses
        _checkpoint(raw_path, raw)
        judgments = [
            mechanics.parse_judgment(response) for response in responses
        ]
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise CrossJudgeExecutionError(
            f"{type(exc).__name__}: {exc}", _usage(model)
        ) from exc

    gates = {
        "exact_10_physical_requests": (
            usage["physical_requests"] == EXPECTED_SERVING_REQUESTS
        ),
        "exact_10_http_attempts": (
            usage["http_attempts"] == EXPECTED_SERVING_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_10_judgments_parse": (
            len(judgments) == EXPECTED_SERVING_REQUESTS
        ),
        "cost_at_most_0_05": (
            usage["adapter_cost_usd"] <= SERVING_MAX_COST_USD
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "serving",
            "model": MODEL_ID,
            "expected_requests": EXPECTED_SERVING_REQUESTS,
            "reasoning_requested": False,
            "scientific_endpoint_evaluated": False,
            "repairs_or_reissues": 0,
        },
        "gates": gates,
        "usage": usage,
    }


def analyze_judgments(
    records: Sequence[dict[str, Any]],
    judgments: Sequence[mechanics.TruthJudgment],
    usage: dict[str, Any],
) -> dict[str, Any]:
    world_metrics = []
    agreements = 0
    total_states = 0
    for record, judgment in zip(records, judgments):
        cross_presence = judgment.present
        original_presence = tuple(record["original_presence"])
        agreements += sum(
            left == right
            for left, right in zip(cross_presence, original_presence)
        )
        total_states += len(cross_presence)
        initial_present = cross_presence[0]
        branch_present = cross_presence[1:]
        gains = tuple(
            score - judgment.best_scores[0]
            for score in judgment.best_scores[1:]
        )
        world_metrics.append(
            {
                "world_id": record["world_id"],
                "task_id": record["task_id"],
                "original_scores": record["original_scores"],
                "cross_scores": judgment.best_scores,
                "original_presence": original_presence,
                "cross_presence": cross_presence,
                "cross_score_gains": gains,
                "truth_entry_branches": [
                    index + 1
                    for index, present in enumerate(branch_present)
                    if not initial_present and present
                ],
            }
        )
    agreement = agreements / total_states
    missing = [
        row for row in world_metrics if not row["cross_presence"][0]
    ]
    entries = [row for row in missing if row["truth_entry_branches"]]
    target = next(
        row for row in world_metrics if row["world_id"] == TARGET_WORLD_ID
    )
    target_successful_roots = [
        index
        for index in TARGET_ROOTS
        if target["cross_presence"][index + 1]
    ]
    target_gain = max(
        target["cross_score_gains"][index] for index in TARGET_ROOTS
    )
    gates = {
        "exact_6_physical_requests": (
            usage["physical_requests"] == EXPECTED_CONFIRMATION_REQUESTS
        ),
        "exact_6_http_attempts": (
            usage["http_attempts"] == EXPECTED_CONFIRMATION_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_6_judgments_parse": len(judgments) == len(records) == 6,
        "cross_judge_finds_initial_missing_world": bool(missing),
        "cross_judge_finds_missing_truth_entry": bool(entries),
        "target_initial_truth_missing": not target["cross_presence"][0],
        "target_os_or_platform_root_recovers_truth": bool(
            target_successful_roots
        ),
        "target_os_or_platform_gain_at_least_8": (
            target_gain >= MINIMUM_TARGET_GAIN
        ),
        "presence_agreement_at_least_0_80": (
            agreement >= MINIMUM_PRESENCE_AGREEMENT
        ),
        "cost_at_most_0_15": (
            usage["adapter_cost_usd"] <= CONFIRMATION_MAX_COST_USD
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "confirmation",
            "model": MODEL_ID,
            "source_public_sha256": SOURCE_PUBLIC_SHA256,
            "source_private_sha256": SOURCE_PRIVATE_SHA256,
            "world_count": len(records),
            "support_states": total_states,
            "truth_match_threshold": mechanics.TRUTH_MATCH_THRESHOLD,
            "expected_requests": EXPECTED_CONFIRMATION_REQUESTS,
            "reasoning_requested": False,
            "regeneration_or_user_requests": 0,
            "repairs_or_reissues": 0,
        },
        "summary": {
            "presence_agreement": agreement,
            "cross_initial_missing_worlds": len(missing),
            "cross_worlds_with_truth_entry": len(entries),
            "target_successful_roots_zero_based": target_successful_roots,
            "target_max_os_or_platform_gain": target_gain,
        },
        "world_metrics": world_metrics,
        "gates": gates,
        "usage": usage,
    }


def run_confirmation(
    records: Sequence[dict[str, Any]],
    model: ChatModel,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "world_ids": [record["world_id"] for record in records],
    }
    try:
        responses = _complete(
            model,
            [
                mechanics.judgment_messages(
                    record["goal_request"],
                    record["task_kind"],
                    record["truth_packet"],
                    record["supports"],
                )
                for record in records
            ],
        )
        raw["responses"] = responses
        _checkpoint(raw_path, raw)
        judgments = [
            mechanics.parse_judgment(response) for response in responses
        ]
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise CrossJudgeExecutionError(
            f"{type(exc).__name__}: {exc}", _usage(model)
        ) from exc
    return analyze_judgments(records, judgments, usage)


class DeterministicCrossJudgeModel:
    def __init__(self, score_rows: Sequence[Sequence[int]]) -> None:
        self.score_rows = [tuple(row) for row in score_rows]
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
        labels = ("initial", "q1", "q2", "q3", "q4")
        for offset, _ in enumerate(batch_messages):
            scores = self.score_rows[
                (self.requests + offset) % len(self.score_rows)
            ]
            value: dict[str, Any] = {}
            for label, score in zip(labels, scores):
                value[f"{label}_best_index"] = (
                    1 if score >= mechanics.TRUTH_MATCH_THRESHOLD else 0
                )
                value[f"{label}_best_score"] = score
                value[f"{label}_reason"] = "Deterministic dry judgment."
            responses.append(json.dumps(value, separators=(",", ":")))
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


def _build_model(config: Config) -> ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        model=MODEL_ID,
        thinking=None,
        reasoning_effort=None,
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.backend != "openrouter":
        raise ValueError("KnowU cross-judge config is not OpenRouter")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("serving", "confirmation"), required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-public", type=Path, required=True)
    parser.add_argument("--source-private", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records, _ = load_frozen_records(
        args.source_public, args.source_private
    )
    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_concurrency = (
        EXPECTED_SERVING_REQUESTS
        if args.stage == "serving"
        else EXPECTED_CONFIRMATION_REQUESTS
    )
    config.openrouter_projected_cost_usd = (
        0.02 if args.stage == "serving" else 0.06
    )
    config.openrouter_run_budget_usd = (
        SERVING_MAX_COST_USD
        if args.stage == "serving"
        else CONFIRMATION_MAX_COST_USD
    )
    config.openrouter_max_output_tokens = 1_100
    config.openrouter_max_retries = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    if args.dry_run:
        score_rows = (
            [(90, 90, 90, 90, 90)] * EXPECTED_SERVING_REQUESTS
            if args.stage == "serving"
            else [record["original_scores"] for record in records]
        )
        model: ChatModel = DeterministicCrossJudgeModel(score_rows)
    else:
        model = _build_model(config)

    try:
        result = (
            run_serving_gate(model, raw_path=raw_path)
            if args.stage == "serving"
            else run_confirmation(records, model, raw_path=raw_path)
        )
        result["protocol"]["private_raw_sha256"] = _sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, CrossJudgeExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = _sha256_file(raw_path)
        _checkpoint(args.output_dir / "GATE_FAILURE.json", failure)
        raise
    output_path = args.output_dir / f"{args.stage.upper()}.json"
    _checkpoint(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_path),
                "summary": result.get("summary"),
                "gates": result["gates"],
                "usage": result["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
