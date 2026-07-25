#!/usr/bin/env python3
"""Run the one-call ClariQ dynamic-support V2 serving gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.clariq_dynamic_support_smoke import (
    DeterministicFixtureModel,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
    TEMPERATURE,
    _question_payload,
    entropy,
    information_gain,
    parse_support,
    support_messages,
)
from scripts.clariq_multisample_likelihood_development import (
    _build_model,
    _checkpoint,
    _usage,
)


INTERFACE_VERSION = "clariq-dynamic-support-v2-serving-1"
MANIFEST_SHA256 = (
    "fa5a34e55ab455359a4a64bd2aba00ea5789f27fb2f2d932ea9e2330cf90ca03"
)
TOPIC_ID = "60"
EXPECTED_REQUESTS = 1
MAX_COST_USD = 0.05
PROJECTED_COST_USD = 0.02


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def load_manifest(path: Path) -> dict[str, Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("ClariQ V2 manifest hash changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] != "passed":
        raise ValueError("ClariQ V2 manifest did not pass")
    task = manifest["mechanics_task"]
    if task["topic_id"] != TOPIC_ID:
        raise ValueError("ClariQ V2 mechanics topic changed")
    if task["root_count"] != 15 or task["branch_count"] != 90:
        raise ValueError("ClariQ V2 task structure changed")
    if task["expected_model_requests"] != 91:
        raise ValueError("ClariQ V2 full-tree request count changed")
    return task


def run_serving(
    config: Config,
    *,
    manifest_path: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    task = load_manifest(manifest_path)
    question_ids = tuple(
        str(root["question_id"]) for root in task["roots"]
    )
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "manifest_sha256": MANIFEST_SHA256,
        "topic_id": TOPIC_ID,
        "branch_requests_sent": 0,
        "endpoint_loaded": False,
    }
    try:
        responses = model.chat_complete_messages_batched(
            [
                support_messages(
                    task,
                    question_ids=question_ids,
                    spaced_codes=True,
                )
            ],
            temperature=TEMPERATURE,
            block_size=1,
            max_new_tokens=MAX_OUTPUT_TOKENS,
        )
        raw["responses"] = responses
        _checkpoint(raw_path, raw)
        if len(responses) != EXPECTED_REQUESTS:
            raise ValueError("ClariQ V2 serving response count changed")
        support = parse_support(
            responses[0],
            questions=_question_payload(task, question_ids),
            spaced_codes=True,
        )
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc
    positive_profiles = {
        prediction
        for probability, prediction in zip(
            support.probabilities,
            support.predictions,
            strict=True,
        )
        if probability > 0.0
    }
    myopic_scores = {
        question_id: information_gain(support, question_id)
        for question_id in question_ids
    }
    support_entropy = entropy(support.probabilities)
    metrics = {
        "positive_response_profile_count": len(positive_profiles),
        "support_entropy_nats": support_entropy,
        "myopic_score_range_nats": (
            max(myopic_scores.values()) - min(myopic_scores.values())
        ),
        "positive_mass_hypothesis_count": sum(
            probability > 0.0 for probability in support.probabilities
        ),
    }
    gates = {
        "exact_physical_requests": (
            usage["physical_requests"] == EXPECTED_REQUESTS
        ),
        "exact_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "support_parses": len(support.hypotheses) == 8,
        "positive_profiles_at_least_4": len(positive_profiles) >= 4,
        "support_entropy_at_least_1_nat": support_entropy >= 1.0,
        "myopic_score_range_at_least_0_05": (
            metrics["myopic_score_range_nats"] >= 0.05
        ),
        "zero_branch_requests": raw["branch_requests_sent"] == 0,
        "endpoint_not_loaded": raw["endpoint_loaded"] is False,
        "cost_at_most_0_05": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "manifest_sha256": MANIFEST_SHA256,
            "model": MODEL_ID,
            "topic_id": TOPIC_ID,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "expected_requests": EXPECTED_REQUESTS,
            "code_format": "single ASCII spaces",
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "branch_requests_sent": 0,
            "endpoint_loaded": False,
            "development_or_holdout_loaded": False,
        },
        "support": {
            "question_ids": list(support.question_ids),
            "hypotheses": [
                {
                    "hypothesis": hypothesis,
                    "probability": probability,
                    "response_codes": prediction,
                }
                for hypothesis, probability, prediction in zip(
                    support.hypotheses,
                    support.probabilities,
                    support.predictions,
                    strict=True,
                )
            ],
        },
        "myopic_scores": myopic_scores,
        "metrics": metrics,
        "gates": gates,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        payload = run_serving(
            config,
            manifest_path=args.manifest,
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
            "manifest_sha256": MANIFEST_SHA256,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "SERVING_FAILURE.json", failure)
        raise
    _checkpoint(args.output_dir / "SERVING.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
