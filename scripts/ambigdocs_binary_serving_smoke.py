#!/usr/bin/env python3
"""Validate semantic question and likelihood generation on AmbigDocs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.pscon_binary_query_serving_smoke import parse_labels, parse_question
from scripts.pscon_semantic_tree_smoke import entropy


INTERFACE_VERSION = "ambigdocs-binary-serving-smoke-1"
SOURCE_REPOSITORY = "https://huggingface.co/datasets/yoonsanglee/AmbigDocs"
SOURCE_REVISION = "19d318a5e6717f63b9b864aa804bc57f69e824df"
DEV_SHA256 = "43ab72b880337fac6442ea04accbda38d6a8e785eb2db4970585fac7ae775d68"
MODEL_ID = "openai/gpt-5.4-mini"
ROW_INDEX = 49
EXPECTED_QID = 43_608
SEED = 24_388
QUESTION_COUNT = 5
EXPECTED_REQUESTS = 10
MAX_COST_USD = 0.15


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


def verify_source(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DEV_SHA256:
        raise ValueError(f"AmbigDocs dev hash is {digest}, expected {DEV_SHA256}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 3610:
        raise ValueError("AmbigDocs dev row count changed")
    row = rows[ROW_INDEX]
    if int(row["qid"]) != EXPECTED_QID or len(row["documents"]) != 6:
        raise ValueError("frozen AmbigDocs serving row changed")
    return row


def _document_records(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "candidate_index": index,
            "title": document["title"],
            "text": document["text"][:1200],
        }
        for index, document in enumerate(row["documents"], start=1)
    ]


def _question_messages(row: dict[str, Any], index: int) -> list[dict[str, str]]:
    request = {
        "ambiguous_question": row["question"],
        "ambiguous_entity": row["ambiguous_entity"],
        "question_index": index,
        "candidate_documents": _document_records(row),
    }
    return [
        {
            "role": "system",
            "content": (
                "Write one useful neutral yes/no clarification question that helps "
                "determine which entity the user means. Use only the ambiguous "
                "question and candidate documents. Ask about one semantic property "
                "the user can answer. Do not mention candidate indices, document "
                "titles, exact entity names, or ask the user to select from a list. "
                "Different question_index values should explore different semantic "
                "distinctions. Output exactly the question on one line and nothing else."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _label_messages(
    row: dict[str, Any],
    question: str,
) -> list[dict[str, str]]:
    request = {
        "clarification_question": question,
        "candidate_documents": _document_records(row),
    }
    return [
        {
            "role": "system",
            "content": (
                "Classify how each candidate document answers the clarification. Use "
                "Y only when the document clearly supports yes, N only when it clearly "
                "supports no, and U when insufficient. Output exactly one Y/N/U "
                "character per candidate in input order, with no spaces or prose."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _usage(model: ChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
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
    source_path: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    row = verify_source(source_path)
    count = len(row["documents"])
    raw: dict[str, Any] = {
        "row_index": ROW_INDEX,
        "qid": EXPECTED_QID,
        "target_sampled": False,
    }
    try:
        question_raw = model.chat_complete_messages_batched(
            [_question_messages(row, index) for index in range(QUESTION_COUNT)],
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=120,
        )
        raw["questions"] = question_raw
        _checkpoint(raw_path, raw)
        if len(question_raw) != QUESTION_COUNT:
            raise ValueError("question response count changed")
        questions = [parse_question(value) for value in question_raw]
        label_raw = model.chat_complete_messages_batched(
            [_label_messages(row, question) for question in questions],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=20,
        )
        raw["labels"] = label_raw
        _checkpoint(raw_path, raw)
        if len(label_raw) != QUESTION_COUNT:
            raise ValueError("label response count changed")
        labels = [parse_labels(value, count) for value in label_raw]
        raw["all_parsed"] = True
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    eigs = [entropy(list(value)) for value in labels]
    informative = [len(set(value)) >= 2 and eig >= 0.30 for value, eig in zip(labels, eigs)]
    kept_eigs = [eig for eig, keep in zip(eigs, informative) if keep]
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_questions_and_labels_parse": len(labels) == QUESTION_COUNT,
        "at_least_four_unique_questions": len(set(questions)) >= 4,
        "at_least_four_unique_partitions": len(set(labels)) >= 4,
        "at_least_four_informative_partitions": sum(informative) >= 4,
        "informative_eig_range_at_least_0_10": (
            len(kept_eigs) >= 2 and max(kept_eigs) - min(kept_eigs) >= 0.10
        ),
        "cost_at_most_0_15": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "dev_sha256": DEV_SHA256,
            "row_index": ROW_INDEX,
            "qid": EXPECTED_QID,
            "support_size": count,
            "seed": SEED,
            "model": MODEL_ID,
            "expected_requests": EXPECTED_REQUESTS,
            "target_sampled": False,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
        },
        "metrics": {
            "unique_question_count": len(set(questions)),
            "unique_partition_count": len(set(labels)),
            "informative_partition_count": sum(informative),
            "mean_eig": statistics.fmean(eigs),
            "eig_range": max(eigs) - min(eigs),
        },
        "queries": [
            {
                "query_index": index,
                "question": question,
                "labels": label,
                "eig": eig,
                "informative": keep,
            }
            for index, (question, label, eig, keep) in enumerate(
                zip(questions, labels, eigs, informative)
            )
        ],
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
            request = json.loads(messages[-1]["content"])
            if "question_index" in request:
                index = int(request["question_index"])
                responses.append(f"Does the intended entity have fixture trait {index}?")
            else:
                index = int(request["clarification_question"].split()[-1][:-1])
                patterns = ("YNNNNN", "YYNNNN", "YYYNNN", "YNYNNN", "YNNYNN")
                responses.append(patterns[index])
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
        raise ValueError("AmbigDocs config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.03
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 5
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = DeterministicFixtureModel() if args.dry_run else _build_model(config)
    try:
        payload = run_smoke(
            config,
            source_path=args.source_path,
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
    output = args.output_dir / "SERVING.json"
    _checkpoint(output, payload)
    print(json.dumps({"status": payload["status"], "metrics": payload["metrics"], "gates": payload["gates"], "usage": payload["usage"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
