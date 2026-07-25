#!/usr/bin/env python3
"""Rescore frozen tau-Knowledge trees with a preregistered second model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.tau_knowledge_first_link_scorer import (
    _score_schema as root_score_schema,
    scorer_messages,
)
from scripts.tau_knowledge_receding_continuation import (
    EXPECTED_REQUESTS,
    FRESH_CONFIRMATION_IDS,
    SMOKE_IDS,
    summarize,
)
from scripts.tau_knowledge_receding_continuation_v3 import (
    _score_schema as focused_score_schema,
    document_count_messages,
)
from scripts.tau_knowledge_retrieval_opportunity import (
    FIRST_QUERY_COUNT,
    FOLLOWUP_QUERY_COUNT,
    GateExecutionError,
    SCHEMA_VERSION,
    _checkpoint,
    _usage_snapshot,
)


INTERFACE_VERSION = "cross-model-1"
MODEL_ID = "anthropic/claude-sonnet-5"
SMOKE_ARTIFACT_SHA256 = (
    "61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01"
)
CONFIRMATION_ARTIFACT_SHA256 = (
    "f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae"
)
NONSEMANTIC_ANALYSIS_SHA256 = (
    "d1d43ecf0f09fd5874e9badd6cff4e9ae5c3da882a3ea408b816225c2d2b29f4"
)
CROSS_MODEL_EXPECTED_REQUESTS = {
    "serving_smoke": 14,
    "confirmation": 140,
}
MAX_NONSEMANTIC_ROOT_ACCURACY = 0.5454545454545454
MAX_NONSEMANTIC_CONTINUATION_ACCURACY = 0.631578947368421
MAX_NONSEMANTIC_ENDPOINT = 22


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coerce_json_int(
    value: Any,
    *,
    minimum: int,
    maximum: int,
    max_digits: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric score")
    if isinstance(value, int):
        parsed = value
    elif (
        isinstance(value, str)
        and value.isdigit()
        and 1 <= len(value) <= max_digits
    ):
        parsed = int(value)
    else:
        raise ValueError("numeric field must be an integer or digit string")
    if not minimum <= parsed <= maximum:
        raise ValueError("numeric field is outside its valid range")
    return parsed


def parse_root_scores(
    text: str,
    *,
    include_followups: bool,
) -> dict[str, Any]:
    payload = _parse_json_object(text)
    expected = root_score_schema(include_followups=include_followups)
    if set(payload) != set(expected):
        raise ValueError("cross-model root response has unexpected keys")
    scores = []
    best_followups = []
    rationales = []
    for index in range(1, FIRST_QUERY_COUNT + 1):
        scores.append(
            coerce_json_int(
                payload[f"root_{index}_score"],
                minimum=0,
                maximum=100,
                max_digits=3,
            )
        )
        if include_followups:
            best_followups.append(
                coerce_json_int(
                    payload[f"root_{index}_best_followup"],
                    minimum=1,
                    maximum=FOLLOWUP_QUERY_COUNT,
                    max_digits=2,
                )
                - 1
            )
        rationale = payload[f"root_{index}_rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("cross-model root rationale must be nonempty")
        rationales.append(rationale.strip())
    return {
        "scores": scores,
        "best_followup_indices": best_followups,
        "rationales": rationales,
    }


def parse_focused_scores(text: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    if set(payload) != set(focused_score_schema()):
        raise ValueError("cross-model focused response has unexpected keys")
    scores = []
    for index in range(1, FIRST_QUERY_COUNT):
        score = coerce_json_int(
            payload[f"followup_{index}_score"],
            minimum=0,
            maximum=99,
            max_digits=2,
        )
        if not (
            0 <= score <= 9
            or 30 <= score <= 39
            or 60 <= score <= 69
            or 90 <= score <= 99
        ):
            raise ValueError("focused score is outside a valid count band")
        scores.append(score)
    return {
        "scores": scores,
        "rationales": ["count-dominant document score"] * len(scores),
    }


def load_records(path: Path, *, stage: str) -> list[dict[str, Any]]:
    expected_hash = (
        SMOKE_ARTIFACT_SHA256
        if stage == "serving_smoke"
        else CONFIRMATION_ARTIFACT_SHA256
    )
    if sha256_file(path) != expected_hash:
        raise ValueError("frozen tau source artifact hash does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("frozen tau source artifact lacks records")
    expected_ids: Sequence[str] = (
        SMOKE_IDS if stage == "serving_smoke" else FRESH_CONFIRMATION_IDS
    )
    if tuple(record.get("task_id") for record in records) != tuple(expected_ids):
        raise ValueError("frozen tau task IDs do not match the stage")
    return records


def summarize_cross_model(
    records: Sequence[dict[str, Any]],
    continuation_scores: Sequence[Sequence[dict[str, Any]]],
    usage: dict[str, Any],
    *,
    stage: str,
    myopic_scores: Sequence[dict[str, Any]],
    nonmyopic_scores: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    compatibility_usage = dict(usage)
    compatibility_usage["physical_requests"] = EXPECTED_REQUESTS[stage]
    summary = summarize(
        records,
        continuation_scores,
        compatibility_usage,
        stage=stage,
        myopic_scores=myopic_scores,
        nonmyopic_scores=nonmyopic_scores,
    )
    gates = summary["gates"]
    gates["exact_physical_request_count"] = (
        int(usage["physical_requests"]) == CROSS_MODEL_EXPECTED_REQUESTS[stage]
    )
    if stage == "confirmation":
        endpoint_total = sum(
            row["nonmyopic_receding_value"]
            for row in summary["policy_diagnostics"]
        )
        summary["cross_model_endpoint_total"] = endpoint_total
        gates["root_accuracy_above_all_nonsemantic_controls"] = (
            summary["nonmyopic_root_pairwise_accuracy"]
            > MAX_NONSEMANTIC_ROOT_ACCURACY
        )
        gates["continuation_accuracy_above_all_nonsemantic_controls"] = (
            summary["focused_pairwise_accuracy"]
            > MAX_NONSEMANTIC_CONTINUATION_ACCURACY
        )
        gates["endpoint_at_least_3_above_best_nonsemantic"] = (
            endpoint_total >= MAX_NONSEMANTIC_ENDPOINT + 3
        )
    gates["all_pass"] = all(
        value for key, value in gates.items() if key != "all_pass"
    )
    return summary


def run_gate(
    config: Config,
    *,
    stage: str,
    input_artifact: Path,
    nonsemantic_analysis: Path,
    raw_checkpoint_path: Path,
    model_id: str = MODEL_ID,
    interface_version: str = INTERFACE_VERSION,
) -> dict[str, Any]:
    if config.model_pairs[0].questioner.model != model_id:
        raise ValueError("cross-model scorer config selects the wrong model")
    if sha256_file(nonsemantic_analysis) != NONSEMANTIC_ANALYSIS_SHA256:
        raise ValueError("frozen nonsemantic analysis hash does not match")
    records = load_records(input_artifact, stage=stage)
    model = build_model_adapter(config.model_pairs[0].questioner, config)
    raw: dict[str, Any] = {}
    try:
        myopic_raw = model.chat_complete_messages_batched(
            [
                scorer_messages(record, include_followups=False)
                for record in records
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["myopic_root_scores"] = myopic_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        myopic_scores = [
            parse_root_scores(text, include_followups=False)
            for text in myopic_raw
        ]

        nonmyopic_raw = model.chat_complete_messages_batched(
            [
                scorer_messages(record, include_followups=True)
                for record in records
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["nonmyopic_root_scores"] = nonmyopic_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        nonmyopic_scores = [
            parse_root_scores(text, include_followups=True)
            for text in nonmyopic_raw
        ]

        focused_keys = [
            (case_index, root_index)
            for case_index in range(len(records))
            for root_index in range(FIRST_QUERY_COUNT)
        ]
        focused_raw = model.chat_complete_messages_batched(
            [
                document_count_messages(records[case_index], root_index)
                for case_index, root_index in focused_keys
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["focused_continuation_scores"] = focused_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        parsed_focused = [parse_focused_scores(text) for text in focused_raw]
        continuation_scores = [
            parsed_focused[
                case_index
                * FIRST_QUERY_COUNT : (case_index + 1) * FIRST_QUERY_COUNT
            ]
            for case_index in range(len(records))
        ]
        usage = _usage_snapshot(model)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    summary = summarize_cross_model(
        records,
        continuation_scores,
        usage,
        stage=stage,
        myopic_scores=myopic_scores,
        nonmyopic_scores=nonmyopic_scores,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "interface_version": interface_version,
            "model": model_id,
            "source_artifact_sha256": sha256_file(input_artifact),
            "nonsemantic_analysis_sha256": NONSEMANTIC_ANALYSIS_SHA256,
            "task_ids": [record["task_id"] for record in records],
            "expected_physical_requests": CROSS_MODEL_EXPECTED_REQUESTS[stage],
            "tree_regeneration_requests": 0,
            "reasoning_requested": False,
            "target_blind_prompts": True,
            "posthoc_open_tree_replication": True,
            "permissive_numeric_representation_parser_frozen": True,
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "myopic_scores": myopic_scores,
        "nonmyopic_scores": nonmyopic_scores,
        "continuation_scores": continuation_scores,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "confirmation"),
        required=True,
    )
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--nonsemantic-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.20
        config.openrouter_run_budget_usd = 0.50
    else:
        config.openrouter_projected_cost_usd = 2.00
        config.openrouter_run_budget_usd = 3.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "CONFIRMATION.json"
    )
    try:
        payload = run_gate(
            config,
            stage=args.stage,
            input_artifact=args.input_artifact,
            nonsemantic_analysis=args.nonsemantic_analysis,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        (args.output_dir / f"{args.stage.upper()}_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = args.output_dir / output_name
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output_path),
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
