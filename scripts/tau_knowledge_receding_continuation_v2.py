#!/usr/bin/env python3
"""Evaluate evidence-only receding continuation selection on tau-Knowledge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.tau_knowledge_first_link_scorer import (
    MAX_SCORER_INPUT_CHARS,
    parse_scores,
    scorer_messages,
)
from scripts.tau_knowledge_receding_continuation import (
    EXPECTED_REQUESTS,
    FRESH_SELECTION_SEED,
    RANDOM_CONTROL_SEED,
    _continuation_schema,
    _generate_fresh_trees,
    compact_continuation_input,
    load_fresh_confirmation,
    load_public_records,
    parse_continuation_scores,
    summarize,
)
from scripts.tau_knowledge_retrieval_opportunity import (
    BM25Corpus,
    FIRST_QUERY_COUNT,
    FOLLOWUP_QUERY_COUNT,
    GateExecutionError,
    HYPOTHESIS_COUNT,
    SCHEMA_VERSION,
    SEARCH_TOP_K,
    TAU_COMMIT,
    _build_model,
    _checkpoint,
    _usage_snapshot,
)


INTERFACE_VERSION = 2


def compact_evidence_input(
    record: dict[str, Any],
    root_index: int,
) -> dict[str, Any]:
    payload = compact_continuation_input(record, root_index)
    for candidate in payload["candidate_followups"]:
        candidate.pop("query")
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > MAX_SCORER_INPUT_CHARS:
        raise ValueError("evidence-only continuation input exceeds character cap")
    return payload


def evidence_continuation_messages(
    record: dict[str, Any],
    root_index: int,
) -> list[dict[str, str]]:
    payload = compact_evidence_input(record, root_index)
    return [
        {
            "role": "system",
            "content": (
                "Choose one next retrieval result after a realized banking-policy "
                "search. Candidate query wording is deliberately hidden. Ground "
                "every score only in facts explicitly present in the supplied "
                "document titles and excerpts. Required-document labels and "
                "evaluation answers are hidden. Return one flat strict JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Score each candidate by the distinct useful policy evidence in "
                "the union of the already acquired first documents and that "
                "candidate's returned documents. Compare actual document evidence "
                "against the customer opening and inferred information needs. A "
                "candidate receives no credit for a topic, rule, prerequisite, "
                "exception, calculation, or procedure that its returned documents "
                "do not explicitly support. Do not infer intended content from "
                "candidate order or imagine a query that might have produced the "
                "documents. Do not reward duplicated evidence, raw document count, "
                "verbosity, or surface relevance. Each rationale must name concrete "
                "supported evidence and, when applicable, the important need it "
                "leaves unresolved. Use the full 0-100 range when warranted. Return "
                "exactly these keys: "
                + json.dumps(
                    _continuation_schema(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + ". Retrieval data: "
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def run_gate(
    config: Config,
    *,
    stage: str,
    tau_root: str | Path,
    input_artifact: str | Path | None,
    raw_checkpoint_path: Path | None,
) -> dict[str, Any]:
    model = _build_model(config)
    raw: dict[str, Any] = {}
    try:
        if stage in {"serving_smoke", "development"}:
            if input_artifact is None:
                raise ValueError(f"{stage} requires --input-artifact")
            records, myopic_scores, nonmyopic_scores = load_public_records(
                input_artifact, stage=stage
            )
        elif stage == "confirmation":
            documents, tasks = load_fresh_confirmation(tau_root)
            records = _generate_fresh_trees(
                model,
                BM25Corpus(documents),
                tasks,
                config,
                raw,
                raw_checkpoint_path,
            )
            myopic_raw = model.chat_complete_messages_batched(
                [
                    scorer_messages(record, include_followups=False)
                    for record in records
                ],
                temperature=0.0,
                block_size=config.batched_block_size,
                max_new_tokens=config.openrouter_max_output_tokens,
            )
            raw["myopic_scores"] = myopic_raw
            _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
            myopic_scores = [
                parse_scores(text, include_followups=False)
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
            raw["nonmyopic_scores"] = nonmyopic_raw
            _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
            nonmyopic_scores = [
                parse_scores(text, include_followups=True)
                for text in nonmyopic_raw
            ]
        else:
            raise ValueError("unknown receding continuation stage")

        focused_keys = [
            (case_index, root_index)
            for case_index in range(len(records))
            for root_index in range(FIRST_QUERY_COUNT)
        ]
        focused_raw = model.chat_complete_messages_batched(
            [
                evidence_continuation_messages(
                    records[case_index], root_index
                )
                for case_index, root_index in focused_keys
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["focused_continuations"] = focused_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        parsed = [
            parse_continuation_scores(text) for text in focused_raw
        ]
        continuation_scores = [
            parsed[
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

    summary = summarize(
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
            "interface_version": INTERFACE_VERSION,
            "source_repository": "https://github.com/sierra-research/tau2-bench",
            "source_commit": TAU_COMMIT,
            "task_ids": [record["task_id"] for record in records],
            "fresh_selection_seed": (
                FRESH_SELECTION_SEED if stage == "confirmation" else None
            ),
            "random_control_seed": RANDOM_CONTROL_SEED,
            "first_query_count": FIRST_QUERY_COUNT,
            "followup_query_count": FOLLOWUP_QUERY_COUNT,
            "search_top_k": SEARCH_TOP_K,
            "hypothesis_count": HYPOTHESIS_COUNT,
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "focused_one_root_per_call": True,
            "candidate_followup_queries_hidden": True,
            "document_evidence_only": True,
            "required_documents_hidden_from_model": True,
            "reasoning_disabled": True,
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "records": records,
        "myopic_scores": myopic_scores,
        "nonmyopic_scores": nonmyopic_scores,
        "continuation_scores": continuation_scores,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "development", "confirmation"),
        required=True,
    )
    parser.add_argument("--input-artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.10
        config.openrouter_run_budget_usd = 0.50
    elif args.stage == "development":
        config.openrouter_projected_cost_usd = 1.00
        config.openrouter_run_budget_usd = 3.00
    else:
        config.openrouter_projected_cost_usd = 3.50
        config.openrouter_run_budget_usd = 8.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = {
        "serving_smoke": "SERVING_SMOKE.json",
        "development": "DEVELOPMENT.json",
        "confirmation": "CONFIRMATION.json",
    }[args.stage]
    try:
        payload = run_gate(
            config,
            stage=args.stage,
            tau_root=args.tau_root,
            input_artifact=args.input_artifact,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
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
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
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
