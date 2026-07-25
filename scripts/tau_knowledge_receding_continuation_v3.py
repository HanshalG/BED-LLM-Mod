#!/usr/bin/env python3
"""Evaluate document-count receding continuation selection on tau-Knowledge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.tau_knowledge_receding_continuation_v2 import (
    compact_evidence_input,
    run_gate,
)
from scripts.tau_knowledge_retrieval_opportunity import (
    FIRST_QUERY_COUNT,
    GateExecutionError,
    SCHEMA_VERSION,
)


INTERFACE_VERSION = 3


def _score_schema() -> dict[str, str]:
    return {
        f"followup_{index}_score": (
            "canonical decimal digit string in 0-9, 30-39, 60-69, or 90-99"
        )
        for index in range(1, FIRST_QUERY_COUNT)
    }


def document_count_messages(
    record: dict[str, Any],
    root_index: int,
) -> list[dict[str, str]]:
    payload = compact_evidence_input(record, root_index)
    return [
        {
            "role": "system",
            "content": (
                "Act as a target-blind semantic document classifier after one "
                "realized banking-policy retrieval. Candidate query wording is "
                "hidden. Judge only facts in supplied document titles and excerpts. "
                "Required-document labels and evaluation answers are hidden. Return "
                "one flat strict JSON object with no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each candidate, count its DISTINCT NEW returned documents that "
                "materially support at least one plausible unresolved customer need. "
                "Use the customer opening and initial information needs as the "
                "primary objective. Refreshed information needs are fallible "
                "path-dependent hypotheses: use their useful detail, but do not let "
                "them erase original goals, named products, comparisons, or parallel "
                "requests. A document counts only when its supplied title or excerpt "
                "contains concrete relevant policy evidence. Ignore documents already "
                "acquired in first_result_refs, duplicate refs, wrong-product analogies, "
                "surface topic matches, imagined query intent, and merely urgent but "
                "unsupported topics. Let N be the count, from 0 through 3. Encode the "
                "score in a count-dominant band: N=0 gives 0-9, N=1 gives 30-39, N=2 "
                "gives 60-69, and N=3 gives 90-99. Within the band, use the final digit "
                "only to prefer more direct and broader evidence. Never let that "
                "tie-break outweigh one additional useful document. Return exactly "
                "these keys and canonical digit-string values: "
                + json.dumps(
                    _score_schema(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + ". Retrieval data: "
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_document_count_scores(text: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    if set(payload) != set(_score_schema()):
        raise ValueError("document-count response has unexpected keys")
    scores = []
    for index in range(1, FIRST_QUERY_COUNT):
        value = payload[f"followup_{index}_score"]
        if (
            not isinstance(value, str)
            or not value.isdigit()
            or (len(value) > 1 and value.startswith("0"))
        ):
            raise ValueError("document-count score is not canonical")
        score = int(value)
        if not (
            0 <= score <= 9
            or 30 <= score <= 39
            or 60 <= score <= 69
            or 90 <= score <= 99
        ):
            raise ValueError("document-count score is outside a valid band")
        scores.append(score)
    return {
        "scores": scores,
        "rationales": ["count-dominant document score"] * len(scores),
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
        config.openrouter_projected_cost_usd = 0.08
        config.openrouter_run_budget_usd = 0.50
    elif args.stage == "development":
        config.openrouter_projected_cost_usd = 0.80
        config.openrouter_run_budget_usd = 3.00
    else:
        config.openrouter_projected_cost_usd = 3.20
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
            message_builder=document_count_messages,
            response_parser=parse_document_count_scores,
            interface_version=INTERFACE_VERSION,
            protocol_extra={
                "count_dominant_document_scoring": True,
                "free_form_rationales_requested": False,
                "refreshed_beliefs_explicitly_fallible": True,
            },
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
