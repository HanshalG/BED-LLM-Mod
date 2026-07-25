#!/usr/bin/env python3
"""Rescore frozen tau-Knowledge trees with compact thinking-Gemma outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from model_factory import build_model_adapter
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.tau_knowledge_cross_model_scorer import (
    coerce_json_int,
    run_gate,
    sha256_file,
)
from scripts.tau_knowledge_first_link_scorer import compact_scorer_input
from scripts.tau_knowledge_gemma26b_thinking_scorer import (
    COST_CAP_USD,
    MODEL_ID,
    apply_thinking_gates,
)
from scripts.tau_knowledge_receding_continuation_v2 import (
    compact_evidence_input,
)
from scripts.tau_knowledge_retrieval_opportunity import (
    FIRST_QUERY_COUNT,
    FOLLOWUP_QUERY_COUNT,
    GateExecutionError,
    SCHEMA_VERSION,
)


INTERFACE_VERSION = "gemma26b-thinking-compact-2"
THINKING_MAX_NEW_TOKENS = 8192
THINKING_FINAL_MAX_NEW_TOKENS = 512


def compact_root_messages(
    record: dict[str, Any],
    *,
    include_followups: bool,
) -> list[dict[str, str]]:
    tree = compact_scorer_input(
        record,
        include_followups=include_followups,
    )
    if include_followups:
        instruction = (
            "Score each first-search root by the best total coverage achievable "
            "after that root and exactly one shown followup. Reward distinct policy "
            "prerequisites, exceptions, eligibility rules, and procedures that "
            "resolve the customer's inferred information needs. Do not reward raw "
            "document count, verbosity, or duplicate coverage. For each root choose "
            "its best shown followup. Return exactly one JSON object with keys "
            '"scores" and "best_followups". scores must be five integers from 0 '
            "through 100 in root order. best_followups must be five integers from 1 "
            f"through {FOLLOWUP_QUERY_COUNT} in root order."
        )
    else:
        instruction = (
            "Score each first-search root using only its shown first results. Reward "
            "distinct policy prerequisites, exceptions, eligibility rules, and "
            "procedures that resolve the customer's inferred information needs. Do "
            "not imagine a followup or reward raw document count, verbosity, or "
            "duplicate coverage. Return exactly one JSON object with only key "
            '"scores", whose value is five integers from 0 through 100 in root order.'
        )
    return [
        {
            "role": "system",
            "content": (
                "Evaluate retrieval coverage for an internal banking support task. "
                "Ground every score in the customer opening and supplied document "
                "text. Required-document labels and evaluation answers are hidden. "
                "Return only the requested compact JSON object, with no rationale or "
                "extra prose."
            ),
        },
        {
            "role": "user",
            "content": (
                instruction
                + " Use the full score range when warranted and distinguish roots "
                "whenever their useful coverage differs. Retrieval data: "
                + json.dumps(tree, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_compact_root_scores(
    text: str,
    *,
    include_followups: bool,
) -> dict[str, Any]:
    payload = _parse_json_object(text)
    expected_keys = (
        {"scores", "best_followups"}
        if include_followups
        else {"scores"}
    )
    if set(payload) != expected_keys:
        raise ValueError("compact root response has unexpected keys")
    raw_scores = payload["scores"]
    if not isinstance(raw_scores, list) or len(raw_scores) != FIRST_QUERY_COUNT:
        raise ValueError("compact root scores must contain exactly five values")
    scores = [
        coerce_json_int(
            value,
            minimum=0,
            maximum=100,
            max_digits=3,
        )
        for value in raw_scores
    ]
    best_followups: list[int] = []
    if include_followups:
        raw_followups = payload["best_followups"]
        if (
            not isinstance(raw_followups, list)
            or len(raw_followups) != FIRST_QUERY_COUNT
        ):
            raise ValueError(
                "compact best-followup list must contain exactly five values"
            )
        best_followups = [
            coerce_json_int(
                value,
                minimum=1,
                maximum=FOLLOWUP_QUERY_COUNT,
                max_digits=2,
            )
            - 1
            for value in raw_followups
        ]
    return {
        "scores": scores,
        "best_followup_indices": best_followups,
        "rationales": ["compact semantic score"] * FIRST_QUERY_COUNT,
    }


def compact_focused_messages(
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
                "only the requested compact JSON object, with no rationale or prose."
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
                "acquired in first_result_refs, duplicate refs, wrong-product "
                "analogies, surface topic matches, imagined query intent, and merely "
                "urgent but unsupported topics. Let N be the count from 0 through 3. "
                "Encode N in a count-dominant band: N=0 gives 0-9, N=1 gives 30-39, "
                "N=2 gives 60-69, and N=3 gives 90-99. The final digit may prefer more "
                "direct or broader evidence but may never outweigh one additional "
                "useful document. Return exactly one JSON object with only key "
                '"scores", whose value is four integers in candidate order. '
                "Retrieval data: "
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_compact_focused_scores(text: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    if set(payload) != {"scores"}:
        raise ValueError("compact focused response has unexpected keys")
    raw_scores = payload["scores"]
    if (
        not isinstance(raw_scores, list)
        or len(raw_scores) != FOLLOWUP_QUERY_COUNT
    ):
        raise ValueError("compact focused scores must contain exactly four values")
    scores = [
        coerce_json_int(
            value,
            minimum=0,
            maximum=99,
            max_digits=2,
        )
        for value in raw_scores
    ]
    if any(
        not (
            0 <= score <= 9
            or 30 <= score <= 39
            or 60 <= score <= 69
            or 90 <= score <= 99
        )
        for score in scores
    ):
        raise ValueError("compact focused score is outside a valid count band")
    return {
        "scores": scores,
        "rationales": ["compact count-dominant score"] * len(scores),
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
    config.openrouter_projected_cost_usd = (
        0.10 if args.stage == "serving_smoke" else 0.75
    )
    config.openrouter_run_budget_usd = COST_CAP_USD[args.stage]
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

    spec = config.model_pairs[0].questioner
    if (
        spec.model != MODEL_ID
        or not spec.thinking
        or spec.thinking_max_new_tokens != THINKING_MAX_NEW_TOKENS
        or spec.thinking_final_max_new_tokens != THINKING_FINAL_MAX_NEW_TOKENS
    ):
        raise ValueError("compact Gemma thinking scorer config does not match")
    model = build_model_adapter(spec, config)
    try:
        payload = run_gate(
            config,
            stage=args.stage,
            input_artifact=args.input_artifact,
            nonsemantic_analysis=args.nonsemantic_analysis,
            raw_checkpoint_path=raw_path,
            model_id=MODEL_ID,
            interface_version=INTERFACE_VERSION,
            model_adapter=model,
            root_message_builder=compact_root_messages,
            root_response_parser=parse_compact_root_scores,
            focused_message_builder=compact_focused_messages,
            focused_response_parser=parse_compact_focused_scores,
        )
        payload = apply_thinking_gates(
            payload,
            stage=args.stage,
            thinking_max_new_tokens=THINKING_MAX_NEW_TOKENS,
            thinking_final_max_new_tokens=THINKING_FINAL_MAX_NEW_TOKENS,
        )
        payload["protocol"].update(
            {
                "compact_array_output": True,
                "free_form_rationales_requested": False,
                "source_trees_and_endpoints_reused_unchanged": True,
            }
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
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
    summary = payload["summary"]
    concise = {
        "status": payload["status"],
        "output": str(output_path),
        "focused_accuracy": summary["focused_pairwise_accuracy"],
        "usage": payload["usage"],
    }
    if args.stage == "confirmation":
        concise.update(
            {
                "root_accuracy": summary[
                    "nonmyopic_root_pairwise_accuracy"
                ],
                "root_accuracy_gain": summary[
                    "root_pairwise_accuracy_gain"
                ],
                "endpoint_total": summary["cross_model_endpoint_total"],
                "myopic_advantage": summary[
                    "end_to_end_total_advantage_over_myopic"
                ],
            }
        )
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
