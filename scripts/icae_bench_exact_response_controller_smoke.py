#!/usr/bin/env python3
"""Qualify an LLM semantic matcher with exact ICAE responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.icae_bench_semantic_serving_smoke import (
    CONCURRENCY,
    GENERIC_UNMATCHED_QUESTION,
    MIN_ANSWER_TERMS_IN_FOLLOWUPS,
    MIN_CHANGED_FOLLOWUPS,
    NUM_TASKS,
    ServingModel,
    _adapter,
    answer_term_count,
    changed_question_count,
    followup_messages,
    initial_messages,
    parse_support,
    planner_response_format,
    strict_json_object,
    usage_summary,
)
from scripts.icae_bench_source_opportunity_audit import sha256_bytes


SCHEMA_VERSION = 1
INTERFACE_VERSION = "icae-exact-response-controller-smoke-1"
EXPECTED_MANIFEST_SHA256 = (
    "47ab7f2fcf985433389f53873fa7fe8ccbebd88dd8390d654de93083bad84f5f"
)
EXPECTED_SOURCE_AUDIT_SHA256 = (
    "f63a313adb8bc3bd1090fe4542d1b38b4c2e63abca30c559447a70a6e6348552"
)
EXCLUDED_OPENED_ALIASES = {"realcode@044", "realcode@276"}
SELECTION_SEED = 50_400
PLANNER_SEED = 50_500
MATCHER_SEED = 50_600
PLANNER_MODEL_ID = "openai/gpt-5.4"
MATCHER_MODEL_ID = "openai/gpt-5.4-mini"
EXPECTED_REQUESTS = 10
PLANNER_MAX_TOKENS = 3000
MATCHER_MAX_TOKENS = 800
PROJECTED_COST_USD = 0.12
RUN_BUDGET_USD = 0.30
MAX_MATCHED_IDS = 3


def select_unopened_mechanics(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in manifest["partitions"]["mechanics"]
        if row["alias"] not in EXCLUDED_OPENED_ALIASES
    ]
    return sorted(
        eligible,
        key=lambda row: (
            hashlib.sha256(
                f"{SELECTION_SEED}:{row['alias']}".encode("utf-8")
            ).hexdigest(),
            row["alias"],
        ),
    )[:NUM_TASKS]


def matcher_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "icae_semantic_trigger_match",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["matched_ids", "fallback"],
                "properties": {
                    "matched_ids": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": MAX_MATCHED_IDS,
                        "items": {"type": "string"},
                    },
                    "fallback": {"type": "boolean"},
                },
            },
        },
    }


def trigger_catalog(record: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = []
    for index, row in enumerate(
        record["oracle_data"]["hidden_constraints"]
    ):
        identifier = str(
            row.get("constraint_id", row.get("id", f"X{index}"))
        )
        catalog.append(
            {
                "id": identifier,
                "trigger_phrases": row["trigger_keywords"],
            }
        )
    return catalog


def matcher_messages(
    record: dict[str, Any],
    question: str,
) -> list[dict[str, str]]:
    request = {
        "task": (
            "Match one free-form clarification question to a released "
            "requirement trigger catalog."
        ),
        "fuzzy_prd": record["fuzzy_prd"],
        "question": question,
        "catalog": trigger_catalog(record),
        "rules": [
            "Match only when the question semantically and explicitly asks about the trigger topic.",
            "Return matched IDs in catalog order.",
            "Return at most three IDs.",
            "If no trigger matches, return an empty list and fallback=true.",
            "If any trigger matches, return fallback=false.",
            "Do not infer or answer the requirement.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict semantic router. Classify only against the "
                "supplied catalog and follow the JSON schema exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def parse_matcher(
    response: str,
    *,
    valid_ids: list[str],
    label: str,
) -> dict[str, Any]:
    value = strict_json_object(response, label=label)
    if set(value) != {"matched_ids", "fallback"}:
        raise ValueError(f"{label} has unexpected fields")
    matched = value["matched_ids"]
    fallback = value["fallback"]
    if not isinstance(matched, list) or len(matched) > MAX_MATCHED_IDS:
        raise ValueError(f"{label}.matched_ids is invalid")
    if not all(isinstance(identifier, str) for identifier in matched):
        raise ValueError(f"{label}.matched_ids must contain strings")
    if len(set(matched)) != len(matched):
        raise ValueError(f"{label}.matched_ids contains duplicates")
    if any(identifier not in valid_ids for identifier in matched):
        raise ValueError(f"{label}.matched_ids contains an unknown ID")
    expected_order = sorted(matched, key=valid_ids.index)
    if matched != expected_order:
        raise ValueError(f"{label}.matched_ids is not in catalog order")
    if not isinstance(fallback, bool) or fallback != (not matched):
        raise ValueError(f"{label}.fallback is inconsistent")
    return {"matched_ids": matched, "fallback": fallback}


def exact_controller_reply(
    record: dict[str, Any],
    matched_ids: list[str],
) -> str:
    if not matched_ids:
        return record["oracle_data"]["fallback_response"]
    by_id = {
        str(row.get("constraint_id", row.get("id", f"X{index}"))): row
        for index, row in enumerate(
            record["oracle_data"]["hidden_constraints"]
        )
    }
    return "\n\n".join(by_id[identifier]["oracle_response"] for identifier in matched_ids)


def run_controller_smoke(
    *,
    repo: Path,
    manifest_path: Path,
    source_audit_path: Path,
    output_dir: Path,
    run_id: str,
    planner_model: ServingModel,
    matcher_model: ServingModel,
) -> dict[str, Any]:
    if sha256_bytes(manifest_path.read_bytes()) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen ICAE manifest hash mismatch")
    if sha256_bytes(source_audit_path.read_bytes()) != EXPECTED_SOURCE_AUDIT_SHA256:
        raise ValueError("Frozen ICAE source-audit hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = select_unopened_mechanics(manifest)
    records = [
        json.loads((repo / row["oracle_record"]).read_text(encoding="utf-8"))
        for row in selected
    ]

    initial_raw = planner_model.chat_complete_messages_batched_structured(
        [initial_messages(record["fuzzy_prd"]) for record in records],
        temperature=0.0,
        block_size=NUM_TASKS,
        response_format=planner_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )
    initial = [
        parse_support(raw, label=f"initial[{index}]")
        for index, raw in enumerate(initial_raw)
    ]
    roots = [support["questions"][0]["question"] for support in initial]
    root_messages = [
        matcher_messages(record, root)
        for record, root in zip(records, roots, strict=True)
    ]

    first_match_raw = matcher_model.chat_complete_messages_batched_structured(
        root_messages,
        temperature=0.0,
        block_size=NUM_TASKS,
        response_format=matcher_response_format(),
        max_new_tokens=MATCHER_MAX_TOKENS,
    )
    repeated_match_raw = matcher_model.chat_complete_messages_batched_structured(
        root_messages,
        temperature=0.0,
        block_size=NUM_TASKS,
        response_format=matcher_response_format(),
        max_new_tokens=MATCHER_MAX_TOKENS,
    )
    generic_match_raw = matcher_model.chat_complete_messages_batched_structured(
        [
            matcher_messages(record, GENERIC_UNMATCHED_QUESTION)
            for record in records
        ],
        temperature=0.0,
        block_size=NUM_TASKS,
        response_format=matcher_response_format(),
        max_new_tokens=MATCHER_MAX_TOKENS,
    )

    valid_ids = [
        [row["id"] for row in trigger_catalog(record)] for record in records
    ]
    first_matches = [
        parse_matcher(raw, valid_ids=ids, label=f"first_match[{index}]")
        for index, (raw, ids) in enumerate(
            zip(first_match_raw, valid_ids, strict=True)
        )
    ]
    repeated_matches = [
        parse_matcher(raw, valid_ids=ids, label=f"repeated_match[{index}]")
        for index, (raw, ids) in enumerate(
            zip(repeated_match_raw, valid_ids, strict=True)
        )
    ]
    generic_matches = [
        parse_matcher(raw, valid_ids=ids, label=f"generic_match[{index}]")
        for index, (raw, ids) in enumerate(
            zip(generic_match_raw, valid_ids, strict=True)
        )
    ]
    replies = [
        exact_controller_reply(record, match["matched_ids"])
        for record, match in zip(records, first_matches, strict=True)
    ]

    followup_raw = planner_model.chat_complete_messages_batched_structured(
        [
            followup_messages(
                record["fuzzy_prd"],
                support,
                root,
                reply,
            )
            for record, support, root, reply in zip(
                records, initial, roots, replies, strict=True
            )
        ],
        temperature=0.0,
        block_size=NUM_TASKS,
        response_format=planner_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )
    followup = [
        parse_support(raw, label=f"followup[{index}]")
        for index, raw in enumerate(followup_raw)
    ]

    task_rows = []
    for row, record, support, first, repeated, generic, reply, refreshed in zip(
        selected,
        records,
        initial,
        first_matches,
        repeated_matches,
        generic_matches,
        replies,
        followup,
        strict=True,
    ):
        task_rows.append(
            {
                "alias": row["alias"],
                "language": row["language"],
                "initial_hypothesis_count": len(support["hypotheses"]),
                "initial_question_count": len(support["questions"]),
                "root_match_count": len(first["matched_ids"]),
                "root_fallback": first["fallback"],
                "matcher_exact_replay": first == repeated,
                "controller_reply_exact_by_construction": (
                    reply
                    == exact_controller_reply(record, first["matched_ids"])
                ),
                "generic_exact_fallback": (
                    generic == {"matched_ids": [], "fallback": True}
                    and exact_controller_reply(record, [])
                    == record["oracle_data"]["fallback_response"]
                ),
                "followup_hypothesis_count": len(refreshed["hypotheses"]),
                "followup_question_count": len(refreshed["questions"]),
                "changed_followup_questions": changed_question_count(
                    support, refreshed
                ),
                "new_answer_terms_used": answer_term_count(
                    fuzzy_prd=record["fuzzy_prd"],
                    initial=support,
                    answer=reply,
                    followup=refreshed,
                ),
            }
        )

    usage = usage_summary([planner_model, matcher_model])
    gates = {
        "exact_request_count": usage["adapter_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": (
            usage["retry_count"] == 0
            and usage["provider_error_retries"] == 0
        ),
        "zero_reasoning_and_forced_exits": (
            usage["adapter_reasoning_tokens"] == 0
            and usage["forced_exits"] == 0
            and usage["forced_final_requests"] == 0
        ),
        "root_questions_match_catalog": all(
            row["root_match_count"] >= 1 and not row["root_fallback"]
            for row in task_rows
        ),
        "semantic_matcher_replays_exactly": all(
            row["matcher_exact_replay"] for row in task_rows
        ),
        "controller_responses_are_exact": all(
            row["controller_reply_exact_by_construction"] for row in task_rows
        ),
        "generic_queries_make_no_progress": all(
            row["generic_exact_fallback"] for row in task_rows
        ),
        "history_changes_question_support": all(
            row["changed_followup_questions"] >= MIN_CHANGED_FOLLOWUPS
            for row in task_rows
        ),
        "followups_use_answer_introduced_terms": all(
            row["new_answer_terms_used"] >= MIN_ANSWER_TERMS_IN_FOLLOWUPS
            for row in task_rows
        ),
        "cost_within_cap": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    all_pass = all(gates.values())
    private = {
        "selected": selected,
        "initial_raw": initial_raw,
        "first_match_raw": first_match_raw,
        "repeated_match_raw": repeated_match_raw,
        "generic_match_raw": generic_match_raw,
        "followup_raw": followup_raw,
    }
    private_path = output_dir / "private" / "RAW_RESPONSES.json"
    checkpoint(private_path, private)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "passed" if all_pass else "gated_null",
        "run_id": run_id,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_audit_sha256": EXPECTED_SOURCE_AUDIT_SHA256,
        "models": {
            "planner": PLANNER_MODEL_ID,
            "semantic_matcher": MATCHER_MODEL_ID,
        },
        "seeds": {
            "selection": SELECTION_SEED,
            "planner": PLANNER_SEED,
            "semantic_matcher": MATCHER_SEED,
        },
        "selected_tasks": [
            {"alias": row["alias"], "language": row["language"]}
            for row in selected
        ],
        "usage": usage,
        "tasks": task_rows,
        "gates": gates,
        "all_pass": all_pass,
        "decision": (
            "authorize_separately_frozen_mechanics_first_link_test"
            if all_pass
            else "close_exact_controller_interface"
        ),
        "private_raw_sha256": sha256_bytes(private_path.read_bytes()),
        "endpoint_accessed": False,
        "development_or_later_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    planner = _adapter(
        model=PLANNER_MODEL_ID,
        run_id=args.run_id,
        output_dir=output_dir,
        request_seed=PLANNER_SEED,
        max_tokens=PLANNER_MAX_TOKENS,
        projected_cost=PROJECTED_COST_USD * 0.8,
    )
    matcher = _adapter(
        model=MATCHER_MODEL_ID,
        run_id=args.run_id,
        output_dir=output_dir,
        request_seed=MATCHER_SEED,
        max_tokens=MATCHER_MAX_TOKENS,
        projected_cost=PROJECTED_COST_USD * 0.2,
    )
    result_path = output_dir / "SERVING.json"
    failure_path = output_dir / "FAILURE.json"
    try:
        payload = run_controller_smoke(
            repo=args.repo.resolve(),
            manifest_path=args.manifest.resolve(),
            source_audit_path=args.source_audit.resolve(),
            output_dir=output_dir,
            run_id=args.run_id,
            planner_model=planner,
            matcher_model=matcher,
        )
        checkpoint(result_path, payload)
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "run_id": args.run_id,
            "error": str(exc),
            "usage": usage_summary([planner, matcher]),
            "endpoint_accessed": False,
            "development_or_later_opened": False,
        }
        checkpoint(failure_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
