#!/usr/bin/env python3
"""Run ICAE first-link instrument V2 with set-valued trigger matches."""

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
from scripts.icae_bench_exact_response_controller_smoke import (
    MAX_MATCHED_IDS,
    matcher_response_format,
    trigger_catalog,
)
from scripts.icae_bench_first_link_instrument_smoke import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_AUDIT_SHA256,
    EVALUATOR_MAX_TOKENS,
    EVALUATOR_MODEL_ID,
    PLANNER_MAX_TOKENS,
    PLANNER_MODEL_ID,
    PROJECTED_COST_USD,
    SCHEMA_VERSION,
    run_instrument_smoke,
)
from scripts.icae_bench_semantic_serving_smoke import (
    _adapter,
    strict_json_object,
    usage_summary,
)


INTERFACE_VERSION = "icae-first-link-instrument-smoke-2"
EXCLUDED_OPENED_ALIASES = {
    "realcode@044",
    "realcode@276",
    "realcode@235",
    "realcode@185",
    "realcode@259",
}
SELECTION_SEED = 51_000
PLANNER_SEED = 51_100
EVALUATOR_SEED = 51_200


def select_instrument_task_v2(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    eligible = [
        row
        for row in manifest["partitions"]["mechanics"]
        if row["alias"] not in EXCLUDED_OPENED_ALIASES
    ]
    return min(
        eligible,
        key=lambda row: (
            hashlib.sha256(
                f"{SELECTION_SEED}:{row['alias']}".encode("utf-8")
            ).hexdigest(),
            row["alias"],
        ),
    )


def matcher_messages_set(
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
            "matched_ids is an unordered set; each ID may appear at most once.",
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


def parse_matcher_set(
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
    if not isinstance(fallback, bool) or fallback != (not matched):
        raise ValueError(f"{label}.fallback is inconsistent")
    canonical = sorted(matched, key=valid_ids.index)
    return {"matched_ids": canonical, "fallback": fallback}


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
        projected_cost=PROJECTED_COST_USD * 0.65,
    )
    evaluator = _adapter(
        model=EVALUATOR_MODEL_ID,
        run_id=args.run_id,
        output_dir=output_dir,
        request_seed=EVALUATOR_SEED,
        max_tokens=EVALUATOR_MAX_TOKENS,
        projected_cost=PROJECTED_COST_USD * 0.35,
    )
    try:
        payload = run_instrument_smoke(
            repo=args.repo.resolve(),
            manifest_path=args.manifest.resolve(),
            source_audit_path=args.source_audit.resolve(),
            output_dir=output_dir,
            run_id=args.run_id,
            planner_model=planner,
            evaluator_model=evaluator,
            task_selector=select_instrument_task_v2,
            match_message_builder=matcher_messages_set,
            match_parser=parse_matcher_set,
            interface_version=INTERFACE_VERSION,
            planner_model_id=PLANNER_MODEL_ID,
            evaluator_model_id=EVALUATOR_MODEL_ID,
            selection_seed=SELECTION_SEED,
            planner_seed=PLANNER_SEED,
            evaluator_seed=EVALUATOR_SEED,
        )
        checkpoint(output_dir / "SERVING.json", payload)
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "run_id": args.run_id,
            "error": str(exc),
            "usage": usage_summary([planner, evaluator]),
            "development_or_later_opened": False,
            "executable_endpoint_opened": False,
        }
        checkpoint(output_dir / "FAILURE.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
