#!/usr/bin/env python3
"""Ablate branch-to-belief alignment on frozen tau-Knowledge trees."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.analyze_tau_knowledge_first_link_confirmation import (
    exact_sign_flip_pvalue,
)
from scripts.tau_knowledge_first_link_scorer import (
    pairwise_ranking_points,
    parse_scores,
    scorer_messages,
)
from scripts.tau_knowledge_receding_continuation import (
    EXPECTED_REQUESTS,
    FRESH_CONFIRMATION_IDS,
    SMOKE_IDS,
    summarize,
)
from scripts.tau_knowledge_receding_continuation_v3 import (
    document_count_messages,
)
from scripts.tau_knowledge_receding_continuation_v3_1 import (
    parse_zero_padding_scores,
)
from scripts.tau_knowledge_retrieval_opportunity import (
    FIRST_QUERY_COUNT,
    GateExecutionError,
    SCHEMA_VERSION,
    _checkpoint,
    _usage_snapshot,
    analyze_record,
)


INTERFACE_VERSION = "belief-alignment-1"
MODEL_ID = "openai/gpt-5.4"
PERMUTATION_SEED = 24343
SMOKE_ARTIFACT_SHA256 = (
    "61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01"
)
CONFIRMATION_ARTIFACT_SHA256 = (
    "f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae"
)
ABLATION_EXPECTED_REQUESTS = {
    "serving_smoke": 12,
    "confirmation": 120,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _records_without_refreshed_beliefs(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    stripped = copy.deepcopy(list(records))
    for record in stripped:
        for branch in record["first_branches"]:
            branch.pop("refreshed_information_need_hypotheses")
    return stripped


def _belief_multiset(record: dict[str, Any]) -> list[str]:
    return sorted(
        json.dumps(
            branch["refreshed_information_need_hypotheses"],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        for branch in record["first_branches"]
    )


def _task_derangement(task_id: str, size: int) -> list[int]:
    digest = hashlib.sha256(
        f"{PERMUTATION_SEED}:{task_id}".encode("ascii")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    indices = list(range(size))
    for _ in range(1000):
        rng.shuffle(indices)
        if all(source != target for target, source in enumerate(indices)):
            return list(indices)
    raise RuntimeError("could not construct deterministic derangement")


def derange_refreshed_beliefs(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transformed = copy.deepcopy(list(records))
    diagnostics = []
    for original, changed in zip(records, transformed, strict=True):
        branches = original["first_branches"]
        permutation = _task_derangement(original["task_id"], len(branches))
        for target, source in enumerate(permutation):
            changed["first_branches"][target][
                "refreshed_information_need_hypotheses"
            ] = copy.deepcopy(
                branches[source]["refreshed_information_need_hypotheses"]
            )
        diagnostics.append(
            {
                "task_id": original["task_id"],
                "source_branch_for_target": permutation,
                "is_derangement": all(
                    source != target
                    for target, source in enumerate(permutation)
                ),
                "belief_multiset_preserved": (
                    _belief_multiset(original) == _belief_multiset(changed)
                ),
            }
        )
    return transformed, diagnostics


def validate_intervention(
    original: Sequence[dict[str, Any]],
    transformed: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    original_nonbelief_hash = _canonical_hash(
        _records_without_refreshed_beliefs(original)
    )
    transformed_nonbelief_hash = _canonical_hash(
        _records_without_refreshed_beliefs(transformed)
    )
    return {
        "all_permutations_are_derangements": all(
            row["is_derangement"] for row in diagnostics
        ),
        "all_belief_multisets_preserved": all(
            row["belief_multiset_preserved"] for row in diagnostics
        ),
        "nonbelief_fields_unchanged": (
            original_nonbelief_hash == transformed_nonbelief_hash
        ),
        "original_nonbelief_sha256": original_nonbelief_hash,
        "transformed_nonbelief_sha256": transformed_nonbelief_hash,
    }


def load_source(path: Path, *, stage: str) -> dict[str, Any]:
    expected_hash = (
        SMOKE_ARTIFACT_SHA256
        if stage == "serving_smoke"
        else CONFIRMATION_ARTIFACT_SHA256
    )
    if sha256_file(path) != expected_hash:
        raise ValueError("frozen tau source artifact hash does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError("frozen tau source artifact did not pass")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("frozen tau source artifact lacks records")
    expected_ids: Sequence[str] = (
        SMOKE_IDS if stage == "serving_smoke" else FRESH_CONFIRMATION_IDS
    )
    if tuple(record.get("task_id") for record in records) != tuple(
        expected_ids
    ):
        raise ValueError("frozen tau task IDs do not match the stage")
    if stage == "confirmation" and not isinstance(
        payload.get("myopic_scores"), list
    ):
        raise ValueError("frozen tau source artifact lacks myopic scores")
    return payload


def _task_ranking_accuracies(
    records: Sequence[dict[str, Any]],
    full_scores: Sequence[dict[str, Any]],
    shuffled_scores: Sequence[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    full = []
    shuffled = []
    for record, full_score, shuffled_score in zip(
        records, full_scores, shuffled_scores, strict=True
    ):
        root_values = [
            max(values) for values in analyze_record(record)["pair_counts"]
        ]
        full_points, comparable = pairwise_ranking_points(
            full_score["scores"], root_values
        )
        shuffled_points, shuffled_comparable = pairwise_ranking_points(
            shuffled_score["scores"], root_values
        )
        if comparable != shuffled_comparable:
            raise ValueError("root comparable counts changed under ablation")
        full.append(full_points / comparable if comparable else 0.0)
        shuffled.append(
            shuffled_points / comparable if comparable else 0.0
        )
    return full, shuffled


def _group_root_rows(summary: dict[str, Any]) -> list[list[dict[str, Any]]]:
    rows = summary["root_diagnostics"]
    return [
        rows[index : index + FIRST_QUERY_COUNT]
        for index in range(0, len(rows), FIRST_QUERY_COUNT)
    ]


def _task_continuation_accuracies(
    full_summary: dict[str, Any],
    shuffled_summary: dict[str, Any],
) -> tuple[list[float], list[float]]:
    full = []
    shuffled = []
    for full_rows, shuffled_rows in zip(
        _group_root_rows(full_summary),
        _group_root_rows(shuffled_summary),
        strict=True,
    ):
        full_comparable = sum(
            row["pairwise_comparable_count"] for row in full_rows
        )
        shuffled_comparable = sum(
            row["pairwise_comparable_count"] for row in shuffled_rows
        )
        if full_comparable != shuffled_comparable:
            raise ValueError(
                "continuation comparable counts changed under ablation"
            )
        full.append(
            sum(row["pairwise_points"] for row in full_rows)
            / full_comparable
            if full_comparable
            else 0.0
        )
        shuffled.append(
            sum(row["pairwise_points"] for row in shuffled_rows)
            / full_comparable
            if full_comparable
            else 0.0
        )
    return full, shuffled


def compare_to_full_alignment(
    source: dict[str, Any],
    shuffled_summary: dict[str, Any],
    shuffled_root_scores: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    full_summary = source["summary"]
    full_root, shuffled_root = _task_ranking_accuracies(
        source["records"],
        source["nonmyopic_scores"],
        shuffled_root_scores,
    )
    full_continuation, shuffled_continuation = (
        _task_continuation_accuracies(full_summary, shuffled_summary)
    )
    full_endpoint = [
        row["nonmyopic_receding_value"]
        for row in full_summary["policy_diagnostics"]
    ]
    shuffled_endpoint = [
        row["nonmyopic_receding_value"]
        for row in shuffled_summary["policy_diagnostics"]
    ]
    root_delta = (
        float(full_summary["nonmyopic_root_pairwise_accuracy"])
        - float(shuffled_summary["nonmyopic_root_pairwise_accuracy"])
    )
    continuation_delta = (
        float(full_summary["focused_pairwise_accuracy"])
        - float(shuffled_summary["focused_pairwise_accuracy"])
    )
    endpoint_delta = sum(full_endpoint) - sum(shuffled_endpoint)
    return {
        "full_root_pairwise_accuracy": full_summary[
            "nonmyopic_root_pairwise_accuracy"
        ],
        "shuffled_root_pairwise_accuracy": shuffled_summary[
            "nonmyopic_root_pairwise_accuracy"
        ],
        "full_minus_shuffled_root_accuracy": root_delta,
        "root_task_sign_flip_one_sided_p": exact_sign_flip_pvalue(
            [
                full_value - shuffled_value
                for full_value, shuffled_value in zip(
                    full_root, shuffled_root, strict=True
                )
            ]
        ),
        "full_continuation_pairwise_accuracy": full_summary[
            "focused_pairwise_accuracy"
        ],
        "shuffled_continuation_pairwise_accuracy": shuffled_summary[
            "focused_pairwise_accuracy"
        ],
        "full_minus_shuffled_continuation_accuracy": continuation_delta,
        "continuation_task_sign_flip_one_sided_p": exact_sign_flip_pvalue(
            [
                full_value - shuffled_value
                for full_value, shuffled_value in zip(
                    full_continuation, shuffled_continuation, strict=True
                )
            ]
        ),
        "full_endpoint_total": sum(full_endpoint),
        "shuffled_endpoint_total": sum(shuffled_endpoint),
        "full_minus_shuffled_endpoint_total": endpoint_delta,
        "endpoint_task_sign_flip_one_sided_p": exact_sign_flip_pvalue(
            [
                full_value - shuffled_value
                for full_value, shuffled_value in zip(
                    full_endpoint, shuffled_endpoint, strict=True
                )
            ]
        ),
    }


def summarize_ablation(
    source: dict[str, Any],
    continuation_scores: Sequence[Sequence[dict[str, Any]]],
    usage: dict[str, Any],
    *,
    stage: str,
    shuffled_root_scores: Sequence[dict[str, Any]],
    intervention: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    compatibility_usage = dict(usage)
    compatibility_usage["physical_requests"] = EXPECTED_REQUESTS[stage]
    summary = summarize(
        source["records"],
        continuation_scores,
        compatibility_usage,
        stage=stage,
        myopic_scores=source["myopic_scores"],
        nonmyopic_scores=shuffled_root_scores,
    )
    summary["reference_efficacy_gates"] = summary.pop("gates")
    gates = {
        "all_cases_complete": len(source["records"])
        == (len(SMOKE_IDS) if stage == "serving_smoke" else 20),
        "all_root_scores_complete": len(shuffled_root_scores)
        == len(source["records"]),
        "all_focused_scores_complete": len(
            summary["root_diagnostics"]
        )
        == len(source["records"]) * FIRST_QUERY_COUNT,
        "exact_physical_request_count": int(usage["physical_requests"])
        == ABLATION_EXPECTED_REQUESTS[stage],
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_permutations_are_derangements": intervention[
            "all_permutations_are_derangements"
        ],
        "all_belief_multisets_preserved": intervention[
            "all_belief_multisets_preserved"
        ],
        "nonbelief_fields_unchanged": intervention[
            "nonbelief_fields_unchanged"
        ],
    }
    comparison = None
    if stage == "confirmation":
        comparison = compare_to_full_alignment(
            source,
            summary,
            shuffled_root_scores,
        )
        gates.update(
            {
                "root_accuracy_drop_at_least_0_05": comparison[
                    "full_minus_shuffled_root_accuracy"
                ]
                >= 0.05,
                "continuation_accuracy_drop_at_least_0_05": comparison[
                    "full_minus_shuffled_continuation_accuracy"
                ]
                >= 0.05,
                "endpoint_drop_at_least_3": comparison[
                    "full_minus_shuffled_endpoint_total"
                ]
                >= 3,
            }
        )
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary, comparison


def run_gate(
    config: Config,
    *,
    stage: str,
    input_artifact: Path,
    raw_checkpoint_path: Path,
) -> dict[str, Any]:
    if config.model_pairs[0].questioner.model != MODEL_ID:
        raise ValueError("ablation config does not select GPT-5.4")
    source = load_source(input_artifact, stage=stage)
    records, permutations = derange_refreshed_beliefs(source["records"])
    intervention = validate_intervention(
        source["records"], records, permutations
    )
    if not all(
        intervention[key]
        for key in (
            "all_permutations_are_derangements",
            "all_belief_multisets_preserved",
            "nonbelief_fields_unchanged",
        )
    ):
        raise ValueError("belief-alignment intervention invariant failed")

    model = build_model_adapter(config.model_pairs[0].questioner, config)
    raw: dict[str, Any] = {}
    try:
        root_raw = model.chat_complete_messages_batched(
            [
                scorer_messages(record, include_followups=True)
                for record in records
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["shuffled_root_scores"] = root_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        root_scores = [
            parse_scores(text, include_followups=True) for text in root_raw
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
        raw["shuffled_continuation_scores"] = focused_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        parsed_focused = [
            parse_zero_padding_scores(text) for text in focused_raw
        ]
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

    summary, comparison = summarize_ablation(
        source,
        continuation_scores,
        usage,
        stage=stage,
        shuffled_root_scores=root_scores,
        intervention=intervention,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "source_artifact_sha256": sha256_file(input_artifact),
            "task_ids": [record["task_id"] for record in records],
            "permutation_seed": PERMUTATION_SEED,
            "expected_physical_requests": ABLATION_EXPECTED_REQUESTS[stage],
            "tree_regeneration_requests": 0,
            "reasoning_requested": False,
            "target_blind_prompts": True,
            "posthoc_open_tree_ablation": True,
            "raw_responses_private_and_untracked": True,
        },
        "intervention": intervention,
        "permutations": permutations,
        "summary": summary,
        "comparison_to_full_alignment": comparison,
        "shuffled_root_scores": root_scores,
        "shuffled_continuation_scores": continuation_scores,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.15
        config.openrouter_run_budget_usd = 0.50
    else:
        config.openrouter_projected_cost_usd = 1.50
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
                "comparison": payload["comparison_to_full_alignment"],
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
