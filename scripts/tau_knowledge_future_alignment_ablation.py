#!/usr/bin/env python3
"""Test whether tau root ranking depends on correct future-subtree alignment."""

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
from scripts.tau_knowledge_belief_alignment_ablation import (
    CONFIRMATION_ARTIFACT_SHA256,
    SMOKE_ARTIFACT_SHA256,
    load_source,
    sha256_file,
)
from scripts.tau_knowledge_first_link_scorer import (
    compact_scorer_input,
    pairwise_ranking_points,
)
from scripts.tau_knowledge_retrieval_opportunity import (
    FIRST_QUERY_COUNT,
    FOLLOWUP_QUERY_COUNT,
    GateExecutionError,
    SCHEMA_VERSION,
    _checkpoint,
    _usage_snapshot,
    analyze_record,
)


INTERFACE_VERSION = "future-alignment-paired-1"
MODEL_ID = "openai/gpt-5.4"
PERMUTATION_SEED = 24_361
BLINDING_SEED = 24_362
EXPECTED_REQUESTS = {"serving_smoke": 2, "confirmation": 20}
MAX_PAIRED_INPUT_CHARS = 200_000


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _future_subtree(branch: dict[str, Any]) -> dict[str, Any]:
    return {
        "refreshed_information_need_hypotheses": branch[
            "refreshed_information_need_hypotheses"
        ],
        "followups": branch["followups"],
    }


def _records_without_future_subtrees(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    stripped = copy.deepcopy(list(records))
    for record in stripped:
        for branch in record["first_branches"]:
            branch.pop("refreshed_information_need_hypotheses")
            branch.pop("followups")
    return stripped


def _future_multiset(record: dict[str, Any]) -> list[str]:
    return sorted(
        json.dumps(
            _future_subtree(branch),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for branch in record["first_branches"]
    )


def _task_derangement(task_id: str, size: int) -> list[int]:
    digest = hashlib.sha256(
        f"{PERMUTATION_SEED}:{task_id}".encode("ascii")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    indices = list(range(size))
    for _ in range(1_000):
        rng.shuffle(indices)
        if all(source != target for target, source in enumerate(indices)):
            return list(indices)
    raise RuntimeError("could not construct future-subtree derangement")


def derange_future_subtrees(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transformed = copy.deepcopy(list(records))
    diagnostics = []
    for original, changed in zip(records, transformed, strict=True):
        branches = original["first_branches"]
        permutation = _task_derangement(original["task_id"], len(branches))
        for target, source in enumerate(permutation):
            subtree = copy.deepcopy(_future_subtree(branches[source]))
            changed_branch = changed["first_branches"][target]
            changed_branch["refreshed_information_need_hypotheses"] = subtree[
                "refreshed_information_need_hypotheses"
            ]
            changed_branch["followups"] = subtree["followups"]
        diagnostics.append(
            {
                "task_id": original["task_id"],
                "source_branch_for_target": permutation,
                "is_derangement": all(
                    source != target
                    for target, source in enumerate(permutation)
                ),
                "future_subtree_multiset_preserved": (
                    _future_multiset(original) == _future_multiset(changed)
                ),
            }
        )
    return transformed, diagnostics


def validate_intervention(
    original: Sequence[dict[str, Any]],
    transformed: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    original_nonfuture_hash = _canonical_hash(
        _records_without_future_subtrees(original)
    )
    transformed_nonfuture_hash = _canonical_hash(
        _records_without_future_subtrees(transformed)
    )
    return {
        "all_permutations_are_derangements": all(
            row["is_derangement"] for row in diagnostics
        ),
        "all_future_subtree_multisets_preserved": all(
            row["future_subtree_multiset_preserved"] for row in diagnostics
        ),
        "nonfuture_fields_unchanged": (
            original_nonfuture_hash == transformed_nonfuture_hash
        ),
        "original_nonfuture_sha256": original_nonfuture_hash,
        "transformed_nonfuture_sha256": transformed_nonfuture_hash,
    }


def blinding_assignments(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    indices = list(range(len(records)))
    random.Random(BLINDING_SEED).shuffle(indices)
    aligned_a = set(indices[: len(indices) // 2])
    return [
        {
            "task_id": record["task_id"],
            "condition_a": "aligned" if index in aligned_a else "shuffled",
            "condition_b": "shuffled" if index in aligned_a else "aligned",
        }
        for index, record in enumerate(records)
    ]


def _paired_schema() -> dict[str, Any]:
    return {
        "condition_a_scores": (
            f"array of {FIRST_QUERY_COUNT} canonical 0-100 digit strings"
        ),
        "condition_a_best_followups": (
            f"array of {FIRST_QUERY_COUNT} canonical 1-"
            f"{FOLLOWUP_QUERY_COUNT} digit strings"
        ),
        "condition_b_scores": (
            f"array of {FIRST_QUERY_COUNT} canonical 0-100 digit strings"
        ),
        "condition_b_best_followups": (
            f"array of {FIRST_QUERY_COUNT} canonical 1-"
            f"{FOLLOWUP_QUERY_COUNT} digit strings"
        ),
    }


def paired_scorer_messages(
    original: dict[str, Any],
    transformed: dict[str, Any],
    assignment: dict[str, Any],
) -> list[dict[str, str]]:
    aligned = compact_scorer_input(original, include_followups=True)
    shuffled = compact_scorer_input(transformed, include_followups=True)
    condition_a = (
        aligned if assignment["condition_a"] == "aligned" else shuffled
    )
    condition_b = (
        aligned if assignment["condition_b"] == "aligned" else shuffled
    )
    payload = {"condition_a": condition_a, "condition_b": condition_b}
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > MAX_PAIRED_INPUT_CHARS:
        raise ValueError("paired scorer input exceeds frozen character cap")
    return [
        {
            "role": "system",
            "content": (
                "Evaluate retrieval coverage for an internal banking support "
                "task. Required-document labels and evaluation answers are "
                "hidden. Return one flat strict JSON object and no prose."
            ),
        },
        {
            "role": "user",
            "content": (
                "The same five first-search roots appear under two candidate "
                "future-branch assignments, condition A and condition B. "
                "Evaluate each condition independently. Score each root by the "
                "best total coverage achievable after that root and exactly one "
                "of its shown followups. Reward distinct policy prerequisites, "
                "exceptions, eligibility rules, and procedures that resolve the "
                "customer's inferred information needs. Do not reward document "
                "count, verbosity, or duplicate coverage. For every root choose "
                "its best shown followup. Use the 0-100 range and distinguish "
                "roots whenever useful coverage differs. Return exactly these "
                "keys: "
                + json.dumps(
                    _paired_schema(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + ". Retrieval data: "
                + encoded
            ),
        },
    ]


def _parse_digit_array(
    value: Any, *, size: int, minimum: int, maximum: int
) -> list[int]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError("paired scorer array has invalid length")
    parsed = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item.isdigit()
            or (len(item) > 1 and item.startswith("0"))
        ):
            raise ValueError("paired scorer values must be canonical digit strings")
        number = int(item)
        if not minimum <= number <= maximum:
            raise ValueError("paired scorer value is outside the allowed range")
        parsed.append(number)
    return parsed


def parse_paired_scores(text: str) -> dict[str, dict[str, list[int]]]:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    try:
        payload, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("paired scorer response is not strict JSON") from exc
    if stripped[end:].strip() or not isinstance(payload, dict):
        raise ValueError("paired scorer response has extra data")
    if set(payload) != set(_paired_schema()):
        raise ValueError("paired scorer response has unexpected keys")
    result = {}
    for condition in ("condition_a", "condition_b"):
        result[condition] = {
            "scores": _parse_digit_array(
                payload[f"{condition}_scores"],
                size=FIRST_QUERY_COUNT,
                minimum=0,
                maximum=100,
            ),
            "best_followup_indices": [
                value - 1
                for value in _parse_digit_array(
                    payload[f"{condition}_best_followups"],
                    size=FIRST_QUERY_COUNT,
                    minimum=1,
                    maximum=FOLLOWUP_QUERY_COUNT,
                )
            ],
        }
    return result


def unblind_scores(
    parsed: dict[str, dict[str, list[int]]],
    assignment: dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    by_label = {
        assignment["condition_a"]: parsed["condition_a"],
        assignment["condition_b"]: parsed["condition_b"],
    }
    return by_label["aligned"], by_label["shuffled"]


def _selected_index(scores: Sequence[int]) -> int:
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def _condition_metrics(
    records: Sequence[dict[str, Any]],
    scores: Sequence[dict[str, list[int]]],
) -> dict[str, Any]:
    total_points = 0.0
    total_comparable = 0
    task_accuracies = []
    selected_values = []
    selected_indices = []
    optimal_count = 0
    varying_count = 0
    root_values_by_task = []
    for record, score in zip(records, scores, strict=True):
        root_values = [
            max(values) for values in analyze_record(record)["pair_counts"]
        ]
        root_values_by_task.append(root_values)
        points, comparable = pairwise_ranking_points(
            score["scores"], root_values
        )
        total_points += points
        total_comparable += comparable
        task_accuracies.append(points / comparable if comparable else 0.0)
        selected = _selected_index(score["scores"])
        selected_indices.append(selected)
        selected_values.append(root_values[selected])
        optimal_count += root_values[selected] == max(root_values)
        varying_count += len(set(score["scores"])) > 1
    return {
        "pairwise_accuracy": (
            total_points / total_comparable if total_comparable else 0.0
        ),
        "pairwise_points": total_points,
        "pairwise_comparable_count": total_comparable,
        "task_pairwise_accuracies": task_accuracies,
        "selected_root_values": selected_values,
        "selected_root_indices": selected_indices,
        "selected_root_total": sum(selected_values),
        "optimal_root_count": optimal_count,
        "varying_score_vector_count": varying_count,
        "root_values_by_task": root_values_by_task,
    }


def _existing_myopic_metrics(
    source: dict[str, Any],
) -> dict[str, Any] | None:
    myopic = source.get("myopic_scores")
    if not isinstance(myopic, list):
        return None
    return _condition_metrics(source["records"], myopic)


def summarize_paired(
    source: dict[str, Any],
    *,
    aligned_scores: Sequence[dict[str, list[int]]],
    shuffled_scores: Sequence[dict[str, list[int]]],
    assignments: Sequence[dict[str, Any]],
    intervention: dict[str, Any],
    usage: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    aligned = _condition_metrics(source["records"], aligned_scores)
    shuffled = _condition_metrics(source["records"], shuffled_scores)
    myopic = _existing_myopic_metrics(source)
    task_deltas = [
        left - right
        for left, right in zip(
            aligned["task_pairwise_accuracies"],
            shuffled["task_pairwise_accuracies"],
            strict=True,
        )
    ]
    endpoint_deltas = [
        left - right
        for left, right in zip(
            aligned["selected_root_values"],
            shuffled["selected_root_values"],
            strict=True,
        )
    ]
    wins = sum(value > 0 for value in endpoint_deltas)
    losses = sum(value < 0 for value in endpoint_deltas)
    ties = sum(value == 0 for value in endpoint_deltas)
    score_vectors_changed = sum(
        aligned_row["scores"] != shuffled_row["scores"]
        for aligned_row, shuffled_row in zip(
            aligned_scores, shuffled_scores, strict=True
        )
    )
    a_aligned = sum(
        row["condition_a"] == "aligned" for row in assignments
    )
    gates = {
        "all_cases_complete": len(aligned_scores)
        == EXPECTED_REQUESTS[stage],
        "exact_physical_request_count": int(usage["physical_requests"])
        == EXPECTED_REQUESTS[stage],
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_permutations_are_derangements": intervention[
            "all_permutations_are_derangements"
        ],
        "all_future_subtree_multisets_preserved": intervention[
            "all_future_subtree_multisets_preserved"
        ],
        "nonfuture_fields_unchanged": intervention[
            "nonfuture_fields_unchanged"
        ],
        "balanced_blinding": a_aligned == len(assignments) // 2,
        "all_aligned_score_vectors_vary": (
            aligned["varying_score_vector_count"] == len(aligned_scores)
        ),
        "all_shuffled_score_vectors_vary": (
            shuffled["varying_score_vector_count"] == len(shuffled_scores)
        ),
    }
    if stage == "confirmation":
        if myopic is None:
            raise ValueError("confirmation source lacks myopic scores")
        gates.update(
            {
                "aligned_pairwise_accuracy_at_least_0_60": (
                    aligned["pairwise_accuracy"] >= 0.60
                ),
                "aligned_minus_myopic_accuracy_at_least_0_05": (
                    aligned["pairwise_accuracy"] - myopic["pairwise_accuracy"]
                    >= 0.05
                ),
                "aligned_minus_shuffled_accuracy_at_least_0_05": (
                    aligned["pairwise_accuracy"]
                    - shuffled["pairwise_accuracy"]
                    >= 0.05
                ),
                "aligned_endpoint_advantage_at_least_3": (
                    aligned["selected_root_total"]
                    - shuffled["selected_root_total"]
                    >= 3
                ),
                "aligned_endpoint_wins_at_least_3": wins >= 3,
                "aligned_endpoint_losses_at_most_2": losses <= 2,
                "score_vectors_change_on_at_least_15_tasks": (
                    score_vectors_changed >= 15
                ),
            }
        )
    gates["all_pass"] = all(gates.values())
    return {
        "aligned": aligned,
        "shuffled": shuffled,
        "existing_myopic": myopic,
        "aligned_minus_shuffled_pairwise_accuracy": (
            aligned["pairwise_accuracy"] - shuffled["pairwise_accuracy"]
        ),
        "aligned_minus_shuffled_endpoint_total": (
            aligned["selected_root_total"] - shuffled["selected_root_total"]
        ),
        "aligned_vs_shuffled_endpoint_wins": wins,
        "aligned_vs_shuffled_endpoint_ties": ties,
        "aligned_vs_shuffled_endpoint_losses": losses,
        "score_vectors_changed_count": score_vectors_changed,
        "task_pairwise_sign_flip_one_sided_p": exact_sign_flip_pvalue(
            task_deltas
        ),
        "endpoint_sign_flip_one_sided_p": exact_sign_flip_pvalue(
            endpoint_deltas
        ),
        "gates": gates,
    }


def run_gate(
    config: Config,
    *,
    stage: str,
    input_artifact: Path,
    raw_checkpoint_path: Path,
) -> dict[str, Any]:
    if config.model_pairs[0].questioner.model != MODEL_ID:
        raise ValueError("future-alignment config does not select GPT-5.4")
    source = load_source(input_artifact, stage=stage)
    records = source["records"]
    transformed, permutations = derange_future_subtrees(records)
    intervention = validate_intervention(
        records, transformed, permutations
    )
    if not all(
        intervention[key]
        for key in (
            "all_permutations_are_derangements",
            "all_future_subtree_multisets_preserved",
            "nonfuture_fields_unchanged",
        )
    ):
        raise ValueError("future-subtree intervention invariant failed")
    assignments = blinding_assignments(records)
    messages = [
        paired_scorer_messages(original, changed, assignment)
        for original, changed, assignment in zip(
            records, transformed, assignments, strict=True
        )
    ]
    model = build_model_adapter(config.model_pairs[0].questioner, config)
    raw: dict[str, Any] = {}
    try:
        responses = model.chat_complete_messages_batched(
            messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["paired_scores"] = responses
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        parsed = [parse_paired_scores(text) for text in responses]
        unblinded = [
            unblind_scores(score, assignment)
            for score, assignment in zip(
                parsed, assignments, strict=True
            )
        ]
        aligned_scores = [values[0] for values in unblinded]
        shuffled_scores = [values[1] for values in unblinded]
        usage = _usage_snapshot(model)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    summary = summarize_paired(
        source,
        aligned_scores=aligned_scores,
        shuffled_scores=shuffled_scores,
        assignments=assignments,
        intervention=intervention,
        usage=usage,
        stage=stage,
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
            "blinding_seed": BLINDING_SEED,
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "tree_regeneration_requests": 0,
            "reasoning_requested": False,
            "target_blind_prompts": True,
            "posthoc_open_tree_causal_ablation": True,
            "raw_responses_private_and_untracked": True,
        },
        "intervention": intervention,
        "permutations": permutations,
        "blinding_assignments": assignments,
        "summary": summary,
        "aligned_scores": aligned_scores,
        "shuffled_scores": shuffled_scores,
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
        config.openrouter_projected_cost_usd = 0.05
        config.openrouter_run_budget_usd = 0.25
    else:
        config.openrouter_projected_cost_usd = 0.40
        config.openrouter_run_budget_usd = 1.00
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
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
