#!/usr/bin/env python3
"""Run a paired causal smoke test of tau path-dependent belief states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.tau_knowledge_cross_model_scorer import (
    SMOKE_ARTIFACT_SHA256,
    load_records,
    sha256_file,
)
from scripts.tau_knowledge_first_link_scorer import pairwise_ranking_points
from scripts.tau_knowledge_retrieval_opportunity import (
    FIRST_QUERY_COUNT,
    GateExecutionError,
    SCHEMA_VERSION,
    _build_model,
    _checkpoint,
    _usage_snapshot,
    analyze_record,
)


INTERFACE_VERSION = "belief-bottleneck-paired-1"
MODEL_ID = "openai/gpt-5.4"
SELECTION_SEED = 24348
EXPECTED_REQUESTS = 10
EVIDENCE_EXCERPT_CHARS = 500
SCORE_BANDS = tuple(
    list(range(0, 10))
    + list(range(30, 40))
    + list(range(60, 70))
    + list(range(90, 100))
)
MAX_COST_USD = 0.25


def aligned_label_schedule(
    *,
    seed: int = SELECTION_SEED,
) -> list[bool]:
    """Return True when the aligned state is label A."""
    labels = [True] * (EXPECTED_REQUESTS // 2) + [False] * (
        EXPECTED_REQUESTS // 2
    )
    random.Random(seed).shuffle(labels)
    return labels


def _clean_excerpt(value: str) -> str:
    return " ".join(value.split())[:EVIDENCE_EXCERPT_CHARS]


def compact_candidate_evidence(
    record: dict[str, Any],
    root_index: int,
) -> list[dict[str, Any]]:
    branch = record["first_branches"][root_index]
    return [
        {
            "candidate": followup_index,
            "documents": [
                {
                    "title": str(result["title"]),
                    "excerpt": _clean_excerpt(str(result["content"])),
                }
                for result in followup["results"]
            ],
        }
        for followup_index, followup in enumerate(
            branch["followups"], start=1
        )
    ]


def score_schema() -> dict[str, str]:
    return {
        f"state_{label}_followup_{index}_score": (
            "canonical decimal digit string in one of the allowed bands"
        )
        for label in ("a", "b")
        for index in range(1, FIRST_QUERY_COUNT)
    }


def scorer_messages(
    record: dict[str, Any],
    root_index: int,
    *,
    aligned_is_a: bool,
) -> list[dict[str, str]]:
    branches = record["first_branches"]
    aligned = branches[root_index][
        "refreshed_information_need_hypotheses"
    ]
    shuffled = branches[(root_index + 1) % FIRST_QUERY_COUNT][
        "refreshed_information_need_hypotheses"
    ]
    state_a, state_b = (
        (aligned, shuffled) if aligned_is_a else (shuffled, aligned)
    )
    payload = {
        "state_a_unresolved_information_needs": state_a,
        "state_b_unresolved_information_needs": state_b,
        "candidate_returned_evidence": compact_candidate_evidence(
            record, root_index
        ),
        "required_output": score_schema(),
    }
    return [
        {
            "role": "system",
            "content": (
                "You rank returned evidence using only the supplied unresolved "
                "information-need state. Score each state independently. A "
                "document is useful only if its supplied title or excerpt "
                "materially supports resolving at least one need in that state. "
                "Count distinct useful documents in each candidate: use 0-9 for "
                "zero, 30-39 for one, 60-69 for two, and 90-99 for three. The "
                "ones digit may express a weak within-count preference, but one "
                "additional useful document must dominate. Return exactly the "
                "requested flat JSON object with canonical digit strings and no "
                "explanation."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        },
    ]


def parse_paired_scores(text: str) -> dict[str, list[int]]:
    payload = _parse_json_object(text)
    if set(payload) != set(score_schema()):
        raise ValueError("paired scorer response has unexpected keys")
    parsed: dict[str, list[int]] = {}
    for label in ("a", "b"):
        scores = []
        for index in range(1, FIRST_QUERY_COUNT):
            value = payload[f"state_{label}_followup_{index}_score"]
            if (
                not isinstance(value, str)
                or not value.isdigit()
                or (len(value) > 1 and value.startswith("0"))
            ):
                raise ValueError("paired score is not a canonical digit string")
            score = int(value)
            if score not in SCORE_BANDS:
                raise ValueError("paired score is outside a count band")
            scores.append(score)
        parsed[label] = scores
    return parsed


def _normalized_state(hypotheses: Sequence[str]) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+", " ".join(hypotheses).casefold())
    )


def prepare_rows(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = aligned_label_schedule()
    rows = []
    label_index = 0
    for record in records:
        branches = record["first_branches"]
        if len(branches) != FIRST_QUERY_COUNT:
            raise ValueError("source record has the wrong root count")
        for root_index in range(FIRST_QUERY_COUNT):
            shuffled_root_index = (root_index + 1) % FIRST_QUERY_COUNT
            aligned_state = branches[root_index][
                "refreshed_information_need_hypotheses"
            ]
            shuffled_state = branches[shuffled_root_index][
                "refreshed_information_need_hypotheses"
            ]
            rows.append(
                {
                    "record": record,
                    "task_id": record["task_id"],
                    "root_index": root_index,
                    "shuffled_root_index": shuffled_root_index,
                    "aligned_is_a": labels[label_index],
                    "state_pair_distinct": (
                        _normalized_state(aligned_state)
                        != _normalized_state(shuffled_state)
                    ),
                }
            )
            label_index += 1
    if len(rows) != EXPECTED_REQUESTS:
        raise ValueError("source artifact does not contain exactly ten roots")
    return rows


def _selected_index(scores: Sequence[int]) -> int:
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def evaluate_row(
    prepared: dict[str, Any],
    parsed: dict[str, list[int]],
) -> dict[str, Any]:
    aligned_label = "a" if prepared["aligned_is_a"] else "b"
    shuffled_label = "b" if prepared["aligned_is_a"] else "a"
    aligned_scores = parsed[aligned_label]
    shuffled_scores = parsed[shuffled_label]
    pair_values = analyze_record(prepared["record"])["pair_counts"][
        prepared["root_index"]
    ]
    aligned_points, aligned_comparable = pairwise_ranking_points(
        aligned_scores, pair_values
    )
    shuffled_points, shuffled_comparable = pairwise_ranking_points(
        shuffled_scores, pair_values
    )
    if aligned_comparable != shuffled_comparable:
        raise ValueError("paired interventions have different comparisons")
    aligned_selected = _selected_index(aligned_scores)
    shuffled_selected = _selected_index(shuffled_scores)
    oracle_value = max(pair_values)
    return {
        "task_id": prepared["task_id"],
        "root_index": prepared["root_index"],
        "shuffled_root_index": prepared["shuffled_root_index"],
        "aligned_label": aligned_label.upper(),
        "state_pair_distinct": prepared["state_pair_distinct"],
        "pair_values": pair_values,
        "aligned_scores": aligned_scores,
        "shuffled_scores": shuffled_scores,
        "aligned_scores_vary": len(set(aligned_scores)) > 1,
        "pairwise_comparable_count": aligned_comparable,
        "aligned_pairwise_points": aligned_points,
        "shuffled_pairwise_points": shuffled_points,
        "aligned_selected_followup_index": aligned_selected,
        "shuffled_selected_followup_index": shuffled_selected,
        "aligned_selected_exact_value": pair_values[aligned_selected],
        "shuffled_selected_exact_value": pair_values[shuffled_selected],
        "aligned_is_oracle_optimal": (
            pair_values[aligned_selected] == oracle_value
        ),
        "shuffled_is_oracle_optimal": (
            pair_values[shuffled_selected] == oracle_value
        ),
    }


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    comparable = sum(row["pairwise_comparable_count"] for row in rows)
    aligned_points = sum(row["aligned_pairwise_points"] for row in rows)
    shuffled_points = sum(row["shuffled_pairwise_points"] for row in rows)
    aligned_accuracy = aligned_points / comparable if comparable else 0.0
    shuffled_accuracy = shuffled_points / comparable if comparable else 0.0
    aligned_optimal = sum(row["aligned_is_oracle_optimal"] for row in rows)
    shuffled_optimal = sum(row["shuffled_is_oracle_optimal"] for row in rows)
    aligned_total = sum(row["aligned_selected_exact_value"] for row in rows)
    shuffled_total = sum(
        row["shuffled_selected_exact_value"] for row in rows
    )
    generator = usage.get("generator", {})
    gates = {
        "exact_10_physical_requests": (
            int(usage.get("physical_requests", -1)) == EXPECTED_REQUESTS
        ),
        "exact_10_http_attempts": (
            int(generator.get("http_attempts", -1)) == EXPECTED_REQUESTS
        ),
        "zero_transport_retries": (
            int(generator.get("retry_count", -1)) == 0
        ),
        "zero_reasoning_tokens": int(usage.get("reasoning_tokens", -1)) == 0,
        "zero_forced_exits": int(generator.get("forced_exits", -1)) == 0,
        "all_responses_parsed_without_repair": len(rows)
        == EXPECTED_REQUESTS,
        "all_state_pairs_distinct": all(
            row["state_pair_distinct"] for row in rows
        ),
        "balanced_blinded_labels": (
            sum(row["aligned_label"] == "A" for row in rows) == 5
            and sum(row["aligned_label"] == "B" for row in rows) == 5
        ),
        "aligned_score_variation_at_least_8": (
            sum(row["aligned_scores_vary"] for row in rows) >= 8
        ),
        "aligned_pairwise_accuracy_at_least_0_60": (
            aligned_accuracy >= 0.60
        ),
        "aligned_optimal_count_at_least_7": aligned_optimal >= 7,
        "aligned_accuracy_gain_at_least_0_10": (
            aligned_accuracy - shuffled_accuracy >= 0.10
        ),
        "aligned_optimal_count_gain_at_least_2": (
            aligned_optimal - shuffled_optimal >= 2
        ),
        "aligned_selected_exact_total_gain_at_least_2": (
            aligned_total - shuffled_total >= 2
        ),
        "cost_at_most_0_25": (
            float(usage.get("adapter_cost_usd", float("inf")))
            <= MAX_COST_USD
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "num_rows": len(rows),
        "pairwise_comparable_count": comparable,
        "aligned_pairwise_accuracy": aligned_accuracy,
        "shuffled_pairwise_accuracy": shuffled_accuracy,
        "aligned_minus_shuffled_pairwise_accuracy": (
            aligned_accuracy - shuffled_accuracy
        ),
        "aligned_optimal_count": aligned_optimal,
        "shuffled_optimal_count": shuffled_optimal,
        "aligned_optimal_count_gain": aligned_optimal - shuffled_optimal,
        "aligned_selected_exact_total": aligned_total,
        "shuffled_selected_exact_total": shuffled_total,
        "aligned_selected_exact_total_gain": aligned_total - shuffled_total,
        "aligned_score_variation_count": sum(
            row["aligned_scores_vary"] for row in rows
        ),
        "gates": gates,
    }


def run_gate(
    config: Config,
    *,
    input_artifact: Path,
    raw_checkpoint_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    if config.model_pairs[0].questioner.model != MODEL_ID:
        raise ValueError("belief-bottleneck config selects the wrong model")
    records = load_records(input_artifact, stage="serving_smoke")
    prepared = prepare_rows(records)
    if not all(row["state_pair_distinct"] for row in prepared):
        raise ValueError("frozen aligned/shuffled state pair is not distinct")
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {}
    try:
        responses = model.chat_complete_messages_batched(
            [
                scorer_messages(
                    row["record"],
                    row["root_index"],
                    aligned_is_a=row["aligned_is_a"],
                )
                for row in prepared
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["paired_scores"] = responses
        _checkpoint(raw_checkpoint_path, stage="serving_smoke", raw=raw)
        parsed = [parse_paired_scores(text) for text in responses]
        usage = _usage_snapshot(model)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc
    rows = [
        evaluate_row(row, scores)
        for row, scores in zip(prepared, parsed, strict=True)
    ]
    summary = summarize_rows(rows, usage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "source_artifact_sha256": SMOKE_ARTIFACT_SHA256,
            "selection_seed": SELECTION_SEED,
            "expected_physical_requests": EXPECTED_REQUESTS,
            "aligned_label_a_count": 5,
            "aligned_label_b_count": 5,
            "shuffled_state_rule": "next root cyclic within task",
            "evidence_excerpt_chars": EVIDENCE_EXCERPT_CHARS,
            "opening_hidden": True,
            "initial_beliefs_hidden": True,
            "all_queries_hidden": True,
            "first_results_hidden": True,
            "document_ids_hidden": True,
            "bm25_scores_hidden": True,
            "endpoints_hidden": True,
            "paired_interventions_in_same_response": True,
            "reasoning_requested": False,
            "repairs_or_scientific_retries": 0,
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "rows": rows,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.15
    config.openrouter_run_budget_usd = MAX_COST_USD
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_gate(
            config,
            input_artifact=args.input_artifact,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        (args.output_dir / "SERVING_SMOKE_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = args.output_dir / "SERVING_SMOKE.json"
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
