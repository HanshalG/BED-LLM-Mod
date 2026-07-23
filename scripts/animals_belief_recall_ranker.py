#!/usr/bin/env python3
"""Rank 20-Questions candidates by target-blind belief-regeneration quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter


SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You rank candidate questions for an open-world 20 Questions agent. "
        "After the chosen question is answered, a separate language model must "
        "regenerate a finite list of plausible animal names consistent with the "
        "whole conversation. Rank questions by expected BELIEF RECALL: how likely "
        "that regenerated list is to contain the unknown true animal. You never "
        "know the true animal. Favor simple, familiar semantic boundaries whose "
        "Yes and No branches both cue coherent and complete sets of animal names. "
        "Penalize obscure, compound, vague, or highly asymmetric boundaries that "
        "make one likely branch hard to enumerate. Predictive branch probabilities "
        "and observed regenerated support sizes are diagnostics, not the objective. "
        "Do not optimize immediate information gain. Return only one bare JSON "
        "object of the form {\"scores\":[number,...]}; one score per candidate in "
        "the original order, with larger meaning better expected belief recall."
    ),
}


def prompt_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build the complete model-visible payload from an explicit safe allowlist."""
    candidates = []
    for index, entry in enumerate(record["candidate_dynamics"]):
        candidates.append(
            {
                "index": index,
                "question": str(entry["question"]),
                "predictive_probability_yes": float(entry["p_yes"]),
                "predictive_probability_no": float(entry["p_no"]),
                "regenerated_support_size_if_yes": int(entry["support_size_if_yes"]),
                "regenerated_support_size_if_no": int(entry["support_size_if_no"]),
            }
        )
    history = [
        {
            "question": str(item["question"]),
            "answer": str(item["answer"]),
        }
        for item in record["history"]
    ]
    return {
        "domain": "animals",
        "history": history,
        "candidates": candidates,
    }


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    payload = prompt_payload(record)
    return [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                "Score every candidate for expected belief recall after the next "
                "answer. The unknown target is not provided.\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_scores(text: str, expected_count: int) -> list[float]:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("ranker response is not bare JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("ranker response must contain only a scores field")
    scores = payload["scores"]
    if not isinstance(scores, list) or len(scores) != expected_count:
        raise ValueError(f"ranker must return exactly {expected_count} scores")
    parsed: list[float] = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("ranker scores must be finite numbers")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("ranker scores must be finite numbers")
        parsed.append(value)
    return parsed


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left)
    right_scale = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else None


def summarize_rankings(records: list[dict[str, Any]]) -> dict[str, Any]:
    ranker_scores: list[float] = []
    immediate_scores: list[float] = []
    coverages: list[float] = []
    ranker_selected: list[float] = []
    immediate_selected: list[float] = []
    oracle_selected: list[float] = []
    ranker_regrets: list[float] = []
    immediate_regrets: list[float] = []
    active_ranker_regrets: list[float] = []
    active_immediate_regrets: list[float] = []

    for record in records:
        dynamics = record["candidate_dynamics"]
        scores = [float(value) for value in record["belief_recall_scores"]]
        eigs = [float(entry["immediate_eig"]) for entry in dynamics]
        state_coverages = [
            float(entry["expected_truth_coverage"]) for entry in dynamics
        ]
        ranker_index = max(range(len(scores)), key=scores.__getitem__)
        immediate_index = max(range(len(eigs)), key=eigs.__getitem__)
        best_coverage = max(state_coverages)
        ranker_coverage = state_coverages[ranker_index]
        immediate_coverage = state_coverages[immediate_index]

        ranker_scores.extend(scores)
        immediate_scores.extend(eigs)
        coverages.extend(state_coverages)
        ranker_selected.append(ranker_coverage)
        immediate_selected.append(immediate_coverage)
        oracle_selected.append(best_coverage)
        ranker_regrets.append(best_coverage - ranker_coverage)
        immediate_regrets.append(best_coverage - immediate_coverage)
        if best_coverage > min(state_coverages):
            active_ranker_regrets.append(best_coverage - ranker_coverage)
            active_immediate_regrets.append(best_coverage - immediate_coverage)

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    ranker_spearman = _pearson(
        _average_ranks(ranker_scores),
        _average_ranks(coverages),
    )
    immediate_spearman = _pearson(
        _average_ranks(immediate_scores),
        _average_ranks(coverages),
    )
    paired_gains = [
        ranker - immediate
        for ranker, immediate in zip(ranker_selected, immediate_selected)
    ]
    return {
        "num_states": len(records),
        "num_active_states": len(active_ranker_regrets),
        "num_candidate_rows": len(coverages),
        "spearman_belief_recall_score_vs_expected_truth_coverage": ranker_spearman,
        "spearman_immediate_eig_vs_expected_truth_coverage": immediate_spearman,
        "mean_selected_expected_truth_coverage_belief_recall": mean(ranker_selected),
        "mean_selected_expected_truth_coverage_immediate_eig": mean(immediate_selected),
        "mean_oracle_candidate_expected_truth_coverage": mean(oracle_selected),
        "mean_paired_selected_coverage_gain": mean(paired_gains),
        "ranker_immediate_wins_ties_losses": [
            sum(value > 1e-12 for value in paired_gains),
            sum(abs(value) <= 1e-12 for value in paired_gains),
            sum(value < -1e-12 for value in paired_gains),
        ],
        "mean_coverage_regret_belief_recall": mean(ranker_regrets),
        "mean_coverage_regret_immediate_eig": mean(immediate_regrets),
        "mean_active_state_regret_belief_recall": mean(active_ranker_regrets),
        "mean_active_state_regret_immediate_eig": mean(active_immediate_regrets),
    }


def load_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(raw)
        source_records = payload.get("records")
        if not isinstance(source_records, list):
            raise ValueError(f"{path} does not contain a records list")
        for record in source_records:
            records.append(
                {
                    "source_run_id": payload.get("run_id"),
                    "source_state_index": record.get("state_index"),
                    "history": record["history"],
                    "candidate_dynamics": record["candidate_dynamics"],
                }
            )
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return records, sources


def run_ranker(
    records: list[dict[str, Any]],
    runtime_config: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not runtime_config.model_pairs:
        raise ValueError("ranker config requires one model pair")
    model = build_model_adapter(
        runtime_config.model_pairs[0].questioner,
        config=runtime_config,
    )
    messages = [build_messages(record) for record in records]
    completions = model.chat_complete_messages_batched(
        messages,
        temperature=runtime_config.generation_temperature_simple,
        block_size=runtime_config.batched_block_size,
        max_new_tokens=runtime_config.openrouter_max_output_tokens,
    )
    if len(completions) != len(records):
        raise RuntimeError("ranker returned the wrong number of completions")

    ranked_records = []
    for record, completion in zip(records, completions):
        candidate_count = len(record["candidate_dynamics"])
        ranked_records.append(
            {
                **record,
                "belief_recall_scores": parse_scores(completion, candidate_count),
                "raw_ranker_response": completion,
                "model_visible_payload": prompt_payload(record),
            }
        )
    return ranked_records, model.usage_snapshot()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_animals_belief_recall_ranker_openrouter.yaml"),
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        default="animals-belief-recall-ranker-development-20260723",
    )
    args = parser.parse_args()

    runtime_config = load_config(str(args.config))
    runtime_config.run_id = args.run_id
    records, sources = load_records(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ranked_records, usage = run_ranker(records, runtime_config)
        summary = summarize_rankings(ranked_records)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "run_id": args.run_id,
            "error": f"{type(exc).__name__}: {exc}",
            "sources": sources,
        }
        (args.output_dir / "RANKING_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    payload = {
        "schema_version": 1,
        "status": "development_only",
        "target_blind_prompt_construction": True,
        "run_id": args.run_id,
        "config_path": str(args.config),
        "sources": sources,
        "summary": summary,
        "records": ranked_records,
        "usage": usage,
    }
    (args.output_dir / "RANKING.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
