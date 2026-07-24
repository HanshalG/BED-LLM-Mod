#!/usr/bin/env python3
"""Aligned semantic CA-BED over one shared bank of country questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.countries_cabed_aligned_v1 import (
    COUNTRIES,
    _bootstrap_mean_ci,
    _canonical_question,
    _is_direct_guess,
    _usage_snapshot,
    classification_messages,
    immediate_eig,
    parse_classification,
    partition,
)
from scripts.movielens_profile_dynamics_gate import _parse_json_object


SCHEMA_VERSION = 2
SELECTION_SEED = 24324
BANK_PROPOSALS = 40
BANK_WIDTH = 32
MIN_BANK_BRANCH_SIZE = 4

SMOKE_STYLES = (
    "Favor broad geopolitical, geographic, linguistic, and economic groupings.",
    "Favor independent physical, historical, cultural, and civic properties.",
)

FORMAL_STYLES = (
    "Use a balanced mix of broad factual properties without relying on trivia.",
    "Favor physical location, borders, coastlines, terrain, and climate.",
    "Favor language families, scripts, religion, and cultural traditions.",
    "Favor government, constitutional structure, and international relations.",
    "Favor population, urbanization, migration, and human geography.",
    "Favor economic sectors, trade, resources, currency, and development.",
    "Favor historical empires, independence, alliances, and former associations.",
    "Favor regional organizations, neighboring regions, and continental groups.",
    "Favor flags, symbols, capitals, and stable civic facts without direct guesses.",
    "Favor travel-relevant geography, transport, landmarks, and natural features.",
    "Use diverse globally applicable properties with substantially different splits.",
    "Use independent country facts spanning several knowledge families.",
)


class CountriesBankGateError(RuntimeError):
    """A failed-closed serving error with any available usage attached."""

    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def stage_styles(stage: str) -> tuple[str, ...]:
    if stage == "serving_smoke":
        return SMOKE_STYLES
    if stage == "formal":
        return FORMAL_STYLES
    raise ValueError("stage must be serving_smoke or formal")


def bank_generation_messages(
    *,
    count: int,
    style: str,
) -> list[dict[str, str]]:
    schema = {"questions": ["one factual Yes/No question"] * count}
    return [
        {
            "role": "system",
            "content": (
                "Generate factual Yes/No questions for country identification. "
                "Return strict JSON only, with no reasoning text."
            ),
        },
        {
            "role": "user",
            "content": (
                "The fixed possible-country support is:\n"
                f"{json.dumps(COUNTRIES, separators=(',', ':'))}\n"
                f"Guidance: {style}\n"
                f"Generate exactly {count} distinct factual Yes/No questions. "
                "Each question must have one well-defined answer for every listed "
                "country. Each property should plausibly be true for at least four "
                "listed countries and false for at least four listed countries. "
                "Do not directly guess a country. Do not repeat or paraphrase "
                "another question. End every question with a question mark. "
                "Do not calculate information gain or choose a policy. Return "
                "exactly this schema:\n"
                + json.dumps(schema, separators=(",", ":"))
            ),
        },
    ]


def parse_bank_questions(text: str, *, count: int) -> tuple[str, ...]:
    rows = _parse_json_object(text).get("questions")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(f"question bank must contain exactly {count} rows")
    seen: set[str] = set()
    questions: list[str] = []
    for value in rows:
        question = " ".join(str(value).strip().split())
        canonical = _canonical_question(question)
        if (
            not canonical
            or canonical in seen
            or not question.endswith("?")
            or _is_direct_guess(question)
        ):
            raise ValueError(
                "bank question is empty, duplicate, malformed, or direct"
            )
        seen.add(canonical)
        questions.append(question)
    return tuple(questions)


def _batched_generate(
    model: Any,
    messages: Sequence[list[dict[str, str]]],
    config: Config,
    *,
    temperature: float,
) -> list[str]:
    return model.chat_complete_messages_batched(
        list(messages),
        temperature=temperature,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )


def _write_raw_checkpoint(
    path: Path | None,
    *,
    stage: str,
    raw: dict[str, Any],
) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "stage": stage, **raw},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_tree(
    *,
    tree_index: int,
    style: str,
    questions: Sequence[str],
    tables: Sequence[Sequence[str]],
) -> dict[str, Any]:
    all_indices = tuple(range(len(COUNTRIES)))
    bank: list[dict[str, Any]] = []
    for proposal_index, (question, answers) in enumerate(
        zip(questions, tables, strict=True)
    ):
        groups = partition(all_indices, answers)
        if min(len(groups["Yes"]), len(groups["No"])) < MIN_BANK_BRANCH_SIZE:
            continue
        bank.append(
            {
                "proposal_index": proposal_index,
                "question": question,
                "answers": list(answers),
            }
        )
        if len(bank) == BANK_WIDTH:
            break
    if len(bank) != BANK_WIDTH:
        raise ValueError(
            f"tree {tree_index} has fewer than {BANK_WIDTH} balanced bank rows"
        )

    roots: list[dict[str, Any]] = []
    for root_index, root in enumerate(bank):
        root_answers = root["answers"]
        groups = partition(all_indices, root_answers)
        root_eig = immediate_eig(all_indices, root_answers)
        future_value = 0.0
        branches = []
        candidate_indices = tuple(
            index for index in range(len(bank)) if index != root_index
        )
        for answer in ("Yes", "No"):
            support = groups[answer]
            scores = [
                immediate_eig(support, bank[index]["answers"])
                for index in candidate_indices
            ]
            selected_offset = int(np.argmax(scores))
            selected_bank_index = candidate_indices[selected_offset]
            future_value += (
                len(support) / len(COUNTRIES)
            ) * scores[selected_offset]
            branches.append(
                {
                    "answer": answer,
                    "support_size": len(support),
                    "candidate_bank_indices": list(candidate_indices),
                    "eig_scores": scores,
                    "selected_bank_index": selected_bank_index,
                    "selected_question": bank[selected_bank_index]["question"],
                }
            )
        roots.append(
            {
                "bank_index": root_index,
                "question": root["question"],
                "immediate_eig": root_eig,
                "depth_two_score": root_eig + future_value,
                "branches": branches,
            }
        )
    return {
        "tree_index": tree_index,
        "style": style,
        "question_proposals": list(questions),
        "bank": bank,
        "roots": roots,
    }


def _build_trees(
    config: Config,
    question_model: Any,
    semantic_model: Any,
    *,
    stage: str,
    styles: Sequence[str],
    raw_checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    raw: dict[str, Any] = {}
    question_raw = _batched_generate(
        question_model,
        [
            bank_generation_messages(count=BANK_PROPOSALS, style=style)
            for style in styles
        ],
        config,
        temperature=0.7,
    )
    raw["question_banks"] = question_raw
    _write_raw_checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
    question_banks = [
        parse_bank_questions(text, count=BANK_PROPOSALS)
        for text in question_raw
    ]

    table_keys: list[tuple[int, int]] = []
    table_questions: list[str] = []
    for tree_index, questions in enumerate(question_banks):
        for proposal_index, question in enumerate(questions):
            table_keys.append((tree_index, proposal_index))
            table_questions.append(question)
    table_raw = _batched_generate(
        semantic_model,
        [classification_messages(question) for question in table_questions],
        config,
        temperature=0.0,
    )
    raw["semantic_tables"] = table_raw
    _write_raw_checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
    parsed_tables = {
        key: parse_classification(text)
        for key, text in zip(table_keys, table_raw, strict=True)
    }

    trees = []
    for tree_index, questions in enumerate(question_banks):
        tables = [
            parsed_tables[tree_index, proposal_index]
            for proposal_index in range(BANK_PROPOSALS)
        ]
        trees.append(
            _build_tree(
                tree_index=tree_index,
                style=styles[tree_index],
                questions=questions,
                tables=tables,
            )
        )
    return trees


def _evaluate_tree(tree: dict[str, Any]) -> dict[str, Any]:
    bank = tree["bank"]
    roots = tree["roots"]
    d1_index = int(np.argmax([root["immediate_eig"] for root in roots]))
    d2_index = int(np.argmax([root["depth_two_score"] for root in roots]))
    random_index = int(
        np.random.default_rng(SELECTION_SEED + tree["tree_index"]).integers(
            0, len(roots)
        )
    )

    def endpoint(root_index: int, target_index: int) -> dict[str, Any]:
        root = roots[root_index]
        root_answers = bank[root["bank_index"]]["answers"]
        root_answer = root_answers[target_index]
        branch = next(
            row for row in root["branches"] if row["answer"] == root_answer
        )
        followup_index = branch["selected_bank_index"]
        followup_answers = bank[followup_index]["answers"]
        followup_answer = followup_answers[target_index]
        final_support = [
            index
            for index in range(len(COUNTRIES))
            if root_answers[index] == root_answer
            and followup_answers[index] == followup_answer
        ]
        final_entropy = math.log(len(final_support))
        return {
            "root_answer": root_answer,
            "followup_bank_index": followup_index,
            "followup_question": bank[followup_index]["question"],
            "followup_answer": followup_answer,
            "final_support_size": len(final_support),
            "final_entropy": final_entropy,
            "truth_nll": final_entropy,
        }

    targets = []
    for target_index, country in enumerate(COUNTRIES):
        targets.append(
            {
                "country": country,
                "depth_one": endpoint(d1_index, target_index),
                "depth_two": endpoint(d2_index, target_index),
                "random_root": endpoint(random_index, target_index),
            }
        )
    d1_entropies = [row["depth_one"]["final_entropy"] for row in targets]
    d2_entropies = [row["depth_two"]["final_entropy"] for row in targets]
    random_entropies = [
        row["random_root"]["final_entropy"] for row in targets
    ]
    return {
        **tree,
        "selections": {
            "depth_one": d1_index,
            "depth_two": d2_index,
            "random_root": random_index,
        },
        "selected_questions": {
            "depth_one": roots[d1_index]["question"],
            "depth_two": roots[d2_index]["question"],
            "random_root": roots[random_index]["question"],
        },
        "distinct_depth_two_root": d1_index != d2_index,
        "mean_final_entropy_depth_one": float(np.mean(d1_entropies)),
        "mean_final_entropy_depth_two": float(np.mean(d2_entropies)),
        "mean_final_entropy_random_root": float(np.mean(random_entropies)),
        "mean_entropy_gain_depth_two_vs_one": float(
            np.mean(np.asarray(d1_entropies) - np.asarray(d2_entropies))
        ),
        "mean_entropy_gain_depth_two_vs_random": float(
            np.mean(np.asarray(random_entropies) - np.asarray(d2_entropies))
        ),
        "target_win_count_vs_one": sum(
            d2 + 1.0e-12 < d1
            for d1, d2 in zip(d1_entropies, d2_entropies, strict=True)
        ),
        "target_loss_count_vs_one": sum(
            d2 > d1 + 1.0e-12
            for d1, d2 in zip(d1_entropies, d2_entropies, strict=True)
        ),
        "targets": targets,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    expected_trees = len(stage_styles(stage))
    expected_requests = expected_trees * (1 + BANK_PROPOSALS)
    gains_one = [
        record["mean_entropy_gain_depth_two_vs_one"] for record in records
    ]
    gains_random = [
        record["mean_entropy_gain_depth_two_vs_random"] for record in records
    ]
    distinct = sum(record["distinct_depth_two_root"] for record in records)
    positive = sum(value > 1.0e-12 for value in gains_one)
    summary: dict[str, Any] = {
        "num_trees": len(records),
        "num_targets_per_tree": len(COUNTRIES),
        "depth_two_distinct_root_count": distinct,
        "positive_tree_count_vs_one": positive,
        "mean_entropy_gain_depth_two_vs_one": float(np.mean(gains_one)),
        "mean_entropy_gain_depth_two_vs_random": float(
            np.mean(gains_random)
        ),
        "total_target_wins_vs_one": sum(
            record["target_win_count_vs_one"] for record in records
        ),
        "total_target_losses_vs_one": sum(
            record["target_loss_count_vs_one"] for record in records
        ),
    }
    base = {
        "all_trees_complete": len(records) == expected_trees
        and all(
            len(record["bank"]) == BANK_WIDTH
            and len(record["roots"]) == BANK_WIDTH
            and all(
                len(branch["candidate_bank_indices"]) == BANK_WIDTH - 1
                for root in record["roots"]
                for branch in root["branches"]
            )
            for record in records
        ),
        "exact_physical_request_count": int(usage["physical_requests"])
        == expected_requests,
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_semantic_tables_binary_and_complete": all(
            len(row["answers"]) == len(COUNTRIES)
            and set(row["answers"]) <= {"Yes", "No"}
            for record in records
            for row in record["bank"]
        ),
        "depth_two_never_worse_than_matched_depth_one": all(
            value >= -1.0e-12 for value in gains_one
        ),
    }
    if stage == "serving_smoke":
        gates = {
            **base,
            "at_least_one_distinct_depth_two_root": distinct >= 1,
            "at_least_one_positive_tree": positive >= 1,
            "mean_depth_two_gain_at_least_0_01": float(np.mean(gains_one))
            >= 0.01,
        }
    else:
        one_ci = _bootstrap_mean_ci(gains_one, seed=SELECTION_SEED + 101)
        random_ci = _bootstrap_mean_ci(
            gains_random, seed=SELECTION_SEED + 102
        )
        summary["entropy_gain_depth_two_vs_one_bootstrap_90ci"] = list(one_ci)
        summary["entropy_gain_depth_two_vs_random_bootstrap_90ci"] = list(
            random_ci
        )
        gates = {
            **base,
            "distinct_depth_two_root_at_least_8": distinct >= 8,
            "positive_tree_count_at_least_8": positive >= 8,
            "mean_depth_two_gain_vs_one_at_least_0_02": float(
                np.mean(gains_one)
            )
            >= 0.02,
            "depth_two_gain_vs_one_ci_positive": one_ci[0] > 0.0,
            "mean_depth_two_gain_vs_random_at_least_0_02": float(
                np.mean(gains_random)
            )
            >= 0.02,
            "depth_two_gain_vs_random_ci_positive": random_ci[0] > 0.0,
        }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_gate(
    config: Config,
    *,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    if len(COUNTRIES) != 64 or len(set(COUNTRIES)) != 64:
        raise ValueError("Countries bank V2 requires 64 distinct countries")
    if len(config.model_pairs) != 1:
        raise ValueError(
            "Countries bank V2 requires one questioner/answerer pair"
        )
    question_model = build_model_adapter(
        config.model_pairs[0].questioner, config
    )
    semantic_model = build_model_adapter(
        config.model_pairs[0].answerer, config
    )
    try:
        trees = _build_trees(
            config,
            question_model,
            semantic_model,
            stage=stage,
            styles=stage_styles(stage),
            raw_checkpoint_path=raw_checkpoint_path,
        )
        records = [_evaluate_tree(tree) for tree in trees]
    except Exception as exc:
        usage = _usage_snapshot(question_model, semantic_model)
        raise CountriesBankGateError(
            f"{type(exc).__name__}: {exc}", usage
        ) from exc
    usage = _usage_snapshot(question_model, semantic_model)
    summary = summarize(records, usage, stage=stage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "domain": "countries",
            "support_size": len(COUNTRIES),
            "bank_proposals": BANK_PROPOSALS,
            "bank_width": BANK_WIDTH,
            "minimum_bank_branch_size": MIN_BANK_BRANCH_SIZE,
            "global_shared_question_bank": True,
            "aligned_target_blind_semantic_tables": True,
            "semantic_tables_define_likelihood_and_observation": True,
            "all_64_targets_evaluated_per_tree": True,
            "matched_compute_myopic_root_control": True,
            "random_root_uses_same_optimal_followups": True,
            "question_generation_temperature": 0.7,
            "semantic_table_temperature": 0.0,
            "no_reasoning": True,
            "expected_physical_requests": len(stage_styles(stage))
            * (1 + BANK_PROPOSALS),
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "formal"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.35
        config.openrouter_run_budget_usd = 1.00
    else:
        config.openrouter_projected_cost_usd = 2.00
        config.openrouter_run_budget_usd = 4.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json" if args.stage == "serving_smoke" else "GATE.json"
    )
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )
    try:
        payload = run_gate(
            config,
            stage=args.stage,
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
            "error": f"{type(exc).__name__}: {exc}",
            "raw_responses_path": str(raw_path),
        }
        if isinstance(exc, CountriesBankGateError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": payload["status"], **payload["summary"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
