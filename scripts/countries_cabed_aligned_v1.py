#!/usr/bin/env python3
"""Aligned semantic CA-BED depth gate over a fixed 64-country support."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.movielens_profile_dynamics_gate import _parse_json_object


SCHEMA_VERSION = 1
SELECTION_SEED = 24323
ROOT_PROPOSALS = 8
ROOT_WIDTH = 4
FOLLOWUP_PROPOSALS = 6
FOLLOWUP_WIDTH = 3
MIN_ROOT_BRANCH_SIZE = 4
BOOTSTRAP_DRAWS = 5000

COUNTRIES = (
    "Argentina",
    "Australia",
    "Austria",
    "Bangladesh",
    "Belgium",
    "Bolivia",
    "Brazil",
    "Canada",
    "Chile",
    "China",
    "Colombia",
    "Costa Rica",
    "Cuba",
    "Czechia",
    "Denmark",
    "Egypt",
    "Ethiopia",
    "Finland",
    "France",
    "Germany",
    "Ghana",
    "Greece",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Kenya",
    "Malaysia",
    "Mexico",
    "Mongolia",
    "Morocco",
    "Nepal",
    "Netherlands",
    "New Zealand",
    "Nigeria",
    "Norway",
    "Pakistan",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Romania",
    "Saudi Arabia",
    "Singapore",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Thailand",
    "Turkey",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Uruguay",
    "Venezuela",
    "Vietnam",
    "Zimbabwe",
)

SMOKE_STYLES = (
    "Use a varied mix of physical geography, region, language, and culture.",
    "Use a varied mix of history, government, economy, and human geography.",
)

FORMAL_STYLES = (
    "Use broad, natural country knowledge without favoring one fact family.",
    "Emphasize physical geography and location while remaining varied.",
    "Emphasize languages and cultural traditions while remaining factual.",
    "Emphasize political institutions and international relations.",
    "Emphasize climate, terrain, coastlines, and natural features.",
    "Emphasize economic structure, resources, and development patterns.",
    "Emphasize population, settlement, and human geography.",
    "Emphasize history and former political associations.",
    "Emphasize regional organizations and neighboring areas.",
    "Emphasize flags, symbols, currencies, and civic facts.",
    "Emphasize travel-relevant geography and widely known landmarks.",
    "Use a balanced mix of independent factual properties and avoid trivia.",
)


def stage_styles(stage: str) -> tuple[str, ...]:
    if stage == "serving_smoke":
        return SMOKE_STYLES
    if stage == "formal":
        return FORMAL_STYLES
    raise ValueError("stage must be serving_smoke or formal")


def _canonical_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip()).casefold().rstrip("?.!")


def _is_direct_guess(question: str) -> bool:
    canonical = _canonical_question(question)
    guesses = set()
    for country in COUNTRIES:
        label = country.casefold()
        guesses.update(
            {
                f"is it {label}",
                f"is the country {label}",
                f"is this country {label}",
            }
        )
    return canonical in guesses


def question_generation_messages(
    entities: Sequence[str],
    *,
    count: int,
    style: str,
    history: Sequence[tuple[str, str]] = (),
    forbidden: Sequence[str] = (),
) -> list[dict[str, str]]:
    schema = {"questions": ["one factual Yes/No question"] * count}
    history_text = (
        "None."
        if not history
        else json.dumps(
            [
                {"question": question, "answer": answer}
                for question, answer in history
            ],
            separators=(",", ":"),
        )
    )
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
                f"Possible countries in the current branch: "
                f"{json.dumps(list(entities), separators=(',', ':'))}\n"
                f"Prior question history: {history_text}\n"
                f"Already used or forbidden questions: "
                f"{json.dumps(list(forbidden), separators=(',', ':'))}\n"
                f"Guidance: {style}\n"
                f"Generate exactly {count} distinct factual Yes/No questions "
                "that help split the listed countries. Questions must have "
                "well-defined answers for every country in the full game, must "
                "not directly guess a country, must not repeat or paraphrase a "
                "forbidden question, and must end with a question mark. Do not "
                "calculate information gain or choose a policy. Return exactly "
                "this schema:\n"
                + json.dumps(schema, separators=(",", ":"))
            ),
        },
    ]


def parse_questions(
    text: str,
    *,
    count: int,
    forbidden: Sequence[str] = (),
) -> tuple[str, ...]:
    rows = _parse_json_object(text).get("questions")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(f"question response must contain exactly {count} rows")
    seen = {_canonical_question(question) for question in forbidden}
    questions: list[str] = []
    for value in rows:
        question = re.sub(r"\s+", " ", str(value).strip())
        canonical = _canonical_question(question)
        if (
            not canonical
            or canonical in seen
            or not question.endswith("?")
            or _is_direct_guess(question)
        ):
            raise ValueError("question is empty, duplicate, malformed, or direct")
        seen.add(canonical)
        questions.append(question)
    return tuple(questions)


def classification_messages(question: str) -> list[dict[str, str]]:
    schema = {
        "answers": [
            {"country": country, "answer": "Yes|No"} for country in COUNTRIES
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "Classify factual country properties. Return strict JSON only, "
                "with no reasoning or explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "For every country below, give the answer an accurate country "
                "20 Questions environment should return. Use exactly Yes or No. "
                "Apply one consistent interpretation to every row. Preserve "
                "country order and spelling; do not omit or add countries.\n"
                f"Countries: {json.dumps(COUNTRIES, separators=(',', ':'))}\n"
                "Return exactly this schema:\n"
                + json.dumps(schema, separators=(",", ":"))
            ),
        },
    ]


def parse_classification(text: str) -> tuple[str, ...]:
    rows = _parse_json_object(text).get("answers")
    if not isinstance(rows, list) or len(rows) != len(COUNTRIES):
        raise ValueError("classification must contain every country")
    answers: list[str] = []
    for country, row in zip(COUNTRIES, rows, strict=True):
        if not isinstance(row, dict) or row.get("country") != country:
            raise ValueError("classification changed country order or spelling")
        answer = str(row.get("answer", "")).strip()
        if answer not in {"Yes", "No"}:
            raise ValueError("classification answer must be Yes or No")
        answers.append(answer)
    return tuple(answers)


def entropy(indices: Sequence[int]) -> float:
    return math.log(len(indices)) if indices else 0.0


def partition(
    indices: Sequence[int],
    answers: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    groups = {
        "Yes": tuple(index for index in indices if answers[index] == "Yes"),
        "No": tuple(index for index in indices if answers[index] == "No"),
    }
    return groups


def immediate_eig(
    indices: Sequence[int],
    answers: Sequence[str],
) -> float:
    if not indices:
        return 0.0
    groups = partition(indices, answers)
    expected_entropy = sum(
        (len(group) / len(indices)) * entropy(group)
        for group in groups.values()
        if group
    )
    return entropy(indices) - expected_entropy


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        array,
        size=(BOOTSTRAP_DRAWS, len(array)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(samples, [0.05, 0.95])
    return float(low), float(high)


def _usage_snapshot(question_model: Any, semantic_model: Any) -> dict[str, Any]:
    question_usage = question_model.usage_snapshot()
    semantic_usage = semantic_model.usage_snapshot()
    return {
        "physical_requests": int(question_usage["adapter_requests"])
        + int(semantic_usage["adapter_requests"]),
        "reasoning_tokens": int(question_usage["adapter_reasoning_tokens"])
        + int(semantic_usage["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(question_usage["adapter_cost_usd"])
        + float(semantic_usage["adapter_cost_usd"]),
        "question_model": question_usage,
        "semantic_model": semantic_usage,
    }


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


def _build_trees(
    config: Config,
    question_model: Any,
    semantic_model: Any,
    *,
    styles: Sequence[str],
    raw: dict[str, Any],
    raw_checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    def checkpoint() -> None:
        if raw_checkpoint_path is None:
            return
        raw_checkpoint_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": (
                        "serving_smoke"
                        if tuple(styles) == SMOKE_STYLES
                        else "formal"
                    ),
                    **raw,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    root_raw = _batched_generate(
        question_model,
        [
            question_generation_messages(
                COUNTRIES,
                count=ROOT_PROPOSALS,
                style=style,
            )
            for style in styles
        ],
        config,
        temperature=0.7,
    )
    raw["root_generation"] = root_raw
    checkpoint()
    root_proposals = [
        parse_questions(text, count=ROOT_PROPOSALS) for text in root_raw
    ]

    root_keys: list[tuple[int, int]] = []
    root_questions: list[str] = []
    for tree_index, questions in enumerate(root_proposals):
        for question_index, question in enumerate(questions):
            root_keys.append((tree_index, question_index))
            root_questions.append(question)
    root_table_raw = _batched_generate(
        semantic_model,
        [classification_messages(question) for question in root_questions],
        config,
        temperature=0.0,
    )
    raw["root_tables"] = root_table_raw
    checkpoint()
    root_tables = {
        key: parse_classification(text)
        for key, text in zip(root_keys, root_table_raw, strict=True)
    }

    selected_roots: list[list[dict[str, Any]]] = []
    all_indices = tuple(range(len(COUNTRIES)))
    for tree_index, questions in enumerate(root_proposals):
        roots = []
        for question_index, question in enumerate(questions):
            answers = root_tables[tree_index, question_index]
            groups = partition(all_indices, answers)
            if min(len(groups["Yes"]), len(groups["No"])) < MIN_ROOT_BRANCH_SIZE:
                continue
            roots.append(
                {
                    "question": question,
                    "answers": answers,
                    "groups": groups,
                }
            )
            if len(roots) == ROOT_WIDTH:
                break
        if len(roots) != ROOT_WIDTH:
            raise ValueError(
                f"tree {tree_index} has fewer than {ROOT_WIDTH} balanced roots"
            )
        selected_roots.append(roots)

    branch_keys: list[tuple[int, int, str]] = []
    branch_messages: list[list[dict[str, str]]] = []
    for tree_index, roots in enumerate(selected_roots):
        forbidden = [root["question"] for root in roots]
        for root_index, root in enumerate(roots):
            for answer in ("Yes", "No"):
                branch_keys.append((tree_index, root_index, answer))
                entities = [
                    COUNTRIES[index] for index in root["groups"][answer]
                ]
                branch_messages.append(
                    question_generation_messages(
                        entities,
                        count=FOLLOWUP_PROPOSALS,
                        style=styles[tree_index],
                        history=((root["question"], answer),),
                        forbidden=forbidden,
                    )
                )
    branch_raw = _batched_generate(
        question_model,
        branch_messages,
        config,
        temperature=0.7,
    )
    raw["branch_generation"] = branch_raw
    checkpoint()

    followups: dict[tuple[int, int, str], tuple[str, ...]] = {}
    seen_by_tree = {
        tree_index: [
            root["question"] for root in selected_roots[tree_index]
        ]
        for tree_index in range(len(styles))
    }
    for key, text in zip(branch_keys, branch_raw, strict=True):
        tree_index, _root_index, _answer = key
        proposals = parse_questions(
            text,
            count=FOLLOWUP_PROPOSALS,
            forbidden=seen_by_tree[tree_index],
        )
        selected = proposals[:FOLLOWUP_WIDTH]
        followups[key] = selected
        seen_by_tree[tree_index].extend(selected)

    followup_keys: list[tuple[int, int, str, int]] = []
    followup_questions: list[str] = []
    for key in branch_keys:
        for followup_index, question in enumerate(followups[key]):
            followup_keys.append((*key, followup_index))
            followup_questions.append(question)
    followup_table_raw = _batched_generate(
        semantic_model,
        [
            classification_messages(question)
            for question in followup_questions
        ],
        config,
        temperature=0.0,
    )
    raw["followup_tables"] = followup_table_raw
    checkpoint()
    followup_tables = {
        key: parse_classification(text)
        for key, text in zip(
            followup_keys, followup_table_raw, strict=True
        )
    }

    trees: list[dict[str, Any]] = []
    for tree_index, roots in enumerate(selected_roots):
        root_records = []
        for root_index, root in enumerate(roots):
            root_eig = immediate_eig(all_indices, root["answers"])
            future_value = 0.0
            branch_records = []
            for answer in ("Yes", "No"):
                branch_indices = root["groups"][answer]
                questions = followups[tree_index, root_index, answer]
                rows = [
                    followup_tables[
                        tree_index, root_index, answer, followup_index
                    ]
                    for followup_index in range(FOLLOWUP_WIDTH)
                ]
                scores = [
                    immediate_eig(branch_indices, row) for row in rows
                ]
                selected_index = int(np.argmax(scores))
                future_value += (
                    len(branch_indices) / len(COUNTRIES)
                ) * scores[selected_index]
                branch_records.append(
                    {
                        "answer": answer,
                        "support": [
                            COUNTRIES[index] for index in branch_indices
                        ],
                        "candidate_questions": list(questions),
                        "candidate_answers": [list(row) for row in rows],
                        "eig_scores": scores,
                        "selected_index": selected_index,
                        "selected_question": questions[selected_index],
                    }
                )
            root_records.append(
                {
                    "question": root["question"],
                    "answers": list(root["answers"]),
                    "immediate_eig": root_eig,
                    "depth_two_score": root_eig + future_value,
                    "branches": branch_records,
                }
            )
        trees.append(
            {
                "tree_index": tree_index,
                "style": styles[tree_index],
                "root_proposals": list(root_proposals[tree_index]),
                "roots": root_records,
            }
        )
    return trees


def _evaluate_tree(tree: dict[str, Any]) -> dict[str, Any]:
    roots = tree["roots"]
    d1_index = int(np.argmax([root["immediate_eig"] for root in roots]))
    d2_index = int(np.argmax([root["depth_two_score"] for root in roots]))
    random_index = int(
        np.random.default_rng(SELECTION_SEED + tree["tree_index"]).integers(
            0, ROOT_WIDTH
        )
    )

    def endpoint(root_index: int, target_index: int) -> dict[str, Any]:
        root = roots[root_index]
        root_answer = root["answers"][target_index]
        branch = next(
            row for row in root["branches"] if row["answer"] == root_answer
        )
        followup_answers = branch["candidate_answers"][
            branch["selected_index"]
        ]
        followup_answer = followup_answers[target_index]
        final_support = [
            index
            for index in range(len(COUNTRIES))
            if root["answers"][index] == root_answer
            and followup_answers[index] == followup_answer
        ]
        final_entropy = math.log(len(final_support))
        return {
            "root_answer": root_answer,
            "followup_question": branch["selected_question"],
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
    expected_requests = expected_trees * (
        1 + ROOT_PROPOSALS + ROOT_WIDTH * 2 + ROOT_WIDTH * 2 * FOLLOWUP_WIDTH
    )
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
            len(record["roots"]) == ROOT_WIDTH
            and all(
                len(branch["candidate_questions"]) == FOLLOWUP_WIDTH
                for root in record["roots"]
                for branch in root["branches"]
            )
            for record in records
        ),
        "exact_physical_request_count": int(usage["physical_requests"])
        == expected_requests,
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_semantic_tables_binary_and_complete": all(
            len(root["answers"]) == len(COUNTRIES)
            and set(root["answers"]) <= {"Yes", "No"}
            and all(
                len(row) == len(COUNTRIES)
                and set(row) <= {"Yes", "No"}
                for branch in root["branches"]
                for row in branch["candidate_answers"]
            )
            for record in records
            for root in record["roots"]
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
        raise ValueError("Countries V1 requires 64 distinct countries")
    if len(config.model_pairs) != 1:
        raise ValueError("Countries V1 requires one questioner/answerer pair")
    question_model = build_model_adapter(
        config.model_pairs[0].questioner, config
    )
    semantic_model = build_model_adapter(
        config.model_pairs[0].answerer, config
    )
    raw: dict[str, Any] = {}
    trees = _build_trees(
        config,
        question_model,
        semantic_model,
        styles=stage_styles(stage),
        raw=raw,
        raw_checkpoint_path=raw_checkpoint_path,
    )
    records = [_evaluate_tree(tree) for tree in trees]
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
            "root_proposals": ROOT_PROPOSALS,
            "root_width": ROOT_WIDTH,
            "followup_proposals": FOLLOWUP_PROPOSALS,
            "followup_width": FOLLOWUP_WIDTH,
            "minimum_root_branch_size": MIN_ROOT_BRANCH_SIZE,
            "aligned_target_blind_semantic_tables": True,
            "semantic_tables_define_likelihood_and_observation": True,
            "all_64_targets_evaluated_per_tree": True,
            "matched_compute_myopic_root_control": True,
            "random_root_uses_same_optimal_followups": True,
            "question_generation_temperature": 0.7,
            "semantic_table_temperature": 0.0,
            "no_reasoning": True,
            "expected_physical_requests": len(stage_styles(stage)) * 41,
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
        config.openrouter_projected_cost_usd = 0.25
        config.openrouter_run_budget_usd = 1.00
    else:
        config.openrouter_projected_cost_usd = 1.50
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
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_responses_path": str(raw_path),
        }
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
