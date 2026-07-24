#!/usr/bin/env python3
"""Test target-blind ranking of path-dependent PAPRIKA clothing supports."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence
import unicodedata

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.animals_belief_recall_ranker import parse_scores
from scripts.movielens_profile_dynamics_gate import _parse_json_object


SCHEMA_VERSION = 1
DATA_SHA256 = "d9b7616d1316886aa1aee84ddc4f126715be4983c50d5160c5c7d223b7e0f16a"
SELECTION_SEED = 24330
BOOTSTRAP_SEED = 24331
SUPPORT_SAMPLES = 3
SUPPORT_SAMPLE_SIZE = 12
CANDIDATE_COUNT = 3
SMOKE_INDICES = (219, 220)
ACTIVITY_INDICES = (217, 221, 28, 20, 27, 21, 214, 26)
FORMAL_INDICES = (
    231,
    0,
    224,
    22,
    19,
    24,
    218,
    222,
    23,
    225,
    25,
    223,
    229,
    216,
    226,
    232,
    18,
    227,
    228,
    230,
)
PREFIX_QUESTIONS = (
    "Is it primarily worn on the upper half of the body?",
    "Is it primarily worn on the feet or lower legs?",
    "Is it mainly an accessory rather than a main garment?",
)
EXPECTED_REQUESTS = {
    "serving_smoke": len(SMOKE_INDICES)
    * (
        1
        + SUPPORT_SAMPLES
        + 1
        + 1
        + CANDIDATE_COUNT * 2 * SUPPORT_SAMPLES
        + 1
        + 1
    ),
    "activity": len(ACTIVITY_INDICES)
    * (
        1
        + SUPPORT_SAMPLES
        + 1
        + 1
        + CANDIDATE_COUNT * 2 * SUPPORT_SAMPLES
        + 1
    ),
    "confirmation": len(FORMAL_INDICES)
    * (
        1
        + SUPPORT_SAMPLES
        + 1
        + 1
        + CANDIDATE_COUNT * 2 * SUPPORT_SAMPLES
        + 1
        + 1
    ),
}


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _normalized(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value))


def selected_indices(stage: str) -> tuple[int, ...]:
    if stage == "serving_smoke":
        return SMOKE_INDICES
    if stage == "activity":
        return ACTIVITY_INDICES
    if stage == "confirmation":
        return FORMAL_INDICES
    raise ValueError("stage must be serving_smoke, activity, or confirmation")


def load_selected_cases(
    data_path: str | Path,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    path = Path(data_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("PAPRIKA Twenty Questions data hash does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("eval")
    if not isinstance(rows, list) or len(rows) != 367:
        raise ValueError("PAPRIKA Twenty Questions eval split is malformed")
    clothing_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("agent") == "clothing"
    ]
    if len(clothing_indices) != 30:
        raise ValueError("expected exactly 30 PAPRIKA eval clothing targets")
    reproduced = list(clothing_indices)
    random.Random(SELECTION_SEED).shuffle(reproduced)
    frozen = list(SMOKE_INDICES + ACTIVITY_INDICES + FORMAL_INDICES)
    if reproduced != frozen:
        raise ValueError("frozen target-blind clothing split does not reproduce")
    selected = [rows[index] for index in selected_indices(stage)]
    if any(
        not isinstance(row.get("env"), str)
        or row.get("agent") != "clothing"
        for row in selected
    ):
        raise ValueError("selected PAPRIKA clothing case is malformed")
    return payload, selected


def support_messages(
    history: Sequence[tuple[str, bool]],
    *,
    current_support: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    payload: dict[str, Any] = {
        "category": "clothing",
        "history": [
            {"question": question, "answer": "Yes" if answer else "No"}
            for question, answer in history
        ],
    }
    if current_support is not None:
        payload["previous_support_for_context"] = list(current_support)
    return [
        {
            "role": "system",
            "content": (
                "You maintain an open-world hypothesis support for Twenty "
                "Questions. Generate plausible concrete targets from semantic "
                "knowledge. The hidden target and evaluation target list are not "
                "available. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate exactly {SUPPORT_SAMPLE_SIZE} distinct specific clothing "
                "items consistent with every answer. Generate the list afresh and "
                "include plausible items that another support might omit; do not "
                "merely copy the previous support. Use ordinary canonical names, "
                "not descriptions or disjunctions. Return exactly "
                '{"possibilities":["item",...]}. Data: '
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_support(text: str) -> list[str]:
    payload = _parse_json_object(text)
    if set(payload) != {"possibilities"}:
        raise ValueError("support response must contain only possibilities")
    values = payload["possibilities"]
    if not isinstance(values, list) or len(values) != SUPPORT_SAMPLE_SIZE:
        raise ValueError(
            f"support must contain exactly {SUPPORT_SAMPLE_SIZE} possibilities"
        )
    cleaned = [
        " ".join(value.split()) if isinstance(value, str) else ""
        for value in values
    ]
    normalized = [_normalized(value) for value in cleaned]
    if (
        any(not value for value in normalized)
        or len(set(normalized)) != SUPPORT_SAMPLE_SIZE
    ):
        raise ValueError("support possibilities must be distinct nonempty strings")
    return cleaned


def union_supports(samples: Sequence[Sequence[str]]) -> list[str]:
    union: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        for value in sample:
            key = _normalized(value)
            if key and key not in seen:
                seen.add(key)
                union.append(value)
    return union


def candidate_messages(
    history: Sequence[tuple[str, bool]],
    current_support: Sequence[str],
) -> list[dict[str, str]]:
    payload = {
        "category": "clothing",
        "history": [
            {"question": question, "answer": "Yes" if answer else "No"}
            for question, answer in history
        ],
        "current_open_world_support": list(current_support),
    }
    return [
        {
            "role": "system",
            "content": (
                "Propose target-blind yes-or-no questions for open-world Twenty "
                "Questions. The hidden target is unavailable. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Propose exactly {CANDIDATE_COUNT} distinct questions that are "
                "answerable Yes or No for any concrete clothing item. Questions "
                "must test different semantic properties, must not repeat history, "
                "and must not directly guess a single item. Prefer questions whose "
                "answers would make a language model recall meaningfully different "
                "plausible items. Return exactly "
                '{"questions":["question",...]}. Data: '
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_candidates(text: str) -> list[str]:
    payload = _parse_json_object(text)
    if set(payload) != {"questions"}:
        raise ValueError("candidate response must contain only questions")
    values = payload["questions"]
    if not isinstance(values, list) or len(values) != CANDIDATE_COUNT:
        raise ValueError(f"candidate response must contain {CANDIDATE_COUNT} questions")
    cleaned = [
        " ".join(value.split()) if isinstance(value, str) else ""
        for value in values
    ]
    normalized = [_normalized(value) for value in cleaned]
    if (
        any(not value for value in normalized)
        or len(set(normalized)) != CANDIDATE_COUNT
        or any(not value.endswith("?") for value in cleaned)
    ):
        raise ValueError("candidate questions must be distinct nonempty questions")
    return cleaned


def support_label_messages(
    current_support: Sequence[str],
    candidates: Sequence[str],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Classify clothing items under yes-or-no semantic questions. "
                "Use ordinary real-world meanings. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For every question, return one boolean per support item in the "
                "same order, where true means the truthful answer is Yes. Return "
                'exactly {"labels":[[true,...],...]}. Data: '
                + json.dumps(
                    {
                        "support": list(current_support),
                        "questions": list(candidates),
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def parse_support_labels(
    text: str,
    *,
    support_size: int,
) -> list[list[bool]]:
    payload = _parse_json_object(text)
    if set(payload) != {"labels"}:
        raise ValueError("support-label response must contain only labels")
    labels = payload["labels"]
    if (
        not isinstance(labels, list)
        or len(labels) != CANDIDATE_COUNT
        or any(
            not isinstance(row, list)
            or len(row) != support_size
            or any(type(value) is not bool for value in row)
            for row in labels
        )
    ):
        raise ValueError("support-label response has the wrong boolean shape")
    return labels


def endpoint_messages(
    target: str,
    current_support: Sequence[str],
    candidates: Sequence[str],
    branch_supports: Sequence[dict[str, Sequence[str]]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Audit a frozen Twenty Questions support experiment. The target is "
                "revealed only now, after every support was generated. Judge ordinary "
                "semantic equivalence conservatively and return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return the truthful boolean answer to each question, whether the "
                "current support contains the target or a clear synonymous clothing "
                "name, and a [yes_branch,no_branch] containment pair for each "
                "candidate. Return exactly "
                '{"target_answers":[true,...],"current_contains_target":false,'
                '"branch_contains_target":[[false,false],...]}. Data: '
                + json.dumps(
                    {
                        "target": target,
                        "current_support": list(current_support),
                        "candidates": list(candidates),
                        "branches": [
                            {
                                "support_if_yes": list(entry["yes"]),
                                "support_if_no": list(entry["no"]),
                            }
                            for entry in branch_supports
                        ],
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def parse_endpoint(text: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    expected = {
        "target_answers",
        "current_contains_target",
        "branch_contains_target",
    }
    if set(payload) != expected:
        raise ValueError("endpoint response has unexpected fields")
    answers = payload["target_answers"]
    branches = payload["branch_contains_target"]
    if (
        not isinstance(answers, list)
        or len(answers) != CANDIDATE_COUNT
        or any(type(value) is not bool for value in answers)
        or type(payload["current_contains_target"]) is not bool
        or not isinstance(branches, list)
        or len(branches) != CANDIDATE_COUNT
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or any(type(value) is not bool for value in row)
            for row in branches
        )
    ):
        raise ValueError("endpoint response has the wrong boolean shape")
    return payload


def ranker_messages(
    history: Sequence[tuple[str, bool]],
    current_support: Sequence[str],
    candidate_rows: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    visible = []
    for index, row in enumerate(candidate_rows):
        visible.append(
            {
                "index": index,
                "question": row["question"],
                "predictive_probability_yes": row["p_yes"],
                "predictive_probability_no": 1.0 - row["p_yes"],
                "regenerated_support_if_yes": row["support_if_yes"],
                "regenerated_support_if_no": row["support_if_no"],
            }
        )
    return [
        {
            "role": "system",
            "content": (
                "Rank candidate questions for an open-world Twenty Questions agent. "
                "The unknown target and evaluation target list are not provided and "
                "the target may be absent from the current support. Score the expected "
                "chance that the regenerated support reached after the truthful answer "
                "contains the unknown target. Judge semantic plausibility, breadth, "
                "omissions, and contradictions in the actual branch supports, weighted "
                "by predictive answer probabilities. Do not optimize immediate "
                "information gain or support size by itself. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return exactly one finite score in [0,1] per candidate, in original "
                'order, as {"scores":[number,...]}. Larger means better expected '
                "future target recall. Data: "
                + json.dumps(
                    {
                        "category": "clothing",
                        "history": [
                            {
                                "question": question,
                                "answer": "Yes" if answer else "No",
                            }
                            for question, answer in history
                        ],
                        "current_support": list(current_support),
                        "candidates": visible,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log(probability) - (
        1.0 - probability
    ) * math.log(1.0 - probability)


def _select(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = np.asarray(_average_ranks(left), dtype=float)
    right_ranks = np.asarray(_average_ranks(right), dtype=float)
    left_ranks -= left_ranks.mean()
    right_ranks -= right_ranks.mean()
    denominator = float(np.linalg.norm(left_ranks) * np.linalg.norm(right_ranks))
    if denominator == 0.0:
        return None
    return float(np.dot(left_ranks, right_ranks) / denominator)


def _bootstrap_mean(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = array[
        rng.integers(0, len(array), size=(10_000, len(array)))
    ].mean(axis=1)
    return {
        "seed": BOOTSTRAP_SEED,
        "num_resamples": 10_000,
        "mean": float(array.mean()),
        "ci90": [
            float(np.quantile(draws, 0.05)),
            float(np.quantile(draws, 0.95)),
        ],
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    endpoint = record["endpoint"]
    realized_coverages = [
        float(branches[0] if answer else branches[1])
        for answer, branches in zip(
            endpoint["target_answers"],
            endpoint["branch_contains_target"],
            strict=True,
        )
    ]
    eig_scores = [
        float(row["immediate_eig"]) for row in record["candidates"]
    ]
    size_scores = [
        float(row["expected_support_size"]) for row in record["candidates"]
    ]
    result: dict[str, Any] = {
        "current_contains_target": bool(endpoint["current_contains_target"]),
        "realized_candidate_coverages": realized_coverages,
        "coverage_spread": max(realized_coverages) - min(realized_coverages),
        "best_candidate_coverage": max(realized_coverages),
        "immediate_eig_selected_index": _select(eig_scores),
        "size_selected_index": _select(size_scores),
    }
    result["immediate_eig_selected_coverage"] = realized_coverages[
        result["immediate_eig_selected_index"]
    ]
    result["size_selected_coverage"] = realized_coverages[
        result["size_selected_index"]
    ]
    if "ranker_scores" in record:
        ranker_index = _select(record["ranker_scores"])
        result["ranker_selected_index"] = ranker_index
        result["ranker_selected_coverage"] = realized_coverages[ranker_index]
        result["ranker_minus_immediate_eig"] = (
            realized_coverages[ranker_index]
            - result["immediate_eig_selected_coverage"]
        )
        result["ranker_minus_size"] = (
            realized_coverages[ranker_index]
            - result["size_selected_coverage"]
        )
    return result


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    diagnostics = [analyze_record(record) for record in records]
    gates: dict[str, bool] = {
        "all_cases_complete": len(records) == len(selected_indices(stage)),
        "exact_physical_request_count": int(usage["physical_requests"])
        == EXPECTED_REQUESTS[stage],
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_supports_nonempty": all(
            record["current_support"]
            and all(
                row["support_if_yes"] and row["support_if_no"]
                for row in record["candidates"]
            )
            for record in records
        ),
    }
    summary: dict[str, Any] = {
        "num_cases": len(records),
        "case_diagnostics": [
            {"case_index": record["case_index"], **row}
            for record, row in zip(records, diagnostics, strict=True)
        ],
    }
    if stage == "serving_smoke":
        gates["all_ranker_scores_finite"] = all(
            len(record.get("ranker_scores", ())) == CANDIDATE_COUNT
            and all(math.isfinite(value) for value in record["ranker_scores"])
            for record in records
        )
    elif stage == "activity":
        omitted = [
            row for row in diagnostics if not row["current_contains_target"]
        ]
        recovered = [
            row for row in omitted if row["best_candidate_coverage"] > 0.0
        ]
        active = [row for row in diagnostics if row["coverage_spread"] > 0.0]
        summary.update(
            {
                "current_target_omission_count": len(omitted),
                "omitted_target_recovery_count": len(recovered),
                "active_candidate_coverage_count": len(active),
            }
        )
        gates.update(
            {
                "current_target_omission_count_at_least_3": len(omitted) >= 3,
                "omitted_target_recovery_count_at_least_2": len(recovered) >= 2,
                "active_candidate_coverage_count_at_least_3": len(active) >= 3,
            }
        )
    else:
        ranker_selected = [
            row["ranker_selected_coverage"] for row in diagnostics
        ]
        eig_selected = [
            row["immediate_eig_selected_coverage"] for row in diagnostics
        ]
        size_selected = [
            row["size_selected_coverage"] for row in diagnostics
        ]
        gain_eig = [
            row["ranker_minus_immediate_eig"] for row in diagnostics
        ]
        gain_size = [row["ranker_minus_size"] for row in diagnostics]
        all_ranker_scores = [
            float(score)
            for record in records
            for score in record["ranker_scores"]
        ]
        all_eig_scores = [
            float(row["immediate_eig"])
            for record in records
            for row in record["candidates"]
        ]
        all_size_scores = [
            float(row["expected_support_size"])
            for record in records
            for row in record["candidates"]
        ]
        all_coverages = [
            float(value)
            for row in diagnostics
            for value in row["realized_candidate_coverages"]
        ]
        ranker_rho = _spearman(all_ranker_scores, all_coverages)
        eig_rho = _spearman(all_eig_scores, all_coverages)
        size_rho = _spearman(all_size_scores, all_coverages)
        bootstrap_eig = _bootstrap_mean(gain_eig)
        bootstrap_size = _bootstrap_mean(gain_size)
        ranker_eig_wtl = [
            sum(value > 0.0 for value in gain_eig),
            sum(value == 0.0 for value in gain_eig),
            sum(value < 0.0 for value in gain_eig),
        ]
        summary.update(
            {
                "mean_selected_realized_coverage_ranker": float(
                    np.mean(ranker_selected)
                ),
                "mean_selected_realized_coverage_immediate_eig": float(
                    np.mean(eig_selected)
                ),
                "mean_selected_realized_coverage_expected_size": float(
                    np.mean(size_selected)
                ),
                "ranker_minus_immediate_eig_bootstrap": bootstrap_eig,
                "ranker_minus_expected_size_bootstrap": bootstrap_size,
                "ranker_immediate_eig_wins_ties_losses": ranker_eig_wtl,
                "ranker_selection_differs_from_immediate_eig_count": sum(
                    row["ranker_selected_index"]
                    != row["immediate_eig_selected_index"]
                    for row in diagnostics
                ),
                "spearman_ranker_vs_realized_coverage": ranker_rho,
                "spearman_immediate_eig_vs_realized_coverage": eig_rho,
                "spearman_expected_size_vs_realized_coverage": size_rho,
            }
        )
        gates.update(
            {
                "ranker_selection_differs_at_least_5": (
                    summary[
                        "ranker_selection_differs_from_immediate_eig_count"
                    ]
                    >= 5
                ),
                "mean_ranker_gain_over_immediate_eig_at_least_0_10": (
                    bootstrap_eig["mean"] >= 0.10
                ),
                "ranker_immediate_eig_ci90_lower_positive": (
                    bootstrap_eig["ci90"][0] > 0.0
                ),
                "ranker_more_wins_than_losses": (
                    ranker_eig_wtl[0] > ranker_eig_wtl[2]
                ),
                "mean_ranker_not_worse_than_expected_size": (
                    bootstrap_size["mean"] >= 0.0
                ),
                "ranker_spearman_positive_and_best": (
                    ranker_rho is not None
                    and ranker_rho > 0.0
                    and (eig_rho is None or ranker_rho > eig_rho)
                    and (size_rho is None or ranker_rho > size_rho)
                ),
            }
        )
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def _build_models(config: Config) -> tuple[Any, Any]:
    if len(config.model_pairs) != 1:
        raise ValueError("PAPRIKA clothing gate requires one model pair")
    generator_spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    judge_spec = replace(
        config.model_pairs[0].answerer,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    return (
        build_model_adapter(generator_spec, config),
        build_model_adapter(judge_spec, config),
    )


def _usage_snapshot(generator: Any, judge: Any) -> dict[str, Any]:
    generator_usage = generator.usage_snapshot()
    judge_usage = judge.usage_snapshot()
    return {
        "physical_requests": int(generator_usage["adapter_requests"])
        + int(judge_usage["adapter_requests"]),
        "reasoning_tokens": int(generator_usage["adapter_reasoning_tokens"])
        + int(judge_usage["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(generator_usage["adapter_cost_usd"])
        + float(judge_usage["adapter_cost_usd"]),
        "generator": generator_usage,
        "judge": judge_usage,
    }


def _checkpoint(path: Path | None, *, stage: str, raw: dict[str, Any]) -> None:
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


def run_gate(
    config: Config,
    *,
    data_path: str | Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    _game_config, cases = load_selected_cases(data_path, stage)
    generator, judge = _build_models(config)
    raw: dict[str, Any] = {}
    try:
        prefix_label_raw = judge.chat_complete_messages_batched(
            [
                support_label_messages(
                    [case["env"]],
                    PREFIX_QUESTIONS,
                )
                for case in cases
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["prefix_labels"] = prefix_label_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        histories = [
            list(
                zip(
                    PREFIX_QUESTIONS,
                    [
                        row[0]
                        for row in parse_support_labels(text, support_size=1)
                    ],
                    strict=True,
                )
            )
            for text in prefix_label_raw
        ]

        initial_keys = [
            (case_offset, sample_index)
            for case_offset in range(len(cases))
            for sample_index in range(SUPPORT_SAMPLES)
        ]
        initial_raw = generator.chat_complete_messages_batched(
            [
                support_messages(histories[case_offset])
                for case_offset, _sample_index in initial_keys
            ],
            temperature=0.6,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial_supports"] = initial_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        initial_samples: dict[int, list[list[str]]] = {
            index: [] for index in range(len(cases))
        }
        for (case_offset, _sample_index), text in zip(
            initial_keys, initial_raw, strict=True
        ):
            initial_samples[case_offset].append(parse_support(text))
        current_supports = [
            union_supports(initial_samples[index])
            for index in range(len(cases))
        ]

        candidate_raw = generator.chat_complete_messages_batched(
            [
                candidate_messages(history, support)
                for history, support in zip(
                    histories, current_supports, strict=True
                )
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["candidates"] = candidate_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        candidates = [parse_candidates(text) for text in candidate_raw]

        current_label_raw = judge.chat_complete_messages_batched(
            [
                support_label_messages(support, questions)
                for support, questions in zip(
                    current_supports, candidates, strict=True
                )
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["current_support_labels"] = current_label_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        support_labels = [
            parse_support_labels(text, support_size=len(support))
            for text, support in zip(
                current_label_raw, current_supports, strict=True
            )
        ]

        branch_keys = [
            (case_offset, candidate_index, answer, sample_index)
            for case_offset in range(len(cases))
            for candidate_index in range(CANDIDATE_COUNT)
            for answer in (True, False)
            for sample_index in range(SUPPORT_SAMPLES)
        ]
        branch_raw = generator.chat_complete_messages_batched(
            [
                support_messages(
                    [
                        *histories[case_offset],
                        (
                            candidates[case_offset][candidate_index],
                            answer,
                        ),
                    ],
                    current_support=current_supports[case_offset],
                )
                for (
                    case_offset,
                    candidate_index,
                    answer,
                    _sample_index,
                ) in branch_keys
            ],
            temperature=0.6,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["branch_supports"] = branch_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        branch_samples: dict[
            tuple[int, int, bool], list[list[str]]
        ] = {}
        for key, text in zip(branch_keys, branch_raw, strict=True):
            case_offset, candidate_index, answer, _sample_index = key
            branch_samples.setdefault(
                (case_offset, candidate_index, answer), []
            ).append(parse_support(text))
        branch_supports = [
            [
                {
                    "yes": union_supports(
                        branch_samples[(case_offset, candidate_index, True)]
                    ),
                    "no": union_supports(
                        branch_samples[(case_offset, candidate_index, False)]
                    ),
                }
                for candidate_index in range(CANDIDATE_COUNT)
            ]
            for case_offset in range(len(cases))
        ]

        endpoint_raw = judge.chat_complete_messages_batched(
            [
                endpoint_messages(
                    case["env"],
                    current_support,
                    questions,
                    branches,
                )
                for case, current_support, questions, branches in zip(
                    cases,
                    current_supports,
                    candidates,
                    branch_supports,
                    strict=True,
                )
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["endpoints"] = endpoint_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        endpoints = [parse_endpoint(text) for text in endpoint_raw]

        records: list[dict[str, Any]] = []
        for case_offset, case in enumerate(cases):
            support_size = len(current_supports[case_offset])
            candidate_rows = []
            for candidate_index in range(CANDIDATE_COUNT):
                labels = support_labels[case_offset][candidate_index]
                p_yes = sum(labels) / support_size
                branch = branch_supports[case_offset][candidate_index]
                candidate_rows.append(
                    {
                        "question": candidates[case_offset][candidate_index],
                        "p_yes": p_yes,
                        "immediate_eig": binary_entropy(p_yes),
                        "support_if_yes": branch["yes"],
                        "support_if_no": branch["no"],
                        "expected_support_size": (
                            p_yes * len(branch["yes"])
                            + (1.0 - p_yes) * len(branch["no"])
                        ),
                    }
                )
            records.append(
                {
                    "case_index": selected_indices(stage)[case_offset],
                    "target": case["env"],
                    "history": [
                        {
                            "question": question,
                            "answer": "Yes" if answer else "No",
                        }
                        for question, answer in histories[case_offset]
                    ],
                    "current_support": current_supports[case_offset],
                    "candidates": candidate_rows,
                    "endpoint": endpoints[case_offset],
                }
            )

        if stage != "activity":
            ranker_raw = generator.chat_complete_messages_batched(
                [
                    ranker_messages(
                        histories[case_offset],
                        current_supports[case_offset],
                        records[case_offset]["candidates"],
                    )
                    for case_offset in range(len(cases))
                ],
                temperature=0.0,
                block_size=config.batched_block_size,
                max_new_tokens=config.openrouter_max_output_tokens,
            )
            raw["ranker"] = ranker_raw
            _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
            for record, text in zip(records, ranker_raw, strict=True):
                scores = parse_scores(text, CANDIDATE_COUNT)
                if any(score < 0.0 or score > 1.0 for score in scores):
                    raise ValueError("ranker scores must lie in [0,1]")
                record["ranker_scores"] = scores

        usage = _usage_snapshot(generator, judge)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(generator, judge),
        ) from exc

    summary = summarize(records, usage, stage=stage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "dataset": "PAPRIKA twenty_questions eval clothing",
            "dataset_sha256": DATA_SHA256,
            "selection_seed": SELECTION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "case_indices": list(selected_indices(stage)),
            "support_samples_per_state": SUPPORT_SAMPLES,
            "support_sample_size": SUPPORT_SAMPLE_SIZE,
            "candidate_count": CANDIDATE_COUNT,
            "prefix_questions": list(PREFIX_QUESTIONS),
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "target_hidden_from_generator_and_ranker": True,
            "target_used_only_by_post_generation_endpoint": True,
            "reasoning_disabled": True,
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
        "--data-path",
        type=Path,
        default=Path(
            "external/paprika/llm_exploration/game/game_configs/"
            "twenty_questions.json"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "activity", "confirmation"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    projected_and_caps = {
        "serving_smoke": (0.50, 1.00),
        "activity": (1.50, 3.00),
        "confirmation": (3.50, 6.00),
    }
    (
        config.openrouter_projected_cost_usd,
        config.openrouter_run_budget_usd,
    ) = projected_and_caps[args.stage]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_names = {
        "serving_smoke": "SERVING_SMOKE.json",
        "activity": "ACTIVITY.json",
        "confirmation": "CONFIRMATION.json",
    }
    try:
        payload = run_gate(
            config,
            data_path=args.data_path,
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
    output_path = args.output_dir / output_names[args.stage]
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
