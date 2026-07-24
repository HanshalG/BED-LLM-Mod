#!/usr/bin/env python3
"""Gate target-blind support-recall ranking on PAPRIKA instruments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.animals_belief_recall_ranker import parse_scores
from scripts.paprika_clothing_support_recall_gate import (
    CANDIDATE_COUNT,
    DATA_SHA256,
    SUPPORT_SAMPLE_SIZE,
    SUPPORT_SAMPLES,
    _average_ranks,
    _build_models,
    _checkpoint,
    _normalized,
    _usage_snapshot,
    analyze_record,
    binary_entropy,
    parse_candidates,
    parse_support,
    parse_support_labels,
    union_supports,
)


SCHEMA_VERSION = 1
SELECTION_SEED = 24332
BOOTSTRAP_SEED = 24333
SMOKE_INDICES = (212, 204)
ACTIVITY_INDICES = (198, 206, 213, 11, 211, 14, 203)
FORMAL_INDICES = (
    13,
    196,
    193,
    210,
    200,
    15,
    195,
    205,
    194,
    207,
    12,
    192,
    209,
    17,
    215,
    10,
    202,
    197,
    16,
    208,
)
PREFIX_QUESTIONS = (
    "Does it produce sound primarily through vibrating strings?",
    "Does the player primarily blow air through or across it to make sound?",
    "Does it produce sound primarily when struck or hit?",
)
EXPECTED_REQUESTS = {
    "serving_smoke": len(SMOKE_INDICES)
    * (
        2
        + SUPPORT_SAMPLES
        + 1
        + 2
        + CANDIDATE_COUNT * 2 * SUPPORT_SAMPLES
        + 2
        + 1
    ),
    "activity": len(ACTIVITY_INDICES)
    * (
        2
        + SUPPORT_SAMPLES
        + 1
        + 2
        + CANDIDATE_COUNT * 2 * SUPPORT_SAMPLES
        + 2
    ),
    "confirmation": len(FORMAL_INDICES)
    * (
        2
        + SUPPORT_SAMPLES
        + 1
        + 2
        + CANDIDATE_COUNT * 2 * SUPPORT_SAMPLES
        + 2
        + 1
    ),
}


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


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
) -> list[dict[str, str]]:
    path = Path(data_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("PAPRIKA Twenty Questions data hash does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("eval")
    if not isinstance(rows, list) or len(rows) != 367:
        raise ValueError("PAPRIKA Twenty Questions eval split is malformed")
    instrument_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("agent") == "instrument"
    ]
    if len(instrument_indices) != 29:
        raise ValueError("expected exactly 29 PAPRIKA eval instrument targets")
    reproduced = list(instrument_indices)
    random.Random(SELECTION_SEED).shuffle(reproduced)
    frozen = list(SMOKE_INDICES + ACTIVITY_INDICES + FORMAL_INDICES)
    if reproduced != frozen:
        raise ValueError("frozen target-blind instrument split does not reproduce")
    selected = [rows[index] for index in selected_indices(stage)]
    if any(
        not isinstance(row.get("env"), str)
        or row.get("agent") != "instrument"
        for row in selected
    ):
        raise ValueError("selected PAPRIKA instrument case is malformed")
    return selected


def support_messages(
    history: Sequence[tuple[str, bool]],
    *,
    current_support: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    payload: dict[str, Any] = {
        "category": "musical instrument",
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
                "Maintain an open-world hypothesis support for Twenty Questions. "
                "Use semantic knowledge; the hidden target and evaluation list are "
                "unavailable. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate exactly {SUPPORT_SAMPLE_SIZE} distinct specific musical "
                "instruments consistent with every answer. Generate afresh, include "
                "plausible instruments another support might omit, and use canonical "
                "names rather than descriptions or disjunctions. Return exactly "
                '{"possibilities":["instrument",...]}. Data: '
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def candidate_messages(
    history: Sequence[tuple[str, bool]],
    current_support: Sequence[str],
) -> list[dict[str, str]]:
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
                f"Propose exactly {CANDIDATE_COUNT} distinct questions answerable "
                "Yes or No for any musical instrument. Test different semantic "
                "properties, do not repeat history, and do not directly guess one "
                "instrument. Prefer questions whose answers make a language model "
                "recall meaningfully different plausible instruments. Return exactly "
                '{"questions":["question",...]}. Data: '
                + json.dumps(
                    {
                        "category": "musical instrument",
                        "history": [
                            {
                                "question": question,
                                "answer": "Yes" if answer else "No",
                            }
                            for question, answer in history
                        ],
                        "current_open_world_support": list(current_support),
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def label_messages(
    items: Sequence[str],
    questions: Sequence[str],
    *,
    no_framed: bool,
) -> list[dict[str, str]]:
    truth_condition = (
        "true means the truthful answer is No"
        if no_framed
        else "true means the truthful answer is Yes"
    )
    return [
        {
            "role": "system",
            "content": (
                "Classify musical instruments under yes-or-no semantic questions "
                "using ordinary real-world meanings. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"For every question return one boolean per item in order, where "
                f"{truth_condition}. Return exactly "
                '{"labels":[[true,...],...]}. Data: '
                + json.dumps(
                    {
                        "items": list(items),
                        "questions": list(questions),
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def recover_yes_labels_from_reversed_no(
    reversed_no_labels: Sequence[Sequence[bool]],
) -> list[list[bool]]:
    return [
        [not value for value in reversed(row)]
        for row in reversed(reversed_no_labels)
    ]


def checked_yes_labels(
    direct_text: str,
    reversed_no_text: str,
    *,
    item_count: int,
) -> list[list[bool]]:
    direct = parse_support_labels(direct_text, support_size=item_count)
    reversed_no = parse_support_labels(
        reversed_no_text,
        support_size=item_count,
    )
    recovered = recover_yes_labels_from_reversed_no(reversed_no)
    if direct != recovered:
        raise ValueError("complementary semantic label passes disagree")
    return direct


def ranker_messages(
    history: Sequence[tuple[str, bool]],
    current_support: Sequence[str],
    candidate_rows: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    candidates = [
        {
            "index": index,
            "question": row["question"],
            "predictive_probability_yes": row["p_yes"],
            "predictive_probability_no": 1.0 - row["p_yes"],
            "regenerated_support_if_yes": row["support_if_yes"],
            "regenerated_support_if_no": row["support_if_no"],
        }
        for index, row in enumerate(candidate_rows)
    ]
    return [
        {
            "role": "system",
            "content": (
                "Rank candidate questions for open-world Twenty Questions. The "
                "unknown target and evaluation list are not provided and the target "
                "may be absent from current support. Score the expected chance that "
                "the regenerated support reached after the truthful answer contains "
                "the unknown target. Judge semantic plausibility, breadth, omissions, "
                "and contradictions in the actual branches, weighted by predictive "
                "answer probabilities. Do not optimize immediate information gain "
                "or support size alone. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return one finite score in [0,1] per candidate in original order as "
                '{"scores":[number,...]}. Data: '
                + json.dumps(
                    {
                        "category": "musical instrument",
                        "history": [
                            {
                                "question": question,
                                "answer": "Yes" if answer else "No",
                            }
                            for question, answer in history
                        ],
                        "current_support": list(current_support),
                        "candidates": candidates,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def exact_contains(support: Sequence[str], target: str) -> bool:
    key = _normalized(target)
    return any(_normalized(value) == key for value in support)


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
        "all_complementary_label_checks_pass": all(
            record["all_label_checks_pass"] for record in records
        ),
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
                "current_target_omission_count_at_least_2": len(omitted) >= 2,
                "omitted_target_recovery_count_at_least_2": len(recovered) >= 2,
                "active_candidate_coverage_count_at_least_3": len(active) >= 3,
            }
        )
    else:
        gain_eig = [
            row["ranker_minus_immediate_eig"] for row in diagnostics
        ]
        gain_size = [row["ranker_minus_size"] for row in diagnostics]
        all_ranker = [
            float(value)
            for record in records
            for value in record["ranker_scores"]
        ]
        all_eig = [
            float(row["immediate_eig"])
            for record in records
            for row in record["candidates"]
        ]
        all_size = [
            float(row["expected_support_size"])
            for record in records
            for row in record["candidates"]
        ]
        all_coverage = [
            float(value)
            for row in diagnostics
            for value in row["realized_candidate_coverages"]
        ]
        ranker_rho = _spearman(all_ranker, all_coverage)
        eig_rho = _spearman(all_eig, all_coverage)
        size_rho = _spearman(all_size, all_coverage)
        bootstrap_eig = _bootstrap_mean(gain_eig)
        bootstrap_size = _bootstrap_mean(gain_size)
        wtl = [
            sum(value > 0.0 for value in gain_eig),
            sum(value == 0.0 for value in gain_eig),
            sum(value < 0.0 for value in gain_eig),
        ]
        summary.update(
            {
                "mean_selected_realized_coverage_ranker": float(
                    np.mean(
                        [row["ranker_selected_coverage"] for row in diagnostics]
                    )
                ),
                "mean_selected_realized_coverage_immediate_eig": float(
                    np.mean(
                        [
                            row["immediate_eig_selected_coverage"]
                            for row in diagnostics
                        ]
                    )
                ),
                "mean_selected_realized_coverage_expected_size": float(
                    np.mean(
                        [row["size_selected_coverage"] for row in diagnostics]
                    )
                ),
                "ranker_minus_immediate_eig_bootstrap": bootstrap_eig,
                "ranker_minus_expected_size_bootstrap": bootstrap_size,
                "ranker_immediate_eig_wins_ties_losses": wtl,
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
                "ranker_more_wins_than_losses": wtl[0] > wtl[2],
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


def run_gate(
    config: Config,
    *,
    data_path: str | Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    cases = load_selected_cases(data_path, stage)
    generator, judge = _build_models(config)
    raw: dict[str, Any] = {}
    try:
        prefix_direct_raw = judge.chat_complete_messages_batched(
            [
                label_messages([case["env"]], PREFIX_QUESTIONS, no_framed=False)
                for case in cases
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        prefix_inverse_raw = judge.chat_complete_messages_batched(
            [
                label_messages(
                    [case["env"]],
                    list(reversed(PREFIX_QUESTIONS)),
                    no_framed=True,
                )
                for case in cases
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["prefix_direct"] = prefix_direct_raw
        raw["prefix_inverse"] = prefix_inverse_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        prefix_labels = [
            checked_yes_labels(direct, inverse, item_count=1)
            for direct, inverse in zip(
                prefix_direct_raw, prefix_inverse_raw, strict=True
            )
        ]
        histories = [
            list(
                zip(
                    PREFIX_QUESTIONS,
                    [row[0] for row in labels],
                    strict=True,
                )
            )
            for labels in prefix_labels
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

        current_direct_raw = judge.chat_complete_messages_batched(
            [
                label_messages(support, questions, no_framed=False)
                for support, questions in zip(
                    current_supports, candidates, strict=True
                )
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        current_inverse_raw = judge.chat_complete_messages_batched(
            [
                label_messages(
                    list(reversed(support)),
                    list(reversed(questions)),
                    no_framed=True,
                )
                for support, questions in zip(
                    current_supports, candidates, strict=True
                )
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["current_direct"] = current_direct_raw
        raw["current_inverse"] = current_inverse_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        support_labels = [
            checked_yes_labels(direct, inverse, item_count=len(support))
            for direct, inverse, support in zip(
                current_direct_raw,
                current_inverse_raw,
                current_supports,
                strict=True,
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
                        (candidates[case_offset][candidate_index], answer),
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
        branches = [
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

        target_direct_raw = judge.chat_complete_messages_batched(
            [
                label_messages([case["env"]], questions, no_framed=False)
                for case, questions in zip(cases, candidates, strict=True)
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        target_inverse_raw = judge.chat_complete_messages_batched(
            [
                label_messages(
                    [case["env"]],
                    list(reversed(questions)),
                    no_framed=True,
                )
                for case, questions in zip(cases, candidates, strict=True)
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["target_direct"] = target_direct_raw
        raw["target_inverse"] = target_inverse_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        target_labels = [
            checked_yes_labels(direct, inverse, item_count=1)
            for direct, inverse in zip(
                target_direct_raw, target_inverse_raw, strict=True
            )
        ]
        target_answers = [
            [row[0] for row in labels] for labels in target_labels
        ]

        records: list[dict[str, Any]] = []
        for case_offset, case in enumerate(cases):
            support_size = len(current_supports[case_offset])
            candidate_rows = []
            for candidate_index in range(CANDIDATE_COUNT):
                labels = support_labels[case_offset][candidate_index]
                p_yes = sum(labels) / support_size
                branch = branches[case_offset][candidate_index]
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
                    "endpoint": {
                        "target_answers": target_answers[case_offset],
                        "current_contains_target": exact_contains(
                            current_supports[case_offset], case["env"]
                        ),
                        "branch_contains_target": [
                            [
                                exact_contains(branch["yes"], case["env"]),
                                exact_contains(branch["no"], case["env"]),
                            ]
                            for branch in branches[case_offset]
                        ],
                    },
                    "all_label_checks_pass": True,
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
            "dataset": "PAPRIKA twenty_questions eval instrument",
            "dataset_sha256": DATA_SHA256,
            "selection_seed": SELECTION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "case_indices": list(selected_indices(stage)),
            "support_samples_per_state": SUPPORT_SAMPLES,
            "support_sample_size": SUPPORT_SAMPLE_SIZE,
            "candidate_count": CANDIDATE_COUNT,
            "prefix_questions": list(PREFIX_QUESTIONS),
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "complementary_label_checks": True,
            "exact_normalized_target_coverage": True,
            "target_hidden_from_generator_and_ranker": True,
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
        "serving_smoke": (0.20, 0.75),
        "activity": (0.75, 2.00),
        "confirmation": (2.00, 4.00),
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
