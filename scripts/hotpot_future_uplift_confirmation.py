#!/usr/bin/env python3
"""Confirm future-uplift selection on fresh HotpotQA training records."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank_bm25 import BM25Okapi

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.hotpot_causal_belief_smoke import (
    GateExecutionError,
    _answer_and_enabling_titles,
    _argmax,
    _candidate_titles,
    _normalized_state,
    _usage_snapshot,
    final_messages,
    initial_messages,
    parse_final,
    parse_initial,
    parse_refresh,
    parse_scorer,
    refresh_messages,
    run_smoke,
    scorer_messages,
    token_f1,
)
from scripts.hotpot_directional_unlock_audit import (
    analyze_row,
    normalize_text,
    ordered_list_hash,
    sha256_file,
)


INTERFACE_VERSION = "hotpot-future-uplift-confirmation-1"
MODEL_ID = "openai/gpt-5.4"
SOURCE_HASHES = (
    "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68",
    "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f",
)
SOURCE_ROWS = (45_224, 45_223)
TOTAL_ROWS = 90_447
BRIDGE_ROWS = 72_991
SELECTION_SEED = 24_419
RANDOM_CONTROL_SEED = 24_420
SPLIT_SIZES = {
    "opportunity": 1_000,
    "development": 100,
    "confirmation": 500,
}
SPLIT_HASHES = {
    "opportunity": "2e63446b32879cae47876214261054ef696e153ebf621042fc5420bc7e1eb60c",
    "development": "01755657af1915c4c3171cbc2454f5c12ea6e48e18f8c5fae7cf7fbb68db3a7a",
    "confirmation": "eb97d88de626bcf6aefa118f0ec19e2d62200495200da4ed63dc5587aa756079",
    "holdout": "f1fc6d1119a8fdda01860055adf1d146f900433c76f1af508bc725ff4ec088d8",
}
METADATA_SHA256 = "e75f2c4ec26b8e3edefb76a1efa81390d9440b634c435ab3a978b5e455c83542"
CONFIRMATION_TASKS = 10
ROOTS_PER_TASK = 4
EXPECTED_CONFIRMATION_REQUESTS = 100
SERVING_COST_CAP = 0.15
CONFIRMATION_COST_CAP = 1.20


def _metadata_hash(rows: Sequence[dict[str, Any]]) -> str:
    encoded = json.dumps(
        list(rows),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadata_splits(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if len(paths) != 2:
        raise ValueError("exactly two HotpotQA training shards are required")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow==21.0.0 is required") from exc
    metadata: list[dict[str, Any]] = []
    for path, expected_hash, expected_rows in zip(
        paths, SOURCE_HASHES, SOURCE_ROWS, strict=True
    ):
        if sha256_file(path) != expected_hash:
            raise ValueError("HotpotQA training shard hash mismatch")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != expected_rows:
            raise ValueError("HotpotQA training shard row count mismatch")
        metadata.extend(
            parquet.read(columns=["id", "type", "level"]).to_pylist()
        )
    if len(metadata) != TOTAL_ROWS:
        raise ValueError("HotpotQA training row count mismatch")
    if sum(row["type"] == "bridge" for row in metadata) != BRIDGE_ROWS:
        raise ValueError("HotpotQA training bridge count mismatch")
    if _metadata_hash(metadata) != METADATA_SHA256:
        raise ValueError("HotpotQA metadata hash mismatch")
    bridge_ids = sorted(
        str(row["id"]) for row in metadata if row["type"] == "bridge"
    )
    random.Random(SELECTION_SEED).shuffle(bridge_ids)
    opportunity_end = SPLIT_SIZES["opportunity"]
    development_end = opportunity_end + SPLIT_SIZES["development"]
    confirmation_end = development_end + SPLIT_SIZES["confirmation"]
    splits = {
        "opportunity": bridge_ids[:opportunity_end],
        "development": bridge_ids[opportunity_end:development_end],
        "confirmation": bridge_ids[development_end:confirmation_end],
        "holdout": bridge_ids[confirmation_end:],
    }
    for name, expected_hash in SPLIT_HASHES.items():
        if ordered_list_hash(splits[name]) != expected_hash:
            raise ValueError(f"HotpotQA training {name} split hash mismatch")
    return metadata, splits


def selected_rows(paths: Sequence[Path], ordered_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Materialize endpoint columns only after an Arrow ID predicate is applied."""
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError("pyarrow==21.0.0 is required") from exc
    ids = list(ordered_ids)
    table = ds.dataset([str(path) for path in paths], format="parquet").to_table(
        filter=ds.field("id").isin(ids)
    )
    rows = table.to_pylist()
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(ids) or set(by_id) != set(ids):
        raise ValueError("filtered HotpotQA rows do not match the frozen IDs")
    return [by_id[task_id] for task_id in ids]


def title_bm25_order(question: str, titles: Sequence[str]) -> list[int]:
    model = BM25Okapi([normalize_text(title).split() for title in titles])
    scores = model.get_scores(normalize_text(question).split())
    return sorted(
        range(len(titles)),
        key=lambda index: (-float(scores[index]), index),
    )


def qualification(row: dict[str, Any]) -> dict[str, Any]:
    diagnostic = analyze_row(row)
    if not diagnostic["strict_unlock"]:
        return {**diagnostic, "qualifies": False}
    titles = [str(value) for value in row["context"]["title"]]
    order = title_bm25_order(str(row["question"]), titles)
    answer_index = int(diagnostic["answer_title_context_index"])
    enabling_index = int(diagnostic["enabling_title_context_index"])
    both_top_four = answer_index in order[:4] and enabling_index in order[:4]
    qualifies = (
        both_top_four and diagnostic["title_bm25_role"] != "enabling"
    )
    return {
        **diagnostic,
        "qualifies": qualifies,
        "both_supports_in_top_four": both_top_four,
        "root_context_indices": order[:4],
    }


def manifest_payload(paths: Sequence[Path]) -> dict[str, Any]:
    metadata, splits = metadata_splits(paths)
    return {
        "schema_version": 1,
        "status": "passed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_hashes": list(SOURCE_HASHES),
            "source_rows": list(SOURCE_ROWS),
            "metadata_columns_read": ["id", "type", "level"],
            "endpoint_columns_read": False,
            "selection_seed": SELECTION_SEED,
            "metadata_sha256": _metadata_hash(metadata),
        },
        "splits": {
            name: {"count": len(values), "ordered_id_sha256": ordered_list_hash(values)}
            for name, values in splits.items()
        },
    }


def opportunity_payload(paths: Sequence[Path]) -> dict[str, Any]:
    _metadata, splits = metadata_splits(paths)
    rows = selected_rows(paths, splits["opportunity"])
    diagnostics = [qualification(row) for row in rows]
    strict = [row for row in diagnostics if row["strict_unlock"]]
    qualifying = [row for row in diagnostics if row["qualifies"]]
    gates = {
        "exact_1000_opportunity_rows": len(rows) == 1_000,
        "strict_directional_unlocks_at_least_250": len(strict) >= 250,
        "qualifying_top_four_misses_at_least_20": len(qualifying) >= 20,
        "every_qualifying_row_has_unit_structural_gain": all(
            row["support_coverage_gain"] == 1 for row in qualifying
        ),
        "development_confirmation_holdout_not_materialized": True,
        "zero_model_calls": True,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "split": "opportunity",
            "split_sha256": SPLIT_HASHES["opportunity"],
            "endpoint_rows_materialized": 1_000,
            "other_split_endpoint_rows_materialized": 0,
            "model_calls": 0,
            "cost_usd": 0.0,
        },
        "metrics": {
            "strict_directional_unlocks": len(strict),
            "qualifying_top_four_misses": len(qualifying),
            "qualifying_neither_title_in_question": sum(
                bool(row["neither_support_title_in_question"])
                for row in qualifying
            ),
            "qualifying_levels": {
                level: sum(row["level"] == level for row in qualifying)
                for level in ("easy", "medium", "hard")
            },
        },
        "gates": gates,
    }


def run_serving(
    config: Config,
    *,
    validation_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    """Exercise the exact all-stage transport on the already-open smoke row."""
    payload = run_smoke(
        config,
        data_path=validation_path,
        raw_path=raw_path,
        model_adapter=model_adapter,
    )
    old_gates = payload["summary"]["gates"]
    continuation_rows = payload["continuation_rows"]
    future_scores = [
        max(row["aligned_scores"]) for row in continuation_rows
    ]
    immediate_scores = payload["initial"]["root_scores"]
    gates = {
        "exact_10_physical_requests": old_gates[
            "exact_10_physical_requests"
        ],
        "exact_10_http_attempts": old_gates["exact_10_http_attempts"],
        "zero_transport_retries": old_gates["zero_transport_retries"],
        "zero_reasoning_tokens": old_gates["zero_reasoning_tokens"],
        "zero_forced_exits": old_gates["zero_forced_exits"],
        "all_responses_parsed_without_repair": old_gates[
            "all_responses_parsed_without_repair"
        ],
        "all_refreshed_states_differ_from_initial": old_gates[
            "all_refreshed_states_differ_from_initial"
        ],
        "all_refreshed_states_pairwise_distinct": old_gates[
            "all_refreshed_states_pairwise_distinct"
        ],
        "balanced_blinding_labels": old_gates["balanced_blinding_labels"],
        "aligned_score_variation_at_least_3_roots": old_gates[
            "aligned_score_variation_at_least_3_roots"
        ],
        "aligned_shuffled_vectors_differ_at_least_3_roots": old_gates[
            "aligned_shuffled_vectors_differ_at_least_3_roots"
        ],
        "aligned_initial_vectors_differ_at_least_3_roots": old_gates[
            "aligned_initial_vectors_differ_at_least_3_roots"
        ],
        "future_scores_have_spread": len(set(future_scores)) >= 2,
        "future_first_root_is_defined": 0
        <= future_first_root(immediate_scores, continuation_rows)
        < ROOTS_PER_TASK,
        "cost_at_most_0_15": float(
            payload["usage"]["adapter_cost_usd"]
        )
        <= SERVING_COST_CAP,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "serving",
            "source": "already_open_validation_smoke_task",
            "scientific_training_record_accessed": False,
            "scientific_endpoint_reported": False,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "repairs_or_scientific_retries": 0,
        },
        "metrics": {
            "logical_stage_counts": {
                "initial": 1,
                "refresh": 4,
                "continuation_scorer": 4,
                "final_answer": 1,
            },
            "future_score_spread": max(future_scores) - min(future_scores),
            "future_first_root_index": future_first_root(
                immediate_scores, continuation_rows
            ),
        },
        "gates": gates,
        "usage": payload["usage"],
    }


def _build_model(config: Config) -> Any:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("Hotpot future-uplift config selects the wrong model")
    return build_model_adapter(spec, config)


def future_first_root(
    immediate_scores: Sequence[int],
    continuation_rows: Sequence[dict[str, Any]],
) -> int:
    future_scores = [max(row["aligned_scores"]) for row in continuation_rows]
    return max(
        range(len(future_scores)),
        key=lambda index: (
            future_scores[index],
            immediate_scores[index],
            -index,
        ),
    )


def policy_selections(
    *,
    immediate_scores: Sequence[int],
    continuation_rows: Sequence[dict[str, Any]],
    random_seed: int,
) -> dict[str, dict[str, int]]:
    aligned_max = [max(row["aligned_scores"]) for row in continuation_rows]
    initial_max = [max(row["initial_scores"]) for row in continuation_rows]
    shuffled_max = [max(row["shuffled_scores"]) for row in continuation_rows]
    roots = {
        "future_uplift": future_first_root(immediate_scores, continuation_rows),
        "myopic": _argmax(immediate_scores),
        "old_total": _argmax(
            [
                immediate + future
                for immediate, future in zip(
                    immediate_scores, aligned_max, strict=True
                )
            ]
        ),
        "fixed": _argmax(
            [
                immediate + future
                for immediate, future in zip(
                    immediate_scores, initial_max, strict=True
                )
            ]
        ),
        "shuffled": _argmax(
            [
                immediate + future
                for immediate, future in zip(
                    immediate_scores, shuffled_max, strict=True
                )
            ]
        ),
        "random": random.Random(random_seed).randrange(ROOTS_PER_TASK),
    }
    selections: dict[str, dict[str, int]] = {}
    for name, root_index in roots.items():
        score_key = {
            "fixed": "initial_scores",
            "shuffled": "shuffled_scores",
        }.get(name, "aligned_scores")
        selections[name] = {
            "root_index": root_index,
            "followup_candidate_index": _argmax(
                continuation_rows[root_index][score_key]
            ),
        }
    return selections


def selection_coverage(
    *,
    selection: dict[str, int],
    root_context_indices: Sequence[int],
    titles: Sequence[str],
    support_titles: set[str],
) -> tuple[int, list[str]]:
    context_index = root_context_indices[selection["root_index"]]
    candidates = _candidate_titles(titles, context_index)
    chosen = [
        titles[context_index],
        candidates[selection["followup_candidate_index"]],
    ]
    return len(set(chosen) & support_titles), chosen


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _blinding(task_index: int, root_index: int) -> bool:
    return (task_index + root_index) % 2 == 0


def _pairwise_accuracy(scores: Sequence[int], values: Sequence[int]) -> tuple[int, int]:
    correct = comparable = 0
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            value_delta = values[left] - values[right]
            score_delta = scores[left] - scores[right]
            if value_delta == 0 or score_delta == 0:
                continue
            comparable += 1
            correct += int((value_delta > 0) == (score_delta > 0))
    return correct, comparable


def _sign_flip_p(differences: Sequence[int]) -> float:
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 1.0
    observed = sum(nonzero)
    extreme = 0
    for mask in range(1 << len(nonzero)):
        total = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(nonzero)
        )
        extreme += int(total >= observed)
    return extreme / (1 << len(nonzero))


def run_confirmation(
    config: Config,
    *,
    paths: Sequence[Path],
    raw_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    _metadata, splits = metadata_splits(paths)
    cohort = selected_rows(paths, splits["confirmation"])
    qualified = [
        (row, qualification(row))
        for row in cohort
        if qualification(row)["qualifies"]
    ]
    if len(qualified) < CONFIRMATION_TASKS:
        return {
            "schema_version": 1,
            "status": "failed_closed",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "confirmation_cohort_rows_materialized": len(cohort),
                "model_constructed": False,
                "model_calls": 0,
            },
            "error": (
                f"confirmation cohort has {len(qualified)} qualifying rows; "
                f"{CONFIRMATION_TASKS} required"
            ),
        }
    selected = qualified[:CONFIRMATION_TASKS]
    model = model_adapter if model_adapter is not None else _build_model(config)
    tasks: list[dict[str, Any]] = []
    for row, diagnostic in selected:
        titles = [str(value) for value in row["context"]["title"]]
        paragraphs = [
            " ".join(str(sentence) for sentence in sentences)
            for sentences in row["context"]["sentences"]
        ]
        roots = [int(value) for value in diagnostic["root_context_indices"]]
        tasks.append(
            {
                "row": row,
                "diagnostic": diagnostic,
                "titles": titles,
                "paragraphs": paragraphs,
                "roots": roots,
            }
        )
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "task_ids": [str(task["row"]["id"]) for task in tasks],
    }
    try:
        initial_responses = model.chat_complete_messages_batched(
            [
                initial_messages(
                    str(task["row"]["question"]),
                    task["titles"],
                    [task["titles"][index] for index in task["roots"]],
                )
                for task in tasks
            ],
            temperature=0.0,
            block_size=CONFIRMATION_TASKS,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial"] = initial_responses
        _checkpoint(raw_path, raw)
        initials = [parse_initial(text) for text in initial_responses]

        refresh_messages_batch = []
        for task, initial in zip(tasks, initials, strict=True):
            for context_index in task["roots"]:
                refresh_messages_batch.append(
                    refresh_messages(
                        str(task["row"]["question"]),
                        initial["hypotheses"],
                        task["titles"][context_index],
                        task["paragraphs"][context_index],
                    )
                )
        refresh_responses = model.chat_complete_messages_batched(
            refresh_messages_batch,
            temperature=0.0,
            block_size=40,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["refreshes"] = refresh_responses
        _checkpoint(raw_path, raw)
        parsed_refreshes = [parse_refresh(text) for text in refresh_responses]
        refreshes = [
            parsed_refreshes[index * 4 : (index + 1) * 4]
            for index in range(CONFIRMATION_TASKS)
        ]

        scorer_batch = []
        for task_index, (task, initial, task_refreshes) in enumerate(
            zip(tasks, initials, refreshes, strict=True)
        ):
            for root_index, context_index in enumerate(task["roots"]):
                scorer_batch.append(
                    scorer_messages(
                        aligned=task_refreshes[root_index],
                        shuffled=task_refreshes[(root_index + 1) % 4],
                        initial=initial["hypotheses"],
                        candidates=_candidate_titles(
                            task["titles"], context_index
                        ),
                        aligned_is_a=_blinding(task_index, root_index),
                    )
                )
        scorer_responses = model.chat_complete_messages_batched(
            scorer_batch,
            temperature=0.0,
            block_size=40,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["scorers"] = scorer_responses
        _checkpoint(raw_path, raw)
        parsed_scorers = [parse_scorer(text, 9) for text in scorer_responses]
        scorer_tasks = [
            parsed_scorers[index * 4 : (index + 1) * 4]
            for index in range(CONFIRMATION_TASKS)
        ]

        records = []
        uplift_evidence = []
        for task_index, (task, initial, scores) in enumerate(
            zip(tasks, initials, scorer_tasks, strict=True)
        ):
            continuation_rows = []
            for root_index, score_vectors in enumerate(scores):
                aligned_label = (
                    "a" if _blinding(task_index, root_index) else "b"
                )
                shuffled_label = "b" if aligned_label == "a" else "a"
                continuation_rows.append(
                    {
                        "aligned_scores": score_vectors[aligned_label],
                        "shuffled_scores": score_vectors[shuffled_label],
                        "initial_scores": score_vectors["c"],
                    }
                )
            policies = policy_selections(
                immediate_scores=initial["root_scores"],
                continuation_rows=continuation_rows,
                random_seed=RANDOM_CONTROL_SEED + task_index,
            )
            support_titles = set(
                str(value)
                for value in task["row"]["supporting_facts"]["title"]
            )
            policy_rows = {}
            for name, selection in policies.items():
                coverage, titles = selection_coverage(
                    selection=selection,
                    root_context_indices=task["roots"],
                    titles=task["titles"],
                    support_titles=support_titles,
                )
                policy_rows[name] = {
                    **selection,
                    "support_coverage": coverage,
                    "selected_titles": titles,
                }
            root_values = []
            for root_index, row_scores in enumerate(continuation_rows):
                followup = _argmax(row_scores["aligned_scores"])
                coverage, _titles = selection_coverage(
                    selection={
                        "root_index": root_index,
                        "followup_candidate_index": followup,
                    },
                    root_context_indices=task["roots"],
                    titles=task["titles"],
                    support_titles=support_titles,
                )
                root_values.append(coverage)
            answer_title, enabling_title = _answer_and_enabling_titles(
                task["row"]
            )
            future_scores = [
                max(row["aligned_scores"]) for row in continuation_rows
            ]
            records.append(
                {
                    "task_id": str(task["row"]["id"]),
                    "answer_title": answer_title,
                    "enabling_title": enabling_title,
                    "root_titles": [
                        task["titles"][index] for index in task["roots"]
                    ],
                    "immediate_scores": initial["root_scores"],
                    "future_scores": future_scores,
                    "old_total_scores": [
                        immediate + future
                        for immediate, future in zip(
                            initial["root_scores"],
                            future_scores,
                            strict=True,
                        )
                    ],
                    "root_values": root_values,
                    "continuation_rows": continuation_rows,
                    "policies": policy_rows,
                }
            )
            uplift_evidence.append(
                [
                    {
                        "title": title,
                        "paragraph": task["paragraphs"][
                            task["titles"].index(title)
                        ],
                    }
                    for title in policy_rows["future_uplift"]["selected_titles"]
                ]
            )

        final_responses = model.chat_complete_messages_batched(
            [
                final_messages(str(task["row"]["question"]), evidence)
                for task, evidence in zip(tasks, uplift_evidence, strict=True)
            ],
            temperature=0.0,
            block_size=CONFIRMATION_TASKS,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["final"] = final_responses
        _checkpoint(raw_path, raw)
        finals = [parse_final(text) for text in final_responses]
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc

    public_records = []
    refresh_changed = pairwise_distinct = 0
    aligned_shuffled_changed = aligned_initial_changed = 0
    pair_counts = {
        "future": [0, 0],
        "myopic": [0, 0],
        "old_total": [0, 0],
    }
    for task_index, (task, initial, task_refreshes, record, final) in enumerate(
        zip(tasks, initials, refreshes, records, finals, strict=True)
    ):
        initial_state = _normalized_state(initial["hypotheses"])
        states = [_normalized_state(values) for values in task_refreshes]
        refresh_changed += sum(state != initial_state for state in states)
        pairwise_distinct += int(len(set(states)) == 4)
        aligned_shuffled_changed += sum(
            row["aligned_scores"] != row["shuffled_scores"]
            for row in record["continuation_rows"]
        )
        aligned_initial_changed += sum(
            row["aligned_scores"] != row["initial_scores"]
            for row in record["continuation_rows"]
        )
        for name, scores in (
            ("future", record["future_scores"]),
            ("myopic", record["immediate_scores"]),
            ("old_total", record["old_total_scores"]),
        ):
            correct, comparable = _pairwise_accuracy(
                scores, record["root_values"]
            )
            pair_counts[name][0] += correct
            pair_counts[name][1] += comparable
        policy_public = {
            name: {
                "root_index": values["root_index"],
                "followup_candidate_index": values[
                    "followup_candidate_index"
                ],
                "support_coverage": values["support_coverage"],
                "root_role": (
                    "enabling"
                    if values["selected_titles"][0]
                    == record["enabling_title"]
                    else "answer"
                    if values["selected_titles"][0] == record["answer_title"]
                    else "distractor"
                ),
                "followup_role": (
                    "answer"
                    if values["selected_titles"][1] == record["answer_title"]
                    else "enabling"
                    if values["selected_titles"][1]
                    == record["enabling_title"]
                    else "distractor"
                ),
            }
            for name, values in record["policies"].items()
        }
        public_records.append(
            {
                "task_id": record["task_id"],
                "future_scores": record["future_scores"],
                "immediate_scores": record["immediate_scores"],
                "old_total_scores": record["old_total_scores"],
                "root_values": record["root_values"],
                "policies": policy_public,
                "final_answer_token_f1": token_f1(
                    final["answer"], str(task["row"]["answer"])
                ),
            }
        )

    def policy_total(name: str) -> int:
        return sum(
            row["policies"][name]["support_coverage"]
            for row in public_records
        )

    def comparison(name: str) -> dict[str, Any]:
        differences = [
            row["policies"]["future_uplift"]["support_coverage"]
            - row["policies"][name]["support_coverage"]
            for row in public_records
        ]
        return {
            "gain": sum(differences),
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
            "exact_one_sided_sign_flip_p": _sign_flip_p(differences),
        }

    comparisons = {
        name: comparison(name) for name in ("myopic", "old_total", "random")
    }
    accuracies = {
        name: (
            correct / comparable if comparable else None
        )
        for name, (correct, comparable) in pair_counts.items()
    }
    generator = usage["generator"]
    gates = {
        "exact_10_selected_from_frozen_500": len(public_records) == 10,
        "exact_100_physical_requests": usage["physical_requests"]
        == EXPECTED_CONFIRMATION_REQUESTS,
        "exact_100_http_attempts": int(generator.get("http_attempts", -1))
        == EXPECTED_CONFIRMATION_REQUESTS,
        "zero_retries": int(generator.get("retry_count", -1)) == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": int(generator.get("forced_exits", -1)) == 0,
        "all_responses_parsed_without_repair": True,
        "at_least_36_of_40_refreshes_change": refresh_changed >= 36,
        "at_least_8_tasks_have_four_distinct_refreshes": pairwise_distinct >= 8,
        "at_least_32_aligned_shuffled_vectors_change": (
            aligned_shuffled_changed >= 32
        ),
        "at_least_32_aligned_initial_vectors_change": (
            aligned_initial_changed >= 32
        ),
        "future_uplift_changes_at_least_4_myopic_roots": sum(
            row["policies"]["future_uplift"]["root_index"]
            != row["policies"]["myopic"]["root_index"]
            for row in public_records
        )
        >= 4,
        "future_uplift_selects_enabling_at_least_5": sum(
            row["policies"]["future_uplift"]["root_role"] == "enabling"
            for row in public_records
        )
        >= 5,
        "enabling_branches_select_answer_at_least_8": sum(
            any(
                policy["root_role"] == "enabling"
                and policy["followup_role"] == "answer"
                for policy in row["policies"].values()
            )
            for row in public_records
        )
        >= 8,
        "future_uplift_covers_at_least_18_supports": (
            policy_total("future_uplift") >= 18
        ),
        "future_uplift_gains_at_least_2_vs_myopic": (
            comparisons["myopic"]["gain"] >= 2
        ),
        "future_uplift_wins_exceed_losses_vs_myopic": (
            comparisons["myopic"]["wins"] > comparisons["myopic"]["losses"]
        ),
        "future_uplift_sign_flip_p_at_most_0_10_vs_myopic": (
            comparisons["myopic"]["exact_one_sided_sign_flip_p"] <= 0.10
        ),
        "future_uplift_gains_at_least_2_vs_old_total": (
            comparisons["old_total"]["gain"] >= 2
        ),
        "future_uplift_wins_exceed_losses_vs_old_total": (
            comparisons["old_total"]["wins"]
            > comparisons["old_total"]["losses"]
        ),
        "future_uplift_gains_at_least_2_vs_random": (
            comparisons["random"]["gain"] >= 2
        ),
        "future_uplift_wins_exceed_losses_vs_random": (
            comparisons["random"]["wins"] > comparisons["random"]["losses"]
        ),
        "future_pairwise_accuracy_at_least_0_60": (
            accuracies["future"] is not None
            and accuracies["future"] >= 0.60
        ),
        "future_accuracy_at_least_0_05_above_myopic": (
            accuracies["future"] is not None
            and accuracies["myopic"] is not None
            and accuracies["future"] >= accuracies["myopic"] + 0.05
        ),
        "mean_final_answer_token_f1_at_least_0_40": (
            sum(row["final_answer_token_f1"] for row in public_records)
            / len(public_records)
            >= 0.40
        ),
        "cost_at_most_1_20": usage["adapter_cost_usd"]
        <= CONFIRMATION_COST_CAP,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "selection_seed": SELECTION_SEED,
            "random_control_seed": RANDOM_CONTROL_SEED,
            "confirmation_split_sha256": SPLIT_HASHES["confirmation"],
            "confirmation_cohort_rows_materialized": 500,
            "qualifying_rows_in_cohort": len(qualified),
            "selected_first_qualifying_count": CONFIRMATION_TASKS,
            "future_uplift_definition": (
                "argmax best aligned continuation score; "
                "immediate score then root order break ties"
            ),
            "endpoint_hidden_from_model": True,
            "repairs_or_scientific_retries": 0,
        },
        "metrics": {
            "policy_support_totals": {
                name: policy_total(name)
                for name in (
                    "future_uplift",
                    "myopic",
                    "old_total",
                    "fixed",
                    "shuffled",
                    "random",
                )
            },
            "comparisons": comparisons,
            "pairwise_root_accuracy": accuracies,
            "pairwise_root_counts": {
                name: {"correct": values[0], "comparable": values[1]}
                for name, values in pair_counts.items()
            },
            "future_uplift_enabling_roots": sum(
                row["policies"]["future_uplift"]["root_role"] == "enabling"
                for row in public_records
            ),
            "future_uplift_vs_myopic_root_changes": sum(
                row["policies"]["future_uplift"]["root_index"]
                != row["policies"]["myopic"]["root_index"]
                for row in public_records
            ),
            "mean_final_answer_token_f1": (
                sum(row["final_answer_token_f1"] for row in public_records)
                / len(public_records)
            ),
            "refreshes_changed": refresh_changed,
            "tasks_with_four_distinct_refreshes": pairwise_distinct,
            "aligned_shuffled_vector_changes": aligned_shuffled_changed,
            "aligned_initial_vector_changes": aligned_initial_changed,
        },
        "records": public_records,
        "gates": gates,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("manifest", "opportunity", "serving", "confirmation"),
        required=True,
    )
    parser.add_argument("--train-shard", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(args.train_shard)
    if args.stage == "manifest":
        payload = manifest_payload(paths)
        output_name = "MANIFEST.json"
    elif args.stage == "opportunity":
        payload = opportunity_payload(paths)
        output_name = "OPPORTUNITY.json"
    elif args.stage == "serving":
        if (
            args.config is None
            or args.private_raw_dir is None
            or not args.run_id
            or args.validation_data is None
        ):
            parser.error(
                "serving requires --config, --private-raw-dir, --run-id, "
                "and --validation-data"
            )
        config = load_config(str(args.config))
        config.run_id = args.run_id
        config.openrouter_budget_usd = min(
            float(config.openrouter_budget_usd), 105.0
        )
        config.openrouter_projected_cost_usd = 0.10
        config.openrouter_run_budget_usd = SERVING_COST_CAP
        config.openrouter_concurrency = 4
        config.log_path = args.output_dir / "run.log"
        private_dir = args.private_raw_dir / args.run_id
        private_dir.mkdir(parents=True, exist_ok=True)
        raw_path = private_dir / "RAW_RESPONSES.json"
        try:
            payload = run_serving(
                config,
                validation_path=args.validation_data,
                raw_path=raw_path,
            )
            payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
        except Exception as exc:
            failure: dict[str, Any] = {
                "schema_version": 1,
                "status": "failed_closed",
                "interface_version": INTERFACE_VERSION,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if isinstance(exc, GateExecutionError):
                failure["usage"] = exc.usage
            if raw_path.exists():
                failure["private_raw_sha256"] = sha256_file(raw_path)
            (args.output_dir / "SERVING_FAILURE.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise
        output_name = "SERVING.json"
    else:
        if args.config is None or args.private_raw_dir is None or not args.run_id:
            parser.error(
                "confirmation requires --config, --private-raw-dir, and --run-id"
            )
        config = load_config(str(args.config))
        config.run_id = args.run_id
        config.openrouter_budget_usd = min(
            float(config.openrouter_budget_usd), 105.0
        )
        config.openrouter_projected_cost_usd = 0.90
        config.openrouter_run_budget_usd = CONFIRMATION_COST_CAP
        config.openrouter_concurrency = 40
        config.log_path = args.output_dir / "run.log"
        private_dir = args.private_raw_dir / args.run_id
        private_dir.mkdir(parents=True, exist_ok=True)
        raw_path = private_dir / "RAW_RESPONSES.json"
        try:
            payload = run_confirmation(
                config,
                paths=paths,
                raw_path=raw_path,
            )
            if raw_path.exists():
                payload["protocol"]["private_raw_sha256"] = sha256_file(
                    raw_path
                )
        except Exception as exc:
            failure: dict[str, Any] = {
                "schema_version": 1,
                "status": "failed_closed",
                "interface_version": INTERFACE_VERSION,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if isinstance(exc, GateExecutionError):
                failure["usage"] = exc.usage
            if raw_path.exists():
                failure["private_raw_sha256"] = sha256_file(raw_path)
            (args.output_dir / "CONFIRMATION_FAILURE.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise
        output_name = "CONFIRMATION.json"
    output = args.output_dir / output_name
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] not in {"passed"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
