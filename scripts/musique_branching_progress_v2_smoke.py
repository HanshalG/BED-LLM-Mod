#!/usr/bin/env python3
"""Run a discrete dependency-progress MuSiQue 4hop3 smoke."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from model_factory import build_model_adapter
from scripts.musique_adaptive_retrieval_belief_gate import BM25Index
from scripts.musique_branching_causal_smoke import (
    FOLLOWUP_COUNT,
    HYPOTHESIS_COUNT,
    MAX_COST_USD,
    MODEL_ID,
    ROOT_COUNT,
    GateExecutionError,
    _argmax,
    _checkpoint,
    _document_map,
    _hypothesis_keys,
    _normalized,
    _paragraph_index,
    _query,
    _state_signature,
    _strict_object,
    _usage_snapshot,
    _validate_unique_queries,
    connected_prefix_length,
    parse_hypotheses,
    parse_refresh,
    refresh_messages,
)
from scripts.musique_branching_unlock_audit import (
    EXPECTED_DEPENDENCIES,
    SOURCE_SHA256,
    analyze_row,
    dependency_sets,
    sha256_file,
    split_ids,
)


INTERFACE_VERSION = "musique-branching-progress-v2-smoke-1"
TASK_IDS = (
    "4hop3__493923_300471_583182_62462",
    "4hop3__21483_551941_773286_24137",
)
EXPECTED_REQUESTS = 16
BLINDING_SEED = 24_356
RANDOM_CONTROL_SEED = 24_357
ALIGNED_IS_A = (
    (True, False, True, False, True, False),
    (False, True, False, True, False, True),
)


def _progress(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("progress band must not be Boolean")
    if isinstance(value, int):
        result = value
    elif (
        isinstance(value, str)
        and value.isdigit()
        and (len(value) == 1 or not value.startswith("0"))
    ):
        result = int(value)
    else:
        raise ValueError(
            "progress band must be an integer or canonical digit string"
        )
    if not 0 <= result <= 4:
        raise ValueError("progress band is outside 0..4")
    return result


def parse_initial(text: str) -> dict[str, Any]:
    payload = _strict_object(text)
    expected = set(_hypothesis_keys()) | {
        f"root_{index}_query" for index in range(1, ROOT_COUNT + 1)
    }
    if set(payload) != expected:
        raise ValueError("V2 initial response has unexpected keys")
    hypotheses = parse_hypotheses(
        {key: payload[key] for key in _hypothesis_keys()}
    )
    queries = [
        _query(payload[f"root_{index}_query"])
        for index in range(1, ROOT_COUNT + 1)
    ]
    _validate_unique_queries(queries)
    return {"hypotheses": hypotheses, "root_queries": queries}


def initial_messages(question: str) -> list[dict[str, str]]:
    schema = {
        **{
            key: "one distinct plausible dependency-chain hypothesis"
            for key in _hypothesis_keys()
        },
        **{
            f"root_{index}_query": "one distinct corpus search query"
            for index in range(1, ROOT_COUNT + 1)
        },
    }
    payload = {"question": question, "required_output": schema}
    return [
        {
            "role": "system",
            "content": (
                "Maintain an open-world belief over the dependency chain needed "
                "to answer a four-hop question. Generate exactly eight distinct "
                "plausible chains and six diverse first corpus-search queries. "
                "Include both immediately answer-relevant searches and prerequisite "
                "searches whose results could formulate a later query. Do not score "
                "the queries and do not use outside factual memory to assert the "
                "final answer. Return exactly the requested flat JSON object with "
                "no markdown or explanation."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def current_key(root: int, state: str) -> str:
    return f"root_{root}_state_{state}_current_progress"


def terminal_key(root: int, state: str, followup: int) -> str:
    return (
        f"root_{root}_state_{state}_followup_{followup}_terminal_progress"
    )


def scorer_keys() -> list[str]:
    return [
        key
        for root in range(1, ROOT_COUNT + 1)
        for state in ("a", "b", "c")
        for key in [
            current_key(root, state),
            *[
                terminal_key(root, state, followup)
                for followup in range(1, FOLLOWUP_COUNT + 1)
            ],
        ]
    ]


def parse_scorer(text: str) -> list[dict[str, dict[str, Any]]]:
    payload = _strict_object(text)
    if set(payload) != set(scorer_keys()):
        raise ValueError("V2 progress scorer has unexpected keys")
    return [
        {
            state: {
                "current": _progress(payload[current_key(root, state)]),
                "terminal": [
                    _progress(payload[terminal_key(root, state, followup)])
                    for followup in range(1, FOLLOWUP_COUNT + 1)
                ],
            }
            for state in ("a", "b", "c")
        }
        for root in range(1, ROOT_COUNT + 1)
    ]


def scorer_messages(
    *,
    initial_hypotheses: Sequence[str],
    refreshes: Sequence[dict[str, Any]],
    followup_titles: Sequence[Sequence[str]],
    aligned_is_a: Sequence[bool],
) -> list[dict[str, str]]:
    roots = []
    for root_index, refresh in enumerate(refreshes):
        aligned = refresh["hypotheses"]
        shuffled = refreshes[(root_index + 1) % ROOT_COUNT]["hypotheses"]
        state_a, state_b = (
            (aligned, shuffled)
            if aligned_is_a[root_index]
            else (shuffled, aligned)
        )
        roots.append(
            {
                "root_id": f"R{root_index + 1}",
                "state_a_unresolved_hypotheses": state_a,
                "state_b_unresolved_hypotheses": state_b,
                "state_c_unresolved_hypotheses": list(initial_hypotheses),
                "followup_candidates": [
                    {
                        "id": f"F{followup_index + 1}",
                        "query": query,
                        "retrieved_title": title,
                    }
                    for followup_index, (query, title) in enumerate(
                        zip(
                            refresh["followup_queries"],
                            followup_titles[root_index],
                            strict=True,
                        )
                    )
                ],
            }
        )
    payload = {
        "progress_scale": {
            "0": "no concrete dependency in the final question is resolved",
            "1": "one useful entity or fact is resolved",
            "2": "a connected two-step dependency chain is resolved",
            "3": "the connected chain plus an independent root is resolved",
            "4": "the final relation or answer is resolved from all dependencies",
        },
        "independent_root_states": roots,
        "required_output": {
            key: "integer 0..4" for key in scorer_keys()
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Estimate dependency progress from each supplied belief state. "
                "For every root/state, assign one current progress band and one "
                "terminal band for each possible follow-up. Use the same 0..4 scale "
                "everywhere; terminal means the total progress after that follow-up, "
                "not an incremental bonus. Treat A, B, and C independently. You do "
                "not know the question, first query, first result, answer, or gold "
                "annotations. Return exactly the requested flat JSON object with no "
                "markdown or explanation."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def policy_selections(
    *,
    progress_rows: Sequence[dict[str, Any]],
    task_index: int,
) -> dict[str, dict[str, int]]:
    root_values = {
        "myopic": [row["aligned"]["current"] for row in progress_rows],
        "model_aware": [
            max(row["aligned"]["terminal"]) for row in progress_rows
        ],
        "fixed": [max(row["initial"]["terminal"]) for row in progress_rows],
        "shuffled": [
            max(row["shuffled"]["terminal"]) for row in progress_rows
        ],
    }
    roots = {name: _argmax(values) for name, values in root_values.items()}
    roots["random"] = random.Random(
        RANDOM_CONTROL_SEED + task_index
    ).randrange(ROOT_COUNT)
    selections = {}
    for name, root_index in roots.items():
        row = progress_rows[root_index]
        state = {
            "fixed": "initial",
            "shuffled": "shuffled",
        }.get(name, "aligned")
        selections[name] = {
            "root_index": root_index,
            "followup_index": _argmax(row[state]["terminal"]),
        }
    return selections


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman(values: Sequence[float], targets: Sequence[float]) -> float:
    if len(values) != len(targets) or len(values) < 2:
        return math.nan
    left = _rankdata(values)
    right = _rankdata(targets)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def load_tasks(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("MuSiQue train source hash mismatch")
    metadata_ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    wanted = set(TASK_IDS)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task_id = str(row["id"])
            metadata_ids.append(task_id)
            if task_id in wanted:
                by_id[task_id] = row
    splits = split_ids(metadata_ids)
    if tuple(splits["development"][2:4]) != TASK_IDS:
        raise ValueError("V2 frozen task order does not reproduce")
    if set(by_id) != wanted:
        raise ValueError("V2 frozen tasks are missing")
    rows = [by_id[task_id] for task_id in TASK_IDS]
    for row in rows:
        if dependency_sets(row) != EXPECTED_DEPENDENCIES:
            raise ValueError("V2 dependency graph does not reproduce")
        if not analyze_row(row)["valid_branching_row"]:
            raise ValueError("V2 task structural eligibility does not reproduce")
    return rows


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
        raise ValueError("MuSiQue V2 config selects the wrong model")
    return build_model_adapter(spec, config)


def run_smoke(
    config: Config,
    *,
    data_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    rows = load_tasks(data_path)
    model = model_adapter if model_adapter is not None else _build_model(config)
    documents = [_document_map(row) for row in rows]
    indices = [BM25Index(value) for value in documents]
    raw: dict[str, Any] = {"task_ids": list(TASK_IDS)}
    try:
        initial_raw = model.chat_complete_messages_batched(
            [initial_messages(str(row["question"])) for row in rows],
            temperature=0.0,
            block_size=2,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial"] = initial_raw
        initials = [parse_initial(text) for text in initial_raw]

        root_doc_ids: list[list[str]] = []
        refresh_batch = []
        for task_index, (row, initial) in enumerate(
            zip(rows, initials, strict=True)
        ):
            task_root_ids = []
            for query in initial["root_queries"]:
                doc_id = indices[task_index].retrieve(query)
                task_root_ids.append(doc_id)
                paragraph = documents[task_index][doc_id]
                refresh_batch.append(
                    refresh_messages(
                        question=str(row["question"]),
                        initial_hypotheses=initial["hypotheses"],
                        root_query=query,
                        root_title=str(paragraph["title"]),
                        root_paragraph=str(paragraph["paragraph_text"]),
                    )
                )
            root_doc_ids.append(task_root_ids)

        refresh_raw = model.chat_complete_messages_batched(
            refresh_batch,
            temperature=0.0,
            block_size=12,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["refreshes"] = refresh_raw
        parsed_refreshes = [parse_refresh(text) for text in refresh_raw]
        refreshes = [
            parsed_refreshes[
                task_index * ROOT_COUNT : (task_index + 1) * ROOT_COUNT
            ]
            for task_index in range(len(rows))
        ]

        followup_doc_ids: list[list[list[str]]] = []
        followup_titles: list[list[list[str]]] = []
        for task_index in range(len(rows)):
            task_ids = []
            task_titles = []
            for root_index, refresh in enumerate(refreshes[task_index]):
                root_id = root_doc_ids[task_index][root_index]
                ids = [
                    indices[task_index].retrieve(query, exclude=(root_id,))
                    for query in refresh["followup_queries"]
                ]
                task_ids.append(ids)
                task_titles.append(
                    [str(documents[task_index][doc_id]["title"]) for doc_id in ids]
                )
            followup_doc_ids.append(task_ids)
            followup_titles.append(task_titles)

        scorer_raw = model.chat_complete_messages_batched(
            [
                scorer_messages(
                    initial_hypotheses=initials[task_index]["hypotheses"],
                    refreshes=refreshes[task_index],
                    followup_titles=followup_titles[task_index],
                    aligned_is_a=ALIGNED_IS_A[task_index],
                )
                for task_index in range(len(rows))
            ],
            temperature=0.0,
            block_size=2,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["scorers"] = scorer_raw
        _checkpoint(raw_path, raw)
        parsed_scorers = [parse_scorer(text) for text in scorer_raw]
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc

    task_records = []
    aligned_variation = 0
    aligned_shuffled_changes = 0
    aligned_initial_changes = 0
    deep_followup_successes = 0
    model_deep_roots = 0
    model_at_least_myopic = 0
    model_strictly_better_myopic = 0
    model_at_least_fixed = 0
    model_at_least_shuffled = 0
    all_refreshes_differ = True
    distinct_refresh_counts = []
    root_retrieval_counts = []
    both_roles_retrieved = []
    pooled_predicted = []
    pooled_exact = []

    for task_index, row in enumerate(rows):
        support_indices = [
            int(step["paragraph_support_idx"])
            for step in row["question_decomposition"]
        ]
        progress_rows = []
        exact_root_endpoints = []
        predicted_root_endpoints = []
        for root_index, states in enumerate(parsed_scorers[task_index]):
            aligned_label = "a" if ALIGNED_IS_A[task_index][root_index] else "b"
            shuffled_label = "b" if ALIGNED_IS_A[task_index][root_index] else "a"
            progress = {
                "root_index": root_index,
                "aligned_label": aligned_label.upper(),
                "aligned": states[aligned_label],
                "shuffled": states[shuffled_label],
                "initial": states["c"],
            }
            progress_rows.append(progress)
            aligned_variation += len(set(progress["aligned"]["terminal"])) > 1
            aligned_shuffled_changes += (
                progress["aligned"]["terminal"]
                != progress["shuffled"]["terminal"]
            )
            aligned_initial_changes += (
                progress["aligned"]["terminal"]
                != progress["initial"]["terminal"]
            )
            first_index = _paragraph_index(
                root_doc_ids[task_index][root_index]
            )
            exact = max(
                connected_prefix_length(
                    first_paragraph_index=first_index,
                    second_paragraph_index=_paragraph_index(doc_id),
                    support_indices=support_indices,
                )
                for doc_id in followup_doc_ids[task_index][root_index]
            )
            predicted = max(progress["aligned"]["terminal"])
            exact_root_endpoints.append(exact)
            predicted_root_endpoints.append(predicted)
            pooled_exact.append(exact)
            pooled_predicted.append(predicted)

        selections = policy_selections(
            progress_rows=progress_rows, task_index=task_index
        )
        policies = {}
        for name, selection in selections.items():
            root_index = selection["root_index"]
            followup_index = selection["followup_index"]
            first_index = _paragraph_index(root_doc_ids[task_index][root_index])
            second_index = _paragraph_index(
                followup_doc_ids[task_index][root_index][followup_index]
            )
            policies[name] = {
                **selection,
                "first_paragraph_index": first_index,
                "second_paragraph_index": second_index,
                "first_title": str(row["paragraphs"][first_index]["title"]),
                "second_title": str(row["paragraphs"][second_index]["title"]),
                "root_role": (
                    "deep_root"
                    if first_index == support_indices[0]
                    else "shallow_root"
                    if first_index == support_indices[2]
                    else "other"
                ),
                "connected_prefix_length": connected_prefix_length(
                    first_paragraph_index=first_index,
                    second_paragraph_index=second_index,
                    support_indices=support_indices,
                ),
            }

        root_paragraph_indices = [
            _paragraph_index(value) for value in root_doc_ids[task_index]
        ]
        deep_candidates = [
            index
            for index, paragraph_index in enumerate(root_paragraph_indices)
            if paragraph_index == support_indices[0]
        ]
        shallow_candidates = [
            index
            for index, paragraph_index in enumerate(root_paragraph_indices)
            if paragraph_index == support_indices[2]
        ]
        task_deep_success = any(
            _paragraph_index(
                followup_doc_ids[task_index][root_index][
                    _argmax(progress_rows[root_index]["aligned"]["terminal"])
                ]
            )
            == support_indices[1]
            for root_index in deep_candidates
        )
        deep_followup_successes += task_deep_success
        model_deep_roots += policies["model_aware"]["root_role"] == "deep_root"
        model_at_least_myopic += (
            policies["model_aware"]["connected_prefix_length"]
            >= policies["myopic"]["connected_prefix_length"]
        )
        model_strictly_better_myopic += (
            policies["model_aware"]["connected_prefix_length"]
            > policies["myopic"]["connected_prefix_length"]
        )
        model_at_least_fixed += (
            policies["model_aware"]["connected_prefix_length"]
            >= policies["fixed"]["connected_prefix_length"]
        )
        model_at_least_shuffled += (
            policies["model_aware"]["connected_prefix_length"]
            >= policies["shuffled"]["connected_prefix_length"]
        )

        initial_signature = _state_signature(initials[task_index]["hypotheses"])
        refresh_signatures = [
            _state_signature(value["hypotheses"])
            for value in refreshes[task_index]
        ]
        all_refreshes_differ &= all(
            value != initial_signature for value in refresh_signatures
        )
        distinct_refresh_counts.append(len(set(refresh_signatures)))
        root_retrieval_counts.append(len(set(root_doc_ids[task_index])))
        both_roles_retrieved.append(bool(deep_candidates and shallow_candidates))
        task_records.append(
            {
                "task_id": str(row["id"]),
                "initial": initials[task_index],
                "root_doc_ids": root_doc_ids[task_index],
                "root_titles": [
                    str(documents[task_index][doc_id]["title"])
                    for doc_id in root_doc_ids[task_index]
                ],
                "refreshes": refreshes[task_index],
                "followup_doc_ids": followup_doc_ids[task_index],
                "followup_titles": followup_titles[task_index],
                "progress_rows": progress_rows,
                "exact_best_prefix_by_root": exact_root_endpoints,
                "predicted_terminal_progress_by_root": predicted_root_endpoints,
                "root_spearman": spearman(
                    predicted_root_endpoints, exact_root_endpoints
                ),
                "deep_root_candidate_indices": deep_candidates,
                "shallow_root_candidate_indices": shallow_candidates,
                "aligned_deep_followup_retrieves_child": task_deep_success,
                "distinct_first_retrieval_count": root_retrieval_counts[-1],
                "distinct_refresh_state_count": distinct_refresh_counts[-1],
                "policies": policies,
            }
        )

    pooled_rho = spearman(pooled_predicted, pooled_exact)
    generator = usage["generator"]
    gates = {
        "exact_16_physical_requests": usage["physical_requests"]
        == EXPECTED_REQUESTS,
        "exact_16_http_attempts": int(generator.get("http_attempts", -1))
        == EXPECTED_REQUESTS,
        "zero_transport_retries": int(generator.get("retry_count", -1)) == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": int(generator.get("forced_exits", -1)) == 0,
        "all_responses_parsed_without_repair": True,
        "at_least_three_distinct_first_documents_each": all(
            value >= 3 for value in root_retrieval_counts
        ),
        "deep_and_shallow_roots_retrieved_each": all(both_roles_retrieved),
        "all_refreshed_states_differ_from_initial": all_refreshes_differ,
        "at_least_five_distinct_refreshes_each": all(
            value >= 5 for value in distinct_refresh_counts
        ),
        "balanced_blinding_labels": (
            sum(sum(task) for task in ALIGNED_IS_A) == 6
        ),
        "aligned_terminal_variation_at_least_10_roots": (
            aligned_variation >= 10
        ),
        "aligned_shuffled_terminal_vectors_differ_at_least_8_roots": (
            aligned_shuffled_changes >= 8
        ),
        "aligned_initial_terminal_vectors_differ_at_least_8_roots": (
            aligned_initial_changes >= 8
        ),
        "aligned_deep_followup_succeeds_at_least_once": (
            deep_followup_successes >= 1
        ),
        "model_aware_selects_deep_root_at_least_once": model_deep_roots >= 1,
        "pooled_root_spearman_at_least_0_30": (
            math.isfinite(pooled_rho) and pooled_rho >= 0.30
        ),
        "model_aware_at_least_myopic_both": model_at_least_myopic == 2,
        "model_aware_strictly_beats_myopic_at_least_once": (
            model_strictly_better_myopic >= 1
        ),
        "model_aware_at_least_fixed_both": model_at_least_fixed == 2,
        "model_aware_at_least_shuffled_both": model_at_least_shuffled == 2,
        "cost_at_most_0_50": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "task_ids": list(TASK_IDS),
            "blinding_seed": BLINDING_SEED,
            "aligned_is_a": [list(values) for values in ALIGNED_IS_A],
            "random_control_seed": RANDOM_CONTROL_SEED,
            "progress_scale": {
                "0": "no concrete dependency resolved",
                "1": "one useful entity or fact resolved",
                "2": "connected two-step dependency resolved",
                "3": "connected chain plus independent root resolved",
                "4": "final relation or answer resolved",
            },
            "question_hidden_from_progress_scorer": True,
            "root_query_hidden_from_progress_scorer": True,
            "root_evidence_hidden_from_progress_scorer": True,
            "annotations_hidden_from_all_model_prompts": True,
            "reasoning_requested": False,
            "repairs_or_scientific_retries": 0,
        },
        "summary": {
            "gates": gates,
            "aligned_terminal_variation_count": aligned_variation,
            "aligned_shuffled_terminal_change_count": (
                aligned_shuffled_changes
            ),
            "aligned_initial_terminal_change_count": aligned_initial_changes,
            "deep_followup_success_count": deep_followup_successes,
            "model_deep_root_count": model_deep_roots,
            "model_at_least_myopic_count": model_at_least_myopic,
            "model_strictly_better_myopic_count": (
                model_strictly_better_myopic
            ),
            "pooled_root_spearman": pooled_rho,
        },
        "tasks": task_records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.20
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 24
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_smoke(config, data_path=args.data, raw_path=raw_path)
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
        (args.output_dir / "SMOKE_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output = args.output_dir / "SMOKE.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
