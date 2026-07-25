#!/usr/bin/env python3
"""Run endpoint-blind HoVer semantic support-regeneration smokes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.hover_path_opportunity_audit import (
    DATABASE_SHA256,
    best_path,
    build_adjacency,
    contains_title,
    normalize_text,
    sha256_file,
)


MODEL_ID = "openai/gpt-5.4"
FIXTURE_SHA256 = (
    "86caf82219fcdbeddcec8462cc225458d3582bd8b438fd15af1ecd43a0efabce"
)
SOURCE_SHA256 = (
    "67c14858f2d7fcdb96b6fe3d538ffcd6f76e3ba594aa2c0cd4359f601101e89d"
)
TASK_IDS = (
    "a88d2342-f506-4b15-8578-fb7861eb54c1",
    "3cc79319-433d-49b2-97f3-953ba925d6bd",
)
HYPOTHESIS_COUNT = 8
PROPOSAL_COUNT = 3
ROOT_COUNT = 10
SERVING_ROOT_COUNT = 8
OBSERVATION_CHARS = 2_400
SERVING_REQUESTS = 10
MECHANICS_REQUESTS = 28
MAX_COST_USD = {"serving": 0.15, "mechanics": 0.50}
RANDOM_SEED = 24_406


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _canonical_score(value: str) -> int:
    if (
        not value.isdigit()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise ValueError("score must be a canonical integer")
    score = int(value)
    if not 0 <= score <= 100:
        raise ValueError("score is outside 0..100")
    return score


def _response_lines(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("response is empty")
    lines = stripped.splitlines()
    if any(not line.strip() or line != line.strip() for line in lines):
        raise ValueError("response contains blank or padded lines")
    return lines


def _parse_hypothesis_lines(
    lines: Sequence[str],
) -> list[dict[str, Any]]:
    if len(lines) != HYPOTHESIS_COUNT:
        raise ValueError("wrong number of hypothesis lines")
    hypotheses = []
    for index, line in enumerate(lines, start=1):
        parts = line.split("|", 2)
        if len(parts) != 3 or parts[0] != f"H{index:02d}":
            raise ValueError("invalid hypothesis line")
        weight = _canonical_score(parts[1])
        text = " ".join(parts[2].split())
        if not text or "|" in text:
            raise ValueError("hypothesis text is invalid")
        hypotheses.append({"weight": weight, "text": text})
    normalized = {
        normalize_text(row["text"]) for row in hypotheses
    }
    if "" in normalized or len(normalized) != HYPOTHESIS_COUNT:
        raise ValueError("hypotheses must be distinct")
    if sum(row["weight"] for row in hypotheses) <= 0:
        raise ValueError("hypothesis weights must have positive mass")
    return hypotheses


def parse_initial(text: str, *, root_count: int) -> dict[str, Any]:
    lines = _response_lines(text)
    expected = HYPOTHESIS_COUNT + root_count
    if len(lines) != expected:
        raise ValueError("initial response has the wrong line count")
    hypotheses = _parse_hypothesis_lines(lines[:HYPOTHESIS_COUNT])
    scores = []
    for index, line in enumerate(
        lines[HYPOTHESIS_COUNT:],
        start=1,
    ):
        parts = line.split("|")
        if len(parts) != 2 or parts[0] != f"R{index:02d}":
            raise ValueError("invalid root-score line")
        scores.append(_canonical_score(parts[1]))
    return {"hypotheses": hypotheses, "immediate_scores": scores}


def parse_refresh(
    text: str,
    *,
    candidate_count: int,
    root_candidate_index: int,
) -> dict[str, Any]:
    lines = _response_lines(text)
    expected = HYPOTHESIS_COUNT + PROPOSAL_COUNT
    if len(lines) != expected:
        raise ValueError("refresh response has the wrong line count")
    hypotheses = _parse_hypothesis_lines(lines[:HYPOTHESIS_COUNT])
    proposals = []
    for index, line in enumerate(
        lines[HYPOTHESIS_COUNT:],
        start=1,
    ):
        parts = line.split("|")
        if len(parts) != 2 or parts[0] != f"N{index:02d}":
            raise ValueError("invalid proposal line")
        candidate = parts[1]
        if (
            len(candidate) != 4
            or candidate[0] != "C"
            or not candidate[1:].isdigit()
        ):
            raise ValueError("proposal must be a Cxxx catalog ID")
        candidate_index = int(candidate[1:]) - 1
        if not 0 <= candidate_index < candidate_count:
            raise ValueError("proposal catalog ID is out of range")
        if candidate_index == root_candidate_index:
            raise ValueError("proposal repeats the revealed root")
        proposals.append(candidate_index)
    if len(set(proposals)) != PROPOSAL_COUNT:
        raise ValueError("proposals must be distinct")
    return {"hypotheses": hypotheses, "proposal_indexes": proposals}


def parse_future_scores(text: str, *, root_count: int) -> list[int]:
    lines = _response_lines(text)
    if len(lines) != root_count:
        raise ValueError("future scorer has the wrong line count")
    scores = []
    for index, line in enumerate(lines, start=1):
        parts = line.split("|")
        if len(parts) != 2 or parts[0] != f"R{index:02d}":
            raise ValueError("invalid future-score line")
        scores.append(_canonical_score(parts[1]))
    return scores


def initial_messages(
    task: Mapping[str, Any],
    *,
    root_count: int,
) -> list[dict[str, str]]:
    roots = [
        {"id": f"R{index + 1:02d}", "title": title}
        for index, title in enumerate(
            task["candidate_titles"][:root_count]
        )
    ]
    grammar = [
        f"H{index:02d}|0..100|one weighted evidence-chain hypothesis"
        for index in range(1, HYPOTHESIS_COUNT + 1)
    ] + [
        f"R{index:02d}|0..100"
        for index in range(1, root_count + 1)
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are the belief machinery for open-domain fact checking. "
                "Generate eight diverse weighted hypotheses about which evidence "
                "documents and relations could verify or refute the claim. The "
                "space is open world: do not limit hypotheses to the shown roots. "
                "Then score each root only for direct evidence value from opening "
                "that one title; do not imagine later searches. Use sharp 0-100 "
                "scores. Output only the exact flat lines requested."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "claim": task["claim"],
                    "root_candidates": roots,
                    "exact_output_lines": grammar,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        },
    ]


def refresh_messages(
    task: Mapping[str, Any],
    *,
    initial_hypotheses: Sequence[Mapping[str, Any]],
    root_index: int,
    root_text: str,
) -> list[dict[str, str]]:
    catalog = [
        {"id": f"C{index + 1:03d}", "title": title}
        for index, title in enumerate(task["candidate_titles"])
    ]
    grammar = [
        f"H{index:02d}|0..100|one updated evidence-chain hypothesis"
        for index in range(1, HYPOTHESIS_COUNT + 1)
    ] + [
        f"N{index:02d}|Cxxx"
        for index in range(1, PROPOSAL_COUNT + 1)
    ]
    return [
        {
            "role": "system",
            "content": (
                "Regenerate the open-world evidence belief after one document is "
                "revealed. Preserve, revise, add, or drop hypotheses based on the "
                "text. The updated state must identify unresolved relations and "
                "plausible evidence chains. Then choose three distinct catalog "
                "documents that would be best to open next. Do not use outside "
                "knowledge as observed evidence. Output only the exact flat lines."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "claim": task["claim"],
                    "prior_weighted_hypotheses": list(initial_hypotheses),
                    "revealed_document": {
                        "title": task["candidate_titles"][root_index],
                        "text": " ".join(root_text.split())[
                            :OBSERVATION_CHARS
                        ],
                    },
                    "candidate_catalog": catalog,
                    "exact_output_lines": grammar,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        },
    ]


def future_scorer_messages(
    task: Mapping[str, Any],
    *,
    bundles: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    roots = []
    for index, bundle in enumerate(bundles):
        roots.append(
            {
                "root_id": f"R{index + 1:02d}",
                "opened_title": task["candidate_titles"][index],
                "weighted_future_hypotheses": bundle["hypotheses"],
                "proposed_next_titles": [
                    task["candidate_titles"][candidate_index]
                    for candidate_index in bundle["proposal_indexes"]
                ],
            }
        )
    grammar = [
        f"R{index:02d}|0..100"
        for index in range(1, len(bundles) + 1)
    ]
    return [
        {
            "role": "system",
            "content": (
                "Score only the incremental future value created after each first "
                "document. High uplift means the supplied future belief is coherent, "
                "covers unresolved parts of the claim, and proposes next documents "
                "that can complete a multi-document evidence chain. Do not reward "
                "the first document's direct evidence again. Treat roots independently "
                "and use a sharp 0-100 range. Output only the exact flat lines."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "claim": task["claim"],
                    "root_future_bundles": roots,
                    "exact_output_lines": grammar,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        },
    ]


def _state_signature(hypotheses: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        (int(row["weight"]), normalize_text(str(row["text"])))
        for row in hypotheses
    ]
    return json.dumps(payload, separators=(",", ":"))


def _argmax(values: Sequence[int | float]) -> int:
    return max(
        range(len(values)),
        key=lambda index: (values[index], -index),
    )


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "generator": snapshot,
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
        raise ValueError("HoVer smoke config selects the wrong model")
    return build_model_adapter(spec, config)


def _checkpoint(path: Path, raw: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_fixture(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != FIXTURE_SHA256:
        raise ValueError("HoVer open mechanics fixture hash changed")
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("endpoint_fields_emitted") is not False:
        raise ValueError("HoVer mechanics fixture contains endpoints")
    tasks = fixture.get("tasks")
    if (
        not isinstance(tasks, list)
        or tuple(task["task_id"] for task in tasks) != TASK_IDS
    ):
        raise ValueError("HoVer mechanics task order changed")
    return tasks


def load_root_texts(
    database_path: Path,
    tasks: Sequence[Mapping[str, Any]],
    *,
    root_count: int,
) -> list[list[str]]:
    if sha256_file(database_path) != DATABASE_SHA256:
        raise ValueError("HoVer Wikipedia database hash changed")
    connection = sqlite3.connect(
        f"file:{database_path.resolve()}?mode=ro",
        uri=True,
    )
    outputs = []
    try:
        for task in tasks:
            rows = []
            for title in task["candidate_titles"][:root_count]:
                hit = connection.execute(
                    "SELECT text FROM documents WHERE id = ?",
                    (title,),
                ).fetchone()
                if hit is None:
                    raise ValueError(f"missing HoVer root document: {title}")
                rows.append(str(hit[0]))
            outputs.append(rows)
    finally:
        connection.close()
    return outputs


def _fixed_bundle(
    task: Mapping[str, Any],
    *,
    initial_hypotheses: Sequence[Mapping[str, Any]],
    root_index: int,
    root_text: str,
) -> dict[str, Any]:
    normalized = normalize_text(root_text)
    proposals = [
        index
        for index, title in enumerate(task["candidate_titles"])
        if index != root_index and contains_title(normalized, title)
    ][:PROPOSAL_COUNT]
    return {
        "hypotheses": list(initial_hypotheses),
        "proposal_indexes": proposals,
    }


def _shuffled_bundles(
    refreshes: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        refreshes[(index + 1) % len(refreshes)]
        for index in range(len(refreshes))
    ]


def run_model_stage(
    config: Config,
    *,
    stage: str,
    fixture_path: Path,
    database_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage not in {"serving", "mechanics"}:
        raise ValueError("stage must be serving or mechanics")
    all_tasks = load_fixture(fixture_path)
    tasks = all_tasks[:1] if stage == "serving" else all_tasks
    root_count = SERVING_ROOT_COUNT if stage == "serving" else ROOT_COUNT
    root_texts = load_root_texts(
        database_path,
        tasks,
        root_count=root_count,
    )
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {
        "stage": stage,
        "task_ids": [task["task_id"] for task in tasks],
    }
    try:
        initial_responses = model.chat_complete_messages_batched(
            [
                initial_messages(task, root_count=root_count)
                for task in tasks
            ],
            temperature=0.0,
            block_size=len(tasks),
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial"] = initial_responses
        initials = [
            parse_initial(response, root_count=root_count)
            for response in initial_responses
        ]
        _checkpoint(raw_path, raw)

        refresh_messages_batch = []
        refresh_locations = []
        for task_index, task in enumerate(tasks):
            for root_index in range(root_count):
                refresh_messages_batch.append(
                    refresh_messages(
                        task,
                        initial_hypotheses=initials[task_index][
                            "hypotheses"
                        ],
                        root_index=root_index,
                        root_text=root_texts[task_index][root_index],
                    )
                )
                refresh_locations.append((task_index, root_index))
        refresh_responses = model.chat_complete_messages_batched(
            refresh_messages_batch,
            temperature=0.0,
            block_size=min(
                len(refresh_messages_batch),
                config.openrouter_concurrency,
            ),
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["refreshes"] = refresh_responses
        refreshes: list[list[dict[str, Any] | None]] = [
            [None] * root_count for _ in tasks
        ]
        for response, (task_index, root_index) in zip(
            refresh_responses,
            refresh_locations,
            strict=True,
        ):
            refreshes[task_index][root_index] = parse_refresh(
                response,
                candidate_count=len(tasks[task_index]["candidate_titles"]),
                root_candidate_index=root_index,
            )
        parsed_refreshes = [
            [row for row in task_rows if row is not None]
            for task_rows in refreshes
        ]
        if any(len(rows) != root_count for rows in parsed_refreshes):
            raise ValueError("parsed refresh matrix is incomplete")
        _checkpoint(raw_path, raw)

        scorer_specs: list[tuple[int, str, list[dict[str, Any]]]] = []
        for task_index, task in enumerate(tasks):
            aligned = parsed_refreshes[task_index]
            scorer_specs.append((task_index, "aligned", aligned))
            if stage == "mechanics":
                scorer_specs.append(
                    (
                        task_index,
                        "shuffled",
                        _shuffled_bundles(aligned),
                    )
                )
                scorer_specs.append(
                    (
                        task_index,
                        "fixed",
                        [
                            _fixed_bundle(
                                task,
                                initial_hypotheses=initials[task_index][
                                    "hypotheses"
                                ],
                                root_index=root_index,
                                root_text=root_texts[task_index][root_index],
                            )
                            for root_index in range(root_count)
                        ],
                    )
                )
        scorer_responses = model.chat_complete_messages_batched(
            [
                future_scorer_messages(
                    tasks[task_index],
                    bundles=bundles,
                )
                for task_index, _, bundles in scorer_specs
            ],
            temperature=0.0,
            block_size=len(scorer_specs),
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["scorers"] = scorer_responses
        scores: list[dict[str, list[int]]] = [
            {} for _ in tasks
        ]
        for response, (task_index, variant, _) in zip(
            scorer_responses,
            scorer_specs,
            strict=True,
        ):
            scores[task_index][variant] = parse_future_scores(
                response,
                root_count=root_count,
            )
        _checkpoint(raw_path, raw)
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    parsed = {
        "tasks": tasks,
        "root_count": root_count,
        "initials": initials,
        "refreshes": parsed_refreshes,
        "future_scores": scores,
    }
    return parsed, usage


def _pairwise_accuracy(
    scores: Sequence[int | float],
    values: Sequence[int],
) -> tuple[float, int]:
    points = 0.0
    comparable = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] == values[right]:
                continue
            comparable += 1
            score_delta = scores[left] - scores[right]
            value_delta = values[left] - values[right]
            if score_delta == 0:
                points += 0.5
            elif (score_delta > 0) == (value_delta > 0):
                points += 1.0
    return (
        points / comparable if comparable else 0.5,
        comparable,
    )


def _belief_support_coverage(
    hypotheses: Sequence[Mapping[str, Any]],
    proposal_indexes: Sequence[int],
    *,
    candidate_titles: Sequence[str],
    support_titles: Sequence[str],
) -> int:
    text = normalize_text(
        " ".join(str(row["text"]) for row in hypotheses)
    )
    represented = {
        title
        for title in support_titles
        if contains_title(text, title)
    }
    represented.update(
        candidate_titles[index]
        for index in proposal_indexes
        if candidate_titles[index] in support_titles
    )
    return len(represented)


def _exact_root_values(
    *,
    task: Mapping[str, Any],
    supporting_titles: Sequence[str],
    database_path: Path,
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[tuple[int, ...]],
]:
    titles = list(task["candidate_titles"])
    connection = sqlite3.connect(
        f"file:{database_path.resolve()}?mode=ro",
        uri=True,
    )
    try:
        text_by_title = {}
        for title in titles:
            hit = connection.execute(
                "SELECT text FROM documents WHERE id = ?",
                (title,),
            ).fetchone()
            if hit is None:
                raise ValueError(f"missing HoVer candidate: {title}")
            text_by_title[title] = str(hit[0])
    finally:
        connection.close()
    adjacency = build_adjacency(titles, text_by_title)
    title_to_index = {
        title: index for index, title in enumerate(titles)
    }
    support_indexes = {
        title_to_index[title]
        for title in supporting_titles
        if title in title_to_index
    }
    immediate_values = []
    values2 = []
    values3 = []
    paths3 = []
    for root_index in range(ROOT_COUNT):
        value2, _ = best_path(
            root_index=root_index,
            adjacency=adjacency,
            support_indexes=support_indexes,
            max_documents=2,
        )
        value3, path3 = best_path(
            root_index=root_index,
            adjacency=adjacency,
            support_indexes=support_indexes,
            max_documents=3,
        )
        immediate_values.append(int(root_index in support_indexes))
        values2.append(value2)
        values3.append(value3)
        paths3.append(path3)
    return immediate_values, values2, values3, paths3


def analyze_mechanics(
    parsed: dict[str, Any],
    usage: dict[str, Any],
    *,
    source_path: Path,
    database_path: Path,
) -> dict[str, Any]:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("HoVer endpoint source hash changed")
    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    source_by_id = {str(row["uid"]): row for row in source_rows}
    task_results = []
    aligned_points = 0.0
    aligned_comparable = 0
    for task_index, task in enumerate(parsed["tasks"]):
        endpoint = source_by_id[task["task_id"]]
        support_titles = list(
            dict.fromkeys(
                str(fact[0])
                for fact in endpoint["supporting_facts"]
            )
        )
        immediate_values, values2, root_values, root_paths = _exact_root_values(
            task=task,
            supporting_titles=support_titles,
            database_path=database_path,
        )
        immediate = parsed["initials"][task_index][
            "immediate_scores"
        ]
        variants = parsed["future_scores"][task_index]
        full_scores = {
            variant: [
                immediate_score + future_score
                for immediate_score, future_score in zip(
                    immediate,
                    future,
                    strict=True,
                )
            ]
            for variant, future in variants.items()
        }
        roots = {
            "myopic": _argmax(immediate),
            "regenerated_d3": _argmax(full_scores["aligned"]),
            "fixed_support": _argmax(full_scores["fixed"]),
            "shuffled_future": _argmax(full_scores["shuffled"]),
            "random": random.Random(
                RANDOM_SEED + task_index
            ).randrange(ROOT_COUNT),
        }
        oracle_root = max(
            range(ROOT_COUNT),
            key=lambda index: (
                root_values[index],
                values2[index],
                immediate_values[index],
                -index,
            ),
        )
        oracle_value = root_values[oracle_root]
        oracle_roots = [
            index
            for index, value in enumerate(root_values)
            if value == oracle_value
        ]
        initial_coverage = _belief_support_coverage(
            parsed["initials"][task_index]["hypotheses"],
            [],
            candidate_titles=task["candidate_titles"],
            support_titles=support_titles,
        )
        oracle_refresh = parsed["refreshes"][task_index][oracle_root]
        refreshed_coverage = _belief_support_coverage(
            oracle_refresh["hypotheses"],
            oracle_refresh["proposal_indexes"],
            candidate_titles=task["candidate_titles"],
            support_titles=support_titles,
        )
        oracle_path = root_paths[oracle_root]
        first_continuation = (
            oracle_path[1] if len(oracle_path) > 1 else None
        )
        continuation_proposed = (
            first_continuation is not None
            and first_continuation
            in oracle_refresh["proposal_indexes"]
        )
        accuracy, comparable = _pairwise_accuracy(
            full_scores["aligned"],
            root_values,
        )
        aligned_points += accuracy * comparable
        aligned_comparable += comparable
        task_results.append(
            {
                "task_id": task["task_id"],
                "root_values": root_values,
                "oracle_value": oracle_value,
                "oracle_roots": oracle_roots,
                "initial_support_coverage": initial_coverage,
                "oracle_refresh_support_coverage": refreshed_coverage,
                "oracle_refresh_increases_support": (
                    refreshed_coverage > initial_coverage
                ),
                "oracle_first_continuation_proposed": (
                    continuation_proposed
                ),
                "future_scores": variants,
                "full_scores": full_scores,
                "selected_roots": roots,
                "selected_values": {
                    policy: root_values[root]
                    for policy, root in roots.items()
                },
                "aligned_pairwise_accuracy": accuracy,
                "aligned_comparable_pairs": comparable,
            }
        )
    policy_means = {
        policy: sum(
            row["selected_values"][policy] for row in task_results
        )
        / len(task_results)
        for policy in (
            "myopic",
            "regenerated_d3",
            "fixed_support",
            "shuffled_future",
            "random",
        )
    }
    initial_signatures = [
        _state_signature(row["hypotheses"])
        for row in parsed["initials"]
    ]
    refresh_signatures = [
        [
            _state_signature(refresh["hypotheses"])
            for refresh in rows
        ]
        for rows in parsed["refreshes"]
    ]
    all_refreshes_changed = all(
        signature != initial_signatures[task_index]
        for task_index, rows in enumerate(refresh_signatures)
        for signature in rows
    )
    distinct_counts = [
        len(set(rows)) for rows in refresh_signatures
    ]
    generator = usage["generator"]
    pooled_accuracy = (
        aligned_points / aligned_comparable
        if aligned_comparable else 0.5
    )
    gates = {
        "exact_28_physical_requests": (
            usage["physical_requests"] == MECHANICS_REQUESTS
        ),
        "exact_28_http_attempts": (
            int(generator.get("http_attempts", -1))
            == MECHANICS_REQUESTS
        ),
        "zero_transport_retries": (
            int(generator.get("retry_count", -1)) == 0
        ),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": (
            int(generator.get("forced_exits", -1)) == 0
        ),
        "all_responses_parsed_without_repair": True,
        "all_20_refreshes_differ_from_initial": all_refreshes_changed,
        "at_least_9_distinct_refreshes_per_task": all(
            count >= 9 for count in distinct_counts
        ),
        "aligned_vectors_nonconstant_both_tasks": all(
            len(set(row["future_scores"]["aligned"])) > 1
            for row in task_results
        ),
        "aligned_differs_shuffled_both_tasks": all(
            row["future_scores"]["aligned"]
            != row["future_scores"]["shuffled"]
            for row in task_results
        ),
        "aligned_differs_fixed_both_tasks": all(
            row["future_scores"]["aligned"]
            != row["future_scores"]["fixed"]
            for row in task_results
        ),
        "oracle_refresh_support_increases_at_least_one": sum(
            row["oracle_refresh_increases_support"]
            for row in task_results
        )
        >= 1,
        "oracle_first_continuation_proposed_at_least_one": sum(
            row["oracle_first_continuation_proposed"]
            for row in task_results
        )
        >= 1,
        "regenerated_changes_myopic_root_at_least_one": sum(
            row["selected_roots"]["regenerated_d3"]
            != row["selected_roots"]["myopic"]
            for row in task_results
        )
        >= 1,
        "regenerated_mean_value_strictly_above_myopic": (
            policy_means["regenerated_d3"] > policy_means["myopic"]
        ),
        "regenerated_mean_value_at_least_fixed": (
            policy_means["regenerated_d3"]
            >= policy_means["fixed_support"]
        ),
        "regenerated_mean_value_at_least_shuffled": (
            policy_means["regenerated_d3"]
            >= policy_means["shuffled_future"]
        ),
        "regenerated_selects_oracle_at_least_one": sum(
            row["selected_roots"]["regenerated_d3"]
            in row["oracle_roots"]
            for row in task_results
        )
        >= 1,
        "pooled_aligned_pairwise_accuracy_at_least_0_55": (
            pooled_accuracy >= 0.55
        ),
        "cost_at_most_0_50": (
            usage["adapter_cost_usd"] <= MAX_COST_USD["mechanics"]
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "stage": "mechanics",
        "model": MODEL_ID,
        "reasoning_requested": False,
        "task_results": task_results,
        "policy_mean_values": policy_means,
        "pooled_aligned_pairwise_accuracy": pooled_accuracy,
        "pooled_aligned_comparable_pairs": aligned_comparable,
        "refresh_distinct_counts": distinct_counts,
        "gates": gates,
        "usage": usage,
    }


def analyze_serving(
    parsed: dict[str, Any],
    usage: dict[str, Any],
) -> dict[str, Any]:
    initial_signature = _state_signature(
        parsed["initials"][0]["hypotheses"]
    )
    refresh_signatures = [
        _state_signature(row["hypotheses"])
        for row in parsed["refreshes"][0]
    ]
    scores = parsed["future_scores"][0]["aligned"]
    generator = usage["generator"]
    gates = {
        "exact_10_physical_requests": (
            usage["physical_requests"] == SERVING_REQUESTS
        ),
        "exact_10_http_attempts": (
            int(generator.get("http_attempts", -1))
            == SERVING_REQUESTS
        ),
        "zero_transport_retries": (
            int(generator.get("retry_count", -1)) == 0
        ),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": (
            int(generator.get("forced_exits", -1)) == 0
        ),
        "all_responses_parsed_without_repair": True,
        "all_8_refreshes_differ_from_initial": all(
            signature != initial_signature
            for signature in refresh_signatures
        ),
        "at_least_7_pairwise_distinct_refreshes": (
            len(set(refresh_signatures)) >= 7
        ),
        "future_score_vector_nonconstant": len(set(scores)) > 1,
        "cost_at_most_0_15": (
            usage["adapter_cost_usd"] <= MAX_COST_USD["serving"]
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "stage": "serving",
        "model": MODEL_ID,
        "endpoint_loaded": False,
        "reasoning_requested": False,
        "refresh_distinct_count": len(set(refresh_signatures)),
        "future_scores": scores,
        "gates": gates,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("serving", "mechanics"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.stage == "mechanics" and args.source_json is None:
        parser.error("--source-json is required for mechanics")
    if args.stage == "serving" and args.source_json is not None:
        parser.error("--source-json is forbidden for serving")

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = (
        0.08 if args.stage == "serving" else 0.25
    )
    config.openrouter_run_budget_usd = MAX_COST_USD[args.stage]
    config.openrouter_concurrency = 24
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        parsed, usage = run_model_stage(
            config,
            stage=args.stage,
            fixture_path=args.fixture,
            database_path=args.database,
            raw_path=raw_path,
        )
        result = (
            analyze_serving(parsed, usage)
            if args.stage == "serving"
            else analyze_mechanics(
                parsed,
                usage,
                source_path=args.source_json,
                database_path=args.database,
            )
        )
        result["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output = args.output_dir / "RESULT.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": args.stage,
                "output": str(output),
                "gates": result["gates"],
                "usage": result["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
