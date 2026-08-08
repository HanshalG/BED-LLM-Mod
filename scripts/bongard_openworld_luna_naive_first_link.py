#!/usr/bin/env python3
"""Run the endpoint-blind Luna reasoning baseline for Bongard first actions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import threading
from typing import Any, Mapping, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-naive-first-link-2"
MODEL_ID = serving.MODEL_ID
REASONING_EFFORT = "medium"
MAX_TOKENS = 8_192
MAX_REQUEST_COST_USD = 0.008
RUN_BUDGET_USD = 0.20
PROJECTED_COST_USD = 0.10
CONCURRENCY = 10
TEMPERATURE = 0.0
SMOKE_REQUESTS = 10
SMOKE_BASE_SEED = 2_026_080_801
BLOCK_BASE_SEEDS = {
    "a": 2_026_081_111,
    "b": 2_026_081_211,
    "c": 2_026_081_311,
    "d": 2_026_081_411,
}
BOOTSTRAP_SEED = 2_026_080_751
MIN_CHANGED_FINAL_HISTORIES = 16
MIN_RELATIVE_BRIER_IMPROVEMENT = 0.03
MIN_BOOTSTRAP_IMPROVEMENT_PROBABILITY = 0.80
PREREGISTRATION = (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_NAIVE_FIRST_LINK_PREREGISTRATION.md"
)
MAIN_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_development64/"
    "PROTOCOL_MANIFEST_V17.json"
)
MAIN_MANIFEST_SHA256 = (
    "7564ced7755f17be13f254067f013130b4beb277313fc51de16b43611a608676"
)


class StructuredModel(Protocol):
    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, Any]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class NaiveCase:
    case_id: str
    task: bed.VisualTask
    seed: int
    display_order: tuple[str, ...]


class LunaReasoningAdapter(serving.LunaVisionAdapter):
    """Retain Luna routing compatibility while enabling medium reasoning."""

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=disable_reasoning,
            response_format=response_format,
        )
        if disable_reasoning:
            payload["reasoning"] = {"enabled": False, "exclude": True}
        else:
            payload["reasoning"] = {
                "effort": REASONING_EFFORT,
                "exclude": True,
            }
        return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def message_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(bed.canonical_json(messages).encode()).hexdigest()


def verify_frozen_protocol() -> dict[str, Any]:
    from scripts import bongard_openworld_luna_naive_first_link_verify as verify

    return verify.verify_protocol_manifest()


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "bongard_naive_first_query",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"first_image_id": {"type": "string"}},
                "required": ["first_image_id"],
                "additionalProperties": False,
            },
        },
    }


def _display_order(task: bed.VisualTask, seed: int) -> tuple[str, ...]:
    candidates = list(task.candidate_ids)
    random.Random(seed).shuffle(candidates)
    return tuple(image_id for image_id, _ in task.initial_history) + tuple(candidates)


def seal_unqueried_labels(task: bed.VisualTask) -> bed.VisualTask:
    """Remove every label except the observations available before querying."""
    initial = dict(task.initial_history)
    return bed.VisualTask(
        task_id=task.task_id,
        image_ids=task.image_ids,
        initial_history=task.initial_history,
        candidate_ids=task.candidate_ids,
        endpoint_ids=task.endpoint_ids,
        image_bytes=task.image_bytes,
        actual_labels={image_id: initial[image_id] for image_id in initial},
        hidden_values=task.hidden_values,
    )


def make_case(task: bed.VisualTask, *, seed: int, suffix: str) -> NaiveCase:
    return NaiveCase(
        case_id=f"{task.task_id}-{suffix}",
        task=task,
        seed=seed,
        display_order=_display_order(task, seed),
    )


def build_messages(case: NaiveCase) -> list[dict[str, Any]]:
    task = case.task
    request = {
        "task": (
            "Choose the single most informative image to label first while "
            "inferring an unknown contrastive visual concept. One adaptive "
            "query will remain after this choice."
        ),
        "task_id": task.task_id,
        "observed_labels": [
            {"image_id": image_id, "label": bed.LABELS[label]}
            for image_id, label in task.initial_history
        ],
        "selectable_image_ids": list(task.candidate_ids),
        "display_order": list(case.display_order),
        "requirements": [
            "Infer candidate rules using both positive and negative examples.",
            "Choose an image whose possible label would best distinguish "
            "plausible rules and improve later classification of unseen images.",
            "Do not infer labels from opaque IDs or display position.",
            "Return only the strict JSON object with first_image_id.",
        ],
    }
    content: list[dict[str, Any]] = [
        {"type": "text", "text": bed.canonical_json(request)}
    ]
    for image_id in case.display_order:
        content.extend(
            [
                {"type": "text", "text": f"IMAGE {image_id}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": bed.image_data_url(task.image_bytes[image_id]),
                        "detail": bed.IMAGE_DETAIL,
                    },
                },
            ]
        )
    return [{"role": "user", "content": content}]


def prompt_errors(case: NaiveCase, messages: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    task = case.task
    text = bed.request_text(messages)
    lowered = text.casefold()
    for value in task.hidden_values:
        if value and value in text:
            errors.append("hidden_source_value")
    if "pos__" in lowered or "neg__" in lowered or "images/" in lowered:
        errors.append("label_bearing_source_path")
    for endpoint_id in task.endpoint_ids:
        if endpoint_id in text:
            errors.append("endpoint_id_exposed")
    if set(task.actual_labels) != {
        image_id for image_id, _ in task.initial_history
    }:
        errors.append("unqueried_labels_materialized")
    try:
        request = bed.strict_json_object(
            str(messages[0]["content"][0]["text"])
        )
    except Exception:
        errors.append("invalid_request_json")
        return sorted(set(errors))
    expected_labels = [
        {"image_id": image_id, "label": bed.LABELS[label]}
        for image_id, label in task.initial_history
    ]
    if request.get("observed_labels") != expected_labels:
        errors.append("observed_label_mismatch")
    if set(request.get("selectable_image_ids") or []) != set(task.candidate_ids):
        errors.append("candidate_set_mismatch")
    if tuple(request.get("display_order") or []) != case.display_order:
        errors.append("display_order_mismatch")
    allowed = {image_id for image_id, _ in task.initial_history} | set(
        task.candidate_ids
    )
    if set(case.display_order) != allowed or len(case.display_order) != len(allowed):
        errors.append("display_image_set_mismatch")
    image_count = sum(
        item.get("type") == "image_url"
        for message in messages
        for item in message.get("content", [])
        if isinstance(item, dict)
    )
    if image_count != len(allowed):
        errors.append("image_count_mismatch")
    return sorted(set(errors))


def parse_choice(response: str, task: bed.VisualTask) -> str:
    value = bed.strict_json_object(response)
    if set(value) != {"first_image_id"}:
        raise ValueError("naive response must contain only first_image_id")
    first = value["first_image_id"]
    if not isinstance(first, str) or first not in task.candidate_ids:
        raise ValueError("naive first_image_id is not selectable")
    return first


def smoke_cases(tasks: Sequence[bed.VisualTask]) -> list[NaiveCase]:
    selected = [
        seal_unqueried_labels(task)
        for task in sorted(tasks, key=lambda task: task.task_id)
    ]
    if len(selected) != 4:
        raise ValueError("naive smoke requires exactly four mechanics tasks")
    cases = []
    for index in range(SMOKE_REQUESTS):
        task = selected[index % len(selected)]
        seed = SMOKE_BASE_SEED + index
        cases.append(make_case(task, seed=seed, suffix=f"smoke-{index:02d}"))
    return cases


def block_cases(tasks: Sequence[bed.VisualTask], block_id: str) -> list[NaiveCase]:
    if block_id not in development.BLOCK_ORDER:
        raise ValueError("unknown development block")
    block = development.development_tasks_for_block(tasks, block_id)
    base = BLOCK_BASE_SEEDS[block_id]
    return [
        make_case(task, seed=base + index, suffix=f"block-{block_id}")
        for index, task in enumerate(block)
    ]


def _adapter(*, output_dir: Path, run_id: str) -> LunaReasoningAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=500.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return LunaReasoningAdapter(
        ModelSpec(
            model=MODEL_ID,
            backend="openrouter",
            max_model_len=1_000_000,
            reasoning_effort=REASONING_EFFORT,
        ),
        config,
        request_seed=SMOKE_BASE_SEED,
    )


def _run_cases(
    *,
    output_dir: Path,
    run_id: str,
    cases: Sequence[NaiveCase],
    stage: str,
    adapter: StructuredModel | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    messages = [build_messages(case) for case in cases]
    errors = [
        prompt_errors(case, message)
        for case, message in zip(cases, messages, strict=True)
    ]
    if any(errors):
        raise ValueError(f"naive prompt privacy audit failed: {errors}")
    model = adapter or _adapter(output_dir=output_dir, run_id=run_id)
    responses = model.chat_complete_seeded_messages_batched_structured(
        messages,
        [case.seed for case in cases],
        temperature=TEMPERATURE,
        response_format=response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    if len(responses) != len(cases):
        raise ValueError("naive adapter returned the wrong response count")
    choices = [
        parse_choice(response, case.task)
        for response, case in zip(responses, cases, strict=True)
    ]
    usage = summarize_usage(model.usage_snapshot())
    expected = len(cases)
    gates = {
        "exact_expected_accepted_requests": usage.get("adapter_requests") == expected,
        "exact_expected_http_attempts": usage.get("http_attempts") == expected,
        "zero_retries": usage.get("retry_count") == 0,
        "zero_provider_error_retries": usage.get("provider_error_retries", 0) == 0,
        "positive_reasoning_tokens": usage.get("adapter_reasoning_tokens", 0) > 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "zero_forced_final_requests": usage.get("forced_final_requests", 0) == 0,
        "all_strict_choices_are_selectable": len(choices) == expected,
        "all_prompts_hide_source_and_endpoint": not any(errors),
        "candidate_and_endpoint_labels_remain_unaccessed": True,
        "cost_at_most_0_20": float(usage.get("run_cost_usd", math.inf))
        <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    raw_path = output_dir / "private/RAW_RESPONSES.json"
    checkpoint(
        raw_path,
        {
            "stage": stage,
            "case_ids": [case.case_id for case in cases],
            "task_ids": [case.task.task_id for case in cases],
            "seeds": [case.seed for case in cases],
            "display_orders": [list(case.display_order) for case in cases],
            "message_sha256": [message_sha256(message) for message in messages],
            "responses": list(responses),
            "candidate_labels_accessed": False,
            "endpoint_labels_accessed": False,
        },
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gated_null",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": PREREGISTRATION,
            "stage": stage,
            "model": MODEL_ID,
            "reasoning_effort": REASONING_EFFORT,
            "max_tokens": MAX_TOKENS,
            "expected_requests": expected,
            "run_budget_usd": RUN_BUDGET_USD,
            "main_development_manifest_sha256": MAIN_MANIFEST_SHA256,
            "candidate_labels_accessed": False,
            "endpoint_labels_accessed": False,
        },
        "usage": usage,
        "gates": gates,
        "choices": [
            {
                "case_id": case.case_id,
                "task_id": case.task.task_id,
                "seed": case.seed,
                "first_image_id": first,
                "request_sha256": message_sha256(message),
            }
            for case, first, message in zip(cases, choices, messages, strict=True)
        ],
        "raw_responses_sha256": sha256_file(raw_path),
    }


def run_smoke(
    *,
    output_dir: Path,
    run_id: str,
    tasks: Sequence[bed.VisualTask] | None = None,
    adapter: StructuredModel | None = None,
) -> dict[str, Any]:
    verify_frozen_protocol()
    result = _run_cases(
        output_dir=output_dir,
        run_id=run_id,
        cases=smoke_cases(tasks or bed.load_mechanics_tasks()),
        stage="smoke",
        adapter=adapter,
    )
    checkpoint(output_dir / "RESULT.json", result)
    return result


def verify_smoke_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    native_v2 = protocol.get("interface_version") == INTERFACE_VERSION
    legacy_replay = None
    if not native_v2:
        from scripts import bongard_openworld_luna_naive_smoke_migration as migration

        if (
            path.resolve() == migration.BANKED_SMOKE_RESULT.resolve()
            and sha256_file(path) == migration.BANKED_RESULT_SHA256
            and protocol.get("interface_version") == migration.OLD_INTERFACE_VERSION
        ):
            legacy_replay = migration.verify_certificate()
            raw_path = migration.PUBLIC_REPLAY_PAYLOAD
    if (
        result.get("status") != "passed"
        or (not native_v2 and legacy_replay is None)
        or protocol.get("stage") != "smoke"
        or protocol.get("model") != MODEL_ID
        or protocol.get("reasoning_effort") != REASONING_EFFORT
        or result.get("raw_responses_sha256") != sha256_file(raw_path)
        or not result.get("gates")
        or not all(result["gates"].values())
    ):
        raise ValueError("naive smoke is not an exact clean pass")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    cases = smoke_cases(bed.load_mechanics_tasks())
    messages = [build_messages(case) for case in cases]
    responses = raw.get("responses") or []
    if (
        raw.get("case_ids") != [case.case_id for case in cases]
        or raw.get("seeds") != [case.seed for case in cases]
        or raw.get("display_orders") != [list(case.display_order) for case in cases]
        or raw.get("message_sha256")
        != [message_sha256(message) for message in messages]
        or raw.get("candidate_labels_accessed") is not False
        or raw.get("endpoint_labels_accessed") is not False
        or len(responses) != SMOKE_REQUESTS
    ):
        raise ValueError("naive smoke raw request manifest changed")
    replayed = [
        parse_choice(response, case.task)
        for response, case in zip(responses, cases, strict=True)
    ]
    if replayed != [row["first_image_id"] for row in result.get("choices") or []]:
        raise ValueError("naive smoke choices do not replay")
    return {
        "verified": True,
        "result_sha256": sha256_file(path),
        "raw_responses_sha256": sha256_file(raw_path),
        "legacy_v1_replayed_under_v2": legacy_replay is not None,
        "migration_certificate_sha256": (
            legacy_replay["certificate_sha256"]
            if legacy_replay is not None
            else None
        ),
    }


def run_block(
    *,
    output_dir: Path,
    run_id: str,
    block_id: str,
    smoke_result: Path,
    tasks: Sequence[bed.VisualTask] | None = None,
    adapter: StructuredModel | None = None,
) -> dict[str, Any]:
    verify_frozen_protocol()
    smoke = verify_smoke_result(smoke_result)
    manifest = development.verify_protocol_manifest(MAIN_MANIFEST)
    if manifest["manifest_sha256"] != MAIN_MANIFEST_SHA256:
        raise ValueError("main development manifest changed")
    source_tasks = sorted(
        list(tasks)
        if tasks is not None
        else bed.load_validation_partition_tasks(
            "development", include_endpoint_labels=False
        ),
        key=lambda task: task.task_id,
    )
    sealed = [seal_unqueried_labels(task) for task in source_tasks]
    cases = block_cases(sealed, block_id)
    result = _run_cases(
        output_dir=output_dir,
        run_id=run_id,
        cases=cases,
        stage=f"block-{block_id}",
        adapter=adapter,
    )
    result["protocol"].update(
        {
            "block_id": block_id,
            "smoke_result_sha256": smoke["result_sha256"],
            "main_development_manifest_sha256": manifest["manifest_sha256"],
        }
    )
    checkpoint(output_dir / "RESULT.json", result)
    return result


def verify_block_result(
    path: Path,
    *,
    smoke_result: Path,
    tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    block_id = protocol.get("block_id")
    smoke = verify_smoke_result(smoke_result)
    if (
        result.get("status") != "passed"
        or protocol.get("interface_version") != INTERFACE_VERSION
        or block_id not in development.BLOCK_ORDER
        or protocol.get("stage") != f"block-{block_id}"
        or protocol.get("smoke_result_sha256") != smoke["result_sha256"]
        or protocol.get("main_development_manifest_sha256") != MAIN_MANIFEST_SHA256
        or not result.get("gates")
        or not all(result["gates"].values())
    ):
        raise ValueError("naive development block is not a clean pass")
    source_tasks = sorted(
        list(tasks)
        if tasks is not None
        else bed.load_validation_partition_tasks(
            "development", include_endpoint_labels=False
        ),
        key=lambda task: task.task_id,
    )
    sealed = [seal_unqueried_labels(task) for task in source_tasks]
    cases = block_cases(sealed, block_id)
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    if sha256_file(raw_path) != result.get("raw_responses_sha256"):
        raise ValueError("naive block raw hash changed")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    messages = [build_messages(case) for case in cases]
    responses = raw.get("responses") or []
    if (
        raw.get("case_ids") != [case.case_id for case in cases]
        or raw.get("seeds") != [case.seed for case in cases]
        or raw.get("display_orders") != [list(case.display_order) for case in cases]
        or raw.get("message_sha256")
        != [message_sha256(message) for message in messages]
        or raw.get("candidate_labels_accessed") is not False
        or raw.get("endpoint_labels_accessed") is not False
        or len(responses) != len(cases)
    ):
        raise ValueError("naive block raw request manifest changed")
    choices = [
        parse_choice(response, case.task)
        for response, case in zip(responses, cases, strict=True)
    ]
    if choices != [row["first_image_id"] for row in result.get("choices") or []]:
        raise ValueError("naive block choices do not replay")
    return {
        "verified": True,
        "block_id": block_id,
        "result_sha256": sha256_file(path),
        "raw_responses_sha256": sha256_file(raw_path),
        "choices": {
            case.task.task_id: first
            for case, first in zip(cases, choices, strict=True)
        },
    }


def analyze(
    *,
    block_results: Sequence[Path],
    smoke_result: Path,
    main_combined_result: Path,
    output_path: Path,
    tasks: Sequence[bed.VisualTask] | None = None,
    main_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verify_frozen_protocol()
    verifications = [
        verify_block_result(path, smoke_result=smoke_result, tasks=tasks)
        for path in block_results
    ]
    by_block = {row["block_id"]: row for row in verifications}
    if set(by_block) != set(development.BLOCK_ORDER) or len(verifications) != 4:
        raise ValueError("naive analysis requires all four unique blocks")
    choice_by_task = {
        task_id: first
        for block_id in development.BLOCK_ORDER
        for task_id, first in by_block[block_id]["choices"].items()
    }
    if len(choice_by_task) != development.TASKS:
        raise ValueError("naive choices do not cover exactly 64 tasks")
    if main_verification is None:
        from scripts import (
            bongard_openworld_luna_development32_daily_execute as main_daily,
        )

        main_verification = main_daily.verify_combined_result(
            result_path=main_combined_result,
            block_results=[
                main_daily.BLOCK_DIRS[block_id] / "RESULT.json"
                for block_id in development.BLOCK_ORDER
            ],
        )
    if (
        main_verification.get("verified") is not True
        or main_verification.get("result_sha256")
        != sha256_file(main_combined_result)
    ):
        raise ValueError("main combined development result did not verify")
    main = json.loads(main_combined_result.read_text(encoding="utf-8"))
    main_protocol = main.get("protocol") or {}
    trees = main.get("trees") or []
    if (
        main_protocol.get("interface_version") != development.INTERFACE_VERSION
        or main_protocol.get("task_count") != development.TASKS
        or len(trees) != development.TASKS
        or len({tree.get("task_id") for tree in trees}) != development.TASKS
        or main_protocol.get("endpoint_labels_accessed_only_after_all_blocks_replayed")
        is not True
    ):
        raise ValueError("main combined development result is invalid")
    rows = []
    for tree in sorted(trees, key=lambda row: row["task_id"]):
        task_id = tree["task_id"]
        first = choice_by_task[task_id]
        action = (tree.get("all_first_action_paths") or {}).get(first)
        if not action or "endpoint" not in action:
            raise ValueError("naive first action is absent from main terminal cache")
        dynamic = tree["policies"]["dynamic_depth2"]
        myopic = tree["policies"]["myopic_width"]
        rows.append(
            {
                "task_id": task_id,
                "naive_first_image_id": first,
                "naive_second_image_id": action["second_image_id"],
                "naive_final_history_key": action["final_history_key"],
                "naive_endpoint": action["endpoint"],
                "dynamic_first_image_id": dynamic["first_image_id"],
                "dynamic_final_history_key": dynamic["final_history_key"],
                "dynamic_endpoint": dynamic["endpoint"],
                "myopic_first_image_id": myopic["first_image_id"],
                "myopic_final_history_key": myopic["final_history_key"],
                "myopic_endpoint": myopic["endpoint"],
            }
        )
    dynamic_minus_naive = {
        metric: development.paired_summary(
            [
                row["dynamic_endpoint"][metric]
                - row["naive_endpoint"][metric]
                for row in rows
            ],
            seed=BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(("mean_brier", "mean_log_loss"))
    }
    myopic_minus_naive = {
        metric: development.paired_summary(
            [
                row["myopic_endpoint"][metric]
                - row["naive_endpoint"][metric]
                for row in rows
            ],
            seed=BOOTSTRAP_SEED + 10 + index,
        )
        for index, metric in enumerate(("mean_brier", "mean_log_loss"))
    }
    naive_brier = statistics.fmean(
        row["naive_endpoint"]["mean_brier"] for row in rows
    )
    dynamic_brier = statistics.fmean(
        row["dynamic_endpoint"]["mean_brier"] for row in rows
    )
    relative_gain = (
        (naive_brier - dynamic_brier) / naive_brier
        if naive_brier > 0
        else -math.inf
    )
    changed = sum(
        row["dynamic_final_history_key"] != row["naive_final_history_key"]
        for row in rows
    )
    gates = {
        "all_four_endpoint_blind_choice_blocks_replay": all(
            row["verified"] for row in verifications
        ),
        "exact_64_unique_task_choices": len(rows) == development.TASKS,
        "at_least_16_dynamic_naive_final_histories_differ": changed
        >= MIN_CHANGED_FINAL_HISTORIES,
        "dynamic_brier_relative_improvement_vs_naive_at_least_3_percent": (
            relative_gain >= MIN_RELATIVE_BRIER_IMPROVEMENT
        ),
        "dynamic_brier_vs_naive_bootstrap_probability_at_least_0_80": (
            dynamic_minus_naive["mean_brier"][
                "bootstrap_probability_improvement"
            ]
            >= MIN_BOOTSTRAP_IMPROVEMENT_PROBABILITY
        ),
        "dynamic_log_loss_is_not_worse_than_naive": (
            dynamic_minus_naive["mean_log_loss"]["mean_difference"] <= 0
        ),
        "all_joined_endpoint_metrics_are_finite": all(
            math.isfinite(float(endpoint[metric]))
            for row in rows
            for endpoint in (
                row["naive_endpoint"],
                row["dynamic_endpoint"],
                row["myopic_endpoint"],
            )
            for metric in ("mean_brier", "mean_log_loss", "accuracy")
        ),
        "main_result_status_and_confirmation_authority_unchanged": True,
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "dynamic_beats_naive_thinking"
            if gates["all_pass"]
            else "naive_thinking_comparison_null"
        ),
        "authorizes_main_confirmation": False,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": PREREGISTRATION,
            "model": MODEL_ID,
            "reasoning_effort": REASONING_EFFORT,
            "task_count": development.TASKS,
            "bootstrap_replicates": development.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "main_result_sha256": sha256_file(main_combined_result),
            "endpoint_access_occurs_only_through_main_combined_result": True,
        },
        "block_verification": {
            row["block_id"]: {
                "result_sha256": row["result_sha256"],
                "raw_responses_sha256": row["raw_responses_sha256"],
            }
            for row in verifications
        },
        "main_result_verification": dict(main_verification),
        "dynamic_minus_naive": dynamic_minus_naive,
        "myopic_minus_naive": myopic_minus_naive,
        "dynamic_vs_naive_relative_brier_improvement": relative_gain,
        "dynamic_vs_naive_changed_first_actions": sum(
            row["dynamic_first_image_id"] != row["naive_first_image_id"]
            for row in rows
        ),
        "dynamic_vs_naive_changed_final_histories": changed,
        "gates": gates,
        "rows": rows,
    }
    checkpoint(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--output-dir", type=Path, required=True)
    smoke_parser.add_argument("--run-id", required=True)
    block_parser = subparsers.add_parser("block")
    block_parser.add_argument("--output-dir", type=Path, required=True)
    block_parser.add_argument("--run-id", required=True)
    block_parser.add_argument(
        "--block-id", choices=development.BLOCK_ORDER, required=True
    )
    block_parser.add_argument("--smoke-result", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument(
        "--block-result", type=Path, action="append", required=True
    )
    analyze_parser.add_argument("--smoke-result", type=Path, required=True)
    analyze_parser.add_argument("--main-combined-result", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke(output_dir=args.output_dir, run_id=args.run_id)
    elif args.command == "block":
        result = run_block(
            output_dir=args.output_dir,
            run_id=args.run_id,
            block_id=args.block_id,
            smoke_result=args.smoke_result,
        )
    else:
        result = analyze(
            block_results=args.block_result,
            smoke_result=args.smoke_result,
            main_combined_result=args.main_combined_result,
            output_path=args.output,
        )
    print(json.dumps({key: result[key] for key in ("status", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
