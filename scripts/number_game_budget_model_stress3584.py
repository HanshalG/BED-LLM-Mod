#!/usr/bin/env python3
"""Run the frozen 3,584-case Number Game budget-model stress gate."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_budget_model_reliability128 as reliability
from scripts.number_game_depth_three_development import (
    MAX_TOKENS,
    TEMPERATURE,
    history_messages,
    initial_messages,
    proposal_response_format,
)
from scripts.number_game_predictive_risk_replication import _adapter
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-budget-model-stress3584-1"
CASE_SELECTION_SEED = 1_081_200
CASE_MANIFEST_SHA256 = (
    "017ef5554906954e8514be487de4cb1924bbf19f295d9977953e486891bb71cd"
)
EXPECTED_INITIAL_REQUESTS = 3_584
INITIAL_CASES = 16
UNSEEN_ONE_CASES = 92
UNSEEN_TWO_CASES = 598
UNSEEN_CONDITIONED_CASES = UNSEEN_ONE_CASES + UNSEEN_TWO_CASES
REPEATED_CONDITIONED_CASES = 2_878
CONDITIONED_CASES = UNSEEN_CONDITIONED_CASES + REPEATED_CONDITIONED_CASES
SEED_GROUPS = 16
CONCURRENCY = 64
ADAPTER_CONCURRENCY = CONCURRENCY // SEED_GROUPS
MAX_FORMAT_RETRIES = 8
MAX_TRANSPORT_RETRIES = 16
MAX_FORCED_EXITS = 8
MIN_INITIAL_VALID = 16
MIN_CONDITIONED_VALID = 4
MIN_MEAN_CONDITIONED_VALID = 8.0
RUN_BUDGET_USD = 1.55
PROJECTED_COST_USD = 1.55
SELECTION_RULE = (
    "pass_then_conditioned_min_mean_then_failure_tail_then_cost"
)
MODEL_SEEDS = {
    "openai/gpt-5.6-luna": tuple(range(1_081_001, 1_081_017)),
    "deepseek/deepseek-v4-flash-0731": tuple(
        range(1_081_101, 1_081_117)
    ),
}
MODEL_TIE_PRIORITY = {
    "openai/gpt-5.6-luna": 0,
    "deepseek/deepseek-v4-flash-0731": 1,
}
SOURCE_TREES = reliability.SOURCE_TREES
SOURCE_TREES_SHA256 = reliability.SOURCE_TREES_SHA256
PREREGISTRATION = REPO_ROOT / (
    "results/nonmyopic/"
    "NUMBER_GAME_BUDGET_MODEL_STRESS3584_PREREGISTRATION.md"
)
PREREGISTRATION_SHA256 = (
    "9be58161b48d7a355d93bc26481cf446ba81d2ef4c7adc2b23e1a6f949d42e51"
)


class StructuredModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _history_key(history: Sequence[tuple[int, bool]]) -> str:
    encoded = ";".join(
        f"{number}:{int(label)}" for number, label in history
    )
    return hashlib.sha256(
        f"{CASE_SELECTION_SEED}|{encoded}".encode("ascii")
    ).hexdigest()


def _all_source_histories(
    source_path: Path,
) -> set[tuple[tuple[int, bool], ...]]:
    if reliability.sha256_file(source_path) != SOURCE_TREES_SHA256:
        raise ValueError("fully fresh source TREES hash changed")
    trees = _load(source_path).get("trees") or []
    one = {
        reliability._parse_first_history(key)
        for tree in trees
        for key in (tree.get("first_branches") or {})
    }
    two = {
        reliability._parse_second_history(key)
        for tree in trees
        for key in (tree.get("second_branches") or {})
    }
    return one | two


def build_stress_cases(
    source_path: Path = SOURCE_TREES,
) -> list[dict[str, Any]]:
    gate_histories = {
        case["observations"]
        for case in reliability.build_cases(source_path)
        if case["observations"]
    }
    unseen = sorted(
        _all_source_histories(source_path) - gate_histories,
        key=_history_key,
    )
    unseen_one = sum(len(history) == 1 for history in unseen)
    unseen_two = sum(len(history) == 2 for history in unseen)
    if (unseen_one, unseen_two) != (UNSEEN_ONE_CASES, UNSEEN_TWO_CASES):
        raise ValueError("unseen history inventory changed")

    records: list[dict[str, Any]] = []
    for seed_group in range(SEED_GROUPS):
        records.append(
            {
                "seed_group": seed_group,
                "observations": tuple(),
                "history_role": "initial",
                "history_index": None,
                "replicate": 0,
            }
        )
    for index, history in enumerate(unseen):
        records.append(
            {
                "seed_group": index % SEED_GROUPS,
                "observations": history,
                "history_role": "unseen_conditioned",
                "history_index": index,
                "replicate": 0,
            }
        )
    unique_group_counts = [
        sum(index % SEED_GROUPS == group for index in range(len(unseen)))
        for group in range(SEED_GROUPS)
    ]
    repeat_quotas = [223 - count for count in unique_group_counts]
    used_groups = {
        index: {index % SEED_GROUPS} for index in range(len(unseen))
    }
    for repeat_index in range(REPEATED_CONDITIONED_CASES):
        index = repeat_index % len(unseen)
        history = unseen[index]
        replicate = repeat_index // len(unseen) + 1
        start = (index + replicate + 1) % SEED_GROUPS
        group = next(
            (
                candidate
                for offset in range(SEED_GROUPS)
                if (
                    (candidate := (start + offset) % SEED_GROUPS)
                    not in used_groups[index]
                    and repeat_quotas[candidate] > 0
                )
            ),
            None,
        )
        if group is None:
            raise AssertionError("could not balance repeated stress histories")
        repeat_quotas[group] -= 1
        used_groups[index].add(group)
        records.append(
            {
                "seed_group": group,
                "observations": history,
                "history_role": "unseen_conditioned_repeat",
                "history_index": index,
                "replicate": replicate,
            }
        )
    if any(repeat_quotas):
        raise AssertionError("stress repeat quotas were not exhausted")
    for case_index, record in enumerate(records):
        observations = record["observations"]
        record["case_index"] = case_index
        record["messages"] = (
            initial_messages()
            if not observations
            else history_messages(observations, enforce_constraints=True)
        )
    if len(records) != EXPECTED_INITIAL_REQUESTS:
        raise AssertionError("stress case count changed")
    if any(
        sum(record["seed_group"] == group for record in records) != 224
        for group in range(SEED_GROUPS)
    ):
        raise AssertionError("stress seed groups are not balanced")
    for index in range(len(unseen)):
        groups = [
            record["seed_group"]
            for record in records
            if record["history_index"] == index
        ]
        if len(groups) != len(set(groups)):
            raise AssertionError("repeated history reused a model seed group")
    if case_manifest_sha256(records) != CASE_MANIFEST_SHA256:
        raise AssertionError("stress case manifest changed")
    return records


def case_manifest_sha256(cases: Sequence[dict[str, Any]]) -> str:
    public = [
        {
            key: case[key]
            for key in (
                "case_index",
                "seed_group",
                "observations",
                "history_role",
                "history_index",
                "replicate",
            )
        }
        for case in cases
    ]
    encoded = json.dumps(
        public,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reliability_metrics(result: dict[str, Any]) -> dict[str, Any]:
    usage = result.get("usage") or {}
    conditioned = result.get("conditioned_valid_summary") or {}
    return {
        "eligible": (
            result.get("status") == "passed"
            and (result.get("gates") or {}).get("all_pass") is True
        ),
        "conditioned_minimum": int(conditioned.get("minimum", -1)),
        "conditioned_mean": float(conditioned.get("mean", -1.0)),
        "initial_parse_failures": int(
            result.get("initial_parse_failures", 10**9)
        ),
        "format_retry_requests": int(
            result.get("format_retry_requests", 10**9)
        ),
        "forced_exits": int(usage.get("forced_exits", 10**9)),
        "transport_retries": int(usage.get("retry_count", 10**9)),
        "run_cost_usd": float(usage.get("run_cost_usd", 1_000_000_000.0)),
    }


def _selection_key(model: str, metrics: dict[str, Any]) -> tuple[Any, ...]:
    return (
        metrics["conditioned_minimum"],
        metrics["conditioned_mean"],
        -metrics["initial_parse_failures"],
        -metrics["format_retry_requests"],
        -metrics["forced_exits"],
        -metrics["transport_retries"],
        -metrics["run_cost_usd"],
        MODEL_TIE_PRIORITY[model],
    )


def select_model(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(results) != set(MODEL_SEEDS):
        raise ValueError("selection requires both frozen model results")
    rows = []
    for model in MODEL_SEEDS:
        result = results[model]
        protocol = result.get("protocol") or {}
        interface_version = (
            protocol.get("interface_version")
            or result.get("interface_version")
        )
        result_model = protocol.get("model") or result.get("model")
        if interface_version != reliability.INTERFACE_VERSION:
            raise ValueError(f"{model} reliability interface changed")
        if result_model != model:
            raise ValueError(f"{model} reliability result model mismatch")
        metrics = _reliability_metrics(result)
        rows.append(
            {
                "model": model,
                "metrics": metrics,
                "selection_key": list(_selection_key(model, metrics)),
            }
        )
    eligible = [row for row in rows if row["metrics"]["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda row: _selection_key(row["model"], row["metrics"]),
        )["model"]
        if eligible
        else None
    )
    return {
        "selection_rule": SELECTION_RULE,
        "selected_model": selected,
        "models": rows,
    }


def validate_tail_authorization(ledger: dict[str, Any]) -> dict[str, Any]:
    if ledger.get("additional_paid_blocks_authorized") is not True:
        raise RuntimeError("daily ledger does not authorize the stress tail")
    gate_entries = [
        item
        for item in (ledger.get("authorized_tail_blocks") or [])
        if item.get("interface") == reliability.INTERFACE_VERSION
    ]
    if len(gate_entries) != len(MODEL_SEEDS):
        raise RuntimeError("daily ledger reliability authorizations changed")
    if any(
        item.get("model") not in MODEL_SEEDS
        or item.get("status") == "authorized_pending"
        for item in gate_entries
    ):
        raise RuntimeError("both reliability gates must be banked first")
    stress = [
        item
        for item in (ledger.get("authorized_tail_blocks") or [])
        if item.get("interface") == INTERFACE_VERSION
    ]
    if len(stress) != 1:
        raise RuntimeError("daily ledger stress authorization changed")
    authorization = stress[0]
    if (
        authorization.get("status")
        != "waiting_for_reliability_results"
        or authorization.get("model") is not None
        or authorization.get("selection_rule") != SELECTION_RULE
        or float(authorization.get("maximum_cost_usd", -1.0))
        != RUN_BUDGET_USD
    ):
        raise RuntimeError("daily ledger stress authorization is not exact")
    return authorization


def stress_gates(
    *,
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    format_retry_requests: int,
    initial_parse_failures: int,
) -> dict[str, bool]:
    initial = [record for record in records if not record["observations"]]
    conditioned = [record for record in records if record["observations"]]
    expected_accepted = EXPECTED_INITIAL_REQUESTS + format_retry_requests
    conditioned_mean = (
        sum(record["valid_unique_count"] for record in conditioned)
        / len(conditioned)
        if conditioned
        else 0.0
    )
    gates = {
        "exactly_3584_final_case_records": (
            len(records) == EXPECTED_INITIAL_REQUESTS
        ),
        "at_most_eight_initial_parse_failures": (
            initial_parse_failures <= MAX_FORMAT_RETRIES
        ),
        "all_parse_failures_retried_once": (
            format_retry_requests == initial_parse_failures
        ),
        "all_final_responses_strictly_parsed": all(
            record["final_parse_error"] is None for record in records
        ),
        "accepted_request_accounting_exact": (
            usage["adapter_requests"] == expected_accepted
        ),
        "http_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "transport_retries_within_cap": (
            usage["retry_count"] <= MAX_TRANSPORT_RETRIES
        ),
        "zero_provider_error_retries": (
            usage["provider_error_retries"] == 0
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "forced_exits_within_cap": usage["forced_exits"] <= MAX_FORCED_EXITS,
        "every_parsed_response_has_24_schema_items": all(
            record["diagnostics"] is not None
            and record["diagnostics"].get("raw_count") == 24
            for record in records
        ),
        "all_initial_supports_have_at_least_16_valid": (
            len(initial) == INITIAL_CASES
            and all(
                record["valid_unique_count"] >= MIN_INITIAL_VALID
                for record in initial
            )
        ),
        "all_conditioned_supports_have_at_least_4_valid": (
            len(conditioned) == CONDITIONED_CASES
            and all(
                record["valid_unique_count"] >= MIN_CONDITIONED_VALID
                for record in conditioned
            )
        ),
        "mean_conditioned_support_at_least_8_valid": (
            conditioned_mean >= MIN_MEAN_CONDITIONED_VALID
        ),
        "within_stress_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _repeat_descriptives(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if record["history_index"] is not None:
            by_index.setdefault(int(record["history_index"]), []).append(record)
    similarities = []
    for index in range(UNSEEN_CONDITIONED_CASES):
        draws = by_index[index]
        if len(draws) < 2:
            raise ValueError("repeated-history draws are incomplete")
        for left_index, left_record in enumerate(draws):
            left = set(left_record["extension_sha256s"])
            for right_record in draws[left_index + 1 :]:
                right = set(right_record["extension_sha256s"])
                union = left | right
                similarities.append(
                    len(left & right) / len(union) if union else 1.0
                )
    return {
        "history_count": len(by_index),
        "pair_count": len(similarities),
        "mean_extension_jaccard_similarity": sum(similarities)
        / len(similarities),
        "minimum_extension_jaccard_similarity": min(similarities),
        "maximum_extension_jaccard_similarity": max(similarities),
        "similarity_is_descriptive_not_gated": True,
    }


def run_stress(
    *,
    output_dir: Path,
    run_id: str,
    selected_model: str,
    selection: dict[str, Any],
    reliability_artifacts: dict[str, str],
    adapters: Sequence[StructuredModel] | None = None,
    source_path: Path = SOURCE_TREES,
) -> dict[str, Any]:
    if selected_model not in MODEL_SEEDS:
        raise ValueError(f"unsupported stress model: {selected_model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    adapters = list(adapters or [])
    if not adapters:
        adapters = [
            _adapter(
                model=selected_model,
                run_id=run_id,
                output_dir=output_dir,
                request_seed=request_seed,
                concurrency=ADAPTER_CONCURRENCY,
                projected_cost=PROJECTED_COST_USD,
                run_budget_usd=RUN_BUDGET_USD,
            )
            for request_seed in MODEL_SEEDS[selected_model]
        ]
    if len(adapters) != SEED_GROUPS:
        raise ValueError("stress gate requires exactly sixteen adapters")
    cases = build_stress_cases(source_path)
    grouped_cases = [
        [case for case in cases if case["seed_group"] == seed_group]
        for seed_group in range(SEED_GROUPS)
    ]
    initial_by_index = reliability._call_seed_groups(
        adapters=adapters,
        grouped_cases=grouped_cases,
    )
    if len(initial_by_index) != EXPECTED_INITIAL_REQUESTS:
        raise ValueError("adapter did not return exactly 3584 initial responses")
    initial_responses = [
        initial_by_index[index] for index in range(EXPECTED_INITIAL_REQUESTS)
    ]

    parsed = []
    failed_indices = []
    for case, response in zip(cases, initial_responses, strict=True):
        support, diagnostic, error = reliability._safe_parse(
            response,
            observations=case["observations"],
        )
        parsed.append([support, diagnostic, error])
        if error is not None:
            failed_indices.append(case["case_index"])

    retry_responses: list[dict[str, Any]] = []
    if 0 < len(failed_indices) <= MAX_FORMAT_RETRIES:
        retry_groups = [
            [
                cases[index]
                for index in failed_indices
                if cases[index]["seed_group"] == seed_group
            ]
            for seed_group in range(SEED_GROUPS)
        ]
        retry_by_index = reliability._call_seed_groups(
            adapters=adapters,
            grouped_cases=retry_groups,
        )
        if len(retry_by_index) != len(failed_indices):
            raise ValueError("adapter omitted a format-retry response")
        for index in failed_indices:
            response = retry_by_index[index]
            retry_responses.append(
                {"case_index": index, "response": response}
            )
            parsed[index] = list(
                reliability._safe_parse(
                    response,
                    observations=cases[index]["observations"],
                )
            )
    checkpoint(
        raw_path,
        {
            "initial_responses": initial_responses,
            "retry_responses": retry_responses,
        },
    )

    retry_index_set = set(failed_indices if retry_responses else [])
    records = []
    for case, (support, diagnostic, final_error) in zip(
        cases, parsed, strict=True
    ):
        records.append(
            {
                "case_index": case["case_index"],
                "seed_group": case["seed_group"],
                "observations": [
                    [number, label] for number, label in case["observations"]
                ],
                "history_role": case["history_role"],
                "history_index": case["history_index"],
                "replicate": case["replicate"],
                "initial_parse_failed": (
                    case["case_index"] in failed_indices
                ),
                "format_retried": (
                    case["case_index"] in retry_index_set
                ),
                "final_parse_error": final_error,
                "valid_unique_count": len(support or []),
                "diagnostics": diagnostic,
                "extension_sha256s": [
                    hypothesis.public_dict()["extension_sha256"]
                    for hypothesis in (support or [])
                ],
            }
        )
    usage = reliability.aggregate_usage(adapters)
    gates = stress_gates(
        records=records,
        usage=usage,
        format_retry_requests=len(retry_responses),
        initial_parse_failures=len(failed_indices),
    )
    conditioned_counts = [
        record["valid_unique_count"]
        for record in records
        if record["observations"]
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gated_null",
        "decision": (
            "eligible_for_policy_scale_budget_model_work"
            if gates["all_pass"]
            else "close_selected_budget_model_at_scale"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "selected_model": selected_model,
            "model_seeds": list(MODEL_SEEDS[selected_model]),
            "case_selection_seed": CASE_SELECTION_SEED,
            "case_manifest_sha256": CASE_MANIFEST_SHA256,
            "source_trees_sha256": SOURCE_TREES_SHA256,
            "temperature": TEMPERATURE,
            "reasoning": False,
            "max_output_tokens": MAX_TOKENS,
            "seed_groups": SEED_GROUPS,
            "concurrency_per_seed_group": ADAPTER_CONCURRENCY,
            "aggregate_concurrency": CONCURRENCY,
            "initial_requests": EXPECTED_INITIAL_REQUESTS,
            "case_mix": {
                "initial": INITIAL_CASES,
                "unseen_one_observation": UNSEEN_ONE_CASES,
                "unseen_two_observation": UNSEEN_TWO_CASES,
                "repeated_conditioned": REPEATED_CONDITIONED_CASES,
            },
            "format_retry_policy": (
                "one_same_prompt_same_temperature_retry_for_strict_parse_"
                "failure_only_if_at_most_eight_failures"
            ),
            "run_budget_usd": RUN_BUDGET_USD,
            "efficacy_used_for_authorization": False,
        },
        "selection": selection,
        "reliability_artifacts": reliability_artifacts,
        "usage": usage,
        "initial_parse_failures": len(failed_indices),
        "format_retry_requests": len(retry_responses),
        "conditioned_valid_summary": {
            "minimum": min(conditioned_counts),
            "mean": sum(conditioned_counts) / len(conditioned_counts),
            "maximum": max(conditioned_counts),
        },
        "repeated_history_descriptives": _repeat_descriptives(records),
        "gates": gates,
        "cases": records,
        "raw_responses_sha256": reliability.sha256_file(raw_path),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def reconcile_daily_ledger(
    *,
    ledger: dict[str, Any],
    selected_model: str | None,
    measured_cost_usd: float,
    live_after: dict[str, float] | None,
    status: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    previous = float(updated.get("recorded_actual_spend_usd", 0.0))
    posted = previous
    if live_after is not None:
        opening = float(updated["opening_total_usage_usd"])
        posted = max(
            0.0,
            float(live_after["total_usage_usd"]) - opening,
        )
    local = previous + measured_cost_usd
    recorded = max(posted, local)
    updated["recorded_actual_spend_usd"] = recorded
    stress = [
        item
        for item in (updated.get("authorized_tail_blocks") or [])
        if item.get("interface") == INTERFACE_VERSION
    ]
    if len(stress) != 1:
        raise RuntimeError("daily ledger stress authorization changed")
    stress[0]["model"] = selected_model
    stress[0]["status"] = status
    stress[0]["actual_cost_usd"] = max(
        float(stress[0].get("actual_cost_usd", 0.0)),
        measured_cost_usd,
    )
    updated["additional_paid_blocks_authorized"] = any(
        item.get("status")
        in {"authorized_pending", "waiting_for_reliability_results"}
        for item in (updated.get("authorized_tail_blocks") or [])
    )
    reconciliation = {
        "posted_spend_since_opening_usd": posted,
        "locally_measured_spend_usd": local,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": max(
            0.0, float(updated["daily_cap_usd"]) - recorded
        ),
    }
    if live_after is not None:
        reconciliation.update(
            {
                "live_total_credits_usd": float(
                    live_after["total_credits_usd"]
                ),
                "live_total_usage_usd": float(
                    live_after["total_usage_usd"]
                ),
                "live_balance_usd": float(live_after["balance_usd"]),
            }
        )
    updated["reconciliation"] = reconciliation
    return updated


def _load_reliability_results(
    paths: dict[str, Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    results = {model: _load(path) for model, path in paths.items()}
    artifacts = {
        model: reliability.sha256_file(path) for model, path in paths.items()
    }
    return results, artifacts


def execute_stress(
    *,
    output_dir: Path,
    run_id: str,
    ledger_path: Path,
    reliability_paths: dict[str, Path],
    live_reader=read_live_credits,
    adapters: Sequence[StructuredModel] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if reliability.sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("stress preregistration hash changed")
    ledger = _load(ledger_path)
    validate_tail_authorization(ledger)
    results, artifacts = _load_reliability_results(reliability_paths)
    selection = select_model(results)
    for item in ledger.get("authorized_tail_blocks") or []:
        model = item.get("model")
        if item.get("interface") == reliability.INTERFACE_VERSION:
            result_status = results[model].get("status")
            allowed_ledger_statuses = (
                {
                    "failed_closed",
                    "failed_closed_posted_spend_reconciled",
                }
                if result_status == "failed_closed"
                else {result_status}
            )
            if item.get("status") not in allowed_ledger_statuses:
                raise RuntimeError(
                    f"ledger and reliability result status differ for {model}"
                )
    selected_model = selection["selected_model"]
    if selected_model is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "gated_null",
            "decision": "no_eligible_model",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "model_calls": 0,
                "run_budget_usd": RUN_BUDGET_USD,
            },
            "selection": selection,
            "reliability_artifacts": artifacts,
            "usage": {"adapter_requests": 0, "run_cost_usd": 0.0},
        }
        checkpoint(output_dir / "RESULT.json", result)
        reconciled = reconcile_daily_ledger(
            ledger=ledger,
            selected_model=None,
            measured_cost_usd=0.0,
            live_after=None,
            status="not_authorized_no_model_passed",
        )
        checkpoint(ledger_path, reconciled)
        return result

    live_before = live_reader()
    require_budget(
        ledger,
        projected_cost_usd=PROJECTED_COST_USD,
        total_usage_usd=float(live_before["total_usage_usd"]),
        now=now,
    )
    if float(live_before["balance_usd"]) + 1e-12 < RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below the stress gate")
    try:
        result = run_stress(
            output_dir=output_dir,
            run_id=run_id,
            selected_model=selected_model,
            selection=selection,
            reliability_artifacts=artifacts,
            adapters=adapters,
        )
    except Exception:
        try:
            live_failure = live_reader()
            reconciled = reconcile_daily_ledger(
                ledger=ledger,
                selected_model=selected_model,
                measured_cost_usd=0.0,
                live_after=live_failure,
                status="failed_closed_posted_spend_reconciled",
            )
            checkpoint(ledger_path, reconciled)
        except Exception:
            pass
        raise
    measured = float(result["usage"]["run_cost_usd"])
    local = reconcile_daily_ledger(
        ledger=ledger,
        selected_model=selected_model,
        measured_cost_usd=measured,
        live_after=live_before,
        status=result["status"],
    )
    checkpoint(ledger_path, local)
    live_after = live_reader()
    reconciled = reconcile_daily_ledger(
        ledger=local,
        selected_model=selected_model,
        measured_cost_usd=0.0,
        live_after=live_after,
        status=result["status"],
    )
    checkpoint(ledger_path, reconciled)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    parser.add_argument("--luna-result", type=Path, required=True)
    parser.add_argument("--deepseek-result", type=Path, required=True)
    args = parser.parse_args()
    reliability_paths = {
        "openai/gpt-5.6-luna": args.luna_result,
        "deepseek/deepseek-v4-flash-0731": args.deepseek_result,
    }
    try:
        result = execute_stress(
            output_dir=args.output_dir,
            run_id=args.run_id,
            ledger_path=args.daily_ledger,
            reliability_paths=reliability_paths,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint(
            args.output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "selected_model": result["selection"]["selected_model"],
                "usage": result["usage"],
                "conditioned_valid_summary": result.get(
                    "conditioned_valid_summary"
                ),
                "gates": result.get("gates"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
