#!/usr/bin/env python3
"""Run the frozen 128-case Number Game budget-model reliability gate."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_depth_three_development import (
    MAX_TOKENS,
    TEMPERATURE,
    history_messages,
    initial_messages,
    parse_proposals,
    proposal_response_format,
)
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.number_game_predictive_risk_replication import _adapter
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-budget-model-reliability128-1"
CASE_SELECTION_SEED = 1_080_800
EXPECTED_INITIAL_REQUESTS = 128
EXPECTED_QWEN_CONTROL_REQUESTS = 3072
INITIAL_CASES = 8
ONE_OBSERVATION_CASES = 40
TWO_OBSERVATION_CASES = 80
MAX_FORMAT_RETRIES = 3
MAX_TRANSPORT_RETRIES = 4
MAX_FORCED_EXITS = 3
MIN_INITIAL_VALID = 16
MIN_CONDITIONED_VALID = 4
MIN_MEAN_CONDITIONED_VALID = 8.0
CONCURRENCY = 64
SEED_GROUPS = 8
ADAPTER_CONCURRENCY = CONCURRENCY // SEED_GROUPS
RUN_BUDGET_USD = 0.10
PROJECTED_COST_USD = 0.10
MODELS = {
    "openai/gpt-5.6-luna": tuple(range(1_080_801, 1_080_809)),
    "deepseek/deepseek-v4-flash-0731": tuple(range(1_080_821, 1_080_829)),
}
SOURCE_TREES = REPO_ROOT / (
    "results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/"
    "number-game-qwen-fully-fresh-daily-stages-20260806T000200Z/"
    "source/TREES.json"
)
SOURCE_TREES_SHA256 = (
    "f812e4a356f5129a6f2f22d5f0f995b76b4ac624a0f312154064ff8c51f2c7b0"
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_completed_qwen_control(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    control = payload.get("control") or {}
    usage = control.get("usage") or {}
    mechanics = control.get("mechanics_gates") or {}
    if usage.get("adapter_requests") != EXPECTED_QWEN_CONTROL_REQUESTS:
        raise ValueError("Qwen control does not contain exactly 3072 requests")
    if not mechanics or not all(mechanics.values()):
        raise ValueError("Qwen control mechanics did not pass")
    if payload.get("decision") != "complete_composite_endpoint":
        raise ValueError("Qwen control composite endpoint is not complete")
    return payload


def validate_tail_authorization(
    ledger: dict[str, Any],
    *,
    model_id: str,
) -> dict[str, Any]:
    if ledger.get("additional_paid_blocks_authorized") is not True:
        raise RuntimeError("daily ledger does not authorize paid tail blocks")
    matches = [
        item
        for item in (ledger.get("authorized_tail_blocks") or [])
        if item.get("model") == model_id
        and item.get("interface") == INTERFACE_VERSION
        and item.get("status") == "authorized_pending"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"daily ledger does not contain one pending authorization for {model_id}"
        )
    authorization = matches[0]
    if float(authorization.get("maximum_cost_usd", -1.0)) != RUN_BUDGET_USD:
        raise RuntimeError("reliability tail cost cap does not match protocol")
    return authorization


def _history_key(history: Sequence[tuple[int, bool]]) -> str:
    encoded = ";".join(
        f"{number}:{int(label)}" for number, label in history
    )
    return hashlib.sha256(
        f"{CASE_SELECTION_SEED}|{encoded}".encode("ascii")
    ).hexdigest()


def _parse_first_history(key: str) -> tuple[tuple[int, bool], ...]:
    number, label = key.split(":")
    return ((int(number), bool(int(label))),)


def _parse_second_history(key: str) -> tuple[tuple[int, bool], ...]:
    first_number, first_label, second_number, second_label = key.split(":")
    return (
        (int(first_number), bool(int(first_label))),
        (int(second_number), bool(int(second_label))),
    )


def build_cases(
    source_path: Path = SOURCE_TREES,
) -> list[dict[str, Any]]:
    if sha256_file(source_path) != SOURCE_TREES_SHA256:
        raise ValueError("fully fresh source TREES hash changed")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    trees = payload.get("trees") or []
    if len(trees) != 32:
        raise ValueError("fully fresh source does not contain exactly 32 trees")

    one_histories = {
        _parse_first_history(key)
        for tree in trees
        for key in (tree.get("first_branches") or {})
    }
    two_histories = {
        _parse_second_history(key)
        for tree in trees
        for key in (tree.get("second_branches") or {})
    }
    if len(one_histories) < ONE_OBSERVATION_CASES:
        raise ValueError("insufficient unique one-observation histories")
    if len(two_histories) < TWO_OBSERVATION_CASES:
        raise ValueError("insufficient unique two-observation histories")

    selected_one = sorted(one_histories, key=_history_key)[
        :ONE_OBSERVATION_CASES
    ]
    selected_two = sorted(two_histories, key=_history_key)[
        :TWO_OBSERVATION_CASES
    ]
    histories = []
    for seed_group in range(SEED_GROUPS):
        histories.extend(
            [
                tuple(),
                *selected_one[seed_group * 5 : (seed_group + 1) * 5],
                *selected_two[seed_group * 10 : (seed_group + 1) * 10],
            ]
        )
    cases = []
    for index, history in enumerate(histories):
        seed_group = index // 16
        cases.append(
            {
                "case_index": index,
                "seed_group": seed_group,
                "observations": history,
                "messages": (
                    initial_messages()
                    if not history
                    else history_messages(history, enforce_constraints=True)
                ),
            }
        )
    if len(cases) != EXPECTED_INITIAL_REQUESTS:
        raise AssertionError("reliability case count changed")
    return cases


def _safe_parse(
    response: str,
    *,
    observations: Sequence[tuple[int, bool]],
) -> tuple[list[Any] | None, dict[str, Any] | None, str | None]:
    try:
        support, diagnostic = parse_proposals(
            response,
            observations=observations,
        )
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    return support, diagnostic, None


def reliability_gates(
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
        "exactly_128_final_case_records": (
            len(records) == EXPECTED_INITIAL_REQUESTS
        ),
        "at_most_three_initial_parse_failures": (
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
        "forced_exits_within_cap": (
            usage["forced_exits"] <= MAX_FORCED_EXITS
        ),
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
            len(conditioned)
            == ONE_OBSERVATION_CASES + TWO_OBSERVATION_CASES
            and all(
                record["valid_unique_count"] >= MIN_CONDITIONED_VALID
                for record in conditioned
            )
        ),
        "mean_conditioned_support_at_least_8_valid": (
            conditioned_mean >= MIN_MEAN_CONDITIONED_VALID
        ),
        "within_model_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def aggregate_usage(adapters: Sequence[StructuredModel]) -> dict[str, Any]:
    snapshots = [
        summarize_usage(adapter.usage_snapshot()) for adapter in adapters
    ]
    return {
        field: sum(snapshot[field] for snapshot in snapshots)
        for field in (
            "adapter_requests",
            "http_attempts",
            "retry_count",
            "provider_error_retries",
            "adapter_reasoning_tokens",
            "forced_exits",
            "run_cost_usd",
            "prompt_tokens",
            "completion_tokens",
        )
    }


def reconcile_daily_ledger(
    *,
    ledger: dict[str, Any],
    model_id: str,
    measured_cost_usd: float,
    live_after: dict[str, float],
    status: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    previous = float(updated.get("recorded_actual_spend_usd", 0.0))
    posted = max(0.0, float(live_after["total_usage_usd"]) - opening)
    local = previous + measured_cost_usd
    recorded = max(posted, local)
    updated["recorded_actual_spend_usd"] = recorded
    model_key = model_id.replace("/", "_").replace(".", "_")
    model_record_key = f"budget_model_reliability128_{model_key}"
    previous_model_cost = float(
        (updated.get(model_record_key) or {}).get("actual_cost_usd", 0.0)
    )
    recorded_model_cost = max(previous_model_cost, measured_cost_usd)
    updated[model_record_key] = {
        "status": status,
        "actual_cost_usd": recorded_model_cost,
        "maximum_cost_usd": RUN_BUDGET_USD,
    }
    for item in updated.get("authorized_tail_blocks") or []:
        if item.get("model") == model_id:
            item["status"] = status
            item["actual_cost_usd"] = max(
                float(item.get("actual_cost_usd") or 0.0),
                measured_cost_usd,
            )
    updated["additional_paid_blocks_authorized"] = any(
        item.get("status")
        in {"authorized_pending", "waiting_for_reliability_results"}
        for item in (updated.get("authorized_tail_blocks") or [])
    )
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live_after["total_credits_usd"]),
        "live_total_usage_usd": float(live_after["total_usage_usd"]),
        "live_balance_usd": float(live_after["balance_usd"]),
        "posted_spend_since_opening_usd": posted,
        "locally_measured_spend_usd": local,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": max(
            0.0, float(updated["daily_cap_usd"]) - recorded
        ),
    }
    return updated


def _call_seed_groups(
    *,
    adapters: Sequence[StructuredModel],
    grouped_cases: Sequence[Sequence[dict[str, Any]]],
) -> dict[int, str]:
    if len(adapters) != len(grouped_cases):
        raise ValueError("adapter and case-group counts do not match")
    responses: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=len(adapters)) as executor:
        futures = {
            executor.submit(
                adapter.chat_complete_messages_batched_structured,
                [case["messages"] for case in cases],
                temperature=TEMPERATURE,
                block_size=len(cases),
                response_format=proposal_response_format(),
                max_new_tokens=MAX_TOKENS,
            ): (group_index, cases)
            for group_index, (adapter, cases) in enumerate(
                zip(adapters, grouped_cases, strict=True)
            )
            if cases
        }
        for future in as_completed(futures):
            group_index, cases = futures[future]
            values = future.result()
            if len(values) != len(cases):
                raise ValueError(
                    f"adapter seed group {group_index} omitted responses"
                )
            for case, response in zip(cases, values, strict=True):
                responses[case["case_index"]] = response
    return responses


def run_reliability(
    *,
    output_dir: Path,
    run_id: str,
    model_id: str,
    adapters: Sequence[StructuredModel] | None = None,
    source_path: Path = SOURCE_TREES,
) -> dict[str, Any]:
    if model_id not in MODELS:
        raise ValueError(f"unsupported reliability model: {model_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    adapters = list(adapters or [])
    if not adapters:
        adapters = [
            _adapter(
                model=model_id,
                run_id=run_id,
                output_dir=output_dir,
                request_seed=request_seed,
                concurrency=ADAPTER_CONCURRENCY,
                projected_cost=PROJECTED_COST_USD,
                run_budget_usd=RUN_BUDGET_USD,
            )
            for request_seed in MODELS[model_id]
        ]
    if len(adapters) != SEED_GROUPS:
        raise ValueError("reliability gate requires exactly eight adapters")
    cases = build_cases(source_path)
    grouped_cases = [
        [case for case in cases if case["seed_group"] == seed_group]
        for seed_group in range(SEED_GROUPS)
    ]
    initial_by_index = _call_seed_groups(
        adapters=adapters,
        grouped_cases=grouped_cases,
    )
    if len(initial_by_index) != EXPECTED_INITIAL_REQUESTS:
        raise ValueError("adapter did not return exactly 128 initial responses")
    initial_responses = [
        initial_by_index[index] for index in range(EXPECTED_INITIAL_REQUESTS)
    ]

    parsed = []
    failed_indices = []
    for case, response in zip(cases, initial_responses, strict=True):
        support, diagnostic, error = _safe_parse(
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
        retry_by_index = _call_seed_groups(
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
                _safe_parse(
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

    records = []
    retry_index_set = set(failed_indices if retry_responses else [])
    for case, (support, diagnostic, final_error) in zip(
        cases, parsed, strict=True
    ):
        records.append(
            {
                "case_index": case["case_index"],
                "seed_group": case["seed_group"],
                "observations": [
                    [number, label]
                    for number, label in case["observations"]
                ],
                "initial_parse_failed": (
                    case["case_index"] in failed_indices
                ),
                "format_retried": case["case_index"] in retry_index_set,
                "final_parse_error": final_error,
                "valid_unique_count": len(support or []),
                "diagnostics": diagnostic,
                "extension_sha256s": [
                    hypothesis.public_dict()["extension_sha256"]
                    for hypothesis in (support or [])
                ],
            }
        )
    usage = aggregate_usage(adapters)
    format_retry_requests = len(retry_responses)
    gates = reliability_gates(
        records=records,
        usage=usage,
        format_retry_requests=format_retry_requests,
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
            "eligible_for_separately_frozen_paired_efficacy"
            if gates["all_pass"]
            else "close_unchanged_interface"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": model_id,
            "model_seeds": list(MODELS[model_id]),
            "case_selection_seed": CASE_SELECTION_SEED,
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
                "one_observation": ONE_OBSERVATION_CASES,
                "two_observations": TWO_OBSERVATION_CASES,
            },
            "format_retry_policy": (
                "one_same_prompt_same_temperature_retry_for_strict_parse_"
                "failure_only_if_at_most_three_failures"
            ),
            "run_budget_usd": RUN_BUDGET_USD,
            "efficacy_used_for_authorization": False,
        },
        "usage": usage,
        "initial_parse_failures": len(failed_indices),
        "format_retry_requests": format_retry_requests,
        "forced_exit_rate": (
            usage["forced_exits"] / usage["adapter_requests"]
            if usage["adapter_requests"]
            else 0.0
        ),
        "conditioned_valid_summary": {
            "minimum": min(conditioned_counts),
            "mean": sum(conditioned_counts) / len(conditioned_counts),
            "maximum": max(conditioned_counts),
        },
        "gates": gates,
        "cases": records,
        "raw_responses_sha256": sha256_file(raw_path),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def execute_reliability(
    *,
    output_dir: Path,
    run_id: str,
    model_id: str,
    ledger_path: Path,
    qwen_control_result: Path,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    reliability_runner: Callable[..., dict[str, Any]] = run_reliability,
) -> dict[str, Any]:
    """Run one authorized gate and always reconcile any posted spend."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    validate_completed_qwen_control(qwen_control_result)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    validate_tail_authorization(ledger, model_id=model_id)
    live = live_reader()
    require_budget(
        ledger,
        projected_cost_usd=PROJECTED_COST_USD,
        total_usage_usd=live["total_usage_usd"],
        now=now,
    )
    if live["balance_usd"] + 1e-12 < RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below the reliability gate")
    try:
        result = reliability_runner(
            output_dir=output_dir,
            run_id=run_id,
            model_id=model_id,
        )
    except Exception as exc:
        reconciliation_error = None
        try:
            live_after = live_reader()
            reconciled = reconcile_daily_ledger(
                ledger=ledger,
                model_id=model_id,
                measured_cost_usd=0.0,
                live_after=live_after,
                status="failed_closed_posted_spend_reconciled",
            )
            checkpoint(ledger_path, reconciled)
        except Exception as reconcile_exc:
            reconciliation_error = (
                f"{type(reconcile_exc).__name__}: {reconcile_exc}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint(
            output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": "failed_closed",
                "model": model_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ledger_reconciliation_error": reconciliation_error,
            },
        )
        raise
    measured_cost = float(result["usage"]["run_cost_usd"])
    locally_reconciled = reconcile_daily_ledger(
        ledger=ledger,
        model_id=model_id,
        measured_cost_usd=measured_cost,
        live_after=live,
        status=result["status"],
    )
    checkpoint(ledger_path, locally_reconciled)
    live_after = live_reader()
    reconciled = reconcile_daily_ledger(
        ledger=locally_reconciled,
        model_id=model_id,
        measured_cost_usd=0.0,
        live_after=live_after,
        status=result["status"],
    )
    checkpoint(ledger_path, reconciled)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    parser.add_argument("--qwen-control-result", type=Path, required=True)
    args = parser.parse_args()
    result = execute_reliability(
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_id=args.model,
        ledger_path=args.daily_ledger,
        qwen_control_result=args.qwen_control_result,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "model": result["protocol"]["model"],
                "usage": result["usage"],
                "initial_parse_failures": result["initial_parse_failures"],
                "format_retry_requests": result["format_retry_requests"],
                "conditioned_valid_summary": result[
                    "conditioned_valid_summary"
                ],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
