from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_budget_model_reliability128 as reliability
from scripts.number_game_generator_aware_bed import compile_expression


def _valid_response(
    observations: Sequence[tuple[int, bool]],
) -> str:
    positives = [number for number, label in observations if label]
    negatives = [number for number, label in observations if not label]
    hypotheses = []
    seen_extensions = set()
    for index in range(240):
        modulus = 2 + index % 49
        remainder = (index // 49 + index * 3 + 1) % modulus
        expression = f"n % {modulus} == {remainder}"
        if positives:
            positive_clause = " or ".join(
                f"n == {number}" for number in positives
            )
            expression = f"({expression} or {positive_clause})"
        for number in negatives:
            expression = f"({expression}) and n != {number}"
        extension = compile_expression(expression)
        if any(extension[number] != label for number, label in observations):
            continue
        if extension in seen_extensions:
            continue
        seen_extensions.add(extension)
        hypotheses.append(
            {"name": f"rule_{index}", "expression": expression}
        )
        if len(hypotheses) == 24:
            response = json.dumps({"hypotheses": hypotheses})
            parsed, _ = reliability.parse_proposals(
                response,
                observations=observations,
            )
            assert len(parsed) == 24
            return response
    raise AssertionError("test generator could not construct 24 valid rules")


def _semantic_failure_response(
    observations: Sequence[tuple[int, bool]],
) -> str:
    number, label = observations[0]
    expression = f"n != {number}" if label else f"n == {number}"
    return json.dumps(
        {
            "hypotheses": [
                {"name": f"bad_{index}", "expression": expression}
                for index in range(24)
            ]
        }
    )


class FakeAdapter:
    def __init__(
        self,
        initial_responses: list[str],
        *,
        retry_responses: list[str] | None = None,
        forced_exits: int = 0,
    ) -> None:
        self.initial_responses = initial_responses
        self.retry_responses = retry_responses or []
        self.forced_exits = forced_exits
        self.calls = 0
        self.adapter_requests = 0

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        **_: Any,
    ) -> list[str]:
        self.adapter_requests += len(batch_messages)
        if self.calls == 0:
            values = self.initial_responses
        else:
            values = self.retry_responses
        self.calls += 1
        if len(values) != len(batch_messages):
            raise AssertionError("fake response count mismatch")
        return values

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.adapter_requests,
            "http_attempts": self.adapter_requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": self.forced_exits,
            "adapter_cost_usd": 0.01,
            "adapter_prompt_tokens": 1000,
            "adapter_completion_tokens": 2000,
        }


def _cases_and_responses() -> tuple[list[dict[str, Any]], list[str]]:
    cases = reliability.build_cases()
    responses = [
        _valid_response(case["observations"])
        for case in cases
    ]
    return cases, responses


def _group_adapters(
    cases: Sequence[dict[str, Any]],
    responses: Sequence[str],
    *,
    retry_by_index: dict[int, str] | None = None,
    forced_exits: int = 0,
) -> list[FakeAdapter]:
    retry_by_index = retry_by_index or {}
    adapters = []
    for seed_group in range(reliability.SEED_GROUPS):
        indices = [
            case["case_index"]
            for case in cases
            if case["seed_group"] == seed_group
        ]
        retries = [
            retry_by_index[index]
            for index in indices
            if index in retry_by_index
        ]
        adapters.append(
            FakeAdapter(
                [responses[index] for index in indices],
                retry_responses=retries,
                forced_exits=forced_exits if seed_group == 0 else 0,
            )
        )
    return adapters


def test_build_cases_is_frozen_and_stratified() -> None:
    first = reliability.build_cases()
    second = reliability.build_cases()

    assert first == second
    assert len(first) == 128
    counts = {
        depth: sum(len(case["observations"]) == depth for case in first)
        for depth in (0, 1, 2)
    }
    assert counts == {0: 8, 1: 40, 2: 80}
    assert len(
        {case["observations"] for case in first if case["observations"]}
    ) == 120
    for seed_group in range(8):
        grouped = [
            case
            for case in first
            if case["seed_group"] == seed_group
        ]
        assert [len(case["observations"]) for case in grouped].count(0) == 1
        assert [len(case["observations"]) for case in grouped].count(1) == 5
        assert [len(case["observations"]) for case in grouped].count(2) == 10


def test_build_cases_rejects_source_hash_change(tmp_path: Path) -> None:
    altered = tmp_path / "TREES.json"
    altered.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="source TREES hash changed"):
        reliability.build_cases(altered)


def test_completed_control_gate_requires_exact_mechanics(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.json"
    payload = {
        "decision": "complete_composite_endpoint",
        "control": {
            "usage": {"adapter_requests": 3072},
            "mechanics_gates": {"requests": True, "parsing": True},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert reliability.validate_completed_qwen_control(path) == payload

    payload["control"]["mechanics_gates"]["parsing"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mechanics did not pass"):
        reliability.validate_completed_qwen_control(path)


def test_completed_control_gate_requires_exact_request_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "RESULT.json"
    path.write_text(
        json.dumps(
            {
                "decision": "complete_composite_endpoint",
                "control": {
                    "usage": {"adapter_requests": 3071},
                    "mechanics_gates": {"all": True},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly 3072"):
        reliability.validate_completed_qwen_control(path)


def test_clean_reliability_run_passes(tmp_path: Path) -> None:
    cases, responses = _cases_and_responses()
    adapters = _group_adapters(cases, responses)

    result = reliability.run_reliability(
        output_dir=tmp_path / "clean",
        run_id="clean",
        model_id="openai/gpt-5.6-luna",
        adapters=adapters,
    )

    assert result["status"] == "passed"
    assert result["decision"] == (
        "eligible_for_separately_frozen_paired_efficacy"
    )
    assert result["initial_parse_failures"] == 0
    assert result["format_retry_requests"] == 0
    assert result["usage"]["adapter_requests"] == 128
    assert result["gates"]["all_pass"]
    assert result["conditioned_valid_summary"]["minimum"] >= 4
    assert (tmp_path / "clean" / "RESULT.json").exists()
    assert (tmp_path / "clean" / "private" / "RAW_RESPONSES.json").exists()
    verification = reliability.replay_reliability_result(
        result_path=tmp_path / "clean" / "RESULT.json",
        model_id="openai/gpt-5.6-luna",
    )
    assert verification["verified"]
    assert verification["request_count"] == 128

    result["cases"][0]["valid_unique_count"] -= 1
    (tmp_path / "clean" / "RESULT.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not replay exactly"):
        reliability.replay_reliability_result(
            result_path=tmp_path / "clean" / "RESULT.json",
            model_id="openai/gpt-5.6-luna",
        )


def test_single_strict_parse_failure_is_retried_once(tmp_path: Path) -> None:
    cases, responses = _cases_and_responses()
    failure_index = 17
    retry = _valid_response(cases[failure_index]["observations"])
    responses[failure_index] = "{not-json"
    adapters = _group_adapters(
        cases,
        responses,
        retry_by_index={failure_index: retry},
    )

    result = reliability.run_reliability(
        output_dir=tmp_path / "retry",
        run_id="retry",
        model_id="openai/gpt-5.6-luna",
        adapters=adapters,
    )

    assert result["status"] == "passed"
    assert result["initial_parse_failures"] == 1
    assert result["format_retry_requests"] == 1
    assert result["usage"]["adapter_requests"] == 129
    assert result["cases"][failure_index]["initial_parse_failed"]
    assert result["cases"][failure_index]["format_retried"]
    assert result["cases"][failure_index]["final_parse_error"] is None
    assert reliability.replay_reliability_result(
        result_path=tmp_path / "retry" / "RESULT.json",
        model_id="openai/gpt-5.6-luna",
    )["verified"]


def test_semantic_support_failure_is_not_retried(tmp_path: Path) -> None:
    cases, responses = _cases_and_responses()
    failure_index = 8
    assert cases[failure_index]["observations"]
    responses[failure_index] = _semantic_failure_response(
        cases[failure_index]["observations"]
    )
    adapters = _group_adapters(cases, responses)

    result = reliability.run_reliability(
        output_dir=tmp_path / "semantic",
        run_id="semantic",
        model_id="deepseek/deepseek-v4-flash-0731",
        adapters=adapters,
    )

    assert result["status"] == "gated_null"
    assert result["initial_parse_failures"] == 0
    assert result["format_retry_requests"] == 0
    assert all(adapter.calls == 1 for adapter in adapters)
    assert not result["gates"][
        "all_conditioned_supports_have_at_least_4_valid"
    ]


def test_more_than_three_parse_failures_close_without_retries(
    tmp_path: Path,
) -> None:
    cases, responses = _cases_and_responses()
    for index in range(4):
        responses[index] = "{not-json"
    adapters = _group_adapters(cases, responses)

    result = reliability.run_reliability(
        output_dir=tmp_path / "too-many",
        run_id="too-many",
        model_id="openai/gpt-5.6-luna",
        adapters=adapters,
    )

    assert result["status"] == "gated_null"
    assert result["initial_parse_failures"] == 4
    assert result["format_retry_requests"] == 0
    assert all(adapter.calls == 1 for adapter in adapters)
    assert result["usage"]["adapter_requests"] == 128


def test_forced_exit_cap_is_a_hard_gate(tmp_path: Path) -> None:
    cases, responses = _cases_and_responses()
    adapters = _group_adapters(cases, responses, forced_exits=4)

    result = reliability.run_reliability(
        output_dir=tmp_path / "forced",
        run_id="forced",
        model_id="openai/gpt-5.6-luna",
        adapters=adapters,
    )

    assert result["status"] == "gated_null"
    assert not result["gates"]["forced_exits_within_cap"]


def test_unknown_model_is_rejected_before_calls(tmp_path: Path) -> None:
    adapters = [FakeAdapter([]) for _ in range(8)]

    with pytest.raises(ValueError, match="unsupported reliability model"):
        reliability.run_reliability(
            output_dir=tmp_path / "unknown",
            run_id="unknown",
            model_id="unknown/model",
            adapters=adapters,
        )

    assert all(adapter.calls == 0 for adapter in adapters)


def test_daily_ledger_reconciliation_accumulates_local_model_cost() -> None:
    ledger = {
        "date": "2026-08-07",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": 3.2,
        "additional_paid_blocks_authorized": True,
        "authorized_tail_blocks": [
            {
                "interface": reliability.INTERFACE_VERSION,
                "model": "openai/gpt-5.6-luna",
                "maximum_cost_usd": 0.1,
                "status": "authorized_pending",
            },
            {
                "interface": reliability.INTERFACE_VERSION,
                "model": "deepseek/deepseek-v4-flash-0731",
                "maximum_cost_usd": 0.1,
                "status": "authorized_pending",
            },
        ],
    }
    reconciled = reliability.reconcile_daily_ledger(
        ledger=ledger,
        model_id="openai/gpt-5.6-luna",
        measured_cost_usd=0.05,
        live_after={
            "total_credits_usd": 130.0,
            "total_usage_usd": 103.1,
            "balance_usd": 26.9,
        },
        status="passed",
    )

    assert reconciled["recorded_actual_spend_usd"] == pytest.approx(3.25)
    assert reconciled["reconciliation"][
        "remaining_daily_allowance_usd"
    ] == pytest.approx(1.75)
    assert reconciled[
        "budget_model_reliability128_openai_gpt-5_6-luna"
    ]["status"] == "passed"
    assert reconciled["additional_paid_blocks_authorized"]
    assert reconciled["authorized_tail_blocks"][0]["status"] == "passed"


def test_daily_ledger_reconciliation_prefers_larger_posted_spend() -> None:
    ledger = {
        "date": "2026-08-07",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": 3.2,
    }
    reconciled = reliability.reconcile_daily_ledger(
        ledger=ledger,
        model_id="deepseek/deepseek-v4-flash-0731",
        measured_cost_usd=0.0,
        live_after={
            "total_credits_usd": 130.0,
            "total_usage_usd": 104.4,
            "balance_usd": 25.6,
        },
        status="failed_closed_posted_spend_reconciled",
    )

    assert reconciled["recorded_actual_spend_usd"] == pytest.approx(4.4)


def test_execute_reliability_uses_shared_ledger_and_reconciles(
    tmp_path: Path,
) -> None:
    control_result = tmp_path / "control.json"
    control_result.write_text(
        json.dumps(
            {
                "decision": "complete_composite_endpoint",
                "control": {
                    "usage": {"adapter_requests": 3072},
                    "mechanics_gates": {"all": True},
                },
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "date": "2026-08-07",
                "timezone": "Europe/London",
                "daily_cap_usd": 5.0,
                "opening_total_usage_usd": 100.0,
                "recorded_actual_spend_usd": 3.2,
                "additional_paid_blocks_authorized": True,
                "authorized_tail_blocks": [
                    {
                        "interface": reliability.INTERFACE_VERSION,
                        "model": "openai/gpt-5.6-luna",
                        "maximum_cost_usd": 0.1,
                        "status": "authorized_pending",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    live_values = iter(
        [
            {
                "total_credits_usd": 130.0,
                "total_usage_usd": 103.2,
                "balance_usd": 26.8,
            },
            {
                "total_credits_usd": 130.0,
                "total_usage_usd": 103.24,
                "balance_usd": 26.76,
            },
        ]
    )

    def runner(**kwargs):
        kwargs["output_dir"].mkdir(parents=True)
        result = {
            "status": "passed",
            "usage": {"run_cost_usd": 0.04},
        }
        (kwargs["output_dir"] / "RESULT.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    result = reliability.execute_reliability(
        output_dir=tmp_path / "reliability",
        run_id="reliability",
        model_id="openai/gpt-5.6-luna",
        ledger_path=ledger_path,
        qwen_control_result=control_result,
        now=datetime(2026, 8, 7, 12, tzinfo=ZoneInfo("Europe/London")),
        live_reader=lambda: next(live_values),
        reliability_runner=runner,
    )

    assert result["status"] == "passed"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(3.24)
    assert ledger["authorized_tail_blocks"][0]["status"] == "passed"


def test_execute_reliability_failure_is_banked_and_reconciled(
    tmp_path: Path,
) -> None:
    control_result = tmp_path / "control.json"
    control_result.write_text(
        json.dumps(
            {
                "decision": "complete_composite_endpoint",
                "control": {
                    "usage": {"adapter_requests": 3072},
                    "mechanics_gates": {"all": True},
                },
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "date": "2026-08-07",
                "timezone": "Europe/London",
                "daily_cap_usd": 5.0,
                "opening_total_usage_usd": 100.0,
                "recorded_actual_spend_usd": 3.2,
                "additional_paid_blocks_authorized": True,
                "authorized_tail_blocks": [
                    {
                        "interface": reliability.INTERFACE_VERSION,
                        "model": "deepseek/deepseek-v4-flash-0731",
                        "maximum_cost_usd": 0.1,
                        "status": "authorized_pending",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    live_values = iter(
        [
            {
                "total_credits_usd": 130.0,
                "total_usage_usd": 103.2,
                "balance_usd": 26.8,
            },
            {
                "total_credits_usd": 130.0,
                "total_usage_usd": 103.23,
                "balance_usd": 26.77,
            },
        ]
    )

    with pytest.raises(RuntimeError, match="serving failed"):
        reliability.execute_reliability(
            output_dir=tmp_path / "failed",
            run_id="failed",
            model_id="deepseek/deepseek-v4-flash-0731",
            ledger_path=ledger_path,
            qwen_control_result=control_result,
            now=datetime(
                2026, 8, 7, 12, tzinfo=ZoneInfo("Europe/London")
            ),
            live_reader=lambda: next(live_values),
            reliability_runner=lambda **_: (_ for _ in ()).throw(
                RuntimeError("serving failed")
            ),
        )

    failure = json.loads(
        (tmp_path / "failed" / "FAILURE.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed_closed"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(3.23)


def test_execute_reliability_checkpoints_local_cost_before_posted_read(
    tmp_path: Path,
) -> None:
    control_result = tmp_path / "control.json"
    control_result.write_text(
        json.dumps(
            {
                "decision": "complete_composite_endpoint",
                "control": {
                    "usage": {"adapter_requests": 3072},
                    "mechanics_gates": {"all": True},
                },
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "date": "2026-08-07",
                "timezone": "Europe/London",
                "daily_cap_usd": 5.0,
                "opening_total_usage_usd": 100.0,
                "recorded_actual_spend_usd": 3.2,
                "additional_paid_blocks_authorized": True,
                "authorized_tail_blocks": [
                    {
                        "interface": reliability.INTERFACE_VERSION,
                        "model": "openai/gpt-5.6-luna",
                        "maximum_cost_usd": 0.1,
                        "status": "authorized_pending",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    def live_reader():
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "total_credits_usd": 130.0,
                "total_usage_usd": 103.2,
                "balance_usd": 26.8,
            }
        raise RuntimeError("credits endpoint unavailable")

    def runner(**kwargs):
        kwargs["output_dir"].mkdir(parents=True)
        result = {"status": "passed", "usage": {"run_cost_usd": 0.04}}
        (kwargs["output_dir"] / "RESULT.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    with pytest.raises(RuntimeError, match="credits endpoint unavailable"):
        reliability.execute_reliability(
            output_dir=tmp_path / "reliability",
            run_id="reliability",
            model_id="openai/gpt-5.6-luna",
            ledger_path=ledger_path,
            qwen_control_result=control_result,
            now=datetime(
                2026, 8, 7, 12, tzinfo=ZoneInfo("Europe/London")
            ),
            live_reader=live_reader,
            reliability_runner=runner,
        )

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(3.24)
    assert ledger["authorized_tail_blocks"][0]["actual_cost_usd"] == 0.04
    assert ledger["authorized_tail_blocks"][0]["status"] == "passed"


def test_tail_authorization_is_exact_and_pending() -> None:
    ledger = {
        "additional_paid_blocks_authorized": True,
        "authorized_tail_blocks": [
            {
                "interface": reliability.INTERFACE_VERSION,
                "model": "openai/gpt-5.6-luna",
                "maximum_cost_usd": reliability.RUN_BUDGET_USD,
                "status": "authorized_pending",
            }
        ],
    }
    authorization = reliability.validate_tail_authorization(
        ledger,
        model_id="openai/gpt-5.6-luna",
    )
    assert authorization["status"] == "authorized_pending"

    ledger["authorized_tail_blocks"][0]["status"] = "passed"
    with pytest.raises(RuntimeError, match="pending authorization"):
        reliability.validate_tail_authorization(
            ledger,
            model_id="openai/gpt-5.6-luna",
        )
