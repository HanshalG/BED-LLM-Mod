from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_budget_model_reliability128 as reliability
from scripts import number_game_budget_model_stress3584 as stress


def _reliability_result(
    model: str,
    *,
    passed: bool = True,
    minimum: int = 6,
    mean: float = 12.0,
    cost: float = 0.04,
) -> dict:
    return {
        "schema_version": 1,
        "status": "passed" if passed else "gated_null",
        "protocol": {
            "interface_version": reliability.INTERFACE_VERSION,
            "model": model,
        },
        "gates": {"all_pass": passed},
        "conditioned_valid_summary": {
            "minimum": minimum,
            "mean": mean,
            "maximum": 24,
        },
        "initial_parse_failures": 0,
        "format_retry_requests": 0,
        "usage": {
            "forced_exits": 0,
            "retry_count": 0,
            "run_cost_usd": cost,
        },
    }


def _ledger(*, luna_status: str, deepseek_status: str) -> dict:
    return {
        "date": "2026-08-07",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": 4.3,
        "additional_paid_blocks_authorized": True,
        "authorized_tail_blocks": [
            {
                "interface": reliability.INTERFACE_VERSION,
                "model": "openai/gpt-5.6-luna",
                "maximum_cost_usd": 0.1,
                "status": luna_status,
            },
            {
                "interface": reliability.INTERFACE_VERSION,
                "model": "deepseek/deepseek-v4-flash-0731",
                "maximum_cost_usd": 0.1,
                "status": deepseek_status,
            },
            {
                "interface": stress.INTERFACE_VERSION,
                "model": None,
                "maximum_cost_usd": stress.RUN_BUDGET_USD,
                "status": "waiting_for_reliability_results",
                "selection_rule": stress.SELECTION_RULE,
            },
        ],
    }


def _write_results(
    tmp_path: Path,
    *,
    luna_passed: bool,
    deepseek_passed: bool,
) -> dict[str, Path]:
    values = {
        "openai/gpt-5.6-luna": _reliability_result(
            "openai/gpt-5.6-luna",
            passed=luna_passed,
        ),
        "deepseek/deepseek-v4-flash-0731": _reliability_result(
            "deepseek/deepseek-v4-flash-0731",
            passed=deepseek_passed,
        ),
    }
    paths = {}
    for index, (model, value) in enumerate(values.items()):
        path = tmp_path / f"reliability-{index}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[model] = path
    return paths


def test_preregistration_and_case_manifest_are_frozen() -> None:
    assert (
        reliability.sha256_file(stress.PREREGISTRATION)
        == stress.PREREGISTRATION_SHA256
    )
    cases = stress.build_stress_cases()
    assert len(cases) == 3584
    assert stress.case_manifest_sha256(cases) == stress.CASE_MANIFEST_SHA256
    assert all(
        sum(case["seed_group"] == group for case in cases) == 224
        for group in range(stress.SEED_GROUPS)
    )
    roles = {
        role: sum(case["history_role"] == role for case in cases)
        for role in {case["history_role"] for case in cases}
    }
    assert roles == {
        "initial": 16,
        "unseen_conditioned": 690,
        "unseen_conditioned_repeat": 2878,
    }


def test_stress_histories_are_disjoint_from_reliability_gate() -> None:
    gate = {
        case["observations"]
        for case in reliability.build_cases()
        if case["observations"]
    }
    unseen = {
        case["observations"]
        for case in stress.build_stress_cases()
        if case["history_role"] == "unseen_conditioned"
    }
    assert len(unseen) == 690
    assert gate.isdisjoint(unseen)


def test_model_selection_uses_frozen_mechanics_order() -> None:
    results = {
        "openai/gpt-5.6-luna": _reliability_result(
            "openai/gpt-5.6-luna"
        ),
        "deepseek/deepseek-v4-flash-0731": _reliability_result(
            "deepseek/deepseek-v4-flash-0731"
        ),
    }
    assert stress.select_model(results)["selected_model"] == (
        "deepseek/deepseek-v4-flash-0731"
    )
    results["openai/gpt-5.6-luna"]["conditioned_valid_summary"][
        "minimum"
    ] = 7
    assert stress.select_model(results)["selected_model"] == (
        "openai/gpt-5.6-luna"
    )
    results["openai/gpt-5.6-luna"]["status"] = "gated_null"
    results["openai/gpt-5.6-luna"]["gates"]["all_pass"] = False
    assert stress.select_model(results)["selected_model"] == (
        "deepseek/deepseek-v4-flash-0731"
    )


def test_failed_closed_reliability_artifact_is_ineligible() -> None:
    results = {
        "openai/gpt-5.6-luna": {
            "interface_version": reliability.INTERFACE_VERSION,
            "model": "openai/gpt-5.6-luna",
            "status": "failed_closed",
        },
        "deepseek/deepseek-v4-flash-0731": _reliability_result(
            "deepseek/deepseek-v4-flash-0731"
        ),
    }

    selection = stress.select_model(results)

    assert selection["selected_model"] == "deepseek/deepseek-v4-flash-0731"
    json.dumps(selection, allow_nan=False)


def test_tail_authorization_requires_both_gate_results_banked() -> None:
    ledger = _ledger(
        luna_status="authorized_pending",
        deepseek_status="passed",
    )
    with pytest.raises(RuntimeError, match="both reliability gates"):
        stress.validate_tail_authorization(ledger)


def _clean_records() -> list[dict]:
    records = []
    for index in range(stress.EXPECTED_INITIAL_REQUESTS):
        records.append(
            {
                "observations": [] if index < stress.INITIAL_CASES else [[1, True]],
                "final_parse_error": None,
                "valid_unique_count": 16 if index < stress.INITIAL_CASES else 8,
                "diagnostics": {"raw_count": 24},
            }
        )
    return records


def test_stress_gates_accept_clean_exact_accounting() -> None:
    usage = {
        "adapter_requests": 3584,
        "http_attempts": 3584,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.4,
    }
    gates = stress.stress_gates(
        records=_clean_records(),
        usage=usage,
        format_retry_requests=0,
        initial_parse_failures=0,
    )
    assert gates["all_pass"]


def test_zero_valid_conditioned_support_fails_stress_gate() -> None:
    records = _clean_records()
    records[-1]["valid_unique_count"] = 0
    usage = {
        "adapter_requests": 3584,
        "http_attempts": 3584,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.4,
    }
    gates = stress.stress_gates(
        records=records,
        usage=usage,
        format_retry_requests=0,
        initial_parse_failures=0,
    )
    assert not gates["all_conditioned_supports_have_at_least_4_valid"]
    assert not gates["all_pass"]


def test_no_eligible_model_closes_without_live_or_model_calls(
    tmp_path: Path,
) -> None:
    paths = _write_results(
        tmp_path,
        luna_passed=False,
        deepseek_passed=False,
    )
    paths["openai/gpt-5.6-luna"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "interface_version": reliability.INTERFACE_VERSION,
                "model": "openai/gpt-5.6-luna",
                "status": "failed_closed",
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            _ledger(
                luna_status="failed_closed_posted_spend_reconciled",
                deepseek_status="gated_null",
            )
        ),
        encoding="utf-8",
    )
    live_calls = []

    result = stress.execute_stress(
        output_dir=tmp_path / "stress",
        run_id="no-eligible",
        ledger_path=ledger_path,
        reliability_paths=paths,
        live_reader=lambda: live_calls.append(True),
    )

    assert result["decision"] == "no_eligible_model"
    assert result["usage"]["adapter_requests"] == 0
    assert live_calls == []
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert not ledger["additional_paid_blocks_authorized"]
    assert ledger["authorized_tail_blocks"][-1]["status"] == (
        "not_authorized_no_model_passed"
    )


def test_second_reconciliation_preserves_measured_stress_cost() -> None:
    ledger = _ledger(luna_status="passed", deepseek_status="gated_null")
    first = stress.reconcile_daily_ledger(
        ledger=ledger,
        selected_model="openai/gpt-5.6-luna",
        measured_cost_usd=0.4,
        live_after={
            "total_credits_usd": 130.0,
            "total_usage_usd": 104.3,
            "balance_usd": 25.7,
        },
        status="passed",
    )
    second = stress.reconcile_daily_ledger(
        ledger=first,
        selected_model="openai/gpt-5.6-luna",
        measured_cost_usd=0.0,
        live_after={
            "total_credits_usd": 130.0,
            "total_usage_usd": 104.7,
            "balance_usd": 25.3,
        },
        status="passed",
    )

    assert second["recorded_actual_spend_usd"] == pytest.approx(4.7)
    assert second["authorized_tail_blocks"][-1][
        "actual_cost_usd"
    ] == pytest.approx(0.4)


def test_selected_execution_reconciles_before_and_after_live_posting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_results(
        tmp_path,
        luna_passed=True,
        deepseek_passed=False,
    )
    ledger_path = tmp_path / "ledger.json"
    ledger = _ledger(luna_status="passed", deepseek_status="gated_null")
    ledger["recorded_actual_spend_usd"] = 3.3
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    run_calls = []

    def run_stress(**kwargs):
        run_calls.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        result = {
            "status": "passed",
            "decision": "eligible_for_policy_scale_budget_model_work",
            "selection": kwargs["selection"],
            "usage": {"adapter_requests": 3584, "run_cost_usd": 0.35},
        }
        (kwargs["output_dir"] / "RESULT.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(stress, "run_stress", run_stress)
    live = iter(
        [
            {
                "total_credits_usd": 130.0,
                "total_usage_usd": 103.3,
                "balance_usd": 26.7,
            },
            {
                "total_credits_usd": 130.0,
                "total_usage_usd": 103.65,
                "balance_usd": 26.35,
            },
        ]
    )

    result = stress.execute_stress(
        output_dir=tmp_path / "stress-selected",
        run_id="selected",
        ledger_path=ledger_path,
        reliability_paths=paths,
        live_reader=lambda: next(live),
        now=datetime(2026, 8, 7, 12, tzinfo=ZoneInfo("Europe/London")),
    )

    assert result["status"] == "passed"
    assert len(run_calls) == 1
    assert run_calls[0]["selected_model"] == "openai/gpt-5.6-luna"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(3.65)
    assert ledger["authorized_tail_blocks"][-1]["status"] == "passed"
    assert ledger["authorized_tail_blocks"][-1][
        "actual_cost_usd"
    ] == pytest.approx(0.35)
    assert not ledger["additional_paid_blocks_authorized"]
