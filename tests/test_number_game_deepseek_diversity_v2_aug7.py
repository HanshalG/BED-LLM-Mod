from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_deepseek_diversity_v2_aug7 as v2
from scripts.number_game_generator_aware_bed import compile_expression


def _now() -> datetime:
    return datetime(2026, 8, 7, 12, tzinfo=ZoneInfo(v2.TIMEZONE))


def _unopened_ledger(tmp_path: Path, monkeypatch) -> Path:
    ledger = json.loads(v2.DAILY_LEDGER.read_text(encoding="utf-8"))
    ledger.pop("deepseek_diversity_v2_blocks", None)
    ledger.pop("deepseek_diversity_v2_preregistration_sha256", None)
    ledger["additional_paid_blocks_authorized"] = False
    ledger["recorded_actual_spend_usd"] = 2.7820341410000027
    ledger["reconciliation"]["remaining_daily_allowance_usd"] = (
        2.2179658589999973
    )
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(v2, "INITIAL_LEDGER_SHA256", v2._sha256(path))
    return path


def _redirect_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(v2, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(v2, "RELIABILITY_DIR", tmp_path / "output/reliability")
    monkeypatch.setattr(v2, "STRESS_DIR", tmp_path / "output/stress")
    monkeypatch.setattr(v2, "WRAPPER_DIR", tmp_path / "output/wrapper")


def _valid_response(observations) -> str:
    expressions = []
    for k in range(101):
        expressions.extend((f"n <= {k}", f"n >= {k}", f"n < {k}", f"n > {k}"))
    for k in range(2, 31):
        expressions.extend(f"n % {k} == {remainder}" for remainder in range(k))
        expressions.extend((f"divisible(n, {k})", f"not divisible(n, {k})"))
    for digit in range(10):
        expressions.extend(
            (f"ends_with(n, {digit})", f"not ends_with(n, {digit})")
        )
    for k in range(1, 30):
        expressions.extend(
            (f"digit_sum(n) <= {k}", f"digit_sum(n) >= {k}", f"digit_sum(n) == {k}")
        )
    selected = []
    seen = set()
    for expression in expressions:
        try:
            extension = compile_expression(expression)
        except ValueError:
            continue
        if extension in seen or not all(
            extension[number] == label for number, label in observations
        ):
            continue
        seen.add(extension)
        selected.append(
            {"name": f"rule {len(selected)}", "expression": expression}
        )
        if len(selected) == 24:
            break
    assert len(selected) == 24
    return json.dumps({"hypotheses": selected})


class _FakeAdapter:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_complete_messages_batched_structured(self, messages, **kwargs):
        assert len(messages) == len(self.responses)
        return list(self.responses)


def test_case_manifests_are_exact_and_conditioned_sets_are_disjoint() -> None:
    old = v2._old_gate_histories()
    gate = {
        case["observations"]
        for case in v2.build_gate_cases()
        if case["observations"]
    }
    stress = {
        case["observations"]
        for case in v2.build_stress_cases()
        if case["history_role"] == "fresh_conditioned"
    }
    assert len(old) == len(gate) == 120
    assert len(stress) == 570
    assert not old & gate
    assert not old & stress
    assert not gate & stress
    assert len(v2.build_stress_cases()) == 7_168


def test_prompt_requires_extension_diversity_and_both_label_directions() -> None:
    content = v2._messages(((7, True), (9, False)))[1]["content"]
    assert "n=7 MUST evaluate to True" in content
    assert "n=9 MUST evaluate to False" in content
    assert "different membership sets" in content
    assert "four hypotheses per family" in content


def test_gate_arithmetic_rejects_one_low_conditioned_support() -> None:
    records = []
    for index in range(128):
        initial = index < 8
        records.append(
            {
                "observations": [] if initial else [[index, True]],
                "valid_unique_count": 20 if initial else 8,
                "final_parse_error": None,
                "diagnostics": {"raw_count": 24},
            }
        )
    usage = {
        "adapter_requests": 128,
        "http_attempts": 128,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.03,
    }
    passed = v2._gates(
        records=records,
        usage=usage,
        expected_requests=128,
        expected_initial=8,
        expected_conditioned=120,
        max_format_retries=3,
        max_transport_retries=4,
        max_forced_exits=3,
        budget_usd=0.1,
        initial_parse_failures=0,
        format_retry_requests=0,
    )
    assert passed["all_pass"] is True
    records[-1]["valid_unique_count"] = 3
    failed = v2._gates(
        records=records,
        usage=usage,
        expected_requests=128,
        expected_initial=8,
        expected_conditioned=120,
        max_format_retries=3,
        max_transport_retries=4,
        max_forced_exits=3,
        budget_usd=0.1,
        initial_parse_failures=0,
        format_retry_requests=0,
    )
    assert failed["all_conditioned_supports_have_at_least_4_valid"] is False
    assert failed["all_pass"] is False


def test_synthetic_gate_runs_and_replays_exactly(tmp_path, monkeypatch) -> None:
    cases = v2.build_gate_cases()
    adapters = []
    for group in range(v2.GATE_GROUPS):
        grouped = [case for case in cases if case["seed_group"] == group]
        adapters.append(
            _FakeAdapter(
                [_valid_response(case["observations"]) for case in grouped]
            )
        )
    monkeypatch.setattr(
        v2,
        "_usage",
        lambda adapters: {
            "adapter_requests": 128,
            "http_attempts": 128,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "run_cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
    )
    result = v2._run_stage(
        output_dir=tmp_path / "gate",
        run_id="synthetic",
        stage="reliability128",
        cases=cases,
        model_seeds=v2.GATE_MODEL_SEEDS,
        max_format_retries=v2.GATE_MAX_FORMAT_RETRIES,
        max_transport_retries=v2.GATE_MAX_TRANSPORT_RETRIES,
        max_forced_exits=v2.GATE_MAX_FORCED_EXITS,
        budget_usd=v2.GATE_BUDGET_USD,
        adapters=adapters,
    )
    assert result["status"] == "passed"
    replay = v2.replay_stage(
        result_path=tmp_path / "gate/RESULT.json",
        cases=cases,
        stage="reliability128",
    )
    assert replay["verified"] is True
    assert replay["requests"] == 128


def test_authorization_is_exactly_remaining_2_20(tmp_path, monkeypatch) -> None:
    ledger_path = _unopened_ledger(tmp_path, monkeypatch)
    _redirect_outputs(tmp_path, monkeypatch)
    ledger = v2._authorize(ledger_path)
    blocks = ledger["deepseek_diversity_v2_blocks"]
    assert sum(block["maximum_cost_usd"] for block in blocks) == pytest.approx(
        2.20
    )
    assert all(block["efficacy_used_for_authorization"] is False for block in blocks)


def test_reliability_exception_closes_stress_authorization(
    tmp_path, monkeypatch
) -> None:
    ledger_path = _unopened_ledger(tmp_path, monkeypatch)
    _redirect_outputs(tmp_path, monkeypatch)
    ledger = v2._authorize(ledger_path)
    updated = v2._reconcile(
        ledger=ledger,
        stage="reliability128",
        measured_cost_usd=0.0,
        live_after={
            "total_credits_usd": 245.0,
            "total_usage_usd": ledger["opening_total_usage_usd"] + 2.8,
            "balance_usd": 24.9,
        },
        status="failed_closed_posted_spend_reconciled",
    )
    statuses = {
        item["stage"]: item["status"]
        for item in updated["deepseek_diversity_v2_blocks"]
    }
    assert statuses["stress7168"] == "not_authorized_reliability_failure"
    assert updated["additional_paid_blocks_authorized"] is False


def test_preflight_is_read_only(tmp_path, monkeypatch) -> None:
    ledger_path = _unopened_ledger(tmp_path, monkeypatch)
    _redirect_outputs(tmp_path, monkeypatch)
    before = ledger_path.read_bytes()
    result = v2.preflight(
        ledger_path=ledger_path,
        live_reader=lambda: {
            "total_credits_usd": 245.0,
            "total_usage_usd": 219.99,
            "balance_usd": 25.01,
        },
        now=_now(),
    )
    assert result["status"] == "ready_without_paid_calls"
    assert result["maximum_new_cost_usd"] == pytest.approx(2.2)
    assert result["post_authorization_slack_usd"] == pytest.approx(
        0.017965859
    )
    assert result["model_calls_made"] == result["files_written"] == 0
    assert ledger_path.read_bytes() == before


def test_reliability_null_closes_stress_without_executor_call(
    tmp_path, monkeypatch
) -> None:
    ledger_path = _unopened_ledger(tmp_path, monkeypatch)
    _redirect_outputs(tmp_path, monkeypatch)
    calls = []

    def fake_paid_stage(**kwargs):
        calls.append(kwargs["stage"])
        return {
            "verified": True,
            "stage": "reliability128",
            "status": "gated_null",
            "requests": 128,
            "cost_usd": 0.03,
            "result_sha256": "a" * 64,
        }

    monkeypatch.setattr(v2, "_execute_paid_stage", fake_paid_stage)
    result = v2.execute(
        ledger_path=ledger_path,
        live_reader=lambda: {
            "total_credits_usd": 245.0,
            "total_usage_usd": 219.99,
            "balance_usd": 25.01,
        },
        now=_now(),
    )
    assert calls == ["reliability128"]
    assert result["stress_verification"]["requests"] == 0
    assert result["authorizes_aug8_diversity"] is False
    stress = json.loads((v2.STRESS_DIR / "RESULT.json").read_text())
    assert stress["decision"] == "not_authorized_reliability_null"


def test_wrong_date_refuses_before_authorization(tmp_path, monkeypatch) -> None:
    ledger_path = _unopened_ledger(tmp_path, monkeypatch)
    _redirect_outputs(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="restricted to Aug 7"):
        v2.execute(
            ledger_path=ledger_path,
            now=datetime(2026, 8, 8, tzinfo=ZoneInfo(v2.TIMEZONE)),
        )
