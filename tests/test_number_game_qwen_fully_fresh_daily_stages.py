from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_qwen_fully_fresh_daily_stages as staged


LONDON = ZoneInfo("Europe/London")


def _ledger(
    date: str,
    *,
    recorded: float = 0.0,
    opening_usage: float = 100.0,
) -> dict:
    return {
        "date": date,
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": opening_usage,
        "recorded_actual_spend_usd": recorded,
    }


def _source_result(
    *,
    mechanics: bool = True,
    roots: int = 21,
    science: bool = True,
) -> dict:
    return {
        "protocol": {},
        "status": "passed" if mechanics and science else "gated_null",
        "usage": {
            "adapter_requests": staged.base.SOURCE_EXPECTED_REQUESTS,
            "run_cost_usd": 4.7,
        },
        "mechanics_gates": {"mechanics": mechanics},
        "myopic_policy_gates": {"myopic": science},
        "dynamic_support": {
            "root_differences": roots,
            "gates": {"dynamic": science},
            "comparison": {},
        },
    }


def _control_result(*, science: bool = True) -> dict:
    return {
        "protocol": {},
        "status": "passed" if science else "gated_null",
        "usage": {
            "adapter_requests": staged.base.CONTROL_EXPECTED_REQUESTS,
            "run_cost_usd": 3.2,
        },
        "mechanics_gates": {"mechanics": True},
        "analysis": {"scientific_gates": {"control": science}},
        "second_draw_novelty": {"used_as_gate": False},
    }


def _source_runner(result: dict):
    def run(*, output_dir, run_id):
        del run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        (output_dir / "TREES.json").write_text(json.dumps({"trees": []}))
        (output_dir / "TARGETS.json").write_text(
            json.dumps({"targets": []})
        )
        return deepcopy(result)

    return run


def _control_runner(result: dict, calls: list[str]):
    def run(
        *,
        source_dir,
        output_dir,
        run_id,
        adapter,
        remaining_credit,
        bootstrap_samples,
    ):
        del source_dir, run_id, adapter, remaining_credit, bootstrap_samples
        calls.append("control")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        (output_dir / "CONTROLS.json").write_text(
            json.dumps({"trees": []})
        )
        return deepcopy(result)

    return run


def test_daily_limits_are_tighter_and_scoped() -> None:
    original = {
        "source": staged.base.SOURCE_RUN_BUDGET_USD,
        "composite": staged.base.COMPOSITE_RUN_BUDGET_USD,
        "balance": staged.base.MIN_STARTING_BALANCE_USD,
    }

    with staged.configured_daily_limits():
        assert staged.base.SOURCE_RUN_BUDGET_USD == 5.0
        assert staged.base.COMPOSITE_RUN_BUDGET_USD == 9.25
        assert staged.base.MIN_STARTING_BALANCE_USD == 5.0

    assert staged.base.SOURCE_RUN_BUDGET_USD == original["source"]
    assert staged.base.COMPOSITE_RUN_BUDGET_USD == original["composite"]
    assert staged.base.MIN_STARTING_BALANCE_USD == original["balance"]


def test_control_authorization_is_invariant_to_source_science() -> None:
    passed = _source_result(science=True)
    null = _source_result(science=False)
    hashes = {"result": "a", "trees": "b", "targets": "c"}

    passed_auth = staged.build_control_authorization(
        source_result=passed,
        source_hashes=hashes,
        source_calendar_date="2026-08-05",
    )
    null_auth = staged.build_control_authorization(
        source_result=null,
        source_hashes=hashes,
        source_calendar_date="2026-08-05",
    )

    assert passed_auth == null_auth
    assert passed_auth["structural_authorization"]["control_authorized"]
    serialized = json.dumps(passed_auth, sort_keys=True)
    assert "myopic" not in serialized
    assert "brier" not in serialized
    assert "confidence" not in serialized


def test_source_stage_writes_authorization_and_no_final_endpoint(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(staged.base, "validate_predecessors", lambda: {})
    now = datetime(2026, 8, 5, 12, tzinfo=LONDON)

    stage = staged.run_source_stage(
        output_dir=tmp_path,
        run_id="daily-source",
        ledger=_ledger("2026-08-05"),
        total_usage_usd=100.0,
        balance_usd=30.0,
        source_runner=_source_runner(_source_result(science=False)),
        now=now,
    )

    assert stage["status"] == "control_authorized"
    assert stage["source_starting_balance_usd"] == 30.0
    assert (tmp_path / "CONTROL_AUTHORIZATION.json").exists()
    assert not (tmp_path / "RESULT.json").exists()
    written = json.loads(
        (tmp_path / "source/RESULT.json").read_text(encoding="utf-8")
    )
    assert written["protocol"]["source_daily_cap_usd"] == 5.0


def test_source_stage_refuses_partial_daily_allowance_before_calls(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(staged.base, "validate_predecessors", lambda: {})
    calls = []

    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("source runner must not be called")

    with pytest.raises(RuntimeError, match="exceeds today's remaining"):
        staged.run_source_stage(
            output_dir=tmp_path,
            run_id="budget-refusal",
            ledger=_ledger("2026-08-05", recorded=0.01),
            total_usage_usd=100.0,
            balance_usd=30.0,
            source_runner=forbidden,
            now=datetime(2026, 8, 5, 12, tzinfo=LONDON),
        )

    assert calls == []


def test_source_structural_failure_banks_final_without_control(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(staged.base, "validate_predecessors", lambda: {})

    stage = staged.run_source_stage(
        output_dir=tmp_path,
        run_id="no-opportunity",
        ledger=_ledger("2026-08-05"),
        total_usage_usd=100.0,
        balance_usd=30.0,
        source_runner=_source_runner(_source_result(roots=19)),
        now=datetime(2026, 8, 5, 12, tzinfo=LONDON),
    )

    assert stage["status"] == "control_not_authorized"
    assert stage["final_result"]["status"] == "opportunity_failed"
    assert (tmp_path / "RESULT.json").exists()
    assert not (tmp_path / "control").exists()


def test_same_day_control_is_refused_before_calls(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(staged.base, "validate_predecessors", lambda: {})
    source_now = datetime(2026, 8, 5, 12, tzinfo=LONDON)
    staged.run_source_stage(
        output_dir=tmp_path,
        run_id="same-day",
        ledger=_ledger("2026-08-05"),
        total_usage_usd=100.0,
        balance_usd=30.0,
        source_runner=_source_runner(_source_result()),
        now=source_now,
    )
    calls = []

    with pytest.raises(RuntimeError, match="requires a later"):
        staged.run_control_stage(
            output_dir=tmp_path,
            run_id="same-day",
            ledger=_ledger("2026-08-05"),
            total_usage_usd=100.0,
            balance_usd=25.0,
            control_runner=_control_runner(_control_result(), calls),
            bootstrap_samples=10,
            now=source_now,
        )

    assert calls == []


def test_authorization_manifest_tamper_stops_before_control(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(staged.base, "validate_predecessors", lambda: {})
    staged.run_source_stage(
        output_dir=tmp_path,
        run_id="auth-tamper",
        ledger=_ledger("2026-08-05"),
        total_usage_usd=100.0,
        balance_usd=30.0,
        source_runner=_source_runner(_source_result()),
        now=datetime(2026, 8, 5, 12, tzinfo=LONDON),
    )
    authorization_path = tmp_path / "CONTROL_AUTHORIZATION.json"
    authorization = json.loads(authorization_path.read_text())
    authorization["structural_authorization"]["root_differences"] = 99
    authorization_path.write_text(json.dumps(authorization))
    calls = []

    with pytest.raises(ValueError, match="authorization artifact hash"):
        staged.run_control_stage(
            output_dir=tmp_path,
            run_id="auth-tamper",
            ledger=_ledger("2026-08-06", opening_usage=105.0),
            total_usage_usd=105.0,
            balance_usd=25.0,
            control_runner=_control_runner(_control_result(), calls),
            bootstrap_samples=10,
            now=datetime(2026, 8, 6, 12, tzinfo=LONDON),
        )

    assert calls == []


def test_later_control_runs_despite_source_science_null(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(staged.base, "validate_predecessors", lambda: {})
    staged.run_source_stage(
        output_dir=tmp_path,
        run_id="science-null",
        ledger=_ledger("2026-08-05"),
        total_usage_usd=100.0,
        balance_usd=30.0,
        source_runner=_source_runner(_source_result(science=False)),
        now=datetime(2026, 8, 5, 12, tzinfo=LONDON),
    )
    calls = []

    result = staged.run_control_stage(
        output_dir=tmp_path,
        run_id="science-null",
        ledger=_ledger("2026-08-06", opening_usage=105.0),
        total_usage_usd=105.0,
        balance_usd=25.0,
        control_runner=_control_runner(_control_result(), calls),
        bootstrap_samples=10,
        now=datetime(2026, 8, 6, 12, tzinfo=LONDON),
    )

    assert calls == ["control"]
    assert result["status"] == "gated_null"
    assert result["decision"] == "complete_composite_endpoint"
    assert result["usage"]["total_requests"] == 6_752
    assert result["protocol"][
        "source_science_was_not_an_authorization_input"
    ]


def test_source_hash_tamper_stops_before_control(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(staged.base, "validate_predecessors", lambda: {})
    staged.run_source_stage(
        output_dir=tmp_path,
        run_id="tamper",
        ledger=_ledger("2026-08-05"),
        total_usage_usd=100.0,
        balance_usd=30.0,
        source_runner=_source_runner(_source_result()),
        now=datetime(2026, 8, 5, 12, tzinfo=LONDON),
    )
    source_path = tmp_path / "source/RESULT.json"
    source_path.write_text(source_path.read_text() + "\n")
    calls = []

    with pytest.raises(ValueError, match="source artifact hashes changed"):
        staged.run_control_stage(
            output_dir=tmp_path,
            run_id="tamper",
            ledger=_ledger("2026-08-06", opening_usage=105.0),
            total_usage_usd=105.0,
            balance_usd=25.0,
            control_runner=_control_runner(_control_result(), calls),
            bootstrap_samples=10,
            now=datetime(2026, 8, 6, 12, tzinfo=LONDON),
        )

    assert calls == []
