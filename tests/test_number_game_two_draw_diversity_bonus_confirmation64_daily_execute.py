from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_two_draw_diversity_bonus_confirmation64_daily_execute as execute
from scripts import number_game_two_draw_diversity_bonus_confirmation64_staged as staged
from scripts import number_game_two_draw_diversity_bonus_confirmation64_verify as verify


LONDON = ZoneInfo("Europe/London")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _control(path: Path, *, verified: bool = True) -> None:
    _write(
        path,
        {
            "status": "complete_verified" if verified else "failed_closed",
            "verification_status": "verified" if verified else None,
            "measured_control_requests": 3072 if verified else 0,
        },
    )


def _live(usage: float = 100.0) -> dict[str, float]:
    return {
        "total_credits_usd": 130.0,
        "total_usage_usd": usage,
        "balance_usd": 130.0 - usage,
    }


def _catalog(*, include_target: bool = True) -> dict:
    model_ids = [execute.PLANNING_MODEL_ID]
    if include_target:
        model_ids.append(execute.TARGET_MODEL_ID)
    return {
        "data": [
            {
                "id": model_id,
                "context_length": 1_000_000,
                "architecture": {"input_modalities": ["text"]},
                "supported_parameters": ["response_format"],
                "top_provider": {"max_completion_tokens": 65_536},
                "pricing": {
                    "prompt": "0.0000003",
                    "completion": "0.0000012",
                },
            }
            for model_id in model_ids
        ]
    }


def _ready_fresh_preflight(**kwargs) -> dict:
    return {
        "status": "ready_without_paid_calls",
        "live_credits": kwargs["live_reader"](),
    }


def _execute_formal_block(**kwargs) -> dict:
    kwargs.setdefault("fresh_preflight", _ready_fresh_preflight)
    return execute.execute_formal_block(**kwargs)


def _preflight(*, block: str, run_dir: Path, ledger: Path, **overrides) -> dict:
    kwargs = {
        "block": block,
        "run_dir": run_dir,
        "ledger_path": ledger,
        "control_execution_path": run_dir.parent / "control.json",
        "live_reader": _live,
        "model_catalog_reader": _catalog,
        "protocol_validator": lambda: {"verified": True},
        "aug7_validator": lambda _: {"verified": True, "status": "complete"},
        "block_a_validator": lambda _: {
            "verified": True,
            "status": "complete",
        },
    }
    kwargs.update(overrides)
    return execute.preflight_formal_block(**kwargs)


def _complete_block_a_predecessor(run_dir: Path, ledger: Path) -> None:
    source_dir = staged.block_directory(run_dir, "a") / "source"
    private_dir = source_dir / "private"
    private_dir.mkdir(parents=True)
    _write(source_dir / "TREES.json", {})
    _write(source_dir / "TARGETS.json", {})
    _write(private_dir / "RAW_RESPONSES.json", {})
    spec = staged.BLOCKS["a"]
    source_result = {
        "protocol": {
            "interface_version": f"{staged.SOURCE_INTERFACE_PREFIX}-a-1",
            "tree_seeds": list(spec["tree_seeds"]),
            "target_seeds": list(spec["target_seeds"]),
            "validation_seeds": verify._expected_validation_seeds(
                spec["validation_seed_start"]
            ),
            "bootstrap_seed": spec["source_bootstrap_seed"],
            "staged_64_confirmation": True,
            "staged_block": "a",
            "block_calendar_date": staged.FORMAL_BLOCK_DATES["a"],
        },
        "usage": {
            "adapter_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "run_cost_usd": 4.2,
        },
        "mechanics_gates": {"mechanics": True},
    }
    _write(source_dir / "RESULT.json", source_result)
    source_artifacts = verify._source_hashes(source_dir)
    stage = {
        "interface_version": staged.INTERFACE_VERSION,
        "status": "block_b_authorized",
        "block": "a",
        "calendar_date": staged.FORMAL_BLOCK_DATES["a"],
        "authorization_inputs": "source mechanics gates and request count only",
        "source_science_was_not_an_authorization_input": True,
        "mechanics_gates": {
            "mechanics": True,
            "accepted_request_count_exact": True,
        },
        "usage": source_result["usage"],
        "source_artifacts": source_artifacts,
    }
    _write(staged.block_stage_path(run_dir, "a"), stage)
    verify.verify_block_a_authorization(run_dir=run_dir)
    _write(
        run_dir / "BLOCK_A_DAILY_EXECUTION.json",
        {
            "status": "complete",
            "run_id": execute.RUN_ID,
            "block": "a",
            "block_stage_status": "block_b_authorized",
            "verification_status": "verified",
            "measured_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "measured_cost_usd": 4.2,
        },
    )
    _write(
        ledger,
        {
            "date": staged.FORMAL_BLOCK_DATES["a"],
            "timezone": "Europe/London",
            "daily_cap_usd": 5.0,
            "recorded_actual_spend_usd": 4.2,
            "first_authorized_block": {
                "interface_version": staged.INTERFACE_VERSION,
                "run_id": execute.RUN_ID,
                "block": "a",
                "status": "complete",
                "actual_cost_usd": 4.2,
            },
            "diversity_bonus_confirmation64_block_a": {
                "status": "complete",
                "actual_cost_usd": 4.2,
            },
        },
    )


def test_protocol_preflight_binds_pre_response_claim_plan(monkeypatch) -> None:
    protocol = execute._verify_protocol_inputs()
    assert protocol["claim_plan_sha256"] == execute.CLAIM_PLAN_SHA256
    monkeypatch.setattr(execute, "CLAIM_PLAN_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="claim plan hash changed"):
        execute._verify_protocol_inputs()


def test_block_a_initializes_exact_ledger_and_executes_once(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)
    calls = []

    def block_executor(**kwargs):
        calls.append(kwargs)
        execution = {
            "status": "complete",
            "block": "a",
            "measured_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "measured_cost_usd": 4.2,
        }
        _write(run_dir / "BLOCK_A_DAILY_EXECUTION.json", execution)
        return execution

    result = _execute_formal_block(
        block="a",
        run_dir=run_dir,
        ledger_path=ledger,
        control_execution_path=control,
        now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
        live_reader=lambda: _live(),
        block_executor=block_executor,
    )

    assert result["status"] == "complete"
    assert len(calls) == 1
    opened = json.loads(ledger.read_text(encoding="utf-8"))
    assert opened["date"] == "2026-08-08"
    assert opened["daily_cap_usd"] == 5.0
    assert opened["opening_total_usage_usd"] == 100.0
    assert opened["first_authorized_block"]["block"] == "a"
    assert opened["first_authorized_block"]["status"] == "complete"
    assert opened["first_authorized_block"]["actual_cost_usd"] == 4.2
    assert not opened["additional_paid_blocks_authorized"]

    _execute_formal_block(
        block="a",
        run_dir=run_dir,
        ledger_path=ledger,
        control_execution_path=control,
        now=datetime(2026, 8, 8, 1, 0, tzinfo=LONDON),
        live_reader=lambda: pytest.fail("resume read live credits"),
        block_executor=lambda **_: pytest.fail("resume repeated block"),
    )
    assert len(calls) == 1


def test_fresh_execution_requires_ready_preflight_before_any_write(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)
    preflight_calls = []

    def waiting_preflight(**kwargs):
        preflight_calls.append(kwargs)
        assert not ledger.exists()
        assert not run_dir.exists()
        return {
            "status": "waiting_for_aug7_control",
            "live_credits": _live(),
        }

    with pytest.raises(
        execute.PreExecutionGateError,
        match="preflight did not authorize execution",
    ):
        execute.execute_formal_block(
            block="a",
            run_dir=run_dir,
            ledger_path=ledger,
            control_execution_path=control,
            now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
            live_reader=lambda: pytest.fail("waiting preflight read live"),
            block_executor=lambda **_: pytest.fail("waiting preflight executed"),
            fresh_preflight=waiting_preflight,
        )

    assert len(preflight_calls) == 1
    assert not ledger.exists()
    assert not run_dir.exists()


def test_fresh_preflight_failure_leaves_no_artifact_or_component_call(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)

    def failed_preflight(**_):
        raise RuntimeError("catalog changed")

    with pytest.raises(execute.PreExecutionGateError, match="catalog changed"):
        execute.execute_formal_block(
            block="a",
            run_dir=run_dir,
            ledger_path=ledger,
            control_execution_path=control,
            now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
            live_reader=lambda: pytest.fail("failed preflight read live"),
            block_executor=lambda **_: pytest.fail("failed preflight executed"),
            fresh_preflight=failed_preflight,
        )

    assert not ledger.exists()
    assert not run_dir.exists()


def test_wrong_day_refuses_before_live_or_ledger(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    _control(control)
    with pytest.raises(RuntimeError, match="only on 2026-08-08"):
        _execute_formal_block(
            block="a",
            run_dir=tmp_path / "run",
            ledger_path=tmp_path / "ledger.json",
            control_execution_path=control,
            now=datetime(2026, 8, 9, 0, 1, tzinfo=LONDON),
            live_reader=lambda: pytest.fail("wrong day read live credits"),
        )
    assert not (tmp_path / "ledger.json").exists()


def test_unverified_aug7_control_refuses_before_opening_day(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    _control(control, verified=False)
    with pytest.raises(RuntimeError, match="not complete and verified"):
        _execute_formal_block(
            block="a",
            run_dir=tmp_path / "run",
            ledger_path=tmp_path / "ledger.json",
            control_execution_path=control,
            now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
            live_reader=lambda: pytest.fail("invalid predecessor read live"),
        )


def test_partial_formal_block_is_never_retried(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    run_dir = tmp_path / "run"
    _control(control)
    _write(run_dir / "block_a" / "source" / "partial.json", {})

    with pytest.raises(RuntimeError, match="partial Block A"):
        _execute_formal_block(
            block="a",
            run_dir=run_dir,
            ledger_path=tmp_path / "ledger.json",
            control_execution_path=control,
            now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
            live_reader=lambda: pytest.fail("partial block read live"),
        )


def test_failed_block_reconciles_posted_spend_and_banks_failure(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)
    live_values = iter([_live(100.0), _live(103.7)])

    with pytest.raises(RuntimeError, match="provider failed"):
        _execute_formal_block(
            block="a",
            run_dir=run_dir,
            ledger_path=ledger,
            control_execution_path=control,
            now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
            live_reader=lambda: next(live_values),
            block_executor=lambda **_: (_ for _ in ()).throw(
                RuntimeError("provider failed")
            ),
        )

    reconciled = json.loads(ledger.read_text(encoding="utf-8"))
    assert reconciled["recorded_actual_spend_usd"] == pytest.approx(3.7)
    assert (
        reconciled["diversity_bonus_confirmation64_block_a"]["status"]
        == "failed_closed_posted_spend_reconciled"
    )
    failure = json.loads(
        (run_dir / "BLOCK_A_RUNNER_FAILURE.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed_closed"
    assert reconciled["first_authorized_block"]["status"] == "failed_closed"


def test_started_ledger_is_never_reused_after_ambiguous_crash(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger_path = tmp_path / "ledger.json"
    _control(control)
    ledger = execute.initialize_daily_ledger(
        path=ledger_path,
        block="a",
        live=_live(),
        local_now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
    )
    ledger["first_authorized_block"]["status"] = "execution_started"
    _write(ledger_path, ledger)

    with pytest.raises(RuntimeError, match="authorization is not reusable"):
        _execute_formal_block(
            block="a",
            run_dir=tmp_path / "run",
            ledger_path=ledger_path,
            control_execution_path=control,
            now=datetime(2026, 8, 8, 0, 2, tzinfo=LONDON),
            live_reader=lambda: _live(),
            block_executor=lambda **_: pytest.fail("ambiguous block rerun"),
        )


def test_block_b_writes_verified_report_before_completion(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)
    _write(run_dir / "BLOCK_A_STAGE.json", {"status": "banked"})
    calls = []

    def block_executor(**_):
        execution = {
            "status": "complete",
            "block": "b",
            "measured_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "measured_cost_usd": 4.2,
        }
        _write(run_dir / "BLOCK_B_DAILY_EXECUTION.json", execution)
        return execution

    def reporter(*, run_dir: Path):
        calls.append(run_dir)
        return {"path": "report.md", "sha256": "report-hash"}

    result = _execute_formal_block(
        block="b",
        run_dir=run_dir,
        ledger_path=ledger,
        control_execution_path=control,
        now=datetime(2026, 8, 9, 0, 1, tzinfo=LONDON),
        live_reader=lambda: _live(),
        block_executor=block_executor,
        reporter=reporter,
    )

    assert calls == [run_dir]
    assert result["verified_report"]["sha256"] == "report-hash"
    stored = json.loads(
        (run_dir / "BLOCK_B_DAILY_EXECUTION.json").read_text(encoding="utf-8")
    )
    assert stored["verified_report"] == result["verified_report"]

    replayed = _execute_formal_block(
        block="b",
        run_dir=run_dir,
        ledger_path=ledger,
        control_execution_path=control,
        now=datetime(2026, 8, 10, 0, 2, tzinfo=LONDON),
        live_reader=lambda: pytest.fail("completed replay read live credits"),
        block_executor=lambda **_: pytest.fail("paid block repeated"),
        reporter=reporter,
    )
    assert calls == [run_dir, run_dir]
    assert replayed["verified_report"] == result["verified_report"]


def test_completed_block_b_refuses_changed_report_record(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)
    _write(run_dir / "BLOCK_A_STAGE.json", {"status": "banked"})

    def block_executor(**_):
        execution = {
            "status": "complete",
            "block": "b",
            "measured_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "measured_cost_usd": 4.2,
        }
        _write(run_dir / "BLOCK_B_DAILY_EXECUTION.json", execution)
        return execution

    _execute_formal_block(
        block="b",
        run_dir=run_dir,
        ledger_path=ledger,
        control_execution_path=control,
        now=datetime(2026, 8, 9, 0, 1, tzinfo=LONDON),
        live_reader=lambda: _live(),
        block_executor=block_executor,
        reporter=lambda **_: {"path": "report.md", "sha256": "original"},
    )
    with pytest.raises(RuntimeError, match="report record changed"):
        _execute_formal_block(
            block="b",
            run_dir=run_dir,
            ledger_path=ledger,
            control_execution_path=control,
            now=datetime(2026, 8, 10, 0, 2, tzinfo=LONDON),
            live_reader=lambda: pytest.fail("completed replay read live"),
            block_executor=lambda **_: pytest.fail("paid block repeated"),
            reporter=lambda **_: {"path": "report.md", "sha256": "changed"},
        )


def test_report_failure_recovers_without_repeating_paid_block(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)
    _write(run_dir / "BLOCK_A_STAGE.json", {"status": "banked"})
    block_calls = []

    def block_executor(**_):
        block_calls.append("b")
        execution = {
            "status": "complete",
            "block": "b",
            "measured_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "measured_cost_usd": 4.2,
        }
        _write(run_dir / "BLOCK_B_DAILY_EXECUTION.json", execution)
        return execution

    with pytest.raises(RuntimeError, match="report failed"):
        _execute_formal_block(
            block="b",
            run_dir=run_dir,
            ledger_path=ledger,
            control_execution_path=control,
            now=datetime(2026, 8, 9, 0, 1, tzinfo=LONDON),
            live_reader=lambda: _live(),
            block_executor=block_executor,
            reporter=lambda **_: (_ for _ in ()).throw(
                RuntimeError("report failed")
            ),
        )

    assert block_calls == ["b"]
    assert not (run_dir / "BLOCK_B_RUNNER_FAILURE.json").exists()
    report_failure = json.loads(
        (run_dir / "BLOCK_B_REPORT_FAILURE.json").read_text(encoding="utf-8")
    )
    assert report_failure["block_execution_remains_complete"]
    current_ledger = json.loads(ledger.read_text(encoding="utf-8"))
    assert current_ledger["first_authorized_block"]["status"] == "complete"

    recovered = _execute_formal_block(
        block="b",
        run_dir=run_dir,
        ledger_path=ledger,
        control_execution_path=control,
        now=datetime(2026, 8, 9, 0, 2, tzinfo=LONDON),
        live_reader=lambda: pytest.fail("report recovery read live credits"),
        block_executor=lambda **_: pytest.fail("paid block repeated"),
        reporter=lambda **_: {"path": "report.md", "sha256": "recovered"},
    )
    assert recovered["verified_report"]["sha256"] == "recovered"
    assert block_calls == ["b"]


def test_block_a_preflight_waits_without_writing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ledger = tmp_path / "ledger.json"
    before = sorted(
        (str(path.relative_to(tmp_path)), path.read_bytes() if path.is_file() else None)
        for path in tmp_path.rglob("*")
    )
    result = _preflight(block="a", run_dir=run_dir, ledger=ledger)
    after = sorted(
        (str(path.relative_to(tmp_path)), path.read_bytes() if path.is_file() else None)
        for path in tmp_path.rglob("*")
    )
    assert after == before
    assert result["status"] == "waiting_for_aug7_control"
    assert result["predecessor"]["verified"] is False
    assert result["budget"]["account_wide_daily_cap_usd"] == 5.0
    assert result["budget"]["expected_requests"] == 3_680
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0


def test_block_a_preflight_becomes_ready_after_verified_control(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    ledger = tmp_path / "ledger.json"
    control = tmp_path / "control.json"
    _write(control, {})
    result = _preflight(
        block="a",
        run_dir=run_dir,
        ledger=ledger,
        control_execution_path=control,
    )
    assert result["status"] == "ready_without_paid_calls"
    assert result["predecessor"]["kind"] == "verified_aug7_control"


def test_block_b_preflight_waits_then_becomes_ready(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ledger = tmp_path / "ledger.json"
    waiting = _preflight(block="b", run_dir=run_dir, ledger=ledger)
    assert waiting["status"] == "waiting_for_block_a"

    _write(run_dir / "BLOCK_A_DAILY_EXECUTION.json", {"status": "complete"})
    ready = _preflight(block="b", run_dir=run_dir, ledger=ledger)
    assert ready["status"] == "ready_without_paid_calls"
    assert ready["predecessor"]["kind"] == "verified_block_a"


def test_preflight_refuses_partial_or_terminal_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ledger = tmp_path / "ledger.json"
    _write(run_dir / "block_a/source/partial.json", {})
    with pytest.raises(RuntimeError, match="partial Block A"):
        _preflight(block="a", run_dir=run_dir, ledger=ledger)


def test_preflight_refuses_existing_target_ledger(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ledger = tmp_path / "ledger.json"
    _write(ledger, {})
    with pytest.raises(RuntimeError, match="daily ledger already exists"):
        _preflight(block="a", run_dir=run_dir, ledger=ledger)


def test_preflight_refuses_low_balance_or_missing_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ledger = tmp_path / "ledger.json"
    with pytest.raises(RuntimeError, match=r"below the \$5 start gate"):
        _preflight(
            block="a",
            run_dir=run_dir,
            ledger=ledger,
            live_reader=lambda: _live(usage=125.001),
        )
    with pytest.raises(RuntimeError, match="does not expose exact model"):
        _preflight(
            block="a",
            run_dir=run_dir,
            ledger=ledger,
            model_catalog_reader=lambda: _catalog(include_target=False),
        )


def test_preflight_rejects_invalid_existing_predecessor(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ledger = tmp_path / "ledger.json"
    control = tmp_path / "control.json"
    _write(control, {})

    def invalid(_: Path) -> dict:
        raise RuntimeError("control replay changed")

    with pytest.raises(RuntimeError, match="control replay changed"):
        _preflight(
            block="a",
            run_dir=run_dir,
            ledger=ledger,
            control_execution_path=control,
            aug7_validator=invalid,
        )


def test_default_block_a_predecessor_replays_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    ledger = tmp_path / "2026-08-08.json"
    _complete_block_a_predecessor(run_dir, ledger)
    monkeypatch.setitem(execute.LEDGER_PATHS, "a", ledger)
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    result = execute._verify_block_a_predecessor(run_dir)
    after = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert result["verified"] is True
    assert result["request_count"] == staged.EXPECTED_REQUESTS_PER_BLOCK
    assert result["cost_usd"] == pytest.approx(4.2)
