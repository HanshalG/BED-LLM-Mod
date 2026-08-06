from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import bongard_openworld_luna_aug10_execute as execute
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving


NOW = datetime(2026, 8, 10, 10, tzinfo=ZoneInfo("Europe/London"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _paths(tmp_path: Path) -> dict:
    return {
        "output_dir": tmp_path / "wrapper",
        "serving_dir": tmp_path / "serving",
        "mechanics_dir": tmp_path / "mechanics",
        "daily_ledger": tmp_path / "ledger.json",
    }


def _opening_ledger() -> dict:
    return {
        "date": "2026-08-10",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "recorded_actual_spend_usd": 0.0,
    }


def _serving_runner(calls: list[str], *, status: str = "passed"):
    def run(*, output_dir: Path, ledger_path: Path, **_):
        calls.append("serving")
        _write(output_dir / "private/RAW_RESPONSES.json", {"serving": True})
        result = {
            "status": status,
            "protocol": {
                "interface_version": serving.INTERFACE_VERSION,
                "model": serving.MODEL_ID,
                "actual_candidate_labels_accessed": False,
                "endpoint_labels_accessed": False,
            },
            "usage": {"run_cost_usd": 0.08},
        }
        _write(output_dir / "RESULT.json", result)
        ledger = json.loads(ledger_path.read_text())
        ledger["recorded_actual_spend_usd"] = 0.08
        ledger["bongard_luna_vlm_serving_smoke"] = {
            "status": status,
            "interface_version": serving.INTERFACE_VERSION,
            "model": serving.MODEL_ID,
            "actual_cost_usd": 0.08,
        }
        _write(ledger_path, ledger)
        return result

    return run


def _mechanics_runner(calls: list[str], *, status: str = "mechanics_pass"):
    def run(*, output_dir: Path, ledger_path: Path, **_):
        calls.append("mechanics")
        _write(output_dir / "private/RAW_RESPONSES.json", {"mechanics": True})
        result = {
            "status": status,
            "protocol": {
                "interface_version": mechanics.INTERFACE_VERSION,
                "model": mechanics.MODEL_ID,
                "development_accessed": False,
                "confirmation_accessed": False,
                "sealed_test_accessed": False,
            },
            "usage": {"run_cost_usd": 0.70},
        }
        _write(output_dir / "RESULT.json", result)
        ledger = json.loads(ledger_path.read_text())
        ledger["recorded_actual_spend_usd"] = 0.78
        ledger["bongard_luna_vlm_mechanics_tree"] = {
            "status": status,
            "interface_version": mechanics.INTERFACE_VERSION,
            "model": mechanics.MODEL_ID,
            "actual_cost_usd": 0.70,
        }
        _write(ledger_path, ledger)
        return result

    return run


def _validate(path: Path, *, status: str) -> dict:
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    return {
        "verified": True,
        "status": status,
        "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_responses_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "cost_usd": float(json.loads(path.read_text())["usage"]["run_cost_usd"]),
    }


def _serving_validator(path: Path) -> dict:
    return _validate(path, status=json.loads(path.read_text())["status"])


def _mechanics_validator(path: Path, *, serving_result: Path) -> dict:
    assert serving_result.name == "RESULT.json"
    return _validate(path, status=json.loads(path.read_text())["status"])


def test_fresh_sequence_runs_serving_then_mechanics(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write(paths["daily_ledger"], _opening_ledger())
    calls = []
    result = execute.execute_aug10_sequence(
        **paths,
        now=NOW,
        serving_runner=_serving_runner(calls),
        mechanics_runner=_mechanics_runner(calls),
        serving_validator=_serving_validator,
        mechanics_validator=_mechanics_validator,
    )
    assert calls == ["serving", "mechanics"]
    assert result["status"] == "complete"
    assert result["authorizes_development"] is True
    assert result["recorded_actual_spend_usd"] == pytest.approx(0.78)
    assert set(result["components"]) == {"serving", "mechanics"}


def test_resume_does_not_repeat_banked_components(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write(paths["daily_ledger"], _opening_ledger())
    first_calls = []
    first = execute.execute_aug10_sequence(
        **paths,
        now=NOW,
        serving_runner=_serving_runner(first_calls),
        mechanics_runner=_mechanics_runner(first_calls),
        serving_validator=_serving_validator,
        mechanics_validator=_mechanics_validator,
    )
    second = execute.execute_aug10_sequence(
        **paths,
        now=NOW,
        serving_runner=lambda **_: pytest.fail("serving repeated"),
        mechanics_runner=lambda **_: pytest.fail("mechanics repeated"),
        serving_validator=_serving_validator,
        mechanics_validator=_mechanics_validator,
    )
    assert first_calls == ["serving", "mechanics"]
    assert second == first


def test_completed_wrapper_refuses_changed_private_raw(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write(paths["daily_ledger"], _opening_ledger())
    calls = []
    execute.execute_aug10_sequence(
        **paths,
        now=NOW,
        serving_runner=_serving_runner(calls),
        mechanics_runner=_mechanics_runner(calls),
        serving_validator=_serving_validator,
        mechanics_validator=_mechanics_validator,
    )
    _write(
        paths["mechanics_dir"] / "private/RAW_RESPONSES.json",
        {"mechanics": "changed"},
    )
    with pytest.raises(RuntimeError, match="mechanics component changed"):
        execute.execute_aug10_sequence(
            **paths,
            now=NOW,
            serving_validator=_serving_validator,
            mechanics_validator=_mechanics_validator,
        )


def test_banked_serving_resumes_at_mechanics(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write(paths["daily_ledger"], _opening_ledger())
    serving_calls = []
    _serving_runner(serving_calls)(
        output_dir=paths["serving_dir"],
        ledger_path=paths["daily_ledger"],
    )
    calls = []
    result = execute.execute_aug10_sequence(
        **paths,
        now=NOW,
        serving_runner=lambda **_: pytest.fail("serving repeated"),
        mechanics_runner=_mechanics_runner(calls),
        serving_validator=_serving_validator,
        mechanics_validator=_mechanics_validator,
    )
    assert calls == ["mechanics"]
    assert result["authorizes_development"] is True


def test_serving_gated_null_stops_before_mechanics(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write(paths["daily_ledger"], _opening_ledger())
    calls = []
    result = execute.execute_aug10_sequence(
        **paths,
        now=NOW,
        serving_runner=_serving_runner(calls, status="gated_null"),
        mechanics_runner=lambda **_: pytest.fail("mechanics must remain closed"),
        serving_validator=_serving_validator,
        mechanics_validator=_mechanics_validator,
    )
    assert calls == ["serving"]
    assert result["status"] == "stopped_after_serving_gated_null"
    assert result["authorizes_development"] is False


def test_mechanics_gated_null_completes_without_development_authority(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write(paths["daily_ledger"], _opening_ledger())
    calls = []
    result = execute.execute_aug10_sequence(
        **paths,
        now=NOW,
        serving_runner=_serving_runner(calls),
        mechanics_runner=_mechanics_runner(calls, status="gated_null"),
        serving_validator=_serving_validator,
        mechanics_validator=_mechanics_validator,
    )
    assert result["status"] == "complete"
    assert result["authorizes_development"] is False


def test_partial_unbanked_component_fails_closed(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write(paths["daily_ledger"], _opening_ledger())
    _write(paths["serving_dir"] / "run.log.json", {"partial": True})
    with pytest.raises(RuntimeError, match="partial unbanked"):
        execute.execute_aug10_sequence(
            **paths,
            now=NOW,
            serving_validator=_serving_validator,
            mechanics_validator=_mechanics_validator,
        )


def test_sequence_refuses_wrong_date_before_calls(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write(paths["daily_ledger"], _opening_ledger())
    with pytest.raises(RuntimeError, match="only on August 10"):
        execute.execute_aug10_sequence(
            **paths,
            now=datetime(2026, 8, 9, 10, tzinfo=ZoneInfo("Europe/London")),
            serving_runner=lambda **_: pytest.fail("paid call opened early"),
            serving_validator=_serving_validator,
            mechanics_validator=_mechanics_validator,
        )
