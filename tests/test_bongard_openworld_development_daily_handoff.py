from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import bongard_openworld_development_daily_handoff as handoff
from scripts import bongard_openworld_luna_naive_first_link as naive
from scripts import bongard_openworld_luna_vlm_development as development


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _bindings() -> dict:
    return {"verified": True}


def _catalog() -> dict:
    return {
        "data": [
            {
                "id": naive.MODEL_ID,
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": ["reasoning", "response_format"],
                "top_provider": {"max_completion_tokens": 128_000},
                "pricing": {"prompt": "0.0000001", "completion": "0.0000006"},
            }
        ]
    }


class Harness:
    def __init__(self, root: Path) -> None:
        self.output = root / "handoff"
        self.main = root / "main"
        self.naive = root / "naive"
        self.main_ledger = root / "main-ledger.json"
        self.naive_ledger = root / "naive-ledger.json"
        self.calls: list[str] = []
        self.main_validations = 0
        self.naive_validations = 0

    def main_runner(self, *, block_id: str) -> dict:
        self.calls.append(f"main:{block_id}")
        _write(self.main / "RESULT.json", {"status": "block_complete"})
        _write(
            self.main / "DAILY_EXECUTION.json",
            {"status": "block_complete_verified"},
        )
        _write(
            self.main_ledger,
            {
                "opening_total_credits_usd": 100.0,
                "opening_total_usage_usd": 80.0,
                "opening_balance_usd": 20.0,
                "recorded_actual_spend_usd": 2.5,
            },
        )
        return {"status": "block_complete"}

    def main_validator(self, block_id: str) -> dict:
        self.main_validations += 1
        return {
            "verification": {"status": "block_complete", "block": block_id},
            "daily": {"status": "block_complete_verified"},
            "paths": {
                "result": self.main / "RESULT.json",
                "daily_execution": self.main / "DAILY_EXECUTION.json",
                "ledger": self.main_ledger,
            },
        }

    def naive_runner(self, *, block_id: str) -> dict:
        self.calls.append(f"naive:{block_id}")
        _write(self.naive / "RESULT.json", {"status": "passed"})
        _write(self.naive / "EXECUTION.json", {"status": "complete_reconciled"})
        _write(
            self.naive_ledger,
            {"recorded_actual_spend_usd": 2.6},
        )
        return {"status": "passed"}

    def naive_validator(self, block_id: str) -> dict:
        self.naive_validations += 1
        return {
            "verification": {"status": "passed", "block": block_id},
            "paths": {
                "result": self.naive / "RESULT.json",
                "execution": self.naive / "EXECUTION.json",
                "ledger": self.naive_ledger,
            },
            "recorded_daily_spend_usd": 2.6,
        }

    def run(self, **overrides) -> dict:
        kwargs = {
            "block_id": "a",
            "output_dir": self.output,
            "main_block_dir": self.main,
            "naive_block_dir": self.naive,
            "main_runner": self.main_runner,
            "naive_runner": self.naive_runner,
            "main_validator": self.main_validator,
            "naive_validator": self.naive_validator,
            "binding_verifier": _bindings,
        }
        kwargs.update(overrides)
        return handoff.run_daily_handoff(**kwargs)


def test_bound_daily_components_are_exact() -> None:
    result = handoff.verify_bindings()
    assert result["protocol"]["sha256"] == handoff.PROTOCOL_SHA256
    assert result["implementations"] == {
        name: {"path": relative, "sha256": expected}
        for name, (relative, expected) in handoff.BOUND_IMPLEMENTATIONS.items()
    }


@pytest.mark.parametrize(
    "main_status", ["waiting_for_aug10", "ready_without_paid_calls"]
)
def test_preflight_is_read_only_and_reports_combined_cap(
    tmp_path: Path, main_status: str
) -> None:
    output = tmp_path / "handoff"
    naive_dir = tmp_path / "naive"
    naive_ledger = tmp_path / "naive-ledger.json"
    result = handoff.preflight_daily_handoff(
        block_id="a",
        output_dir=output,
        naive_block_dir=naive_dir,
        naive_ledger=naive_ledger,
        main_preflight=lambda **_: {
            "status": main_status,
            "model_calls_made": 0,
            "files_written": 0,
        },
        catalog_reader=_catalog,
        binding_verifier=_bindings,
    )

    assert result["status"] == main_status
    assert result["budget"] == {
        "account_wide_daily_cap_usd": 5.0,
        "main_run_cap_usd": 4.75,
        "naive_run_cap_usd": 0.2,
        "combined_run_cap_usd": 4.95,
        "unallocated_cap_usd": pytest.approx(0.05),
        "naive_reauthorizes_after_observed_main_spend": True,
    }
    assert result["naive_full_preflight_deferred_until_main_complete"] is True
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0
    assert not output.exists()
    assert not naive_dir.exists()
    assert not naive_ledger.exists()


def test_preflight_refuses_nonpristine_naive_path(tmp_path: Path) -> None:
    naive_dir = tmp_path / "naive"
    _write(naive_dir / "partial.json", {"partial": True})
    with pytest.raises(RuntimeError, match="naive block path is not pristine"):
        handoff.preflight_daily_handoff(
            block_id="a",
            output_dir=tmp_path / "handoff",
            naive_block_dir=naive_dir,
            naive_ledger=tmp_path / "ledger.json",
            main_preflight=lambda **_: pytest.fail("main preflight opened"),
            binding_verifier=_bindings,
        )


@pytest.mark.parametrize("block_id", development.BLOCK_ORDER)
def test_complete_handoff_always_runs_main_then_naive(
    tmp_path: Path, block_id: str
) -> None:
    harness = Harness(tmp_path)
    result = harness.run(block_id=block_id)

    assert harness.calls == [f"main:{block_id}", f"naive:{block_id}"]
    assert result["status"] == "paired_daily_complete"
    assert result["main_completed_before_naive"] is True
    assert result["naive_required_regardless_of_main_endpoint"] is True
    assert result["recorded_daily_spend_usd"] == pytest.approx(2.6)
    assert result["authorizes_paid_calls"] is False
    assert result["authorizes_rerun"] is False
    assert result["authorizes_later_block"] is False
    assert result["this_record_authorizes_confirmation"] is False
    assert set(result["components"]) == {
        "main_result",
        "main_daily_execution",
        "main_ledger",
        "naive_result",
        "naive_execution",
        "naive_ledger",
    }


def test_banked_main_failure_never_opens_naive(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    _write(harness.main / "FAILURE.json", {"status": "failed_closed"})
    result = harness.run(
        main_runner=lambda **_: pytest.fail("main repeated"),
        naive_runner=lambda **_: pytest.fail("naive opened"),
    )

    assert result["status"] == "failed_closed"
    assert result["failed_stage"] == "main"
    assert set(result["components"]) == {"main_failure"}
    assert result["authorizes_later_block"] is False


def test_main_exception_with_banked_failure_never_opens_naive(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)

    def fail_main(**_) -> None:
        _write(harness.main / "FAILURE.json", {"status": "failed_closed"})
        raise RuntimeError("main transport failed")

    result = harness.run(
        main_runner=fail_main,
        naive_runner=lambda **_: pytest.fail("naive opened"),
    )

    assert result["status"] == "failed_closed"
    assert result["failed_stage"] == "main"
    assert result["error_type"] == "RuntimeError"


def test_naive_failure_retains_main_and_closes_later_blocks(tmp_path: Path) -> None:
    harness = Harness(tmp_path)

    def fail_naive(**_) -> None:
        _write(harness.naive / "FAILURE.json", {"status": "failed_closed"})
        raise RuntimeError("naive transport failed")

    result = harness.run(naive_runner=fail_naive)

    assert harness.calls == ["main:a"]
    assert result["status"] == "failed_closed"
    assert result["failed_stage"] == "naive"
    assert result["error_type"] == "RuntimeError"
    assert set(result["components"]) == {
        "main_result",
        "main_daily_execution",
        "main_ledger",
        "naive_failure",
    }
    assert result["authorizes_later_block"] is False
    assert result["this_record_authorizes_confirmation"] is False


def test_unbanked_child_exception_does_not_create_handoff(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    with pytest.raises(RuntimeError, match="unexpected main exception"):
        harness.run(
            main_runner=lambda **_: (_ for _ in ()).throw(
                RuntimeError("unexpected main exception")
            )
        )
    assert not harness.output.exists()


def test_existing_handoff_revalidates_without_paid_runner(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    first = harness.run()
    calls_after_first = list(harness.calls)
    replay = harness.run(
        main_runner=lambda **_: pytest.fail("main repeated"),
        naive_runner=lambda **_: pytest.fail("naive repeated"),
    )

    assert replay == first
    assert harness.calls == calls_after_first
    assert harness.main_validations == 2
    assert harness.naive_validations == 2


def test_existing_handoff_rejects_changed_naive_ledger(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.run()
    _write(harness.naive_ledger, {"recorded_actual_spend_usd": 4.0})

    with pytest.raises(ValueError, match="component changed"):
        harness.run(
            main_runner=lambda **_: pytest.fail("main repeated"),
            naive_runner=lambda **_: pytest.fail("naive repeated"),
        )
