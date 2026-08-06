from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import bongard_openworld_luna_development32_daily_execute as execute
from scripts import bongard_openworld_luna_vlm_development as development


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _now(block_id: str) -> datetime:
    year, month, day = map(
        int, development.BLOCK_EARLIEST_DATES[block_id].split("-")
    )
    return datetime(year, month, day, 10, tzinfo=ZoneInfo(execute.TIMEZONE))


def _catalog() -> dict:
    return {
        "data": [
            {
                "id": development.MODEL_ID,
                "context_length": 1_050_000,
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": ["structured_outputs"],
                "top_provider": {"max_completion_tokens": 128_000},
                "pricing": {
                    "prompt": "0.0000001",
                    "completion": "0.0000006",
                },
            }
        ]
    }


class Harness:
    def __init__(self, root: Path):
        self.root = root
        self.block_dirs = {
            block_id: root / f"block-{block_id}"
            for block_id in development.BLOCK_ORDER
        }
        self.result_paths = {
            block_id: path / "RESULT.json"
            for block_id, path in self.block_dirs.items()
        }
        self.ledger_paths = {
            block_id: root / f"ledger-{block_id}.json"
            for block_id in development.BLOCK_ORDER
        }
        self.protocol_manifest = root / "PROTOCOL_MANIFEST.json"
        self.combined_result = root / "COMBINED_RESULT.json"
        self.executor_calls: list[str] = []
        self.preflight_calls: list[str] = []
        self.analyzer_calls = 0
        self.combined_validator_calls = 0

    @staticmethod
    def manifest_validator(path: Path) -> dict:
        assert path.name == "PROTOCOL_MANIFEST.json"
        return {"verified": True, "manifest_sha256": "manifest-sha"}

    @staticmethod
    def aug10_validator(**_) -> dict:
        return {
            "verified": True,
            "wrapper_result_sha256": "aug10-wrapper-sha",
            "mechanics_result_sha256": "mechanics-sha",
        }

    def fresh_preflight(self, *, block_id: str, live_reader, **_) -> dict:
        self.preflight_calls.append(block_id)
        return {
            "status": "ready_without_paid_calls",
            "live_credits": live_reader(),
        }

    def block_executor(
        self,
        *,
        output_dir: Path,
        block_id: str,
        ledger_path: Path,
        **_,
    ) -> dict:
        self.executor_calls.append(block_id)
        result = {
            "status": "block_mechanics_pass",
            "block_id": block_id,
        }
        _write(output_dir / "RESULT.json", result)
        block_cost = 0.25
        recorded = 0.25
        ledger = {
            "date": development.BLOCK_EARLIEST_DATES[block_id],
            "timezone": execute.TIMEZONE,
            "daily_cap_usd": 5.0,
            "recorded_actual_spend_usd": recorded,
            "account_wide_usage_counts_against_cap": True,
            "unspent_allowance_does_not_roll_over": True,
            "additional_paid_blocks_authorized": False,
            "first_authorized_block": {
                "interface_version": development.INTERFACE_VERSION,
                "block_id": block_id,
                "model": development.MODEL_ID,
                "maximum_cost_usd": development.RUN_BUDGET_USD,
                "actual_cost_usd": block_cost,
                "status": "block_mechanics_pass",
            },
            f"bongard_luna_vlm_development_block_{block_id}": {
                "interface_version": development.INTERFACE_VERSION,
                "model": development.MODEL_ID,
                "maximum_cost_usd": development.RUN_BUDGET_USD,
                "actual_cost_usd": block_cost,
                "status": "block_mechanics_pass",
            },
            "reconciliation": {
                "remaining_daily_allowance_usd": 5.0 - recorded,
            },
        }
        _write(ledger_path, ledger)
        _write(
            output_dir / "EXECUTION.json",
            {
                "interface_version": development.INTERFACE_VERSION,
                "block_id": block_id,
                "status": "complete_reconciled",
                "result_sha256": execute._sha256(output_dir / "RESULT.json"),
                "ledger_sha256": execute._sha256(ledger_path),
            },
        )
        return result

    @staticmethod
    def block_validator(
        *, path: Path, block_id: str, ledger_path: Path
    ) -> dict:
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["block_id"] == block_id
        ledger = execute.validate_block_ledger(
            path=ledger_path, block_id=block_id
        )
        execution_path = path.parent / "EXECUTION.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        assert execution["result_sha256"] == execute._sha256(path)
        assert execution["ledger_sha256"] == ledger["ledger_sha256"]
        return {
            "verified": True,
            "block_id": block_id,
            "result_sha256": execute._sha256(path),
            "raw_responses_sha256": f"raw-{block_id}",
            "execution_sha256": execute._sha256(execution_path),
            "protocol_manifest_sha256": "manifest-sha",
            "ledger_sha256": ledger["ledger_sha256"],
            "recorded_daily_spend_usd": ledger[
                "recorded_daily_spend_usd"
            ],
        }

    def analyzer(self, *, block_results, output_path: Path) -> dict:
        self.analyzer_calls += 1
        assert list(block_results) == [
            self.result_paths[item] for item in development.BLOCK_ORDER
        ]
        result = {
            "status": "development_signal",
            "authorizes_confirmation_preregistration": True,
            "block_sha256": {
                item: execute._sha256(self.result_paths[item])
                for item in development.BLOCK_ORDER
            },
        }
        _write(output_path, result)
        return result

    def combined_validator(self, *, result_path: Path, block_results) -> dict:
        self.combined_validator_calls += 1
        assert list(block_results) == [
            self.result_paths[item] for item in development.BLOCK_ORDER
        ]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return {
            "verified": True,
            "status": result["status"],
            "result_sha256": execute._sha256(result_path),
            "authorizes_confirmation_preregistration": result[
                "authorizes_confirmation_preregistration"
            ],
        }

    def run(self, block_id: str, *, now: datetime | None = None) -> dict:
        return execute.execute_daily_block(
            block_id=block_id,
            block_dir=self.block_dirs[block_id],
            ledger_path=self.ledger_paths[block_id],
            ledger_paths=self.ledger_paths,
            protocol_manifest=self.protocol_manifest,
            aug10_result=self.root / "aug10.json",
            mechanics_result=self.root / "mechanics.json",
            combined_result=self.combined_result,
            block_result_paths=self.result_paths,
            now=now or _now(block_id),
            manifest_validator=self.manifest_validator,
            aug10_validator=self.aug10_validator,
            block_executor=self.block_executor,
            block_validator=self.block_validator,
            analyzer=self.analyzer,
            combined_validator=self.combined_validator,
            live_reader=lambda: {
                "total_credits_usd": 100.0,
                "total_usage_usd": 80.0,
                "balance_usd": 20.0,
            },
            fresh_preflight=self.fresh_preflight,
        )


def test_wrong_date_refuses_before_any_component(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    with pytest.raises(RuntimeError, match="can run only"):
        harness.run("a", now=_now("b"))
    assert harness.executor_calls == []
    assert harness.preflight_calls == []
    assert not harness.block_dirs["a"].exists()


def test_runtime_preflight_is_read_only_and_exact(tmp_path: Path) -> None:
    block_dir = tmp_path / "block-a"
    ledger = tmp_path / "ledger-a.json"
    result = execute.preflight_fresh_block_runtime(
        block_id="a",
        block_dir=block_dir,
        ledger_path=ledger,
        live_reader=lambda: {
            "total_credits_usd": 100.0,
            "total_usage_usd": 80.0,
            "balance_usd": 20.0,
        },
        model_catalog_reader=_catalog,
    )
    assert result["status"] == "ready_without_paid_calls"
    assert result["budget"]["block_maximum_cost_usd"] == 4.75
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0
    assert not block_dir.exists()
    assert not ledger.exists()


def test_runtime_preflight_refuses_low_balance_and_nonpristine_path(
    tmp_path: Path,
) -> None:
    block_dir = tmp_path / "block-a"
    ledger = tmp_path / "ledger-a.json"
    with pytest.raises(RuntimeError, match=r"below the \$5 start gate"):
        execute.preflight_fresh_block_runtime(
            block_id="a",
            block_dir=block_dir,
            ledger_path=ledger,
            live_reader=lambda: {
                "total_credits_usd": 100.0,
                "total_usage_usd": 95.01,
                "balance_usd": 4.99,
            },
            model_catalog_reader=_catalog,
        )
    _write(block_dir / "partial.json", {})
    with pytest.raises(RuntimeError, match="path is not pristine"):
        execute.preflight_fresh_block_runtime(
            block_id="a",
            block_dir=block_dir,
            ledger_path=ledger,
            live_reader=lambda: pytest.fail("dirty path read live"),
            model_catalog_reader=lambda: pytest.fail("dirty path read catalog"),
        )


def test_fresh_block_and_resume_do_not_repeat_model_execution(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    first = harness.run("a")
    second = harness.run("a")
    assert first == second
    assert harness.executor_calls == ["a"]
    assert harness.preflight_calls == ["a"]
    assert first["status"] == "block_complete_verified"
    assert first["combined_endpoint_accessed"] is False
    assert not harness.combined_result.exists()


def test_failed_fresh_runtime_preflight_writes_nothing(tmp_path: Path) -> None:
    harness = Harness(tmp_path)

    def failed_preflight(**_):
        raise RuntimeError("Luna endpoint changed")

    with pytest.raises(
        execute.PreExecutionGateError, match="Luna endpoint changed"
    ):
        execute.execute_daily_block(
            block_id="a",
            block_dir=harness.block_dirs["a"],
            ledger_path=harness.ledger_paths["a"],
            ledger_paths=harness.ledger_paths,
            protocol_manifest=harness.protocol_manifest,
            aug10_result=tmp_path / "aug10.json",
            mechanics_result=tmp_path / "mechanics.json",
            combined_result=harness.combined_result,
            block_result_paths=harness.result_paths,
            now=_now("a"),
            manifest_validator=harness.manifest_validator,
            aug10_validator=harness.aug10_validator,
            block_executor=lambda **_: pytest.fail("development block opened"),
            block_validator=harness.block_validator,
            analyzer=harness.analyzer,
            combined_validator=harness.combined_validator,
            live_reader=lambda: pytest.fail("failed preflight read live"),
            fresh_preflight=failed_preflight,
        )

    assert not harness.block_dirs["a"].exists()
    assert not harness.ledger_paths["a"].exists()


def test_later_block_requires_prior_verified_daily_wrapper(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    with pytest.raises(RuntimeError, match="required prior block a is missing"):
        harness.run("b")
    harness.run("a")
    (harness.block_dirs["a"] / "DAILY_EXECUTION.json").unlink()
    with pytest.raises(RuntimeError, match="required prior daily block a is missing"):
        harness.run("b")
    assert harness.executor_calls == ["a"]


def test_early_combined_artifact_refuses_before_block_execution(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    _write(harness.combined_result, {"forbidden": True})
    with pytest.raises(RuntimeError, match="exists before block D"):
        harness.run("a")
    assert harness.executor_calls == []


def test_combined_artifact_before_block_d_refuses_before_paid_execution(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    for block_id in development.BLOCK_ORDER[:-1]:
        harness.run(block_id)
    _write(harness.combined_result, {"forbidden": True})
    with pytest.raises(RuntimeError, match="before block D completed"):
        harness.run("d")
    assert harness.executor_calls == ["a", "b", "c"]


@pytest.mark.parametrize("artifact", ["partial.txt", "FAILURE.json"])
def test_partial_or_failed_block_refuses_without_rerun(
    tmp_path: Path, artifact: str
) -> None:
    harness = Harness(tmp_path)
    _write(harness.block_dirs["a"] / artifact, {"partial": True})
    with pytest.raises(RuntimeError):
        harness.run("a")
    assert harness.executor_calls == []


def test_block_d_opens_combined_endpoint_once_and_revalidates_on_resume(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    for block_id in development.BLOCK_ORDER:
        result = harness.run(block_id)
        if block_id != "d":
            assert result["combined_endpoint_accessed"] is False
            assert not harness.combined_result.exists()
    assert harness.executor_calls == list(development.BLOCK_ORDER)
    assert harness.analyzer_calls == 1
    assert result["status"] == "complete_combined_verified"
    assert result["combined_endpoint_accessed"] is True
    assert result["authorizes_confirmation_preregistration"] is True
    assert set(result["all_block_verifications"]) == set(
        development.BLOCK_ORDER
    )

    resumed = harness.run("d")
    assert resumed == result
    assert harness.executor_calls == list(development.BLOCK_ORDER)
    assert harness.analyzer_calls == 1
    assert harness.combined_validator_calls == 2


def test_completed_block_d_refuses_missing_combined_artifact(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    for block_id in development.BLOCK_ORDER:
        harness.run(block_id)
    harness.combined_result.unlink()
    with pytest.raises(RuntimeError, match="combined result artifact is missing"):
        harness.run("d")
    assert harness.analyzer_calls == 1


def test_tampered_prior_endpoint_access_refuses_next_block(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.run("a")
    daily_path = harness.block_dirs["a"] / "DAILY_EXECUTION.json"
    daily = json.loads(daily_path.read_text(encoding="utf-8"))
    daily["combined_endpoint_accessed"] = True
    daily["status"] = "complete_combined_verified"
    daily["combined_verification"] = {"verified": True}
    daily["authorizes_confirmation_preregistration"] = False
    _write(daily_path, daily)
    with pytest.raises(RuntimeError, match="banked daily development"):
        harness.run("b")
    assert harness.executor_calls == ["a"]


def test_block_ledger_rejects_overspend(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.run("a")
    ledger_path = harness.ledger_paths["a"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["recorded_actual_spend_usd"] = 5.01
    ledger["reconciliation"]["remaining_daily_allowance_usd"] = 0.0
    _write(ledger_path, ledger)
    with pytest.raises(RuntimeError, match="ledger is invalid"):
        execute.validate_block_ledger(path=ledger_path, block_id="a")
