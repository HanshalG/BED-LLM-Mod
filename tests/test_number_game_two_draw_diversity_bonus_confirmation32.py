from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_two_draw_diversity_bonus_confirmation32 as confirmation


LONDON = ZoneInfo("Europe/London")


def _summary(*, depth_two_difference: float = -0.006) -> dict:
    return {
        "comparisons": {
            "crossfit_depth_two": {
                "relative_brier_reduction": 0.05,
                "tree_bootstrap_95pct": [-0.01, -0.001],
                "wins": 15,
                "losses": 8,
                "changed_roots": 23,
                "mean_candidate_minus_baseline_brier": depth_two_difference,
            },
            "original_depth_three": {
                "relative_brier_reduction": 0.01,
                "tree_bootstrap_95pct": [-0.004, 0.001],
                "wins": 8,
                "losses": 4,
                "changed_roots": 12,
                "mean_candidate_minus_baseline_brier": -0.001,
            },
        }
    }


def _ledger(*, recorded: float = 0.0) -> dict:
    return {
        "date": "2026-08-08",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": recorded,
    }


def _source_result() -> dict:
    return {
        "status": "gated_null",
        "protocol": {},
        "mechanics_gates": {"mechanics": True},
        "usage": {
            "adapter_requests": confirmation.EXPECTED_REQUESTS,
            "run_cost_usd": 4.2,
        },
    }


def test_fresh_seed_configuration_is_scoped() -> None:
    original = {
        "trees": confirmation.base.TREE_SEEDS,
        "targets": confirmation.base.TARGET_SEEDS,
        "validation": confirmation.base.VALIDATION_SEED_START,
    }
    with confirmation.configured_fresh_source():
        assert confirmation.base.TREE_SEEDS == confirmation.TREE_SEEDS
        assert confirmation.base.TARGET_SEEDS == confirmation.TARGET_SEEDS
        assert (
            confirmation.base.VALIDATION_SEED_START
            == confirmation.VALIDATION_SEED_START
        )
    assert confirmation.base.TREE_SEEDS == original["trees"]
    assert confirmation.base.TARGET_SEEDS == original["targets"]
    assert confirmation.base.VALIDATION_SEED_START == original["validation"]


def test_preregistration_hash_is_frozen() -> None:
    assert (
        confirmation.audit.sha256_file(confirmation.PREREGISTRATION)
        == confirmation.PREREGISTRATION_SHA256
    )


def test_scientific_gates_encode_monotonic_depth_and_nonworsening() -> None:
    assert all(confirmation.scientific_gates(_summary()).values())
    failed = _summary()
    failed["comparisons"]["crossfit_depth_two"][
        "tree_bootstrap_95pct"
    ] = [-0.01, 0.001]
    assert not confirmation.scientific_gates(failed)[
        "depth_three_vs_depth_two_interval_below_zero"
    ]


def test_budget_refuses_before_source_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        confirmation.audit,
        "sha256_file",
        lambda path: confirmation.PREREGISTRATION_SHA256,
    )
    monkeypatch.setattr(confirmation.base, "validate_predecessors", lambda: {})
    calls = []

    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("source must not run")

    with pytest.raises(RuntimeError, match="exceeds today's remaining"):
        confirmation.run_confirmation(
            output_dir=tmp_path,
            run_id="refuse",
            ledger=_ledger(recorded=0.01),
            total_usage_usd=100.0,
            balance_usd=20.0,
            source_runner=forbidden,
            now=datetime(2026, 8, 8, 12, tzinfo=LONDON),
        )
    assert calls == []


def test_confirmation_builds_frozen_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        confirmation.audit,
        "sha256_file",
        lambda path: confirmation.PREREGISTRATION_SHA256,
    )
    monkeypatch.setattr(confirmation.base, "validate_predecessors", lambda: {})

    def source_runner(*, output_dir: Path, run_id: str):
        del run_id
        output_dir.mkdir(parents=True)
        return deepcopy(_source_result())

    def scorer(source_dir: Path):
        assert source_dir == tmp_path / "source"
        return {
            "source_artifacts": {"result_sha256": "a"},
            "summary": _summary(),
            "rank_metrics": {},
            "rows": [],
        }

    result = confirmation.run_confirmation(
        output_dir=tmp_path,
        run_id="confirm",
        ledger=_ledger(),
        total_usage_usd=100.0,
        balance_usd=20.0,
        source_runner=source_runner,
        scorer=scorer,
        now=datetime(2026, 8, 8, 12, tzinfo=LONDON),
    )

    assert result["status"] == "passed"
    assert result["protocol"]["diversity_coefficient"] == -0.5
    assert result["protocol"]["tree_seeds"] == list(confirmation.TREE_SEEDS)
    assert (tmp_path / "RESULT.json").exists()


def test_reconcile_ledger_uses_larger_posted_account_spend() -> None:
    reconciled = confirmation.reconcile_ledger(
        ledger=_ledger(),
        measured_cost_usd=4.2,
        live_after={
            "total_credits_usd": 130.0,
            "total_usage_usd": 104.5,
            "balance_usd": 25.5,
        },
    )
    assert reconciled["recorded_actual_spend_usd"] == 4.5
    assert reconciled["reconciliation"]["remaining_daily_allowance_usd"] == 0.5
    assert not reconciled["additional_paid_blocks_authorized"]


def test_source_scoring_uses_only_frozen_coefficients(
    tmp_path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "source"
    (source_dir / "private").mkdir(parents=True)
    for path in (
        source_dir / "RESULT.json",
        source_dir / "TREES.json",
        source_dir / "private/RAW_RESPONSES.json",
    ):
        path.write_text("{}", encoding="utf-8")
    calls = {}
    rows = [
        {
            "coefficient_grid_roots": {"0.0": 1, "-0.5": 2},
            "adjusted_scores": {1: 0.0, 2: -1.0},
        }
    ]

    def load_source(spec, *, coefficients):
        calls["coefficients"] = coefficients
        return deepcopy(rows)

    def source_summary(value, *, seed, include_coefficient_grid):
        calls["summary"] = (seed, include_coefficient_grid)
        assert value == rows
        return {"comparisons": {}}

    monkeypatch.setattr(confirmation.audit, "sha256_file", lambda path: "h")
    monkeypatch.setattr(confirmation.audit, "load_source", load_source)
    monkeypatch.setattr(confirmation.audit, "source_summary", source_summary)
    monkeypatch.setattr(confirmation, "_mean_rank_metrics", lambda value: {})

    scored = confirmation.score_source_directory(source_dir)

    assert calls["coefficients"] == (0.0, -0.5)
    assert calls["summary"] == (confirmation.BOOTSTRAP_SEED, False)
    assert "coefficient_grid_roots" not in scored["rows"][0]
    assert scored["rows"][0]["adjusted_scores"] == {
        "1": 0.0,
        "2": -1.0,
    }


def test_daily_execution_checkpoints_spend_before_verification(
    tmp_path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(_ledger()), encoding="utf-8")
    run_dir = tmp_path / "run"
    live_values = iter(
        [
            {
                "total_credits_usd": 130.0,
                "total_usage_usd": 100.0,
                "balance_usd": 30.0,
            },
            {
                "total_credits_usd": 130.0,
                "total_usage_usd": 104.2,
                "balance_usd": 25.8,
            },
        ]
    )

    def runner(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True)
        (output_dir / "RESULT.json").write_text("{}", encoding="utf-8")
        return {
            "status": "passed",
            "usage": {
                "run_cost_usd": 4.2,
                "adapter_requests": confirmation.EXPECTED_REQUESTS,
            },
        }

    def verifier(*, run_dir: Path):
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger["recorded_actual_spend_usd"] == 4.2
        result = {"status": "verified"}
        (run_dir / "VERIFICATION.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    execution = confirmation.execute_daily(
        output_dir=run_dir,
        run_id="daily",
        ledger_path=ledger_path,
        live_reader=lambda: next(live_values),
        runner=runner,
        verifier=verifier,
    )

    assert execution["verification_status"] == "verified"
    assert execution["recorded_actual_spend_usd"] == pytest.approx(4.2)
