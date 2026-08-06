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
