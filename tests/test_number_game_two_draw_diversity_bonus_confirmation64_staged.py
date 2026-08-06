from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_two_draw_diversity_bonus_confirmation64_staged as staged


LONDON = ZoneInfo("Europe/London")


def _ledger(date: str, *, recorded: float = 0.0) -> dict:
    return {
        "date": date,
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": recorded,
    }


def _stage(block: str, date: str, *, mechanics: bool = True) -> dict:
    return {
        "schema_version": 1,
        "interface_version": staged.INTERFACE_VERSION,
        "status": (
            "block_b_authorized"
            if block == "a" and mechanics
            else "block_complete"
            if mechanics
            else "mechanics_failed"
        ),
        "run_id": "staged",
        "block": block,
        "calendar_date": date,
        "authorization_inputs": "source mechanics gates and request count only",
        "source_science_was_not_an_authorization_input": True,
        "mechanics_gates": {"mechanics": mechanics},
        "usage": {
            "adapter_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "run_cost_usd": 4.2,
        },
        "source_artifacts": {"result_sha256": block},
    }


def _summary() -> dict:
    return {
        "comparisons": {
            "crossfit_depth_two": {
                "relative_brier_reduction": 0.05,
                "tree_bootstrap_95pct": [-0.01, -0.001],
                "wins": 30,
                "losses": 18,
            },
            "original_depth_three": {
                "changed_roots": 20,
                "mean_candidate_minus_baseline_brier": -0.001,
            },
        }
    }


def test_preregistration_and_seed_blocks_are_frozen_and_disjoint() -> None:
    assert staged.audit.sha256_file(staged.PREREGISTRATION) == staged.PREREGISTRATION_SHA256
    assert set(staged.BLOCKS["a"]["tree_seeds"]).isdisjoint(
        staged.BLOCKS["b"]["tree_seeds"]
    )
    assert set(staged.BLOCKS["a"]["target_seeds"]).isdisjoint(
        staged.BLOCKS["b"]["target_seeds"]
    )


def test_scientific_gates_use_primary_and_nonworsening_only() -> None:
    assert all(staged.scientific_gates(_summary()).values())
    failed = _summary()
    failed["comparisons"]["crossfit_depth_two"]["wins"] = 17
    failed["comparisons"]["crossfit_depth_two"]["losses"] = 18
    assert not staged.scientific_gates(failed)[
        "depth_three_wins_exceed_losses_vs_depth_two"
    ]


def test_block_b_authorization_uses_only_mechanics_and_later_date(tmp_path) -> None:
    stage = _stage("a", "2026-08-08")
    (tmp_path / "BLOCK_A_STAGE.json").write_text(
        json.dumps(stage), encoding="utf-8"
    )
    assert staged._block_b_authorization(
        run_dir=tmp_path,
        block_b_date="2026-08-09",
    ) == stage
    with pytest.raises(RuntimeError, match="later"):
        staged._block_b_authorization(
            run_dir=tmp_path,
            block_b_date="2026-08-08",
        )


def test_block_refuses_partial_daily_allowance_before_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(staged, "_validate_preregistration", lambda: None)
    monkeypatch.setattr(staged.base, "validate_predecessors", lambda: {})
    calls = []

    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("source must not run")

    with pytest.raises(RuntimeError, match="exceeds today's remaining"):
        staged.run_block(
            run_dir=tmp_path,
            run_id="refuse",
            block="a",
            ledger=_ledger("2026-08-08", recorded=0.01),
            total_usage_usd=100.0,
            balance_usd=20.0,
            source_runner=forbidden,
            now=datetime(2026, 8, 8, 12, tzinfo=LONDON),
        )
    assert calls == []


def test_combined_result_is_the_only_scientific_decision(tmp_path, monkeypatch) -> None:
    for block, date in (("a", "2026-08-08"), ("b", "2026-08-09")):
        (tmp_path / f"BLOCK_{block.upper()}_STAGE.json").write_text(
            json.dumps(_stage(block, date)), encoding="utf-8"
        )
    rows = [
        {
            "tree_seed": index,
            "root_rows": [],
            "adjusted_scores": {},
        }
        for index in range(32)
    ]

    def scored(run_dir, block):
        del run_dir
        return {
            "rows": [row | {"block": block} for row in deepcopy(rows)],
            "source_artifacts": {"result_sha256": block},
        }

    monkeypatch.setattr(staged, "_scored_block", scored)
    monkeypatch.setattr(
        staged.audit,
        "source_summary",
        lambda rows, **kwargs: _summary(),
    )
    monkeypatch.setattr(staged.component, "_mean_rank_metrics", lambda rows: {})
    result = staged.build_combined_result(run_dir=tmp_path, run_id="combined")

    assert result["status"] == "passed"
    assert result["protocol"]["tree_count"] == 64
    assert result["usage"]["adapter_requests"] == 7_360
    assert len(result["rows"]) == 64


def test_block_b_checkpoints_spend_before_scoring_and_verification(
    tmp_path,
    monkeypatch,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(_ledger("2026-08-09")),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
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

    def block_runner(**kwargs):
        del kwargs
        return _stage("b", "2026-08-09")

    def build_combined_result(**kwargs):
        del kwargs
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger["recorded_actual_spend_usd"] == 4.2
        return {"status": "passed"}

    def verifier(*, run_dir: Path):
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger["recorded_actual_spend_usd"] == 4.2
        return {"status": "verified"}

    monkeypatch.setattr(
        staged,
        "build_combined_result",
        build_combined_result,
    )
    execution = staged.execute_daily_block(
        run_dir=run_dir,
        run_id="daily-b",
        block="b",
        ledger_path=ledger_path,
        live_reader=lambda: next(live_values),
        block_runner=block_runner,
        verifier=verifier,
    )

    assert execution["verification_status"] == "verified"
    assert execution["recorded_actual_spend_usd"] == pytest.approx(4.2)


def test_failed_block_reconciliation_records_posted_account_spend() -> None:
    reconciled = staged.reconcile_ledger(
        ledger=_ledger("2026-08-09"),
        block="b",
        measured_cost_usd=0.0,
        live_after={
            "total_credits_usd": 130.0,
            "total_usage_usd": 103.7,
            "balance_usd": 26.3,
        },
        status="failed_closed_posted_spend_reconciled",
    )
    assert reconciled["recorded_actual_spend_usd"] == pytest.approx(3.7)
    assert reconciled["diversity_bonus_confirmation64_block_b"]["status"] == (
        "failed_closed_posted_spend_reconciled"
    )
