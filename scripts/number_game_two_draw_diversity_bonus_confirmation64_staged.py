#!/usr/bin/env python3
"""Run the prospective diversity-bonus confirmation in two daily blocks."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_qwen_fully_fresh_source_control32 as base
from scripts import number_game_two_draw_diversity_bonus_audit as audit
from scripts import number_game_two_draw_diversity_bonus_confirmation32 as component
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-two-draw-diversity-bonus-confirmation64-staged-1"
SOURCE_INTERFACE_PREFIX = (
    "number-game-two-draw-diversity-bonus-confirmation64-source"
)
TREE_COUNT = 64
BLOCK_TREE_COUNT = 32
EXPECTED_REQUESTS_PER_BLOCK = 3_680
EXPECTED_REQUESTS_TOTAL = 7_360
DAILY_CAP_USD = 5.0
MIN_STARTING_BALANCE_USD = 5.0
COMBINED_BOOTSTRAP_SEED = 112_800
PREREGISTRATION = component.PREREGISTRATION
PREREGISTRATION_SHA256 = (
    "49b1a8bd783f8cbf54ba561ec55567bc4143af3b4358397cab4840d7d778cc5d"
)

BLOCKS = {
    "a": {
        "tree_seeds": tuple(range(110_000, 110_032)),
        "target_seeds": tuple(range(110_100, 110_132)),
        "validation_seed_start": 110_200,
        "source_bootstrap_seed": 110_800,
    },
    "b": {
        "tree_seeds": tuple(range(111_000, 111_032)),
        "target_seeds": tuple(range(111_100, 111_132)),
        "validation_seed_start": 111_200,
        "source_bootstrap_seed": 111_800,
    },
}


def block_directory(run_dir: Path, block: str) -> Path:
    return run_dir / f"block_{block}"


def block_stage_path(run_dir: Path, block: str) -> Path:
    return run_dir / f"BLOCK_{block.upper()}_STAGE.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_preregistration() -> None:
    if audit.sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("staged-64 preregistration hash changed")


def _validate_daily_budget(
    *,
    ledger: dict[str, Any],
    total_usage_usd: float,
    balance_usd: float,
    now: datetime | None,
) -> None:
    require_budget(
        ledger,
        projected_cost_usd=DAILY_CAP_USD,
        total_usage_usd=total_usage_usd,
        now=now,
    )
    if balance_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            f"OpenRouter balance ${balance_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} block gate"
        )


def _block_b_authorization(
    *,
    run_dir: Path,
    block_b_date: str,
) -> dict[str, Any]:
    stage = _load(block_stage_path(run_dir, "a"))
    if stage.get("status") != "block_b_authorized":
        raise RuntimeError("Block A did not mechanically authorize Block B")
    if not all((stage.get("mechanics_gates") or {}).values()):
        raise RuntimeError("Block A mechanics are not all passing")
    if str(block_b_date) <= str(stage["calendar_date"]):
        raise RuntimeError("Block B requires a later Europe/London date")
    forbidden = ("brier", "comparison", "scientific", "selected_root")
    serialized = json.dumps(stage, sort_keys=True).lower()
    if any(term in serialized for term in forbidden):
        raise ValueError("Block A authorization contains scientific values")
    return stage


def _source_hashes(source_dir: Path) -> dict[str, str]:
    return {
        "result_sha256": audit.sha256_file(source_dir / "RESULT.json"),
        "trees_sha256": audit.sha256_file(source_dir / "TREES.json"),
        "targets_sha256": audit.sha256_file(source_dir / "TARGETS.json"),
        "raw_sha256": audit.sha256_file(
            source_dir / "private" / "RAW_RESPONSES.json"
        ),
    }


def run_block(
    *,
    run_dir: Path,
    run_id: str,
    block: str,
    ledger: dict[str, Any],
    total_usage_usd: float,
    balance_usd: float,
    source_runner: Callable[..., dict[str, Any]] = base.run_fresh_source,
    now: datetime | None = None,
) -> dict[str, Any]:
    if block not in BLOCKS:
        raise ValueError(f"unknown block: {block}")
    _validate_preregistration()
    base.validate_predecessors()
    _validate_daily_budget(
        ledger=ledger,
        total_usage_usd=total_usage_usd,
        balance_usd=balance_usd,
        now=now,
    )
    if block == "a":
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not run_dir.exists():
            raise FileNotFoundError("Block B requires an existing Block A run")
        _block_b_authorization(
            run_dir=run_dir,
            block_b_date=str(ledger["date"]),
        )
    output_dir = block_directory(run_dir, block)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"block directory is not empty: {output_dir}")
    source_dir = output_dir / "source"
    spec = BLOCKS[block]
    with component.configured_fresh_source(
        tree_seeds=spec["tree_seeds"],
        target_seeds=spec["target_seeds"],
        validation_seed_start=spec["validation_seed_start"],
        bootstrap_seed=spec["source_bootstrap_seed"],
        source_interface_version=f"{SOURCE_INTERFACE_PREFIX}-{block}-1",
    ):
        source_result = source_runner(
            output_dir=source_dir,
            run_id=f"{run_id}-block-{block}-source",
        )
    source_result["protocol"].update(
        {
            "staged_64_confirmation": True,
            "staged_block": block,
            "block_calendar_date": str(ledger["date"]),
        }
    )
    checkpoint(source_dir / "RESULT.json", source_result)
    exact_requests = (
        int(source_result["usage"]["adapter_requests"])
        == EXPECTED_REQUESTS_PER_BLOCK
    )
    mechanics = dict(source_result["mechanics_gates"])
    mechanics["accepted_request_count_exact"] = exact_requests
    mechanics_pass = bool(mechanics) and all(mechanics.values())
    stage = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "block_b_authorized"
            if block == "a" and mechanics_pass
            else "block_complete"
            if block == "b" and mechanics_pass
            else "mechanics_failed"
        ),
        "run_id": run_id,
        "block": block,
        "calendar_date": str(ledger["date"]),
        "authorization_inputs": "source mechanics gates and request count only",
        "source_science_was_not_an_authorization_input": True,
        "mechanics_gates": mechanics,
        "usage": source_result["usage"],
        "source_artifacts": _source_hashes(source_dir),
    }
    checkpoint(block_stage_path(run_dir, block), stage)
    return stage


def _scored_block(run_dir: Path, block: str) -> dict[str, Any]:
    source_dir = block_directory(run_dir, block) / "source"
    scoring = component.score_source_directory(
        source_dir,
        bootstrap_seed=BLOCKS[block]["source_bootstrap_seed"],
    )
    rows = []
    for row in scoring["rows"]:
        item = dict(row)
        item["block"] = block
        item["source"] = f"prospective_block_{block}"
        rows.append(item)
    return scoring | {"rows": rows}


def scientific_gates(summary: dict[str, Any]) -> dict[str, bool]:
    depth_two = summary["comparisons"]["crossfit_depth_two"]
    original = summary["comparisons"]["original_depth_three"]
    return {
        "depth_three_reduction_vs_depth_two_at_least_three_percent": (
            depth_two["relative_brier_reduction"] >= 0.03
        ),
        "depth_three_vs_depth_two_interval_below_zero": (
            depth_two["tree_bootstrap_95pct"][1] < 0.0
        ),
        "depth_three_wins_exceed_losses_vs_depth_two": (
            depth_two["wins"] > depth_two["losses"]
        ),
        "bonus_changes_at_least_sixteen_original_roots": (
            original["changed_roots"] >= 16
        ),
        "bonus_mean_brier_not_worse_than_original": (
            original["mean_candidate_minus_baseline_brier"] <= 0.0
        ),
    }


def build_combined_result(*, run_dir: Path, run_id: str) -> dict[str, Any]:
    stages = {block: _load(block_stage_path(run_dir, block)) for block in BLOCKS}
    scored = {block: _scored_block(run_dir, block) for block in BLOCKS}
    rows = [row for block in BLOCKS for row in scored[block]["rows"]]
    if len(rows) != TREE_COUNT:
        raise ValueError(f"expected {TREE_COUNT} combined rows, got {len(rows)}")
    summary = audit.source_summary(
        rows,
        seed=COMBINED_BOOTSTRAP_SEED,
        include_coefficient_grid=False,
    )
    gates = scientific_gates(summary)
    mechanics_pass = all(
        all(stage["mechanics_gates"].values()) for stage in stages.values()
    )
    status = (
        "passed"
        if mechanics_pass and all(gates.values())
        else "mechanics_failed"
        if not mechanics_pass
        else "gated_null"
    )
    usage = {
        "adapter_requests": sum(
            int(stage["usage"]["adapter_requests"])
            for stage in stages.values()
        ),
        "run_cost_usd": sum(
            float(stage["usage"]["run_cost_usd"])
            for stage in stages.values()
        ),
        "block_cost_usd": {
            block: float(stages[block]["usage"]["run_cost_usd"])
            for block in BLOCKS
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "run_id": run_id,
        "protocol": {
            "analysis_was_preregistered": True,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "blocks_were_mandatory_after_block_a_mechanics": True,
            "block_a_science_was_not_an_authorization_input": True,
            "tree_count": TREE_COUNT,
            "tree_seeds": {
                block: list(BLOCKS[block]["tree_seeds"])
                for block in BLOCKS
            },
            "target_seeds": {
                block: list(BLOCKS[block]["target_seeds"])
                for block in BLOCKS
            },
            "validation_seed_starts": {
                block: BLOCKS[block]["validation_seed_start"]
                for block in BLOCKS
            },
            "source_bootstrap_seeds": {
                block: BLOCKS[block]["source_bootstrap_seed"]
                for block in BLOCKS
            },
            "combined_bootstrap_seed": COMBINED_BOOTSTRAP_SEED,
            "bootstrap_samples": audit.BOOTSTRAP_SAMPLES,
            "diversity_coefficient": audit.DIVERSITY_COEFFICIENT,
            "no_coefficient_sweep_on_fresh_data": True,
            "expected_requests_total": EXPECTED_REQUESTS_TOTAL,
        },
        "usage": usage,
        "block_stages": stages,
        "source_artifacts": {
            block: scored[block]["source_artifacts"] for block in BLOCKS
        },
        "scientific_gates": gates,
        "comparisons": summary["comparisons"],
        "rank_metrics": component._mean_rank_metrics(rows),
        "rows": rows,
    }


def reconcile_ledger(
    *,
    ledger: dict[str, Any],
    block: str,
    measured_cost_usd: float,
    live_after: dict[str, float],
    status: str = "complete",
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    previous = float(updated.get("recorded_actual_spend_usd", 0.0))
    posted = max(0.0, float(live_after["total_usage_usd"]) - opening)
    local = previous + measured_cost_usd
    recorded = max(posted, local)
    updated["recorded_actual_spend_usd"] = recorded
    updated[f"diversity_bonus_confirmation64_block_{block}"] = {
        "status": status,
        "actual_cost_usd": measured_cost_usd,
        "maximum_cost_usd": DAILY_CAP_USD,
    }
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live_after["total_credits_usd"]),
        "live_total_usage_usd": float(live_after["total_usage_usd"]),
        "live_balance_usd": float(live_after["balance_usd"]),
        "posted_spend_since_opening_usd": posted,
        "locally_measured_spend_usd": local,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": max(
            0.0, float(updated["daily_cap_usd"]) - recorded
        ),
    }
    updated["additional_paid_blocks_authorized"] = False
    return updated


def execute_daily_block(
    *,
    run_dir: Path,
    run_id: str,
    block: str,
    ledger_path: Path,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    block_runner: Callable[..., dict[str, Any]] = run_block,
    verifier: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ledger = _load(ledger_path)
    live_before = live_reader()
    stage = block_runner(
        run_dir=run_dir,
        run_id=run_id,
        block=block,
        ledger=ledger,
        total_usage_usd=float(live_before["total_usage_usd"]),
        balance_usd=float(live_before["balance_usd"]),
    )
    measured_cost = float(stage["usage"]["run_cost_usd"])
    local = reconcile_ledger(
        ledger=ledger,
        block=block,
        measured_cost_usd=measured_cost,
        live_after=live_before,
    )
    checkpoint(ledger_path, local)
    scientific_status = None
    verification = None
    if block == "b":
        result = build_combined_result(run_dir=run_dir, run_id=run_id)
        checkpoint(run_dir / "RESULT.json", result)
        scientific_status = result["status"]
        if verifier is None:
            from scripts.number_game_two_draw_diversity_bonus_confirmation64_verify import (
                verify_completed_confirmation,
            )

            verifier = verify_completed_confirmation
        verification = verifier(run_dir=run_dir)
    live_after = live_reader()
    reconciled = reconcile_ledger(
        ledger=local,
        block=block,
        measured_cost_usd=0.0,
        live_after=live_after,
    )
    checkpoint(ledger_path, reconciled)
    execution = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete",
        "run_id": run_id,
        "block": block,
        "block_stage_status": stage["status"],
        "scientific_status": scientific_status,
        "verification_status": (
            verification["status"] if verification is not None else None
        ),
        "measured_cost_usd": measured_cost,
        "measured_requests": int(stage["usage"]["adapter_requests"]),
        "recorded_actual_spend_usd": reconciled[
            "recorded_actual_spend_usd"
        ],
        "remaining_daily_allowance_usd": reconciled["reconciliation"][
            "remaining_daily_allowance_usd"
        ],
        "additional_paid_blocks_authorized": False,
    }
    checkpoint(
        run_dir / f"BLOCK_{block.upper()}_DAILY_EXECUTION.json",
        execution,
    )
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block", choices=tuple(BLOCKS), required=True)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    args = parser.parse_args()
    try:
        execution = execute_daily_block(
            run_dir=args.run_dir,
            run_id=args.run_id,
            block=args.block,
            ledger_path=args.daily_ledger,
        )
    except Exception as exc:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        reconciliation_error = None
        try:
            ledger = _load(args.daily_ledger)
            live = read_live_credits()
            reconciled = reconcile_ledger(
                ledger=ledger,
                block=args.block,
                measured_cost_usd=0.0,
                live_after=live,
                status="failed_closed_posted_spend_reconciled",
            )
            checkpoint(args.daily_ledger, reconciled)
        except Exception as reconcile_exc:
            reconciliation_error = (
                f"{type(reconcile_exc).__name__}: {reconcile_exc}"
            )
        checkpoint(
            args.run_dir / f"BLOCK_{args.block.upper()}_RUNNER_FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ledger_reconciliation_error": reconciliation_error,
            },
        )
        raise
    print(json.dumps(execution, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
