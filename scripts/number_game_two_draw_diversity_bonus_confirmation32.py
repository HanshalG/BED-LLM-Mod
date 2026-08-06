#!/usr/bin/env python3
"""Run the prospective Qwen two-draw diversity-bonus confirmation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_qwen_fully_fresh_source_control32 as base
from scripts import number_game_two_draw_diversity_bonus_audit as audit
from scripts.number_game_ranking_fidelity_audit import spearman_correlation
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-two-draw-diversity-bonus-confirmation32-1"
SOURCE_INTERFACE_VERSION = (
    "number-game-two-draw-diversity-bonus-confirmation32-source-1"
)
TREE_SEEDS = tuple(range(110_000, 110_032))
TARGET_SEEDS = tuple(range(110_100, 110_132))
VALIDATION_SEED_START = 110_200
BOOTSTRAP_SEED = 110_800
TREE_COUNT = 32
EXPECTED_REQUESTS = 3_680
DAILY_CAP_USD = 5.0
MIN_STARTING_BALANCE_USD = 5.0
MIN_RELATIVE_REDUCTION_VS_DEPTH_TWO = 0.03
MIN_WINS_VS_DEPTH_TWO = 14
MIN_CHANGED_ROOTS_VS_ORIGINAL = 8
PREREGISTRATION = REPO_ROOT / (
    "results/nonmyopic/"
    "NUMBER_GAME_TWO_DRAW_DIVERSITY_BONUS_PROSPECTIVE_PREREGISTRATION.md"
)
PREREGISTRATION_SHA256 = (
    "f68839781cf79a51fd7e5e898ec6eff8c2c70a7dbc3c2b4fb2e39835ba261412"
)


@contextmanager
def configured_fresh_source() -> Iterator[None]:
    overrides = {
        "SOURCE_INTERFACE_VERSION": SOURCE_INTERFACE_VERSION,
        "TREE_SEEDS": TREE_SEEDS,
        "TARGET_SEEDS": TARGET_SEEDS,
        "VALIDATION_SEED_START": VALIDATION_SEED_START,
        "SOURCE_BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "SOURCE_RUN_BUDGET_USD": DAILY_CAP_USD,
        "MIN_STARTING_BALANCE_USD": MIN_STARTING_BALANCE_USD,
    }
    originals = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def _comparison(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return summary["comparisons"][name]


def scientific_gates(summary: dict[str, Any]) -> dict[str, bool]:
    depth_two = _comparison(summary, "crossfit_depth_two")
    original = _comparison(summary, "original_depth_three")
    return {
        "depth_three_reduction_vs_depth_two_at_least_three_percent": (
            depth_two["relative_brier_reduction"]
            >= MIN_RELATIVE_REDUCTION_VS_DEPTH_TWO
        ),
        "depth_three_vs_depth_two_interval_below_zero": (
            depth_two["tree_bootstrap_95pct"][1] < 0.0
        ),
        "depth_three_wins_vs_depth_two_at_least_fourteen": (
            depth_two["wins"] >= MIN_WINS_VS_DEPTH_TWO
        ),
        "bonus_changes_at_least_eight_original_roots": (
            original["changed_roots"] >= MIN_CHANGED_ROOTS_VS_ORIGINAL
        ),
        "bonus_mean_brier_not_worse_than_original": (
            original["mean_candidate_minus_baseline_brier"] <= 0.0
        ),
        "bonus_changed_root_wins_exceed_losses": (
            original["wins"] > original["losses"]
        ),
    }


def _mean_rank_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    original_correlations = []
    adjusted_correlations = []
    original_regrets = []
    adjusted_regrets = []
    for row in rows:
        root_rows = row["root_rows"]
        roots = [int(item["root"]) for item in root_rows]
        realized = {
            int(item["root"]): float(item["realized_brier"])
            for item in root_rows
        }
        predicted = {
            int(item["root"]): float(item["predicted_brier"])
            for item in root_rows
        }
        adjusted = {
            int(root): float(value)
            for root, value in row["adjusted_scores"].items()
        }
        original_correlations.append(
            spearman_correlation(
                [predicted[root] for root in roots],
                [realized[root] for root in roots],
            )
        )
        adjusted_correlations.append(
            spearman_correlation(
                [adjusted[root] for root in roots],
                [realized[root] for root in roots],
            )
        )
        oracle_brier = min(realized.values())
        original_regrets.append(
            realized[int(row["original_root"])] - oracle_brier
        )
        adjusted_regrets.append(
            realized[int(row["bonus_root"])] - oracle_brier
        )
    return {
        "original_mean_candidate_root_spearman": statistics.fmean(
            original_correlations
        ),
        "bonus_mean_candidate_root_spearman": statistics.fmean(
            adjusted_correlations
        ),
        "original_mean_candidate_set_oracle_regret": statistics.fmean(
            original_regrets
        ),
        "bonus_mean_candidate_set_oracle_regret": statistics.fmean(
            adjusted_regrets
        ),
    }


def score_source_directory(source_dir: Path) -> dict[str, Any]:
    spec = {
        "name": "prospective_confirmation32",
        "role": "prospective_confirmation",
        "directory": source_dir,
        "tree_count": TREE_COUNT,
        "result_sha256": audit.sha256_file(source_dir / "RESULT.json"),
        "trees_sha256": audit.sha256_file(source_dir / "TREES.json"),
        "raw_sha256": audit.sha256_file(
            source_dir / "private" / "RAW_RESPONSES.json"
        ),
    }
    rows = audit.load_source(
        spec,
        coefficients=(0.0, audit.DIVERSITY_COEFFICIENT),
    )
    summary = audit.source_summary(
        rows,
        seed=BOOTSTRAP_SEED,
        include_coefficient_grid=False,
    )
    public_rows = []
    for row in rows:
        public_row = dict(row)
        public_row.pop("coefficient_grid_roots")
        public_row["adjusted_scores"] = {
            str(root): float(value)
            for root, value in public_row["adjusted_scores"].items()
        }
        public_rows.append(public_row)
    return {
        "source_artifacts": {
            key: spec[key]
            for key in ("result_sha256", "trees_sha256", "raw_sha256")
        },
        "summary": summary,
        "rank_metrics": _mean_rank_metrics(rows),
        "rows": public_rows,
    }


def build_result(
    *,
    run_id: str,
    source_result: dict[str, Any],
    scoring: dict[str, Any],
    calendar_date: str,
) -> dict[str, Any]:
    summary = scoring["summary"]
    gates = scientific_gates(summary)
    mechanics = source_result["mechanics_gates"]
    exact_requests = (
        int(source_result["usage"]["adapter_requests"])
        == EXPECTED_REQUESTS
    )
    status = (
        "passed"
        if all(mechanics.values()) and exact_requests and all(gates.values())
        else "gated_null"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "run_id": run_id,
        "protocol": {
            "analysis_was_preregistered": True,
            "preregistration": str(PREREGISTRATION.relative_to(REPO_ROOT)),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "calendar_date": calendar_date,
            "timezone": "Europe/London",
            "daily_cap_usd": DAILY_CAP_USD,
            "tree_seeds": list(TREE_SEEDS),
            "target_seeds": list(TARGET_SEEDS),
            "validation_seed_start": VALIDATION_SEED_START,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": audit.BOOTSTRAP_SAMPLES,
            "diversity_coefficient": audit.DIVERSITY_COEFFICIENT,
            "accepted_requests_expected": EXPECTED_REQUESTS,
            "no_coefficient_sweep_on_fresh_data": True,
        },
        "usage": source_result["usage"],
        "mechanics_gates": mechanics
        | {"accepted_request_count_exact": exact_requests},
        "scientific_gates": gates,
        "source_original_status": source_result["status"],
        "source_artifacts": scoring["source_artifacts"],
        "comparisons": summary["comparisons"],
        "rank_metrics": scoring["rank_metrics"],
        "rows": scoring["rows"],
    }


def run_confirmation(
    *,
    output_dir: Path,
    run_id: str,
    ledger: dict[str, Any],
    total_usage_usd: float,
    balance_usd: float,
    source_runner: Callable[..., dict[str, Any]] = base.run_fresh_source,
    scorer: Callable[[Path], dict[str, Any]] = score_source_directory,
    now: datetime | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if audit.sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("diversity-bonus preregistration hash changed")
    base.validate_predecessors()
    require_budget(
        ledger,
        projected_cost_usd=DAILY_CAP_USD,
        total_usage_usd=total_usage_usd,
        now=now,
    )
    if balance_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            f"OpenRouter balance ${balance_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} gate"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source"
    with configured_fresh_source():
        source_result = source_runner(
            output_dir=source_dir,
            run_id=f"{run_id}-source",
        )
    scoring = scorer(source_dir)
    result = build_result(
        run_id=run_id,
        source_result=source_result,
        scoring=scoring,
        calendar_date=str(ledger["date"]),
    )
    checkpoint(output_dir / "RESULT.json", result)
    return result


def reconcile_ledger(
    *,
    ledger: dict[str, Any],
    measured_cost_usd: float,
    live_after: dict[str, float],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    previous = float(updated.get("recorded_actual_spend_usd", 0.0))
    posted = max(0.0, float(live_after["total_usage_usd"]) - opening)
    local = previous + measured_cost_usd
    recorded = max(posted, local)
    updated["recorded_actual_spend_usd"] = recorded
    updated["diversity_bonus_confirmation32"] = {
        "status": "complete",
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


def execute_daily(
    *,
    output_dir: Path,
    run_id: str,
    ledger_path: Path,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    runner: Callable[..., dict[str, Any]] = run_confirmation,
    verifier: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    live_before = live_reader()
    result = runner(
        output_dir=output_dir,
        run_id=run_id,
        ledger=ledger,
        total_usage_usd=float(live_before["total_usage_usd"]),
        balance_usd=float(live_before["balance_usd"]),
    )
    measured_cost = float(result["usage"]["run_cost_usd"])
    local = reconcile_ledger(
        ledger=ledger,
        measured_cost_usd=measured_cost,
        live_after=live_before,
    )
    checkpoint(ledger_path, local)
    if verifier is None:
        from scripts.number_game_two_draw_diversity_bonus_confirmation32_verify import (
            verify_completed_confirmation,
        )

        verifier = verify_completed_confirmation
    verification = verifier(run_dir=output_dir)
    live_after = live_reader()
    reconciled = reconcile_ledger(
        ledger=local,
        measured_cost_usd=0.0,
        live_after=live_after,
    )
    checkpoint(ledger_path, reconciled)
    execution = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete",
        "scientific_status": result["status"],
        "run_id": run_id,
        "measured_cost_usd": measured_cost,
        "measured_requests": int(result["usage"]["adapter_requests"]),
        "recorded_actual_spend_usd": reconciled[
            "recorded_actual_spend_usd"
        ],
        "remaining_daily_allowance_usd": reconciled["reconciliation"][
            "remaining_daily_allowance_usd"
        ],
        "additional_paid_blocks_authorized": False,
        "verification_status": verification["status"],
        "verification_sha256": audit.sha256_file(
            output_dir / "VERIFICATION.json"
        ),
        "result_sha256": audit.sha256_file(output_dir / "RESULT.json"),
    }
    checkpoint(output_dir / "DAILY_EXECUTION.json", execution)
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    args = parser.parse_args()
    try:
        execution = execute_daily(
            output_dir=args.output_dir,
            run_id=args.run_id,
            ledger_path=args.daily_ledger,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        reconciliation_error = None
        try:
            ledger = json.loads(
                args.daily_ledger.read_text(encoding="utf-8")
            )
            live = read_live_credits()
            reconciled = reconcile_ledger(
                ledger=ledger,
                measured_cost_usd=0.0,
                live_after=live,
            )
            checkpoint(args.daily_ledger, reconciled)
        except Exception as reconcile_exc:
            reconciliation_error = (
                f"{type(reconcile_exc).__name__}: {reconcile_exc}"
            )
        checkpoint(
            args.output_dir / "RUNNER_FAILURE.json",
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
