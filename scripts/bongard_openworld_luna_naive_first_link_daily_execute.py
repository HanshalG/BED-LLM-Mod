#!/usr/bin/env python3
"""Execute the frozen Luna naive first-link smoke or daily choice block."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_development32_daily_execute as main_daily
from scripts import bongard_openworld_luna_naive_first_link as naive
from scripts import bongard_openworld_luna_naive_first_link_verify as protocol_verify
from scripts import bongard_openworld_luna_vlm_development as development
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-naive-first-link-daily-execute-3"
TIMEZONE = "Europe/London"
SMOKE_DATE = "2026-08-08"
ROOT = REPO_ROOT / "results/nonmyopic/bongard_openworld_luna_naive_first_link"
SMOKE_DIR = ROOT / "smoke-20260808"
SMOKE_RESULT = SMOKE_DIR / "RESULT.json"
SMOKE_LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-08-naive-first-link.json"
)
BLOCK_DIRS = {
    block_id: ROOT
    / f"block-{block_id}-{development.BLOCK_EARLIEST_DATES[block_id].replace('-', '')}"
    for block_id in development.BLOCK_ORDER
}
SUPPLEMENTAL_LEDGERS = {
    block_id: REPO_ROOT
    / (
        "results/nonmyopic/openrouter_daily_budget/"
        f"{development.BLOCK_EARLIEST_DATES[block_id]}-naive-first-link.json"
    )
    for block_id in development.BLOCK_ORDER
}
MIN_RESERVED_PROMPT_TOKENS = 30_000


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _date(now: datetime | None = None) -> str:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    return local.date().isoformat()


def _validate_date(expected: str, now: datetime | None = None) -> None:
    if _date(now) != expected:
        raise RuntimeError(f"naive first-link stage can run only on {expected}")


def _validate_model_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        row for row in catalog.get("data", []) if row.get("id") == naive.MODEL_ID
    ]
    if len(matches) != 1:
        raise RuntimeError("exact Luna endpoint is unavailable")
    model = matches[0]
    architecture = model.get("architecture") or {}
    modalities = set(architecture.get("input_modalities") or [])
    supported = set(model.get("supported_parameters") or [])
    top = model.get("top_provider") or {}
    if not {"text", "image"}.issubset(modalities):
        raise RuntimeError("Luna no longer supports multimodal prompts")
    if "reasoning" not in supported and "reasoning_effort" not in supported:
        raise RuntimeError("Luna no longer advertises reasoning control")
    if not ({"response_format", "structured_outputs"} & supported):
        raise RuntimeError("Luna no longer advertises structured output")
    if int(top.get("max_completion_tokens") or 0) < naive.MAX_TOKENS:
        raise RuntimeError("Luna completion limit is below the frozen baseline")
    pricing = model.get("pricing") or {}
    prompt = float(pricing.get("prompt", math.nan))
    completion = float(pricing.get("completion", math.nan))
    if not all(math.isfinite(value) and value >= 0 for value in (prompt, completion)):
        raise RuntimeError("Luna live pricing is invalid")
    residual = naive.MAX_REQUEST_COST_USD - completion * naive.MAX_TOKENS
    covered_prompt = math.inf if prompt == 0 else residual / prompt
    if residual < 0 or covered_prompt + 1e-9 < MIN_RESERVED_PROMPT_TOKENS:
        raise RuntimeError(
            "naive per-attempt reservation no longer covers prompt/output"
        )
    return {
        "id": model["id"],
        "input_modalities": sorted(modalities),
        "reasoning_supported": True,
        "structured_output_supported": True,
        "max_completion_tokens": int(top["max_completion_tokens"]),
        "prompt_usd_per_million_tokens": prompt * 1_000_000,
        "completion_usd_per_million_tokens": completion * 1_000_000,
        "maximum_request_cost_usd": naive.MAX_REQUEST_COST_USD,
        "covered_prompt_tokens_at_live_price": covered_prompt,
    }


def _day_spent(
    *, opening_usage: float, recorded_spend: float, live_usage: float
) -> float:
    return max(recorded_spend, max(0.0, live_usage - opening_usage))


def _preflight_common(
    *,
    expected_date: str,
    output_dir: Path,
    supplemental_ledger: Path,
    opening_usage: float,
    recorded_spend: float,
    live_reader: Callable[[], dict[str, float]],
    catalog_reader: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    baseline_protocol = protocol_verify.verify_protocol_manifest()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise RuntimeError(f"naive output path is not pristine: {output_dir}")
    if supplemental_ledger.exists():
        raise RuntimeError("naive supplemental daily ledger already exists")
    manifest = development.verify_protocol_manifest(naive.MAIN_MANIFEST)
    if manifest["manifest_sha256"] != naive.MAIN_MANIFEST_SHA256:
        raise RuntimeError("main Bongard development manifest changed")
    model = _validate_model_catalog(catalog_reader())
    live = live_reader()
    spent = _day_spent(
        opening_usage=opening_usage,
        recorded_spend=recorded_spend,
        live_usage=float(live["total_usage_usd"]),
    )
    if spent + naive.RUN_BUDGET_USD > 5.0 + 1e-12:
        raise RuntimeError("naive baseline cap exceeds remaining account-wide day")
    if float(live["balance_usd"]) + 1e-12 < naive.RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below naive run cap")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": expected_date,
        "model": model,
        "naive_protocol_manifest_sha256": baseline_protocol[
            "manifest_sha256"
        ],
        "main_development_manifest_sha256": manifest["manifest_sha256"],
        "live_credits": live,
        "budget": {
            "daily_cap_usd": 5.0,
            "spent_before_naive_usd": spent,
            "naive_run_cap_usd": naive.RUN_BUDGET_USD,
            "remaining_after_full_naive_cap_usd": 5.0
            - spent
            - naive.RUN_BUDGET_USD,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def preflight_smoke(
    *,
    now: datetime | None = None,
    output_dir: Path = SMOKE_DIR,
    supplemental_ledger: Path = SMOKE_LEDGER,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = aug10.read_openrouter_model_catalog,
) -> dict[str, Any]:
    _validate_date(SMOKE_DATE, now)
    live = live_reader()
    return _preflight_common(
        expected_date=SMOKE_DATE,
        output_dir=output_dir,
        supplemental_ledger=supplemental_ledger,
        opening_usage=float(live["total_usage_usd"]),
        recorded_spend=0.0,
        live_reader=lambda: live,
        catalog_reader=catalog_reader,
    )


def _main_predecessor(block_id: str) -> dict[str, Any]:
    result_path = main_daily.BLOCK_DIRS[block_id] / "RESULT.json"
    verification = main_daily.validate_block_result(
        path=result_path,
        block_id=block_id,
        ledger_path=main_daily.LEDGERS[block_id],
    )
    daily_path = result_path.parent / "DAILY_EXECUTION.json"
    if not daily_path.is_file():
        raise RuntimeError("main development daily execution is absent")
    return {
        "result_path": str(result_path),
        "result_sha256": verification["result_sha256"],
        "ledger_path": str(main_daily.LEDGERS[block_id]),
        "ledger_sha256": verification["ledger_sha256"],
        "daily_execution_path": str(daily_path),
        "daily_execution_sha256": development.sha256_file(daily_path),
    }


def _naive_predecessor(block_id: str) -> dict[str, Any]:
    result_path = BLOCK_DIRS[block_id] / "RESULT.json"
    ledger_path = SUPPLEMENTAL_LEDGERS[block_id]
    execution_path = result_path.parent / "EXECUTION.json"
    if not all(path.is_file() for path in (result_path, ledger_path, execution_path)):
        raise RuntimeError(f"required prior naive block {block_id} is missing")
    verification = naive.verify_block_result(
        result_path,
        smoke_result=SMOKE_RESULT,
    )
    ledger = _load(ledger_path)
    execution = _load(execution_path)
    baseline = ledger.get("naive_first_link") or {}
    recorded = float(ledger.get("recorded_actual_spend_usd", math.inf))
    actual = float(baseline.get("actual_cost_usd", math.inf))
    if (
        ledger.get("interface_version") != INTERFACE_VERSION
        or ledger.get("date") != development.BLOCK_EARLIEST_DATES[block_id]
        or float(ledger.get("daily_cap_usd", 0.0)) != 5.0
        or ledger.get("account_wide_usage_counts_against_cap") is not True
        or ledger.get("opening_boundary_derived_from_main_reconciled_ledger")
        is not True
        or ledger.get("opening_boundary_amendment_sha256")
        != main_daily.BUDGET_CHAIN_AMENDMENT_SHA256
        or baseline.get("status") != "passed"
        or baseline.get("model") != naive.MODEL_ID
        or baseline.get("reasoning_effort") != naive.REASONING_EFFORT
        or not math.isfinite(actual)
        or not 0.0 <= actual <= naive.RUN_BUDGET_USD
        or not math.isfinite(recorded)
        or not 0.0 <= recorded <= 5.0
        or execution.get("interface_version") != INTERFACE_VERSION
        or execution.get("status") != "complete_reconciled"
        or execution.get("result_sha256") != verification["result_sha256"]
        or execution.get("ledger_sha256")
        != development.sha256_file(ledger_path)
    ):
        raise RuntimeError(f"required prior naive block {block_id} is invalid")
    return {
        "block_id": block_id,
        "result_sha256": verification["result_sha256"],
        "raw_responses_sha256": verification["raw_responses_sha256"],
        "ledger_sha256": development.sha256_file(ledger_path),
        "execution_sha256": development.sha256_file(execution_path),
    }


def preflight_block(
    *,
    block_id: str,
    now: datetime | None = None,
    output_dir: Path | None = None,
    supplemental_ledger: Path | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = aug10.read_openrouter_model_catalog,
) -> dict[str, Any]:
    if block_id not in development.BLOCK_ORDER:
        raise ValueError("unknown naive development block")
    expected_date = development.BLOCK_EARLIEST_DATES[block_id]
    _validate_date(expected_date, now)
    smoke = naive.verify_smoke_result(SMOKE_RESULT)
    predecessor = _main_predecessor(block_id)
    target_index = development.BLOCK_ORDER.index(block_id)
    prior_naive_blocks = [
        _naive_predecessor(prior_id)
        for prior_id in development.BLOCK_ORDER[:target_index]
    ]
    for future_id in development.BLOCK_ORDER[target_index + 1 :]:
        future_dir = BLOCK_DIRS[future_id]
        if future_dir.exists() and (
            not future_dir.is_dir() or any(future_dir.iterdir())
        ):
            raise RuntimeError(f"future naive block {future_id} already exists")
        if SUPPLEMENTAL_LEDGERS[future_id].exists():
            raise RuntimeError(f"future naive ledger {future_id} already exists")
    main_ledger = _load(main_daily.LEDGERS[block_id])
    result = _preflight_common(
        expected_date=expected_date,
        output_dir=output_dir or BLOCK_DIRS[block_id],
        supplemental_ledger=(
            supplemental_ledger or SUPPLEMENTAL_LEDGERS[block_id]
        ),
        opening_usage=float(main_ledger["opening_total_usage_usd"]),
        recorded_spend=float(main_ledger["recorded_actual_spend_usd"]),
        live_reader=live_reader,
        catalog_reader=catalog_reader,
    )
    result.update(
        {
            "block_id": block_id,
            "smoke_result_sha256": smoke["result_sha256"],
            "main_predecessor": predecessor,
            "prior_naive_blocks": prior_naive_blocks,
        }
    )
    return result


def _initial_ledger(
    *,
    preflight: Mapping[str, Any],
    opening_usage: float,
    opening_credits: float,
    opening_balance: float,
    predecessor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    spent = float(preflight["budget"]["spent_before_naive_usd"])
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "date": preflight["date"],
        "timezone": TIMEZONE,
        "daily_cap_usd": 5.0,
        "opening_total_credits_usd": opening_credits,
        "opening_total_usage_usd": opening_usage,
        "opening_balance_usd": opening_balance,
        "recorded_actual_spend_usd": spent,
        "account_wide_usage_counts_against_cap": True,
        "opening_boundary_derived_from_main_reconciled_ledger": (
            predecessor is not None
        ),
        "opening_boundary_amendment_sha256": (
            main_daily.BUDGET_CHAIN_AMENDMENT_SHA256
            if predecessor is not None
            else None
        ),
        "execution_opening_total_usage_usd": float(
            preflight["live_credits"]["total_usage_usd"]
        ),
        "execution_opening_balance_usd": float(
            preflight["live_credits"]["balance_usd"]
        ),
        "unspent_allowance_does_not_roll_over": True,
        "main_predecessor": predecessor,
        "naive_first_link": {
            "status": "authorized_pending",
            "maximum_cost_usd": naive.RUN_BUDGET_USD,
            "maximum_request_cost_usd": naive.MAX_REQUEST_COST_USD,
            "model": naive.MODEL_ID,
            "reasoning_effort": naive.REASONING_EFFORT,
        },
    }


def _reconcile(
    *,
    ledger: Mapping[str, Any],
    measured_cost: float,
    live: Mapping[str, float],
    status: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    prior = float(updated["recorded_actual_spend_usd"])
    posted = max(0.0, float(live["total_usage_usd"]) - opening)
    recorded = max(posted, prior + measured_cost)
    if recorded > 5.0 + 1e-12:
        raise RuntimeError("naive reconciliation exceeds daily cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["naive_first_link"].update(
        {"status": status, "actual_cost_usd": measured_cost}
    )
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live["total_credits_usd"]),
        "live_total_usage_usd": float(live["total_usage_usd"]),
        "live_balance_usd": float(live["balance_usd"]),
        "posted_spend_since_day_opening_usd": posted,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": 5.0 - recorded,
    }
    return updated


def _execute(
    *,
    preflight: Mapping[str, Any],
    output_dir: Path,
    ledger_path: Path,
    run_id: str,
    runner: Callable[..., dict[str, Any]],
    runner_kwargs: Mapping[str, Any],
    live_reader: Callable[[], dict[str, float]],
) -> dict[str, Any]:
    live = preflight["live_credits"]
    predecessor = preflight.get("main_predecessor")
    if predecessor:
        main_ledger_path = Path(predecessor["ledger_path"])
        main_ledger = _load(main_ledger_path)
        if (
            development.sha256_file(main_ledger_path)
            != predecessor["ledger_sha256"]
        ):
            raise RuntimeError("main development ledger changed before naive dispatch")
        opening_usage = float(main_ledger["opening_total_usage_usd"])
        opening_credits = float(main_ledger["opening_total_credits_usd"])
        opening_balance = float(main_ledger["opening_balance_usd"])
        live = live_reader()
        live_usage = float(live["total_usage_usd"])
        live_balance = float(live["balance_usd"])
        if not all(
            math.isfinite(value)
            for value in (
                float(live["total_credits_usd"]),
                live_usage,
                live_balance,
            )
        ):
            raise RuntimeError("live OpenRouter credit values are non-finite")
        if live_usage + 1e-12 < opening_usage:
            raise RuntimeError("live cumulative usage is below the main boundary")
        spent = _day_spent(
            opening_usage=opening_usage,
            recorded_spend=float(main_ledger["recorded_actual_spend_usd"]),
            live_usage=live_usage,
        )
        if spent + naive.RUN_BUDGET_USD > 5.0 + 1e-12:
            raise RuntimeError(
                "naive baseline cap exceeds the remaining account-wide day"
            )
        if live_balance + 1e-12 < naive.RUN_BUDGET_USD:
            raise RuntimeError("OpenRouter balance is below naive run cap")
        preflight = json.loads(json.dumps(preflight))
        preflight["live_credits"] = live
        preflight["budget"]["spent_before_naive_usd"] = spent
        preflight["budget"]["remaining_after_full_naive_cap_usd"] = (
            5.0 - spent - naive.RUN_BUDGET_USD
        )
    else:
        opening_usage = float(live["total_usage_usd"])
        opening_credits = float(live["total_credits_usd"])
        opening_balance = float(live["balance_usd"])
    ledger = _initial_ledger(
        preflight=preflight,
        opening_usage=opening_usage,
        opening_credits=opening_credits,
        opening_balance=opening_balance,
        predecessor=predecessor,
    )
    checkpoint(ledger_path, ledger)
    try:
        result = runner(output_dir=output_dir, run_id=run_id, **runner_kwargs)
    except Exception as exc:
        reconciled = _reconcile(
            ledger=ledger,
            measured_cost=0.0,
            live=live_reader(),
            status="failed_closed",
        )
        checkpoint(ledger_path, reconciled)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint(
            output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ledger_sha256": development.sha256_file(ledger_path),
            },
        )
        raise
    measured = float(result["usage"]["run_cost_usd"])
    local = _reconcile(
        ledger=ledger,
        measured_cost=measured,
        live=live,
        status=result["status"],
    )
    checkpoint(ledger_path, local)
    final = _reconcile(
        ledger=local,
        measured_cost=0.0,
        live=live_reader(),
        status=result["status"],
    )
    checkpoint(ledger_path, final)
    checkpoint(
        output_dir / "EXECUTION.json",
        {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "complete_reconciled",
            "result_sha256": development.sha256_file(output_dir / "RESULT.json"),
            "ledger_sha256": development.sha256_file(ledger_path),
            "recorded_daily_spend_usd": final["recorded_actual_spend_usd"],
            "remaining_daily_allowance_usd": final["reconciliation"][
                "remaining_daily_allowance_usd"
            ],
        },
    )
    return result


def execute_smoke(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
) -> dict[str, Any]:
    preflight = preflight_smoke(now=now, live_reader=live_reader)
    return _execute(
        preflight=preflight,
        output_dir=SMOKE_DIR,
        ledger_path=SMOKE_LEDGER,
        run_id="bongard-openworld-luna-naive-first-link-smoke-20260808",
        runner=naive.run_smoke,
        runner_kwargs={},
        live_reader=live_reader,
    )


def execute_block(
    *,
    block_id: str,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
) -> dict[str, Any]:
    preflight = preflight_block(
        block_id=block_id, now=now, live_reader=live_reader
    )
    return _execute(
        preflight=preflight,
        output_dir=BLOCK_DIRS[block_id],
        ledger_path=SUPPLEMENTAL_LEDGERS[block_id],
        run_id=(
            f"bongard-openworld-luna-naive-first-link-{block_id}-"
            f"{development.BLOCK_EARLIEST_DATES[block_id].replace('-', '')}"
        ),
        runner=naive.run_block,
        runner_kwargs={"block_id": block_id, "smoke_result": SMOKE_RESULT},
        live_reader=live_reader,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--block", choices=development.BLOCK_ORDER)
    args = parser.parse_args()
    if args.smoke:
        result = preflight_smoke() if args.preflight else execute_smoke()
    else:
        result = (
            preflight_block(block_id=args.block)
            if args.preflight
            else execute_block(block_id=args.block)
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
