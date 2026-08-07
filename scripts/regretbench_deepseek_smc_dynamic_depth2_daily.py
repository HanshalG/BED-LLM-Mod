#!/usr/bin/env python3
"""Execute the sealed Aug 9 RegretBench SMC depth-two policy."""

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

from helpers import Config, ModelSpec
from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_naive_first_link_daily_execute as baseline_daily
from scripts import regretbench_deepseek_dynamic_depth2_confirmation_daily as confirmation_daily
from scripts import regretbench_deepseek_dynamic_depth2_policy as transport
from scripts import regretbench_deepseek_dynamic_depth2_policy_daily as primary_policy_daily
from scripts import regretbench_deepseek_smc_dynamic_depth2_experiment as experiment
from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as core
from scripts import regretbench_deepseek_smc_dynamic_depth2_verify as verifier
from scripts import regretbench_deepseek_smc_support_recovery as smc_support
from scripts import regretbench_deepseek_smc_support_recovery_daily as smc_daily
from scripts import regretbench_deepseek_smc_support_recovery_verify as smc_verify
from scripts import regretbench_deepseek_support_recovery as primary
from scripts import regretbench_deepseek_support_recovery_daily as primary_daily
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_qwen_history_blind_serving_smoke import (
    PerRequestSeedStructuredAdapter,
)
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-smc-dynamic-depth2-daily-1"
DATE = "2026-08-09"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
ROOT = REPO_ROOT / (
    "results/nonmyopic/regretbench_deepseek_smc_dynamic_depth2_policy"
)
SMOKE_DIR = ROOT / "smoke-20260809"
NAIVE_SMOKE_DIR = ROOT / "naive-smoke-20260809"
DEVELOPMENT_DIR = ROOT / "development-20260809"
DAILY_RESULT = ROOT / "DAILY_RESULT.json"
LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-09-regretbench-smc-dynamic-policy.json"
)
PRIMARY_SMOKE_DIR = primary_daily.SMOKE_DIR
PRIMARY_DEVELOPMENT_DIR = primary_daily.DEVELOPMENT_DIR
CORE_SHA256 = (
    "71f79258184bfa2f00e06f8c82836b1d5fbd1d3895ebc4105cb3256360c843fa"
)
PRODUCER_SHA256 = (
    "6525ce70ca1f14553e2de3e2e1f79ac43c6647a95c61d1b7fed88a051c91949a"
)
TRANSPORT_SHA256 = (
    "c097468b24acf9ff33743b4f8798c5d1915e142e2bf39aa998b2186812fba2f3"
)
SMC_SUPPORT_CORE_SHA256 = (
    "91be9699391aa67070174ca3be2fdb7b6cd9e1ae210fcbb0f5c7e35ba601f280"
)
SMC_SUPPORT_DAILY_SHA256 = (
    "18805dbd9f79e002fd6eaa6df9aa2c8da1afdc76cd360b0f9f26e7b2f64d7c84"
)
SMC_SUPPORT_VERIFIER_SHA256 = (
    "fab0b932c1c00ccb2a7333f63394e16b930cd238c3d5f1da73c8692992fa1e2a"
)
PRIMARY_DAILY_SHA256 = (
    "ad8c3f0ad907ea1042c08607976cca4bf900dc533347421806d7385616bf6eb7"
)
LUNA_CATALOG_DAILY_SHA256 = (
    "db8d13229444abbf3f549967af4e70271103b6f5abd917fcce88573ec44b95c9"
)
CATALOG_READER_SHA256 = (
    "5e4d2322f776bae63978073d64047190916fe8b2589f034db606f9bffdaa03c5"
)
PRIMARY_POLICY_DAILY_SHA256 = (
    "67d22c89385765bd0651a056c74d2960f70e13960f9bbf225dbc8e6ad0cb868c"
)
CONFIRMATION_DAILY_SHA256 = (
    "b58e4fd6d4da6d3df465e359d9251098c3aebc19c8645d4cddc1b57da8d8aad4"
)
VERIFIER_SHA256 = (
    "a76e61149f1537a048d6338d19f0a51b2aa6d396df06c5c45928763616a70afa"
)
MAX_REQUEST_COST_USD = transport.MAX_REQUEST_COST_USD
MIN_RESERVED_PROMPT_TOKENS = primary_daily.MIN_RESERVED_PROMPT_TOKENS
POLICY_WORST_CASE_USD = (
    experiment.SMOKE_BUDGET_USD
    + transport.NAIVE_SMOKE_BUDGET_USD
    + experiment.DEVELOPMENT_BUDGET_USD
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_date(now: datetime | None = None) -> datetime:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"RegretBench SMC policy can run only on {DATE}")
    return local


def _pristine(path: Path) -> bool:
    if not path.exists():
        return True
    if path.is_file():
        return False
    return not any(path.iterdir())


def _forbidden_descendants() -> list[Path]:
    return [
        primary_policy_daily.SMOKE_DIR,
        primary_policy_daily.NAIVE_SMOKE_DIR,
        primary_policy_daily.DEVELOPMENT_DIR,
        primary_policy_daily.LEDGER,
        primary_policy_daily.ROOT / "DAILY_RESULT.json",
        confirmation_daily.RUN_DIR,
        confirmation_daily.DAILY_RESULT,
        confirmation_daily.LEDGER,
    ]


def _assert_no_competing_descendant() -> None:
    opened = [str(path) for path in _forbidden_descendants() if not _pristine(path)]
    if opened:
        raise RuntimeError("a primary policy or confirmation descendant is already open")


def _validate_hashes() -> None:
    core.validate_protocol_binding()
    expected = {
        Path(core.__file__).resolve(): CORE_SHA256,
        Path(experiment.__file__).resolve(): PRODUCER_SHA256,
        Path(transport.__file__).resolve(): TRANSPORT_SHA256,
        Path(smc_support.__file__).resolve(): SMC_SUPPORT_CORE_SHA256,
        Path(smc_daily.__file__).resolve(): SMC_SUPPORT_DAILY_SHA256,
        Path(smc_verify.__file__).resolve(): SMC_SUPPORT_VERIFIER_SHA256,
        Path(primary_daily.__file__).resolve(): PRIMARY_DAILY_SHA256,
        Path(baseline_daily.__file__).resolve(): LUNA_CATALOG_DAILY_SHA256,
        Path(aug10.__file__).resolve(): CATALOG_READER_SHA256,
        Path(primary_policy_daily.__file__).resolve(): PRIMARY_POLICY_DAILY_SHA256,
        Path(confirmation_daily.__file__).resolve(): CONFIRMATION_DAILY_SHA256,
    }
    expected[Path(verifier.__file__).resolve()] = VERIFIER_SHA256
    for path, digest in expected.items():
        if primary.sha256_file(path) != digest:
            raise RuntimeError(f"SMC policy bound file changed: {path.name}")


def validate_smc_predecessor() -> dict[str, Any]:
    authorization = core.validate_smc_support_predecessor(
        result_path=smc_daily.RUN_DIR / "RESULT.json",
        verification_path=smc_daily.RUN_DIR / "VERIFICATION.json",
        daily_result_path=smc_daily.DAILY_RESULT,
        ledger_path=smc_daily.LEDGER,
    )
    support_verification = _load(smc_daily.RUN_DIR / "VERIFICATION.json")
    raw_path = smc_daily.RUN_DIR / "private/RAW_RESPONSES.json"
    controls_path = smc_daily.RUN_DIR / "private/CONTROLS.json"
    artifacts = support_verification.get("artifact_sha256") or {}
    if (
        not raw_path.is_file()
        or not controls_path.is_file()
        or artifacts.get("private/RAW_RESPONSES.json")
        != primary.sha256_file(raw_path)
        or artifacts.get("private/CONTROLS.json")
        != primary.sha256_file(controls_path)
    ):
        raise RuntimeError("verified SMC support private artifacts are incomplete")
    _assert_no_competing_descendant()
    return {
        **authorization,
        "support_raw_sha256": primary.sha256_file(raw_path),
        "support_controls_sha256": primary.sha256_file(controls_path),
        "support_ledger": _load(smc_daily.LEDGER),
    }


def _validated_existing_complete() -> dict[str, Any] | None:
    if not DAILY_RESULT.exists():
        return None
    if not DAILY_RESULT.is_file():
        raise RuntimeError("SMC policy daily result path is invalid")
    predecessor = validate_smc_predecessor()
    daily = _load(DAILY_RESULT)
    required = [
        SMOKE_DIR / "RESULT.json",
        SMOKE_DIR / "VERIFICATION.json",
        DEVELOPMENT_DIR / "RESULT.json",
        DEVELOPMENT_DIR / "VERIFICATION.json",
        LEDGER,
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError("SMC policy completed artifacts are incomplete")
    smoke_replay = verifier.verify_smoke(
        SMOKE_DIR, primary_dir=PRIMARY_SMOKE_DIR
    )
    development_replay = verifier.verify(
        DEVELOPMENT_DIR, primary_dir=PRIMARY_DEVELOPMENT_DIR
    )
    stored_smoke_replay = _load(SMOKE_DIR / "VERIFICATION.json")
    stored_development_replay = _load(DEVELOPMENT_DIR / "VERIFICATION.json")
    development = _load(DEVELOPMENT_DIR / "RESULT.json")
    ledger = _load(LEDGER)
    if (
        smoke_replay != stored_smoke_replay
        or development_replay != stored_development_replay
        or smoke_replay.get("status") != "verified"
        or development_replay.get("status") != "verified"
        or daily.get("status") != "complete_reconciled"
        or daily.get("smoke_status") != "passed"
        or daily.get("development_status") != development.get("status")
        or daily.get("development_opened") is not True
        or daily.get("confirmation_opened") is not False
        or daily.get("independent_replay_passed") is not True
        or daily.get("smoke_result_sha256")
        != primary.sha256_file(SMOKE_DIR / "RESULT.json")
        or daily.get("smoke_verification_sha256")
        != primary.sha256_file(SMOKE_DIR / "VERIFICATION.json")
        or daily.get("development_result_sha256")
        != primary.sha256_file(DEVELOPMENT_DIR / "RESULT.json")
        or daily.get("development_verification_sha256")
        != primary.sha256_file(DEVELOPMENT_DIR / "VERIFICATION.json")
        or daily.get("ledger_sha256") != primary.sha256_file(LEDGER)
        or ledger.get("date") != DATE
        or ledger.get("timezone") != TIMEZONE
        or float(ledger.get("daily_cap_usd", 0.0)) != DAILY_CAP_USD
        or ledger.get("account_wide_usage_counts_against_cap") is not True
        or ledger.get("predecessor", {}).get("result_sha256")
        != predecessor["result_sha256"]
    ):
        raise RuntimeError("SMC policy completed artifacts do not replay")
    return {**daily, "resume_status": "already_complete_verified"}


def _validate_deepseek(catalog: Mapping[str, Any]) -> dict[str, Any]:
    model = primary_daily.validate_deepseek_model_catalog(catalog)
    completion = float(model["completion_usd_per_million_tokens"]) / 1_000_000
    prompt = float(model["prompt_usd_per_million_tokens"]) / 1_000_000
    residual = MAX_REQUEST_COST_USD - completion * core.MAX_TOKENS
    covered_prompt = math.inf if prompt == 0 else residual / prompt
    if (
        int(model["max_completion_tokens"]) < core.MAX_TOKENS
        or residual < 0
        or covered_prompt + 1e-9 < MIN_RESERVED_PROMPT_TOKENS
    ):
        raise RuntimeError(
            "DeepSeek per-attempt reservation does not cover the SMC policy"
        )
    return {
        **model,
        "status": "available",
        "policy_max_tokens": core.MAX_TOKENS,
        "covered_prompt_tokens_at_policy_limit": covered_prompt,
    }


def _day_boundary(ledger: Mapping[str, Any]) -> float:
    return float(ledger["opening_total_usage_usd"])


def _spent(ledger: Mapping[str, Any], live: Mapping[str, float]) -> float:
    posted = max(
        0.0,
        float(live["total_usage_usd"]) - _day_boundary(ledger),
    )
    return max(posted, float(ledger["recorded_actual_spend_usd"]))


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = aug10.read_openrouter_model_catalog,
) -> dict[str, Any]:
    local = _validate_date(now)
    _validate_hashes()
    existing = _validated_existing_complete()
    if existing is not None:
        return {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "already_complete_verified",
            "date": local.date().isoformat(),
            "daily_result": existing,
            "model_calls_made": 0,
            "files_written": 0,
        }
    predecessor = validate_smc_predecessor()
    for path in (SMOKE_DIR, NAIVE_SMOKE_DIR, DEVELOPMENT_DIR, DAILY_RESULT, LEDGER):
        if not _pristine(path):
            raise RuntimeError(f"SMC policy output path is not pristine: {path}")
    catalog = catalog_reader()
    deepseek = _validate_deepseek(catalog)
    try:
        luna = baseline_daily._validate_model_catalog(catalog)
    except (RuntimeError, TypeError, ValueError) as exc:
        luna = {
            "id": transport.NAIVE_MODEL_ID,
            "status": "unavailable",
            "can_affect_primary_status": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    else:
        luna["status"] = "available"
    live = live_reader()
    spent = _spent(predecessor["support_ledger"], live)
    if spent + POLICY_WORST_CASE_USD > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("SMC policy exceeds the remaining account-wide day")
    if float(live["balance_usd"]) + 1e-12 < POLICY_WORST_CASE_USD:
        raise RuntimeError("OpenRouter balance is below the SMC policy caps")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": local.date().isoformat(),
        "models": {"deepseek": deepseek, "luna": luna},
        "predecessor": {
            key: value
            for key, value in predecessor.items()
            if key != "support_ledger"
        },
        "live_credits": live,
        "budget": {
            "daily_cap_usd": DAILY_CAP_USD,
            "opening_total_usage_boundary_usd": _day_boundary(
                predecessor["support_ledger"]
            ),
            "spent_before_policy_usd": spent,
            "enriched_smoke_cap_usd": experiment.SMOKE_BUDGET_USD,
            "naive_smoke_cap_usd": transport.NAIVE_SMOKE_BUDGET_USD,
            "development_cap_usd": experiment.DEVELOPMENT_BUDGET_USD,
            "policy_worst_case_usd": POLICY_WORST_CASE_USD,
            "remaining_after_full_caps_usd": (
                DAILY_CAP_USD - spent - POLICY_WORST_CASE_USD
            ),
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def _initial_ledger(ready: Mapping[str, Any]) -> dict[str, Any]:
    live = ready["live_credits"]
    boundary = ready["budget"]["opening_total_usage_boundary_usd"]
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "date": DATE,
        "timezone": TIMEZONE,
        "daily_cap_usd": DAILY_CAP_USD,
        "opening_total_credits_usd": float(live["total_credits_usd"]),
        "opening_total_usage_usd": boundary,
        "opening_balance_usd": float(live["total_credits_usd"]) - boundary,
        "recorded_actual_spend_usd": ready["budget"]["spent_before_policy_usd"],
        "account_wide_usage_counts_against_cap": True,
        "opening_boundary_inherited_from_aug9_smc_support": True,
        "unspent_allowance_does_not_roll_over": True,
        "predecessor": ready["predecessor"],
        "stages": {
            "enriched_smoke": {
                "status": "authorized_pending",
                "maximum_cost_usd": experiment.SMOKE_BUDGET_USD,
            },
            "naive_smoke": {
                "status": "enriched_smoke_gated",
                "maximum_cost_usd": transport.NAIVE_SMOKE_BUDGET_USD,
            },
            "development": {
                "status": "smokes_gated",
                "maximum_cost_usd": experiment.DEVELOPMENT_BUDGET_USD,
            },
        },
    }


def _budget_status(
    ledger: Mapping[str, Any],
    *,
    projected: float,
    live: Mapping[str, float],
    now: datetime | None,
) -> dict[str, Any]:
    status = require_budget(
        dict(ledger),
        projected_cost_usd=projected,
        total_usage_usd=float(live["total_usage_usd"]),
        now=now,
    )
    status.update(live)
    return status


def _reconcile(
    ledger: Mapping[str, Any],
    *,
    stage: str,
    status: str,
    measured_cost: float,
    live: Mapping[str, float],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    prior = float(updated["recorded_actual_spend_usd"])
    posted = max(0.0, float(live["total_usage_usd"]) - opening)
    recorded = max(posted, prior + measured_cost)
    if recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("SMC policy reconciliation exceeds the daily cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["stages"][stage].update(
        {"status": status, "actual_cost_usd": measured_cost}
    )
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live["total_credits_usd"]),
        "live_total_usage_usd": float(live["total_usage_usd"]),
        "live_balance_usd": float(live["balance_usd"]),
        "posted_spend_since_day_boundary_usd": posted,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": DAILY_CAP_USD - recorded,
    }
    return updated


def build_deepseek_adapter(
    *, stage: str, run_id: str, output_dir: Path
) -> PerRequestSeedStructuredAdapter:
    if stage not in {"smoke", "development"}:
        raise ValueError("invalid SMC policy adapter stage")
    smoke = stage == "smoke"
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=275.0,
        openrouter_run_budget_usd=(
            experiment.SMOKE_BUDGET_USD
            if smoke
            else experiment.DEVELOPMENT_BUDGET_USD
        ),
        openrouter_projected_cost_usd=(0.02 if smoke else 3.10),
        openrouter_concurrency=10 if smoke else 128,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=core.MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return PerRequestSeedStructuredAdapter(
        ModelSpec(model=core.MODEL_ID, backend="openrouter", max_model_len=65_536),
        config,
    )


def _usage(adapter: Any | None) -> dict[str, Any]:
    if adapter is None:
        return experiment._empty_usage()
    return experiment.summarize_usage(adapter.usage_snapshot())


def _cost(adapter: Any | None) -> float:
    return float(_usage(adapter)["run_cost_usd"])


def execute(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
) -> dict[str, Any]:
    ready = preflight(now=now, live_reader=live_reader)
    if ready["status"] == "already_complete_verified":
        return dict(ready["daily_result"])
    ledger = _initial_ledger(ready)
    checkpoint(LEDGER, ledger)

    smoke_live = live_reader()
    smoke_budget = _budget_status(
        ledger,
        projected=experiment.SMOKE_BUDGET_USD,
        live=smoke_live,
        now=now,
    )
    smoke_adapter = build_deepseek_adapter(
        stage="smoke",
        run_id="regretbench-deepseek-smc-policy-smoke-20260809",
        output_dir=SMOKE_DIR,
    )
    try:
        smoke = experiment.run_smoke(
            output_dir=SMOKE_DIR,
            adapter=smoke_adapter,
            primary_smoke_dir=PRIMARY_SMOKE_DIR,
            smc_result_path=smc_daily.RUN_DIR / "RESULT.json",
            smc_verification_path=smc_daily.RUN_DIR / "VERIFICATION.json",
            smc_daily_result_path=smc_daily.DAILY_RESULT,
            smc_ledger_path=smc_daily.LEDGER,
            daily_budget_status=smoke_budget,
        )
        smoke_verification = verifier.verify_smoke(
            SMOKE_DIR, primary_dir=PRIMARY_SMOKE_DIR
        )
        checkpoint(SMOKE_DIR / "VERIFICATION.json", smoke_verification)
        if smoke_verification["status"] != "verified":
            raise RuntimeError("SMC policy smoke independent replay failed")
    except Exception:
        ledger = _reconcile(
            ledger,
            stage="enriched_smoke",
            status="failed_closed",
            measured_cost=_cost(smoke_adapter),
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        raise
    ledger = _reconcile(
        ledger,
        stage="enriched_smoke",
        status=smoke["status"],
        measured_cost=float(smoke["usage"]["run_cost_usd"]),
        live=live_reader(),
    )
    checkpoint(LEDGER, ledger)
    if smoke["status"] != "passed":
        daily = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "smoke_stopped",
            "smoke_status": smoke["status"],
            "smoke_result_sha256": primary.sha256_file(SMOKE_DIR / "RESULT.json"),
            "smoke_verification_sha256": primary.sha256_file(
                SMOKE_DIR / "VERIFICATION.json"
            ),
            "development_opened": False,
            "confirmation_opened": False,
            "ledger_sha256": primary.sha256_file(LEDGER),
        }
        checkpoint(DAILY_RESULT, daily)
        return daily

    luna_available = ready["models"]["luna"]["status"] == "available"
    naive_smoke_status = "unavailable_preflight"
    naive_smoke_path = ROOT / "NAIVE_SMOKE_STATUS.json"
    if luna_available:
        naive_live = live_reader()
        naive_budget = _budget_status(
            ledger,
            projected=transport.NAIVE_SMOKE_BUDGET_USD,
            live=naive_live,
            now=now,
        )
        ledger["stages"]["naive_smoke"]["status"] = "authorized_pending"
        checkpoint(LEDGER, ledger)
        naive_smoke_adapter = transport.build_naive_adapter(
            stage="smoke",
            run_id="regretbench-luna-smc-policy-naive-smoke-20260809",
            output_dir=NAIVE_SMOKE_DIR,
        )
        try:
            naive_smoke = experiment.run_naive_smoke(
                output_dir=NAIVE_SMOKE_DIR,
                adapter=naive_smoke_adapter,
                policy_smoke_result=SMOKE_DIR / "RESULT.json",
                daily_budget_status=naive_budget,
            )
        except Exception as exc:
            naive_smoke_status = "failed_closed"
            naive_smoke_path = NAIVE_SMOKE_DIR / "FAILURE.json"
            checkpoint(
                naive_smoke_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "interface_version": INTERFACE_VERSION,
                    "status": naive_smoke_status,
                    "authorizes": "nothing",
                    "can_affect_primary_status": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        else:
            naive_smoke_status = naive_smoke["status"]
            naive_smoke_path = NAIVE_SMOKE_DIR / "RESULT.json"
        ledger = _reconcile(
            ledger,
            stage="naive_smoke",
            status=naive_smoke_status,
            measured_cost=_cost(naive_smoke_adapter),
            live=live_reader(),
        )
    else:
        ledger["stages"]["naive_smoke"].update(
            {"status": naive_smoke_status, "actual_cost_usd": 0.0}
        )
    checkpoint(LEDGER, ledger)
    naive_enabled = naive_smoke_status == "passed"
    if not naive_enabled:
        checkpoint(
            ROOT / "NAIVE_SMOKE_STATUS.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": naive_smoke_status,
                "development_opened": True,
                "naive_baseline_enabled": False,
                "can_affect_primary_status": False,
                "naive_smoke_artifact": str(naive_smoke_path),
                "catalog_status": ready["models"]["luna"],
            },
        )

    development_live = live_reader()
    development_budget = _budget_status(
        ledger,
        projected=experiment.DEVELOPMENT_BUDGET_USD,
        live=development_live,
        now=now,
    )
    ledger["stages"]["development"]["status"] = "authorized_pending"
    checkpoint(LEDGER, ledger)
    run_id = "regretbench-deepseek-smc-dynamic-policy-development-20260809"
    development_adapter = build_deepseek_adapter(
        stage="development", run_id=run_id, output_dir=DEVELOPMENT_DIR
    )
    naive_adapter = (
        transport.build_naive_adapter(
            stage="development", run_id=run_id, output_dir=DEVELOPMENT_DIR
        )
        if naive_enabled
        else None
    )
    endpoint_adapter = (
        build_deepseek_adapter(
            stage="development", run_id=run_id, output_dir=DEVELOPMENT_DIR
        )
        if naive_enabled
        else None
    )
    try:
        tree = experiment.build_development_planning_tree(
            output_dir=DEVELOPMENT_DIR,
            adapter=development_adapter,
            primary_development_dir=PRIMARY_DEVELOPMENT_DIR,
        )
        realized = experiment.run_realized_primary(
            output_dir=DEVELOPMENT_DIR,
            adapter=development_adapter,
            tree=tree,
        )
        naive_result = None
        naive_error = None
        naive_usage_on_error = None
        endpoint_usage_on_error = None
        if naive_enabled:
            try:
                naive_result = experiment.run_naive_baseline(
                    output_dir=DEVELOPMENT_DIR,
                    contexts=tree["contexts"],
                    initial_supports=tree["initial_supports"],
                    naive_adapter=naive_adapter,
                    endpoint_adapter=endpoint_adapter,
                )
            except Exception as exc:
                naive_error = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                naive_usage_on_error = _usage(naive_adapter)
                endpoint_usage_on_error = _usage(endpoint_adapter)
        policy_smoke = experiment.validate_policy_smoke(
            SMOKE_DIR / "RESULT.json"
        )
        naive_smoke = (
            experiment.validate_naive_smoke(naive_smoke_path)
            if naive_enabled
            else {"status": naive_smoke_status, "path": str(naive_smoke_path)}
        )
        development = experiment.finalize_development_result(
            output_dir=DEVELOPMENT_DIR,
            primary_result=realized,
            policy_smoke=policy_smoke,
            naive_smoke=naive_smoke,
            naive_result=naive_result,
            naive_error=naive_error,
            naive_usage_on_error=naive_usage_on_error,
            endpoint_usage_on_error=endpoint_usage_on_error,
            primary_privacy=[*tree["privacy"], *realized["actual_privacy"]],
            naive_privacy=(naive_result or {}).get("naive_privacy", []),
            endpoint_privacy=(naive_result or {}).get("endpoint_privacy", []),
            daily_budget_status=development_budget,
        )
        development_verification = verifier.verify(
            DEVELOPMENT_DIR, primary_dir=PRIMARY_DEVELOPMENT_DIR
        )
        checkpoint(
            DEVELOPMENT_DIR / "VERIFICATION.json", development_verification
        )
        if development_verification["status"] != "verified":
            raise RuntimeError("SMC policy development independent replay failed")
    except Exception:
        measured = sum(
            _cost(adapter)
            for adapter in (development_adapter, naive_adapter, endpoint_adapter)
        )
        ledger = _reconcile(
            ledger,
            stage="development",
            status="failed_closed",
            measured_cost=measured,
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        raise
    ledger = _reconcile(
        ledger,
        stage="development",
        status=development["status"],
        measured_cost=float(development["usage"]["combined_cost_usd"]),
        live=live_reader(),
    )
    checkpoint(LEDGER, ledger)
    daily = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete_reconciled",
        "smoke_status": smoke["status"],
        "smoke_result_sha256": primary.sha256_file(SMOKE_DIR / "RESULT.json"),
        "smoke_verification_sha256": primary.sha256_file(
            SMOKE_DIR / "VERIFICATION.json"
        ),
        "naive_smoke_status": naive_smoke_status,
        "naive_baseline_enabled": naive_enabled,
        "development_status": development["status"],
        "development_result_sha256": primary.sha256_file(
            DEVELOPMENT_DIR / "RESULT.json"
        ),
        "development_verification_sha256": primary.sha256_file(
            DEVELOPMENT_DIR / "VERIFICATION.json"
        ),
        "independent_replay_passed": True,
        "development_opened": True,
        "confirmation_opened": False,
        "authorizes": development["authorizes"],
        "ledger_sha256": primary.sha256_file(LEDGER),
        "recorded_daily_spend_usd": ledger["recorded_actual_spend_usd"],
    }
    checkpoint(DAILY_RESULT, daily)
    return daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        result = preflight() if args.preflight else execute()
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "development_opened": False,
            "confirmation_opened": False,
        }
        if not args.preflight:
            ROOT.mkdir(parents=True, exist_ok=True)
            checkpoint(ROOT / "DAILY_FAILURE.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {
        "ready_without_paid_calls",
        "already_complete_verified",
        "complete_reconciled",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
