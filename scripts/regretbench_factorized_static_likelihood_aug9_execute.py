#!/usr/bin/env python3
"""Run the conditional Aug 9 RegretBench factorized exact-10 smoke."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import regretbench_branch_draw_decision as draw_decision
from scripts import regretbench_branch_draw_fidelity as draw_fidelity
from scripts import regretbench_deepseek_dynamic_depth2_confirmation_daily as confirmation_daily
from scripts import regretbench_deepseek_dynamic_depth2_policy as policy
from scripts import regretbench_deepseek_dynamic_depth2_policy_daily as policy_daily
from scripts import regretbench_deepseek_frozen_report as frozen_report
from scripts import regretbench_deepseek_paper_fragment as paper_fragment
from scripts import regretbench_deepseek_smc_support_recovery_daily as smc_daily
from scripts import regretbench_deepseek_support_recovery as primary
from scripts import regretbench_deepseek_support_recovery_daily as primary_daily
from scripts import regretbench_factorized_static_likelihood_smoke as smoke
from scripts import regretbench_factorized_static_likelihood_smoke_verify as verifier
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-factorized-static-likelihood-aug9-execute-1"
DATE = "2026-08-09"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
RUN_CAP_USD = smoke.SMOKE_BUDGET_USD
EXECUTION_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_FACTORIZED_STATIC_LIKELIHOOD_AUG9_EXECUTION_20260807.md"
)
EXECUTION_PROTOCOL_SHA256 = "108b0f6a0f6691dea32495c988185b570a1a299404ca54fd8e0a68cd79a53075"
CORE_BINDING = REPO_ROOT / (
    "results/nonmyopic/regretbench_factorized_static_likelihood_smoke/"
    "CORE_BINDING.json"
)
ROOT = REPO_ROOT / (
    "results/nonmyopic/regretbench_factorized_static_likelihood_smoke"
)
RUN_DIR = ROOT / "smoke-20260809"
DAILY_RESULT = ROOT / "DAILY_RESULT_20260809.json"
DAILY_FAILURE = ROOT / "DAILY_FAILURE_20260809.json"
LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-09-regretbench-factorized-static-smoke.json"
)
FIDELITY = REPO_ROOT / (
    "results/nonmyopic/regretbench_branch_draw_fidelity/"
    "original-development-20260808.json"
)
DECISION = REPO_ROOT / (
    "results/nonmyopic/regretbench_branch_draw_fidelity/"
    "original-development-20260808-decision.json"
)
DECISION_MARKDOWN = REPO_ROOT / (
    "results/nonmyopic/regretbench_branch_draw_fidelity/"
    "original-development-20260808-decision.md"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pristine(path: Path) -> bool:
    if not path.exists():
        return True
    if path.is_file():
        return False
    return not any(path.iterdir())


def _validate_date(now: datetime | None = None) -> datetime:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"factorized smoke can run only on {DATE}")
    return local


def validate_bindings() -> dict[str, Any]:
    if _sha256(EXECUTION_PROTOCOL) != EXECUTION_PROTOCOL_SHA256:
        raise RuntimeError("factorized Aug 9 execution protocol changed")
    smoke.validate_protocol()
    binding = _load(CORE_BINDING)
    expected = {
        "protocol": smoke.PROTOCOL,
        "producer": Path(smoke.__file__).resolve(),
        "independent_verifier": Path(verifier.__file__).resolve(),
        "execution_protocol": EXECUTION_PROTOCOL,
        "dated_budget_wrapper": Path(__file__).resolve(),
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)):
            raise RuntimeError(f"factorized {name} binding path changed")
        if row.get("sha256") != _sha256(path):
            raise RuntimeError(f"factorized {name} binding hash changed")
    if (
        binding.get("paid_calls_authorized") != smoke.EXPECTED_REQUESTS
        or binding.get("independent_verifier_implemented") is not True
        or binding.get("dated_budget_wrapper_implemented") is not True
        or binding.get("current_regretbench_chain_changed") is not False
    ):
        raise RuntimeError("factorized core binding metadata changed")
    return {
        "status": "verified_frozen_execution",
        "execution_protocol_sha256": EXECUTION_PROTOCOL_SHA256,
        "core_binding_sha256": _sha256(CORE_BINDING),
        "model_calls_made": 0,
        "cost_usd": 0.0,
    }


def _assert_descendants_pristine() -> None:
    paths = [
        confirmation_daily.RUN_DIR,
        confirmation_daily.DAILY_RESULT,
        confirmation_daily.LEDGER,
        smc_daily.RUN_DIR,
        smc_daily.DAILY_RESULT,
        smc_daily.LEDGER,
    ]
    opened = [str(path) for path in paths if not _pristine(path)]
    if opened:
        raise RuntimeError("confirmation or SMC contingency is already open")


def validate_policy_null_predecessor() -> dict[str, Any]:
    support = policy_daily.validate_recovery_predecessor()["support"]
    daily = _load(policy_daily.ROOT / "DAILY_RESULT.json")
    ledger = _load(policy_daily.LEDGER)
    result, verification, result_sha, verification_sha = (
        frozen_report._validate_verified_result(
            policy_daily.DEVELOPMENT_DIR, stage="development"
        )
    )
    if (
        result.get("status") != "gated_null"
        or daily.get("status") != "complete_reconciled"
        or daily.get("development_status") != "gated_null"
        or daily.get("independent_replay_passed") is not True
        or daily.get("development_result_sha256") != result_sha
        or daily.get("development_verification_sha256") != verification_sha
        or daily.get("ledger_sha256") != _sha256(policy_daily.LEDGER)
        or daily.get("confirmation_opened") is not False
        or ledger.get("date") != "2026-08-08"
        or ledger.get("timezone") != TIMEZONE
        or float(ledger.get("daily_cap_usd", 0.0)) != DAILY_CAP_USD
        or ledger.get("account_wide_usage_counts_against_cap") is not True
    ):
        raise RuntimeError("dynamic policy predecessor is not a verified daily null")

    saved_report = _load(policy_daily.DEVELOPMENT_DIR / "FROZEN_REPORT.json")
    expected_report = frozen_report.build_report(
        policy_daily.DEVELOPMENT_DIR, stage="development"
    )
    if (
        _canonical(saved_report) != _canonical(expected_report)
        or saved_report.get("claim_tier")
        != "development_policy_null_confirmation_forbidden"
        or (policy_daily.DEVELOPMENT_DIR / "FROZEN_REPORT.md").read_text(
            encoding="utf-8"
        )
        != frozen_report.render_markdown(expected_report)
    ):
        raise RuntimeError("dynamic policy frozen null report changed")

    expected_tex, expected_meta = paper_fragment.build_fragment(
        policy_daily.DEVELOPMENT_DIR, stage="development"
    )
    saved_meta = _load(paper_fragment.DEFAULT_OUTPUT.with_suffix(".json"))
    if (
        paper_fragment.DEFAULT_OUTPUT.read_text(encoding="utf-8") != expected_tex
        or any(saved_meta.get(key) != value for key, value in expected_meta.items())
        or saved_meta.get("tex_sha256") != _sha256(paper_fragment.DEFAULT_OUTPUT)
    ):
        raise RuntimeError("dynamic policy paper fragment changed")

    expected_fidelity = draw_fidelity.run(policy_daily.DEVELOPMENT_DIR)
    saved_fidelity = _load(FIDELITY)
    if _canonical(saved_fidelity) != _canonical(expected_fidelity):
        raise RuntimeError("branch-draw fidelity artifact changed")
    expected_decision = draw_decision.run(FIDELITY)
    saved_decision = _load(DECISION)
    if (
        _canonical(saved_decision) != _canonical(expected_decision)
        or DECISION_MARKDOWN.read_text(encoding="utf-8")
        != draw_decision.render_markdown(expected_decision)
        or saved_decision.get("can_change_status_authorization_or_claim_tier")
        is not False
    ):
        raise RuntimeError("branch-draw decision artifact changed")
    _assert_descendants_pristine()
    return {
        "support": support,
        "support_smoke_result": primary_daily.SMOKE_DIR / "RESULT.json",
        "support_development_result": primary_daily.DEVELOPMENT_DIR / "RESULT.json",
        "primary_smoke_dir": primary_daily.SMOKE_DIR,
        "policy_result_sha256": result_sha,
        "policy_verification_sha256": verification_sha,
        "policy_daily_result_sha256": _sha256(
            policy_daily.ROOT / "DAILY_RESULT.json"
        ),
        "policy_ledger_sha256": _sha256(policy_daily.LEDGER),
        "frozen_report_sha256": _sha256(
            policy_daily.DEVELOPMENT_DIR / "FROZEN_REPORT.json"
        ),
        "paper_fragment_sha256": _sha256(paper_fragment.DEFAULT_OUTPUT),
        "fidelity_sha256": _sha256(FIDELITY),
        "decision_sha256": _sha256(DECISION),
        "aug8_close_usage_boundary_usd": float(
            ledger["opening_total_usage_usd"]
        )
        + float(ledger["recorded_actual_spend_usd"]),
    }


def _public_predecessor(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key
        not in {
            "support_smoke_result",
            "support_development_result",
            "primary_smoke_dir",
        }
    }


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader=None,
) -> dict[str, Any]:
    local = _validate_date(now)
    bindings = validate_bindings()
    predecessor = validate_policy_null_predecessor()
    for path in (RUN_DIR, DAILY_RESULT, DAILY_FAILURE, LEDGER):
        if not _pristine(path):
            raise RuntimeError(f"factorized output path is not pristine: {path}")
    if catalog_reader is None:
        from scripts import bongard_openworld_luna_aug10_execute as catalog

        catalog_reader = catalog.read_openrouter_model_catalog
    model = primary_daily.validate_deepseek_model_catalog(catalog_reader())
    live = live_reader()
    boundary = predecessor["aug8_close_usage_boundary_usd"]
    if float(live["total_usage_usd"]) + 1e-12 < boundary:
        raise RuntimeError("live usage is below the reconciled Aug 8 boundary")
    spent = float(live["total_usage_usd"]) - boundary
    if spent + RUN_CAP_USD > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("factorized smoke exceeds remaining account-wide day")
    if float(live["balance_usd"]) + 1e-12 < RUN_CAP_USD:
        raise RuntimeError("OpenRouter balance is below factorized smoke cap")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": local.date().isoformat(),
        "bindings": bindings,
        "model": model,
        "predecessor": _public_predecessor(predecessor),
        "budget": {
            "daily_cap_usd": DAILY_CAP_USD,
            "opening_total_usage_boundary_usd": boundary,
            "spent_before_factorized_smoke_usd": spent,
            "run_cap_usd": RUN_CAP_USD,
            "remaining_after_full_cap_usd": DAILY_CAP_USD - spent - RUN_CAP_USD,
        },
        "live_credits": live,
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
        "recorded_actual_spend_usd": ready["budget"][
            "spent_before_factorized_smoke_usd"
        ],
        "account_wide_usage_counts_against_cap": True,
        "opening_boundary_derived_from_reconciled_aug8_close": True,
        "unspent_allowance_does_not_roll_over": True,
        "predecessor": ready["predecessor"],
        "stage": {
            "status": "authorized_pending",
            "maximum_cost_usd": RUN_CAP_USD,
        },
    }


def _budget_status(
    ledger: Mapping[str, Any], live: Mapping[str, float], *, now: datetime | None
) -> dict[str, Any]:
    status = require_budget(
        dict(ledger),
        projected_cost_usd=RUN_CAP_USD,
        total_usage_usd=float(live["total_usage_usd"]),
        now=now,
    )
    status.update(live)
    return status


def _adapter_cost(adapter: Any) -> float:
    return float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def _reconcile(
    ledger: Mapping[str, Any], *, status: str, measured: float, live: Mapping[str, float]
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    prior = float(updated["recorded_actual_spend_usd"])
    posted = max(0.0, float(live["total_usage_usd"]) - opening)
    recorded = max(posted, prior + measured)
    if recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("factorized reconciliation exceeds daily cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["stage"].update({"status": status, "actual_cost_usd": measured})
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live["total_credits_usd"]),
        "live_total_usage_usd": float(live["total_usage_usd"]),
        "live_balance_usd": float(live["balance_usd"]),
        "posted_spend_since_boundary_usd": posted,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": DAILY_CAP_USD - recorded,
    }
    return updated


def _validate_completed() -> dict[str, Any]:
    daily = _load(DAILY_RESULT)
    ledger = _load(LEDGER)
    expected_predecessor = ledger.get("predecessor", {}).get("support") or {}
    replay = verifier.verify_smoke(
        RUN_DIR,
        primary_dir=primary_daily.SMOKE_DIR,
        expected_predecessor=expected_predecessor,
    )
    saved_verification = _load(RUN_DIR / "VERIFICATION.json")
    if (
        daily.get("status") != "complete_reconciled"
        or daily.get("smoke_result_sha256") != _sha256(RUN_DIR / "RESULT.json")
        or daily.get("smoke_verification_sha256")
        != _sha256(RUN_DIR / "VERIFICATION.json")
        or daily.get("ledger_sha256") != _sha256(LEDGER)
        or _canonical(saved_verification) != _canonical(replay)
        or replay.get("status") != "verified"
        or daily.get("development_opened") is not False
        or daily.get("confirmation_opened") is not False
    ):
        raise RuntimeError("completed factorized smoke does not replay exactly")
    return daily


def execute(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader=None,
    adapter_builder=None,
) -> dict[str, Any]:
    if DAILY_RESULT.is_file():
        return _validate_completed()
    ready = preflight(now=now, live_reader=live_reader, catalog_reader=catalog_reader)
    predecessor = validate_policy_null_predecessor()
    ledger = _initial_ledger(ready)
    checkpoint(LEDGER, ledger)
    if adapter_builder is None:
        adapter_builder = policy.build_adapter
    adapter = adapter_builder(
        stage="smoke",
        run_id="regretbench-factorized-static-likelihood-smoke-20260809",
        output_dir=RUN_DIR,
    )
    try:
        live = live_reader()
        result = smoke.run_smoke(
            output_dir=RUN_DIR,
            adapter=adapter,
            primary_smoke_dir=predecessor["primary_smoke_dir"],
            support_smoke_result=predecessor["support_smoke_result"],
            support_development_result=predecessor["support_development_result"],
            daily_budget_status=_budget_status(ledger, live, now=now),
        )
        replay = verifier.verify_smoke(
            RUN_DIR,
            primary_dir=predecessor["primary_smoke_dir"],
            expected_predecessor=predecessor["support"],
        )
        checkpoint(RUN_DIR / "VERIFICATION.json", replay)
        if replay.get("status") != "verified":
            raise RuntimeError("factorized smoke independent replay failed")
    except Exception as exc:
        ledger = _reconcile(
            ledger,
            status="failed_closed",
            measured=_adapter_cost(adapter),
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "authorizes": "nothing",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "development_opened": False,
            "confirmation_opened": False,
            "ledger_sha256": _sha256(LEDGER),
        }
        checkpoint(DAILY_FAILURE, failure)
        raise
    ledger = _reconcile(
        ledger,
        status=result["status"],
        measured=float(result["usage"]["run_cost_usd"]),
        live=live_reader(),
    )
    checkpoint(LEDGER, ledger)
    daily = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete_reconciled",
        "smoke_status": result["status"],
        "authorizes": result["authorizes"],
        "smoke_result_sha256": _sha256(RUN_DIR / "RESULT.json"),
        "smoke_verification_sha256": _sha256(RUN_DIR / "VERIFICATION.json"),
        "independent_replay_passed": True,
        "development_opened": False,
        "confirmation_opened": False,
        "ledger_sha256": _sha256(LEDGER),
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
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
