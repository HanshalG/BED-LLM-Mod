#!/usr/bin/env python3
"""Execute the sealed Aug 9 RegretBench SMC support-recovery contingency."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import regretbench_deepseek_dynamic_depth2_confirmation_daily as confirmation_daily
from scripts import regretbench_deepseek_dynamic_depth2_policy_daily as policy_daily
from scripts import regretbench_deepseek_smc_support_recovery as core
from scripts import regretbench_deepseek_smc_support_recovery_verify as verifier
from scripts import regretbench_deepseek_support_recovery as primary
from scripts import regretbench_deepseek_support_recovery_daily as primary_daily
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-smc-support-recovery-daily-1"
DATE = "2026-08-09"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
ROOT = REPO_ROOT / (
    "results/nonmyopic/regretbench_deepseek_smc_support_recovery"
)
RUN_DIR = ROOT / "development-20260809"
DAILY_RESULT = ROOT / "DAILY_RESULT.json"
LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-09-regretbench-smc-support-recovery.json"
)
PRIMARY_DIR = primary_daily.DEVELOPMENT_DIR
CORE_SHA256 = (
    "91be9699391aa67070174ca3be2fdb7b6cd9e1ae210fcbb0f5c7e35ba601f280"
)
VERIFIER_SHA256 = (
    "fab0b932c1c00ccb2a7333f63394e16b930cd238c3d5f1da73c8692992fa1e2a"
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
        raise RuntimeError(f"RegretBench SMC contingency can run only on {DATE}")
    return local


def _pristine(path: Path) -> bool:
    if not path.exists():
        return True
    if path.is_file():
        return False
    return not any(path.iterdir())


def _forbidden_primary_descendants() -> list[Path]:
    return [
        policy_daily.SMOKE_DIR,
        policy_daily.NAIVE_SMOKE_DIR,
        policy_daily.DEVELOPMENT_DIR,
        policy_daily.LEDGER,
        policy_daily.ROOT / "DAILY_RESULT.json",
        confirmation_daily.RUN_DIR,
        confirmation_daily.DAILY_RESULT,
        confirmation_daily.LEDGER,
    ]


def _assert_no_primary_descendants_open() -> None:
    opened = [
        str(path)
        for path in _forbidden_primary_descendants()
        if not _pristine(path)
    ]
    if opened:
        raise RuntimeError("primary policy or confirmation artifact is already open")


def validate_primary_predecessor() -> dict[str, Any]:
    result_path = PRIMARY_DIR / "RESULT.json"
    verification_path = PRIMARY_DIR / "VERIFICATION.json"
    authorization = core.validate_primary_null_predecessor(
        result_path, verification_path
    )
    daily_result = _load(primary_daily.ROOT / "DAILY_RESULT.json")
    ledger = _load(primary_daily.LEDGER)
    verification = _load(verification_path)
    primary_raw = PRIMARY_DIR / "private" / "RAW_RESPONSES.json"
    primary_controls = PRIMARY_DIR / "private" / "CONTROLS.json"
    if not primary_raw.is_file() or not primary_controls.is_file():
        raise RuntimeError("primary private artifacts are incomplete")
    verified_artifacts = verification.get("artifact_sha256") or {}
    if (
        daily_result.get("status") != "complete_reconciled"
        or daily_result.get("development_status") != "gated_null"
        or daily_result.get("development_opened") is not True
        or daily_result.get("policy_endpoint_opened") is not False
        or daily_result.get("confirmation_opened") is not False
        or daily_result.get("independent_replay_passed") is not True
        or daily_result.get("development_result_sha256")
        != primary.sha256_file(result_path)
        or daily_result.get("development_verification_sha256")
        != primary.sha256_file(verification_path)
        or daily_result.get("ledger_sha256")
        != primary.sha256_file(primary_daily.LEDGER)
        or ledger.get("date") != "2026-08-08"
        or ledger.get("timezone") != TIMEZONE
        or float(ledger.get("daily_cap_usd", 0.0)) != DAILY_CAP_USD
        or ledger.get("account_wide_usage_counts_against_cap") is not True
        or ledger.get("stages", {}).get("smoke", {}).get("status") != "passed"
        or ledger.get("stages", {}).get("development", {}).get("status")
        != "gated_null"
        or verified_artifacts.get("private/RAW_RESPONSES.json")
        != primary.sha256_file(primary_raw)
        or verified_artifacts.get("private/CONTROLS.json")
        != primary.sha256_file(primary_controls)
    ):
        raise RuntimeError("primary support predecessor is not a clean daily null")
    _assert_no_primary_descendants_open()
    return {
        **authorization,
        "daily_result_sha256": primary.sha256_file(
            primary_daily.ROOT / "DAILY_RESULT.json"
        ),
        "daily_ledger_sha256": primary.sha256_file(primary_daily.LEDGER),
        "primary_raw_sha256": primary.sha256_file(primary_raw),
        "primary_controls_sha256": primary.sha256_file(primary_controls),
        "primary_ledger": ledger,
    }


def _day_boundary(ledger: Mapping[str, Any]) -> float:
    return float(ledger["opening_total_usage_usd"]) + float(
        ledger["recorded_actual_spend_usd"]
    )


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = aug10.read_openrouter_model_catalog,
) -> dict[str, Any]:
    local = _validate_date(now)
    core.validate_protocol_binding()
    if primary.sha256_file(Path(core.__file__).resolve()) != CORE_SHA256:
        raise RuntimeError("SMC support core binding changed")
    if (
        primary.sha256_file(Path(primary_daily.result_verify.__file__).resolve())
        != primary_daily.RESULT_VERIFIER_SHA256
    ):
        raise RuntimeError("primary independent verifier binding changed")
    if primary.sha256_file(Path(verifier.__file__).resolve()) != VERIFIER_SHA256:
        raise RuntimeError("SMC independent verifier binding changed")
    predecessor = validate_primary_predecessor()
    for path in (RUN_DIR, DAILY_RESULT, LEDGER):
        if not _pristine(path):
            raise RuntimeError(f"SMC output path is not pristine: {path}")
    model = primary_daily.validate_deepseek_model_catalog(catalog_reader())
    live = live_reader()
    boundary = _day_boundary(predecessor["primary_ledger"])
    prior_usage = max(0.0, float(live["total_usage_usd"]) - boundary)
    if prior_usage + core.RUN_BUDGET_USD > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("SMC contingency exceeds remaining account-wide day")
    if float(live["balance_usd"]) + 1e-12 < core.RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below the SMC run cap")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": local.date().isoformat(),
        "model": model,
        "predecessor": {
            key: value
            for key, value in predecessor.items()
            if key != "primary_ledger"
        },
        "budget": {
            "daily_cap_usd": DAILY_CAP_USD,
            "opening_total_usage_boundary_usd": boundary,
            "spent_before_smc_usd": prior_usage,
            "run_cap_usd": core.RUN_BUDGET_USD,
            "projected_cost_usd": core.PROJECTED_COST_USD,
            "remaining_after_full_cap_usd": (
                DAILY_CAP_USD - prior_usage - core.RUN_BUDGET_USD
            ),
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
        "recorded_actual_spend_usd": ready["budget"]["spent_before_smc_usd"],
        "account_wide_usage_counts_against_cap": True,
        "opening_boundary_derived_from_reconciled_aug8_close": True,
        "unspent_allowance_does_not_roll_over": True,
        "predecessor": ready["predecessor"],
        "stage": {
            "status": "authorized_pending",
            "maximum_cost_usd": core.RUN_BUDGET_USD,
            "projected_cost_usd": core.PROJECTED_COST_USD,
        },
    }


def _budget_status(
    ledger: Mapping[str, Any],
    live: Mapping[str, float],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    status = require_budget(
        dict(ledger),
        projected_cost_usd=core.RUN_BUDGET_USD,
        total_usage_usd=float(live["total_usage_usd"]),
        now=now,
    )
    status.update(live)
    return status


def _reconcile(
    ledger: Mapping[str, Any],
    *,
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
        raise RuntimeError("SMC reconciliation exceeds the daily cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["stage"].update(
        {"status": status, "actual_cost_usd": measured_cost}
    )
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live["total_credits_usd"]),
        "live_total_usage_usd": float(live["total_usage_usd"]),
        "live_balance_usd": float(live["balance_usd"]),
        "posted_spend_since_boundary_usd": posted,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": DAILY_CAP_USD - recorded,
    }
    return updated


def _call_batch(
    adapter: Any,
    messages: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
) -> list[str]:
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        messages,
        seeds,
        temperature=core.TEMPERATURE,
        response_format=core.child_response_format(),
        max_new_tokens=core.MAX_TOKENS,
    )
    if len(responses) != len(messages):
        raise ValueError("adapter returned the wrong SMC response count")
    return list(responses)


def run_experiment(
    *,
    output_dir: Path,
    adapter: Any,
    daily_budget_status: Mapping[str, Any],
    bootstrap_samples: int = core.BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    core.validate_protocol_binding()
    predecessor = core.validate_primary_null_predecessor(
        PRIMARY_DIR / "RESULT.json", PRIMARY_DIR / "VERIFICATION.json"
    )
    primary_raw_path = PRIMARY_DIR / "private" / "RAW_RESPONSES.json"
    primary_controls_path = PRIMARY_DIR / "private" / "CONTROLS.json"
    raw_primary = _load(primary_raw_path)
    controls_primary = _load(primary_controls_path)
    raw_roots = raw_primary.get("root") or []
    cigs = primary.load_stage_cigs("development")
    if len(raw_roots) != 64 or len(cigs) != 64:
        raise ValueError("primary parent cohort changed")
    control_by_id = {
        row["task_id"]: row for row in controls_primary.get("roots", [])
    }
    messages = []
    seeds = []
    privacy = []
    parents = []
    contexts = []
    for index, (cig, raw_root) in enumerate(zip(cigs, raw_roots, strict=True)):
        parent = core.parse_parent_population(raw_root)
        control = control_by_id.get(cig.cig_id)
        if control is None:
            raise ValueError(f"primary control is missing: {cig.cig_id}")
        root_support = primary.parse_support(raw_root)
        if root_support["questions"][0] != control["question"]:
            raise ValueError("primary selected question changed")
        dialogue = [
            {"role": "assistant", "content": control["question"]},
            {"role": "user", "content": control["mapping"]["answer"]},
        ]
        conditioned_messages, conditioned_audit = core.messages_for(
            cig, dialogue, parent
        )
        blind_messages, blind_audit = core.messages_for(cig, [], parent)
        seed = core.branch_seed(index)
        messages.extend([conditioned_messages, blind_messages])
        seeds.extend([seed, seed])
        privacy.extend([conditioned_audit, blind_audit])
        parents.append(parent)
        contexts.append(
            {
                "cig": cig,
                "control": control,
                "root_support": root_support,
            }
        )
    raw_children = _call_batch(adapter, messages, seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(
        private_dir / "RAW_RESPONSES.json",
        {
            "interface_version": INTERFACE_VERSION,
            "primary_raw_sha256": primary.sha256_file(primary_raw_path),
            "branches": raw_children,
        },
    )
    rows = []
    internal_rows = []
    for index, (context, parent) in enumerate(zip(contexts, parents, strict=True)):
        conditioned = core.parse_child_support(raw_children[2 * index], parent)
        blind = core.parse_child_support(raw_children[2 * index + 1], parent)
        aliases = str(context["control"]["aliases"])
        row = {
            "task_id": context["cig"].cig_id,
            "task_index": index,
            "supported": bool(context["control"]["mapping"]["supported"]),
            "root_covered": primary.truth_covered(context["root_support"], aliases),
            "refresh_seed": seeds[2 * index],
            "parent_population_sha256": parent["raw_parent_sha256"],
            "conditioned_dispatch_index": 2 * index,
            "blind_dispatch_index": 2 * index + 1,
            "conditioned_retained_count": conditioned["diagnostic"][
                "retained_count"
            ],
            "blind_retained_count": blind["diagnostic"]["retained_count"],
            "conditioned_covered": core.truth_covered(conditioned, aliases),
            "blind_covered": core.truth_covered(blind, aliases),
        }
        rows.append(row)
        internal_rows.append(
            {
                **row,
                "conditioned_support": conditioned,
                "blind_support": blind,
            }
        )
    usage = primary.summarize_usage(adapter.usage_snapshot())
    gates = core.mechanics_gates(rows=internal_rows, privacy=privacy, usage=usage)
    science = (
        core.scientific_summary(rows, samples=bootstrap_samples)
        if gates["all_pass"]
        else None
    )
    status = "mechanics_failed"
    if gates["all_pass"]:
        status = "passed" if science and science["gates"]["all_pass"] else "gated_null"
    authorizes = (
        "separately_preregistered_smc_policy_only"
        if status == "passed"
        else "nothing"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": authorizes,
        "protocol": {
            "model": core.MODEL_ID,
            "reasoning": "disabled_excluded",
            "temperature": core.TEMPERATURE,
            "max_tokens": core.MAX_TOKENS,
            "expected_requests": 128,
            "parent_particles": core.PARENT_PARTICLES,
            "child_particles": core.CHILD_PARTICLES,
            "matched_refresh_seed": True,
            "conditioned_blind_dispatch_adjacent": True,
            "initial_support_calls_repeated": False,
            "hidden_cig_exposed_to_model": False,
            "primary_policy_endpoint_opened": False,
            "primary_confirmation_opened": False,
            "smc_policy_endpoint_opened": False,
            "support_recovery_endpoint_accessed": True,
            "protocol_sha256": core.PROTOCOL_SHA256,
            "primary_result_sha256": predecessor["result_sha256"],
            "primary_verification_sha256": predecessor["verification_sha256"],
            "primary_raw_sha256": primary.sha256_file(primary_raw_path),
            "primary_controls_sha256": primary.sha256_file(primary_controls_path),
        },
        "daily_budget_status": dict(daily_budget_status),
        "usage": usage,
        "mechanics_gates": gates,
        "science": science,
        "tasks": rows,
    }
    checkpoint(output_dir / "RESULT.json", result)
    checkpoint(
        private_dir / "CONTROLS.json",
        {
            "primary_raw_sha256": primary.sha256_file(primary_raw_path),
            "primary_controls_sha256": primary.sha256_file(primary_controls_path),
            "privacy": privacy,
        },
    )
    return result


def execute(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
) -> dict[str, Any]:
    ready = preflight(now=now, live_reader=live_reader)
    ledger = _initial_ledger(ready)
    checkpoint(LEDGER, ledger)
    adapter = primary.build_adapter(
        stage="development",
        run_id="regretbench-deepseek-smc-support-recovery-20260809",
        output_dir=RUN_DIR,
    )
    try:
        dispatch_live = live_reader()
        result = run_experiment(
            output_dir=RUN_DIR,
            adapter=adapter,
            daily_budget_status=_budget_status(
                ledger, dispatch_live, now=now
            ),
        )
        verification = verifier.verify(RUN_DIR)
        checkpoint(RUN_DIR / "VERIFICATION.json", verification)
        if verification["status"] != "verified":
            raise RuntimeError("SMC independent replay failed")
    except Exception:
        usage = primary.summarize_usage(adapter.usage_snapshot())
        ledger = _reconcile(
            ledger,
            status="failed_closed",
            measured_cost=float(usage["run_cost_usd"]),
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        raise
    ledger = _reconcile(
        ledger,
        status=result["status"],
        measured_cost=float(result["usage"]["run_cost_usd"]),
        live=live_reader(),
    )
    checkpoint(LEDGER, ledger)
    daily = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete_reconciled",
        "development_status": result["status"],
        "result_sha256": primary.sha256_file(RUN_DIR / "RESULT.json"),
        "verification_sha256": primary.sha256_file(
            RUN_DIR / "VERIFICATION.json"
        ),
        "independent_replay_passed": True,
        "smc_policy_endpoint_opened": False,
        "primary_policy_endpoint_opened": False,
        "primary_confirmation_opened": False,
        "authorizes": result["authorizes"],
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
            "smc_policy_endpoint_opened": False,
        }
        if not args.preflight:
            ROOT.mkdir(parents=True, exist_ok=True)
            checkpoint(ROOT / "DAILY_FAILURE.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"ready_without_paid_calls", "complete_reconciled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
