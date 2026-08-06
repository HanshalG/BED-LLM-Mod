#!/usr/bin/env python3
"""Execute one frozen daily Bongard development block and gated analysis."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import tempfile
import sys
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-development32-daily-execute-1"
TIMEZONE = "Europe/London"
ROOT = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_development32"
)
PROTOCOL_MANIFEST = ROOT / "PROTOCOL_MANIFEST.json"
AUG10_RESULT = aug10.OUTPUT_DIR / "RESULT.json"
MECHANICS_RESULT = aug10.MECHANICS_DIR / "RESULT.json"
BLOCK_DIRS = {
    block_id: ROOT
    / f"block-{block_id}-{development.BLOCK_EARLIEST_DATES[block_id].replace('-', '')}"
    for block_id in development.BLOCK_ORDER
}
BLOCK_RUN_IDS = {
    block_id: (
        f"bongard-openworld-luna-vlm-development32-{block_id}-"
        f"{development.BLOCK_EARLIEST_DATES[block_id].replace('-', '')}"
    )
    for block_id in development.BLOCK_ORDER
}
LEDGERS = {
    block_id: REPO_ROOT
    / (
        "results/nonmyopic/openrouter_daily_budget/"
        f"{development.BLOCK_EARLIEST_DATES[block_id]}.json"
    )
    for block_id in development.BLOCK_ORDER
}
COMBINED_RESULT = ROOT / "COMBINED_RESULT.json"


class PreExecutionGateError(RuntimeError):
    """Raised when an unopened development block fails its runtime gate."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return development.sha256_file(path)


def preflight_fresh_block_runtime(
    *,
    block_id: str,
    block_dir: Path,
    ledger_path: Path,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    model_catalog_reader: Callable[[], dict[str, Any]] = (
        aug10.read_openrouter_model_catalog
    ),
) -> dict[str, Any]:
    """Validate volatile launch conditions without writing or model calls."""
    if block_id not in development.BLOCK_ORDER:
        raise ValueError(f"unknown development block {block_id!r}")
    if ledger_path.exists():
        raise RuntimeError(f"development block {block_id} ledger already exists")
    if block_dir.exists() and (not block_dir.is_dir() or any(block_dir.iterdir())):
        raise RuntimeError(f"development block {block_id} path is not pristine")
    model = aug10._validate_model_catalog(model_catalog_reader())
    live = live_reader()
    values = [
        float(live[field])
        for field in ("total_credits_usd", "total_usage_usd", "balance_usd")
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("live OpenRouter credit values are non-finite")
    if float(live["balance_usd"]) + 1e-12 < 5.0:
        raise RuntimeError("live OpenRouter balance is below the $5 start gate")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "block_id": block_id,
        "date": development.BLOCK_EARLIEST_DATES[block_id],
        "model": model,
        "live_credits": live,
        "budget": {
            "account_wide_daily_cap_usd": 5.0,
            "block_maximum_cost_usd": development.RUN_BUDGET_USD,
            "unspent_allowance_does_not_roll_over": True,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def _validate_date(block_id: str, now: datetime | None = None) -> None:
    if block_id not in development.BLOCK_ORDER:
        raise ValueError(f"unknown development block {block_id!r}")
    timezone = ZoneInfo(TIMEZONE)
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    expected = development.BLOCK_EARLIEST_DATES[block_id]
    if local_now.date().isoformat() != expected:
        raise RuntimeError(f"development block {block_id} can run only on {expected}")


def validate_aug10_authorization(
    *, wrapper_result: Path = AUG10_RESULT, mechanics_result: Path = MECHANICS_RESULT
) -> dict[str, Any]:
    wrapper = _load(wrapper_result)
    mechanics_component = (wrapper.get("components") or {}).get("mechanics") or {}
    if (
        wrapper.get("interface_version") != aug10.INTERFACE_VERSION
        or wrapper.get("status") != "complete"
        or wrapper.get("authorizes_development") is not True
        or Path(mechanics_component.get("artifact", "")).resolve()
        != mechanics_result.resolve()
        or mechanics_component.get("artifact_sha256") != _sha256(mechanics_result)
        or mechanics_component.get("status") != "mechanics_pass"
        or mechanics_component.get("verified") is not True
    ):
        raise RuntimeError("August 10 wrapper does not authorize development")
    mechanics_verification = development.verify_mechanics_result(mechanics_result)
    if mechanics_verification["result_sha256"] != mechanics_component[
        "artifact_sha256"
    ]:
        raise RuntimeError("August 10 mechanics result changed")
    return {
        "verified": True,
        "wrapper_result_sha256": _sha256(wrapper_result),
        "mechanics_result_sha256": mechanics_verification["result_sha256"],
    }


def _block_result_paths() -> dict[str, Path]:
    return {
        block_id: BLOCK_DIRS[block_id] / "RESULT.json"
        for block_id in development.BLOCK_ORDER
    }


def validate_block_result(
    *,
    path: Path,
    block_id: str,
    ledger_path: Path | None = None,
    planning_tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    tasks = list(planning_tasks) if planning_tasks is not None else (
        bed.load_validation_partition_tasks(
            "development", include_endpoint_labels=False
        )
    )
    replay = development.replay_block(
        result_path=path, all_development_tasks=tasks
    )
    if replay["block_id"] != block_id:
        raise RuntimeError("development result block ID changed")
    execution_path = path.parent / "EXECUTION.json"
    execution = _load(execution_path)
    ledger_verification = (
        validate_block_ledger(path=ledger_path, block_id=block_id)
        if ledger_path is not None
        else None
    )
    if (
        execution.get("interface_version") != development.INTERFACE_VERSION
        or execution.get("block_id") != block_id
        or execution.get("status") != "complete_reconciled"
        or execution.get("result_sha256") != replay["result_sha256"]
        or (
            ledger_verification is not None
            and execution.get("ledger_sha256")
            != ledger_verification["ledger_sha256"]
        )
    ):
        raise RuntimeError("development block execution is not reconciled")
    verification = {
        "verified": True,
        "block_id": block_id,
        "result_sha256": replay["result_sha256"],
        "raw_responses_sha256": replay["raw_responses_sha256"],
        "execution_sha256": _sha256(execution_path),
        "protocol_manifest_sha256": replay["protocol_manifest_sha256"],
    }
    if ledger_verification is not None:
        verification["ledger_sha256"] = ledger_verification["ledger_sha256"]
        verification["recorded_daily_spend_usd"] = ledger_verification[
            "recorded_daily_spend_usd"
        ]
    return verification


def validate_block_ledger(*, path: Path, block_id: str) -> dict[str, Any]:
    ledger = _load(path)
    authorization = ledger.get("first_authorized_block") or {}
    block_record = ledger.get(
        f"bongard_luna_vlm_development_block_{block_id}"
    ) or {}
    reconciliation = ledger.get("reconciliation") or {}
    recorded = float(ledger.get("recorded_actual_spend_usd", math.inf))
    block_cost = float(block_record.get("actual_cost_usd", math.inf))
    remaining = float(
        reconciliation.get("remaining_daily_allowance_usd", math.inf)
    )
    if (
        block_id not in development.BLOCK_ORDER
        or ledger.get("date") != development.BLOCK_EARLIEST_DATES[block_id]
        or ledger.get("timezone") != TIMEZONE
        or float(ledger.get("daily_cap_usd", 0.0)) != 5.0
        or ledger.get("account_wide_usage_counts_against_cap") is not True
        or ledger.get("unspent_allowance_does_not_roll_over") is not True
        or ledger.get("additional_paid_blocks_authorized") is not False
        or not math.isfinite(recorded)
        or not 0.0 <= recorded <= 5.0
        or not math.isfinite(block_cost)
        or not 0.0 <= block_cost <= development.RUN_BUDGET_USD
        or authorization.get("interface_version")
        != development.INTERFACE_VERSION
        or authorization.get("block_id") != block_id
        or authorization.get("model") != development.MODEL_ID
        or float(authorization.get("maximum_cost_usd", 0.0))
        != development.RUN_BUDGET_USD
        or authorization.get("status") != "block_mechanics_pass"
        or float(authorization.get("actual_cost_usd", math.inf)) != block_cost
        or block_record.get("interface_version")
        != development.INTERFACE_VERSION
        or block_record.get("model") != development.MODEL_ID
        or float(block_record.get("maximum_cost_usd", 0.0))
        != development.RUN_BUDGET_USD
        or block_record.get("status") != "block_mechanics_pass"
        or not math.isfinite(remaining)
        or remaining != max(0.0, 5.0 - recorded)
    ):
        raise RuntimeError(f"development block {block_id} ledger is invalid")
    return {
        "verified": True,
        "ledger_sha256": _sha256(path),
        "recorded_daily_spend_usd": recorded,
        "block_cost_usd": block_cost,
    }


def verify_combined_result(
    *,
    result_path: Path,
    block_results: Sequence[Path],
    all_development_tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    observed = _load(result_path)
    with tempfile.TemporaryDirectory(prefix="bongard-development-replay-") as tmp:
        replay = development.analyze_combined(
            block_results=block_results,
            output_path=Path(tmp) / "COMBINED_RESULT.json",
            all_development_tasks=all_development_tasks,
        )
    if bed.canonical_json(replay) != bed.canonical_json(observed):
        raise RuntimeError("combined development result does not independently replay")
    return {
        "verified": True,
        "status": observed["status"],
        "result_sha256": _sha256(result_path),
        "authorizes_confirmation_preregistration": observed[
            "authorizes_confirmation_preregistration"
        ],
    }


def _validate_daily_record(
    *,
    path: Path,
    block_id: str,
    result_path: Path,
    ledger_path: Path,
    manifest_verification: Mapping[str, Any],
    aug10_verification: Mapping[str, Any],
    block_verification: Mapping[str, Any],
) -> dict[str, Any]:
    daily = _load(path)
    combined_accessed = daily.get("combined_endpoint_accessed")
    expected_status = (
        "complete_combined_verified"
        if combined_accessed is True
        else "block_complete_verified"
    )
    if (
        daily.get("schema_version") != SCHEMA_VERSION
        or daily.get("interface_version") != INTERFACE_VERSION
        or daily.get("status") != expected_status
        or daily.get("block_id") != block_id
        or daily.get("date") != development.BLOCK_EARLIEST_DATES[block_id]
        or daily.get("block_result_path") != str(result_path)
        or daily.get("ledger_path") != str(ledger_path)
        or daily.get("protocol_manifest_sha256")
        != manifest_verification["manifest_sha256"]
        or daily.get("aug10_verification") != aug10_verification
        or daily.get("block_verification") != block_verification
        or type(combined_accessed) is not bool
        or daily.get("confirmation_accessed") is not False
        or daily.get("sealed_test_accessed") is not False
        or (
            block_id != development.BLOCK_ORDER[-1]
            and combined_accessed is not False
        )
        or (
            combined_accessed is True
            and (
                not isinstance(daily.get("combined_verification"), dict)
                or "authorizes_confirmation_preregistration" not in daily
            )
        )
    ):
        raise RuntimeError("banked daily development execution changed")
    return daily


def preflight_daily_block(
    *,
    block_id: str,
    block_dir: Path | None = None,
    ledger_path: Path | None = None,
    protocol_manifest: Path = PROTOCOL_MANIFEST,
    aug10_result: Path = AUG10_RESULT,
    mechanics_result: Path = MECHANICS_RESULT,
    combined_result: Path = COMBINED_RESULT,
    block_result_paths: Mapping[str, Path] | None = None,
    ledger_paths: Mapping[str, Path] | None = None,
    manifest_validator: Callable[..., dict[str, Any]] = (
        development.verify_protocol_manifest
    ),
    aug10_validator: Callable[..., dict[str, Any]] = validate_aug10_authorization,
    block_validator: Callable[..., dict[str, Any]] = validate_block_result,
    daily_validator: Callable[..., dict[str, Any]] = _validate_daily_record,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    model_catalog_reader: Callable[[], dict[str, Any]] = (
        aug10.read_openrouter_model_catalog
    ),
) -> dict[str, Any]:
    """Report complete predecessor and runtime readiness without writes."""
    if block_id not in development.BLOCK_ORDER:
        raise ValueError(f"unknown development block {block_id!r}")
    block_dir = block_dir or BLOCK_DIRS[block_id]
    ledgers = dict(ledger_paths or LEDGERS)
    if set(ledgers) != set(development.BLOCK_ORDER):
        raise ValueError("development preflight requires all four ledger paths")
    ledger_path = ledger_path or ledgers[block_id]
    ledgers[block_id] = ledger_path
    paths = dict(block_result_paths or _block_result_paths())
    if set(paths) != set(development.BLOCK_ORDER):
        raise ValueError("development preflight requires all four result paths")
    if paths[block_id] != block_dir / "RESULT.json":
        raise ValueError("block directory and result path do not agree")
    if combined_result.exists():
        raise RuntimeError("combined endpoint result exists before fresh block")

    target_index = development.BLOCK_ORDER.index(block_id)
    for future_id in development.BLOCK_ORDER[target_index + 1 :]:
        future_result = paths[future_id]
        future_dir = future_result.parent
        if (
            ledgers[future_id].exists()
            or future_result.exists()
            or (future_dir.exists() and any(future_dir.iterdir()))
        ):
            raise RuntimeError(
                f"future development block {future_id} exists out of order"
            )

    manifest_verification = manifest_validator(protocol_manifest)
    runtime = preflight_fresh_block_runtime(
        block_id=block_id,
        block_dir=block_dir,
        ledger_path=ledger_path,
        live_reader=live_reader,
        model_catalog_reader=model_catalog_reader,
    )

    aug10_exists = aug10_result.is_file()
    mechanics_exists = mechanics_result.is_file()
    prior_blocks: dict[str, Any] = {}
    if aug10_exists != mechanics_exists:
        raise RuntimeError("August 10 development predecessor is partial")
    if aug10_exists:
        aug10_verification: dict[str, Any] = aug10_validator(
            wrapper_result=aug10_result,
            mechanics_result=mechanics_result,
        )
    else:
        aug10_verification = {
            "verified": False,
            "status": "waiting_for_aug10",
            "wrapper_result": str(aug10_result),
            "mechanics_result": str(mechanics_result),
        }

    waiting_for = None if aug10_verification["verified"] else "aug10"
    required = development.BLOCK_ORDER[:target_index]
    for position, prior_id in enumerate(required):
        result_path = paths[prior_id]
        daily_path = result_path.parent / "DAILY_EXECUTION.json"
        prior_ledger = ledgers[prior_id]
        present = (
            result_path.is_file(),
            daily_path.is_file(),
            prior_ledger.is_file(),
        )
        if any(present) and not all(present):
            raise RuntimeError(
                f"development predecessor block {prior_id} is partial"
            )
        if not any(present):
            later_required = required[position + 1 :]
            if any(
                paths[item].exists()
                or ledgers[item].exists()
                or (
                    paths[item].parent.exists()
                    and any(paths[item].parent.iterdir())
                )
                for item in later_required
            ):
                raise RuntimeError("development predecessors exist out of order")
            if waiting_for is None:
                waiting_for = f"block_{prior_id}"
            break
        if not aug10_verification["verified"]:
            raise RuntimeError("development block exists before August 10 approval")
        verification = block_validator(
            path=result_path,
            block_id=prior_id,
            ledger_path=prior_ledger,
        )
        daily_validator(
            path=daily_path,
            block_id=prior_id,
            result_path=result_path,
            ledger_path=prior_ledger,
            manifest_verification=manifest_verification,
            aug10_verification=aug10_verification,
            block_verification=verification,
        )
        prior_blocks[prior_id] = verification

    status = (
        "ready_without_paid_calls"
        if waiting_for is None
        else f"waiting_for_{waiting_for}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "block_id": block_id,
        "date": development.BLOCK_EARLIEST_DATES[block_id],
        "protocol_manifest": manifest_verification,
        "aug10_predecessor": aug10_verification,
        "prior_blocks": prior_blocks,
        "runtime": runtime,
        "model": runtime["model"],
        "live_credits": runtime["live_credits"],
        "budget": runtime["budget"],
        "model_calls_made": 0,
        "files_written": 0,
    }


def execute_daily_block(
    *,
    block_id: str,
    block_dir: Path | None = None,
    ledger_path: Path | None = None,
    protocol_manifest: Path = PROTOCOL_MANIFEST,
    aug10_result: Path = AUG10_RESULT,
    mechanics_result: Path = MECHANICS_RESULT,
    combined_result: Path = COMBINED_RESULT,
    block_result_paths: Mapping[str, Path] | None = None,
    ledger_paths: Mapping[str, Path] | None = None,
    now: datetime | None = None,
    manifest_validator: Callable[..., dict[str, Any]] = (
        development.verify_protocol_manifest
    ),
    aug10_validator: Callable[..., dict[str, Any]] = validate_aug10_authorization,
    block_executor: Callable[..., dict[str, Any]] = development.execute_block,
    block_validator: Callable[..., dict[str, Any]] = validate_block_result,
    analyzer: Callable[..., dict[str, Any]] = development.analyze_combined,
    combined_validator: Callable[..., dict[str, Any]] = verify_combined_result,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    fresh_preflight: Callable[..., dict[str, Any]] = (
        preflight_fresh_block_runtime
    ),
) -> dict[str, Any]:
    _validate_date(block_id, now)
    block_dir = block_dir or BLOCK_DIRS[block_id]
    ledgers = dict(ledger_paths or LEDGERS)
    if set(ledgers) != set(development.BLOCK_ORDER):
        raise ValueError("daily driver requires all four fixed ledger paths")
    ledger_path = ledger_path or ledgers[block_id]
    ledgers[block_id] = ledger_path
    paths = dict(block_result_paths or _block_result_paths())
    if set(paths) != set(development.BLOCK_ORDER):
        raise ValueError("daily driver requires all four fixed result paths")
    if paths[block_id] != block_dir / "RESULT.json":
        raise ValueError("block directory and result path do not agree")
    if (
        block_id == development.BLOCK_ORDER[-1]
        and combined_result.exists()
        and not paths[block_id].exists()
    ):
        raise RuntimeError("combined endpoint result exists before block D completed")
    if block_id != development.BLOCK_ORDER[-1] and combined_result.exists():
        raise RuntimeError("combined endpoint result exists before block D")
    manifest_verification = manifest_validator(protocol_manifest)
    aug10_verification = aug10_validator(
        wrapper_result=aug10_result,
        mechanics_result=mechanics_result,
    )
    required = development.BLOCK_ORDER[: development.BLOCK_ORDER.index(block_id)]
    previous = [paths[item] for item in required]
    for prior_id, path in zip(required, previous, strict=True):
        if not path.is_file():
            raise RuntimeError(f"required prior block {prior_id} is missing")
        verification = block_validator(
            path=path,
            block_id=prior_id,
            ledger_path=ledgers[prior_id],
        )
        prior_daily_path = path.parent / "DAILY_EXECUTION.json"
        if not prior_daily_path.is_file():
            raise RuntimeError(f"required prior daily block {prior_id} is missing")
        _validate_daily_record(
            path=prior_daily_path,
            block_id=prior_id,
            result_path=path,
            ledger_path=ledgers[prior_id],
            manifest_verification=manifest_verification,
            aug10_verification=aug10_verification,
            block_verification=verification,
        )

    daily_path = block_dir / "DAILY_EXECUTION.json"
    result_path = paths[block_id]
    if daily_path.exists():
        verification = block_validator(
            path=result_path,
            block_id=block_id,
            ledger_path=ledger_path,
        )
        daily = _validate_daily_record(
            path=daily_path,
            block_id=block_id,
            result_path=result_path,
            ledger_path=ledger_path,
            manifest_verification=manifest_verification,
            aug10_verification=aug10_verification,
            block_verification=verification,
        )
    else:
        failure_path = block_dir / "FAILURE.json"
        if failure_path.exists():
            raise RuntimeError(f"development block {block_id} already failed closed")
        if result_path.exists():
            if not (block_dir / "EXECUTION.json").is_file():
                raise RuntimeError("block result exists without reconciled execution")
        elif block_dir.exists() and any(block_dir.iterdir()):
            raise RuntimeError("partial development block exists without a result")
        else:
            if not ledger_path.exists():
                try:
                    preflight = fresh_preflight(
                        block_id=block_id,
                        block_dir=block_dir,
                        ledger_path=ledger_path,
                        live_reader=live_reader,
                    )
                except Exception as exc:
                    raise PreExecutionGateError(str(exc)) from exc
                if preflight.get("status") != "ready_without_paid_calls":
                    raise PreExecutionGateError(
                        f"fresh development block {block_id} was not authorized"
                    )
                live_opening = preflight.get("live_credits")
                if not isinstance(live_opening, dict):
                    raise PreExecutionGateError(
                        "fresh preflight omitted the live credit snapshot"
                    )
                development._initialize_daily_ledger(
                    path=ledger_path,
                    live=live_opening,
                    block_id=block_id,
                    now=now,
                )
            block_executor(
                output_dir=block_dir,
                run_id=BLOCK_RUN_IDS[block_id],
                block_id=block_id,
                mechanics_result=mechanics_result,
                protocol_manifest=protocol_manifest,
                previous_results=previous,
                ledger_path=ledger_path,
                now=now,
                live_reader=live_reader,
            )
        verification = block_validator(
            path=result_path,
            block_id=block_id,
            ledger_path=ledger_path,
        )
        daily = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "block_complete_verified",
            "block_id": block_id,
            "date": development.BLOCK_EARLIEST_DATES[block_id],
            "block_result_path": str(result_path),
            "ledger_path": str(ledger_path),
            "protocol_manifest_sha256": manifest_verification[
                "manifest_sha256"
            ],
            "aug10_verification": aug10_verification,
            "block_verification": verification,
            "combined_endpoint_accessed": False,
            "confirmation_accessed": False,
            "sealed_test_accessed": False,
        }
        checkpoint(daily_path, daily)

    if block_id != development.BLOCK_ORDER[-1]:
        if combined_result.exists():
            raise RuntimeError("combined endpoint result opened before block D")
        return daily

    all_paths = [paths[item] for item in development.BLOCK_ORDER]
    all_verifications = {}
    for item, path in zip(development.BLOCK_ORDER, all_paths, strict=True):
        all_verifications[item] = block_validator(
            path=path,
            block_id=item,
            ledger_path=ledgers[item],
        )
    if daily["combined_endpoint_accessed"] is True and not combined_result.exists():
        raise RuntimeError("verified combined result artifact is missing")
    if not combined_result.exists():
        analyzer(block_results=all_paths, output_path=combined_result)
    combined_verification = combined_validator(
        result_path=combined_result, block_results=all_paths
    )
    if daily["combined_endpoint_accessed"] is True:
        if (
            daily.get("combined_verification") != combined_verification
            or daily.get("all_block_verifications") != all_verifications
            or daily.get("authorizes_confirmation_preregistration")
            != combined_verification[
                "authorizes_confirmation_preregistration"
            ]
        ):
            raise RuntimeError("banked combined development execution changed")
        return daily
    daily["status"] = "complete_combined_verified"
    daily["combined_endpoint_accessed"] = True
    daily["all_block_verifications"] = all_verifications
    daily["combined_verification"] = combined_verification
    daily["authorizes_confirmation_preregistration"] = combined_verification[
        "authorizes_confirmation_preregistration"
    ]
    checkpoint(daily_path, daily)
    return daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=development.BLOCK_ORDER, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    function = preflight_daily_block if args.preflight else execute_daily_block
    result = function(block_id=args.block)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
