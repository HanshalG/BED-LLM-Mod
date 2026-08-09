#!/usr/bin/env python3
"""Run one frozen Bongard development block with its naive baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_development32_daily_execute as main_daily
from scripts import bongard_openworld_luna_naive_first_link as naive
from scripts import bongard_openworld_luna_naive_first_link_daily_execute as naive_daily
from scripts import bongard_openworld_luna_vlm_development as development


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-development-daily-handoff-1"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_DEVELOPMENT_DAILY_HANDOFF_PROTOCOL_20260809.md"
)
PROTOCOL_SHA256 = (
    "6453a42e866f025d461f72888d41b958aa0d00f6750e434f7200627580a3a2a4"
)
BOUND_IMPLEMENTATIONS = {
    "main_daily": (
        "scripts/bongard_openworld_luna_development32_daily_execute.py",
        "f7caea8467c5f9d88d3634742e8e842edf9c50fd53855f1a38c81ce9601ef5fb",
    ),
    "naive_daily": (
        "scripts/bongard_openworld_luna_naive_first_link_daily_execute.py",
        "38f817234967f333c23cfcf5b1e83db44a3e0cd80da2bcbdc06bb3aab75e1d74",
    ),
}
ROOT = REPO_ROOT / "results/nonmyopic/bongard_openworld_development_daily_handoff"
OUTPUT_DIRS = {
    block_id: ROOT
    / (
        f"block-{block_id}-"
        f"{development.BLOCK_EARLIEST_DATES[block_id].replace('-', '')}"
    )
    for block_id in development.BLOCK_ORDER
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _component(path: Path, *, status: Any = None) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "status": status,
    }


def verify_bindings() -> dict[str, Any]:
    protocol_hash = _sha256(PROTOCOL)
    if protocol_hash != PROTOCOL_SHA256:
        raise ValueError("development daily-handoff protocol changed")
    observed = {
        name: {"path": relative, "sha256": _sha256(REPO_ROOT / relative)}
        for name, (relative, _) in BOUND_IMPLEMENTATIONS.items()
    }
    expected = {
        name: {"path": relative, "sha256": expected_hash}
        for name, (relative, expected_hash) in BOUND_IMPLEMENTATIONS.items()
    }
    if observed != expected:
        raise ValueError(
            "bound development daily implementation changed: "
            f"expected {expected}, observed {observed}"
        )
    return {
        "protocol": {"path": str(PROTOCOL), "sha256": protocol_hash},
        "implementations": observed,
    }


def preflight_daily_handoff(
    *,
    block_id: str,
    output_dir: Path | None = None,
    naive_block_dir: Path | None = None,
    naive_ledger: Path | None = None,
    main_preflight: Callable[..., dict[str, Any]] = main_daily.preflight_daily_block,
    catalog_reader: Callable[[], dict[str, Any]] = (
        main_daily.aug10.read_openrouter_model_catalog
    ),
    binding_verifier: Callable[[], dict[str, Any]] = verify_bindings,
) -> dict[str, Any]:
    if block_id not in development.BLOCK_ORDER:
        raise ValueError("unknown development daily-handoff block")
    output_dir = output_dir or OUTPUT_DIRS[block_id]
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise RuntimeError(f"development handoff path is not pristine: {output_dir}")
    naive_dir = naive_block_dir or naive_daily.BLOCK_DIRS[block_id]
    naive_ledger = naive_ledger or naive_daily.SUPPLEMENTAL_LEDGERS[block_id]
    if naive_dir.exists() and (not naive_dir.is_dir() or any(naive_dir.iterdir())):
        raise RuntimeError(f"naive block path is not pristine: {naive_dir}")
    if naive_ledger.exists():
        raise RuntimeError("naive supplemental ledger already exists")
    bindings = binding_verifier()
    main = main_preflight(block_id=block_id)
    model = naive_daily._validate_model_catalog(catalog_reader())
    protocol = naive_daily.protocol_verify.verify_protocol_manifest()
    smoke = naive.verify_smoke_result(naive_daily.SMOKE_RESULT)
    combined_cap = development.RUN_BUDGET_USD + naive.RUN_BUDGET_USD
    if not math.isclose(combined_cap, 4.95, abs_tol=1e-12):
        raise ValueError("combined development and naive run cap changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": main["status"],
        "block_id": block_id,
        "date": development.BLOCK_EARLIEST_DATES[block_id],
        "bindings": bindings,
        "main_preflight": main,
        "naive_model": model,
        "naive_protocol_manifest_sha256": protocol["manifest_sha256"],
        "naive_smoke_result_sha256": smoke["result_sha256"],
        "budget": {
            "account_wide_daily_cap_usd": 5.0,
            "main_run_cap_usd": development.RUN_BUDGET_USD,
            "naive_run_cap_usd": naive.RUN_BUDGET_USD,
            "combined_run_cap_usd": combined_cap,
            "unallocated_cap_usd": 5.0 - combined_cap,
            "naive_reauthorizes_after_observed_main_spend": True,
        },
        "naive_full_preflight_deferred_until_main_complete": True,
        "model_calls_made": 0,
        "files_written": 0,
    }


def _validate_main_bundle(block_id: str) -> dict[str, Any]:
    result_path = main_daily.BLOCK_DIRS[block_id] / "RESULT.json"
    daily_path = main_daily.BLOCK_DIRS[block_id] / "DAILY_EXECUTION.json"
    ledger_path = main_daily.LEDGERS[block_id]
    verification = main_daily.validate_block_result(
        path=result_path,
        block_id=block_id,
        ledger_path=ledger_path,
    )
    manifest = development.verify_protocol_manifest(main_daily.PROTOCOL_MANIFEST)
    aug10 = main_daily.validate_aug10_authorization(
        wrapper_result=main_daily.AUG10_RESULT,
        mechanics_result=main_daily.MECHANICS_RESULT,
    )
    daily = main_daily._validate_daily_record(
        path=daily_path,
        block_id=block_id,
        result_path=result_path,
        ledger_path=ledger_path,
        manifest_verification=manifest,
        aug10_verification=aug10,
        block_verification=verification,
    )
    return {
        "verification": verification,
        "daily": daily,
        "paths": {
            "result": result_path,
            "daily_execution": daily_path,
            "ledger": ledger_path,
        },
    }


def _validate_naive_bundle(block_id: str) -> dict[str, Any]:
    predecessor = naive_daily._main_predecessor(block_id)
    verification = naive_daily._naive_predecessor(block_id)
    result_path = naive_daily.BLOCK_DIRS[block_id] / "RESULT.json"
    execution_path = naive_daily.BLOCK_DIRS[block_id] / "EXECUTION.json"
    ledger_path = naive_daily.SUPPLEMENTAL_LEDGERS[block_id]
    main_ledger = _load(main_daily.LEDGERS[block_id])
    ledger = _load(ledger_path)
    if ledger.get("main_predecessor") != predecessor:
        raise ValueError("naive ledger main predecessor changed")
    for field in (
        "opening_total_credits_usd",
        "opening_total_usage_usd",
        "opening_balance_usd",
    ):
        if float(ledger.get(field, math.nan)) != float(
            main_ledger.get(field, math.nan)
        ):
            raise ValueError("naive ledger does not share main account opening")
    recorded = float(ledger.get("recorded_actual_spend_usd", math.inf))
    if not 0.0 <= recorded <= 5.0:
        raise ValueError("paired development day exceeds the account-wide cap")
    return {
        "verification": verification,
        "paths": {
            "result": result_path,
            "execution": execution_path,
            "ledger": ledger_path,
        },
        "recorded_daily_spend_usd": recorded,
    }


def _records(bundle: Mapping[str, Any], *, statuses: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _component(path, status=statuses.get(name))
        for name, path in bundle["paths"].items()
    }


def _bank_failure(
    *,
    path: Path,
    block_id: str,
    bindings: Mapping[str, Any],
    failed_stage: str,
    error: Exception | None,
    components: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "failed_closed",
        "block_id": block_id,
        "date": development.BLOCK_EARLIEST_DATES[block_id],
        "bindings": dict(bindings),
        "failed_stage": failed_stage,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else "banked child failure",
        "components": dict(components),
        "authorizes_paid_calls": False,
        "authorizes_rerun": False,
        "authorizes_later_block": False,
        "this_record_authorizes_confirmation": False,
    }
    _write_once(path, record)
    return record


def _validate_terminal(
    *,
    record: Mapping[str, Any],
    path: Path,
    block_id: str,
    bindings: Mapping[str, Any],
    main_validator: Callable[[str], dict[str, Any]],
    naive_validator: Callable[[str], dict[str, Any]],
) -> None:
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("interface_version") != INTERFACE_VERSION
        or record.get("block_id") != block_id
        or record.get("bindings") != bindings
        or record.get("status") not in {"paired_daily_complete", "failed_closed"}
        or record.get("authorizes_paid_calls") is not False
        or record.get("authorizes_rerun") is not False
        or record.get("this_record_authorizes_confirmation") is not False
    ):
        raise ValueError("banked development daily handoff changed")
    components = record.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("banked development handoff components are missing")
    for component in components.values():
        component_path = Path(str(component.get("path", "")))
        if (
            not component_path.is_file()
            or component.get("sha256") != _sha256(component_path)
        ):
            raise ValueError("banked development handoff component changed")
    if "main_result" in components:
        main_validator(block_id)
    if "naive_result" in components:
        naive_validator(block_id)
    if path.name == "RESULT.json" and record.get("status") != "paired_daily_complete":
        raise ValueError("development handoff result status changed")
    if path.name == "FAILURE.json" and record.get("status") != "failed_closed":
        raise ValueError("development handoff failure status changed")


def run_daily_handoff(
    *,
    block_id: str,
    output_dir: Path | None = None,
    main_block_dir: Path | None = None,
    naive_block_dir: Path | None = None,
    main_runner: Callable[..., dict[str, Any]] = main_daily.execute_daily_block,
    naive_runner: Callable[..., dict[str, Any]] = naive_daily.execute_block,
    main_validator: Callable[[str], dict[str, Any]] = _validate_main_bundle,
    naive_validator: Callable[[str], dict[str, Any]] = _validate_naive_bundle,
    binding_verifier: Callable[[], dict[str, Any]] = verify_bindings,
) -> dict[str, Any]:
    if block_id not in development.BLOCK_ORDER:
        raise ValueError("unknown development daily-handoff block")
    output_dir = output_dir or OUTPUT_DIRS[block_id]
    main_block_dir = main_block_dir or main_daily.BLOCK_DIRS[block_id]
    naive_block_dir = naive_block_dir or naive_daily.BLOCK_DIRS[block_id]
    bindings = binding_verifier()
    result_path = output_dir / "RESULT.json"
    failure_path = output_dir / "FAILURE.json"
    terminal = [path for path in (result_path, failure_path) if path.exists()]
    if len(terminal) > 1:
        raise RuntimeError("ambiguous development daily-handoff artifacts")
    if terminal:
        record = _load(terminal[0])
        _validate_terminal(
            record=record,
            path=terminal[0],
            block_id=block_id,
            bindings=bindings,
            main_validator=main_validator,
            naive_validator=naive_validator,
        )
        return record

    components: dict[str, Any] = {}
    main_failure = main_block_dir / "FAILURE.json"
    if main_failure.exists():
        components["main_failure"] = _component(main_failure, status="failed_closed")
        return _bank_failure(
            path=failure_path,
            block_id=block_id,
            bindings=bindings,
            failed_stage="main",
            error=None,
            components=components,
        )
    try:
        main_runner(block_id=block_id)
        main = main_validator(block_id)
    except Exception as exc:
        if not main_failure.exists():
            raise
        components["main_failure"] = _component(main_failure, status="failed_closed")
        return _bank_failure(
            path=failure_path,
            block_id=block_id,
            bindings=bindings,
            failed_stage="main",
            error=exc,
            components=components,
        )
    components.update(
        {
            f"main_{name}": value
            for name, value in _records(
                main,
                statuses={
                    "result": main["verification"].get("status"),
                    "daily_execution": main["daily"].get("status"),
                    "ledger": "reconciled",
                },
            ).items()
        }
    )

    naive_failure = naive_block_dir / "FAILURE.json"
    if naive_failure.exists():
        components["naive_failure"] = _component(
            naive_failure, status="failed_closed"
        )
        return _bank_failure(
            path=failure_path,
            block_id=block_id,
            bindings=bindings,
            failed_stage="naive",
            error=None,
            components=components,
        )
    try:
        naive_runner(block_id=block_id)
        paired = naive_validator(block_id)
    except Exception as exc:
        if not naive_failure.exists():
            raise
        components["naive_failure"] = _component(
            naive_failure, status="failed_closed"
        )
        return _bank_failure(
            path=failure_path,
            block_id=block_id,
            bindings=bindings,
            failed_stage="naive",
            error=exc,
            components=components,
        )
    components.update(
        {
            f"naive_{name}": value
            for name, value in _records(
                paired,
                statuses={
                    "result": paired["verification"].get("status"),
                    "execution": "complete_reconciled",
                    "ledger": "reconciled",
                },
            ).items()
        }
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "paired_daily_complete",
        "block_id": block_id,
        "date": development.BLOCK_EARLIEST_DATES[block_id],
        "bindings": bindings,
        "components": components,
        "recorded_daily_spend_usd": paired["recorded_daily_spend_usd"],
        "main_completed_before_naive": True,
        "naive_required_regardless_of_main_endpoint": True,
        "authorizes_paid_calls": False,
        "authorizes_rerun": False,
        "authorizes_later_block": False,
        "this_record_authorizes_confirmation": False,
    }
    _write_once(result_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=development.BLOCK_ORDER, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = (
        preflight_daily_handoff(block_id=args.block)
        if args.preflight
        else run_daily_handoff(block_id=args.block)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "failed_closed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
