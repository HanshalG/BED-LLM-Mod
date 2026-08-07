#!/usr/bin/env python3
"""Bank the frozen Bongard development claim and verify its confirmation handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_claim_report as claim_report
from scripts import bongard_openworld_luna_confirmation64 as confirmation
from scripts import bongard_openworld_luna_development32_daily_execute as daily
from scripts import bongard_openworld_luna_vlm_development as development


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-development-claim-finalize-1"
CLAIM_REPORT = daily.ROOT / "CLAIM_REPORT.json"


def _block_results() -> dict[str, Path]:
    return {
        block_id: daily.BLOCK_DIRS[block_id] / "RESULT.json"
        for block_id in development.BLOCK_ORDER
    }


def finalize_development_claim(
    *,
    combined_result: Path = daily.COMBINED_RESULT,
    claim_report_path: Path = CLAIM_REPORT,
    block_results: Mapping[str, Path] | None = None,
    ledgers: Mapping[str, Path] | None = None,
    manifest_validator: Callable[..., dict[str, Any]] = (
        development.verify_protocol_manifest
    ),
    aug10_validator: Callable[..., dict[str, Any]] = (
        daily.validate_aug10_authorization
    ),
    block_validator: Callable[..., dict[str, Any]] = daily.validate_block_result,
    daily_validator: Callable[..., dict[str, Any]] = daily._validate_daily_record,
    combined_validator: Callable[..., dict[str, Any]] = daily.verify_combined_result,
    authorization_validator: Callable[[], dict[str, Any]] = (
        confirmation.verify_development_authorization
    ),
) -> dict[str, Any]:
    paths = dict(block_results or _block_results())
    ledger_paths = dict(ledgers or daily.LEDGERS)
    expected_blocks = set(development.BLOCK_ORDER)
    if set(paths) != expected_blocks or set(ledger_paths) != expected_blocks:
        raise ValueError("claim finalization requires all four frozen blocks")
    if not combined_result.is_file():
        raise FileNotFoundError("combined development result is missing")

    manifest = manifest_validator(daily.PROTOCOL_MANIFEST)
    aug10 = aug10_validator(
        wrapper_result=daily.AUG10_RESULT,
        mechanics_result=daily.MECHANICS_RESULT,
    )
    block_verifications = {}
    for block_id in development.BLOCK_ORDER:
        result_path = paths[block_id]
        ledger_path = ledger_paths[block_id]
        daily_path = result_path.parent / "DAILY_EXECUTION.json"
        if not (result_path.is_file() and ledger_path.is_file() and daily_path.is_file()):
            raise FileNotFoundError(f"development block {block_id} is incomplete")
        verification = block_validator(
            path=result_path,
            block_id=block_id,
            ledger_path=ledger_path,
        )
        daily_record = daily_validator(
            path=daily_path,
            block_id=block_id,
            result_path=result_path,
            ledger_path=ledger_path,
            manifest_verification=manifest,
            aug10_verification=aug10,
            block_verification=verification,
        )
        if (
            block_id == development.BLOCK_ORDER[-1]
            and daily_record.get("combined_endpoint_accessed") is not True
        ):
            raise RuntimeError("development block D has not finalized its combined result")
        block_verifications[block_id] = verification

    ordered_paths: Sequence[Path] = [
        paths[block_id] for block_id in development.BLOCK_ORDER
    ]
    combined_verification = combined_validator(
        result_path=combined_result,
        block_results=ordered_paths,
    )
    combined = json.loads(combined_result.read_text(encoding="utf-8"))
    report = claim_report.build_claim_report(
        combined,
        result_sha256=development.sha256_file(combined_result),
        independent_verification=combined_verification,
    )
    report = claim_report.bank_claim_report(claim_report_path, report)

    authorization = None
    if report["claim_tier"] == "full_llm_native_development_signal":
        authorization = authorization_validator()
        if authorization.get("verified") is not True:
            raise RuntimeError("full development tier did not authorize confirmation")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "confirmation_handoff_verified"
            if authorization is not None
            else "development_claim_banked_confirmation_forbidden"
        ),
        "claim_tier": report["claim_tier"],
        "claim_report_path": str(claim_report_path),
        "claim_report_sha256": development.sha256_file(claim_report_path),
        "combined_verification": combined_verification,
        "block_verifications": block_verifications,
        "confirmation_authorization": authorization,
        "model_calls_made": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(finalize_development_claim(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
