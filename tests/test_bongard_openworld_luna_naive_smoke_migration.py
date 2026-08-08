from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import bongard_openworld_luna_naive_first_link as naive
from scripts import bongard_openworld_luna_naive_first_link_daily_execute as daily
from scripts import bongard_openworld_luna_naive_first_link_verify as protocol_verify
from scripts import bongard_openworld_luna_naive_smoke_migration as migration


def _catalog() -> dict:
    return {
        "data": [
            {
                "id": naive.MODEL_ID,
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": [
                    "reasoning",
                    "response_format",
                    "structured_outputs",
                ],
                "top_provider": {"max_completion_tokens": 128_000},
                "pricing": {"prompt": "0.0000001", "completion": "0.0000006"},
            }
        ]
    }


def test_banked_v1_smoke_replays_under_exact_current_requests() -> None:
    certificate = migration.verify_certificate()
    smoke = naive.verify_smoke_result(migration.BANKED_SMOKE_RESULT)
    manifest = protocol_verify.verify_protocol_manifest()

    assert certificate["verified"] is True
    assert certificate["model_calls_made"] == 0
    assert certificate["cost_usd"] == 0.0
    assert certificate["banked_raw_responses_sha256"] == (
        migration.BANKED_RAW_SHA256
    )
    assert migration.sha256_file(migration.PUBLIC_REPLAY_PAYLOAD) == (
        migration.BANKED_RAW_SHA256
    )
    assert smoke["legacy_v1_replayed_under_v2"] is True
    assert smoke["migration_certificate_sha256"] == certificate[
        "certificate_sha256"
    ]
    assert manifest["banked_smoke_replayed_under_current_requests"] is True


def test_replay_certificate_tamper_fails_closed(tmp_path: Path) -> None:
    certificate = json.loads(migration.CERTIFICATE.read_text(encoding="utf-8"))
    certificate["checks"]["exact_request_seeds"] = False
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(certificate), encoding="utf-8")

    with pytest.raises(RuntimeError, match="certificate changed"):
        migration.verify_certificate(path)


def test_old_result_or_public_payload_tamper_fails_closed(tmp_path: Path) -> None:
    result = json.loads(
        migration.BANKED_SMOKE_RESULT.read_text(encoding="utf-8")
    )
    result["choices"][0]["first_image_id"] = "image-00"
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="result changed"):
        migration.build_certificate(result_path=result_path)

    payload = json.loads(
        migration.PUBLIC_REPLAY_PAYLOAD.read_text(encoding="utf-8")
    )
    payload["seeds"][0] += 1
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="raw responses changed"):
        migration.build_certificate(raw_path=payload_path)


def test_arbitrary_v1_smoke_cannot_use_banked_exception(tmp_path: Path) -> None:
    copied = tmp_path / "RESULT.json"
    copied.write_bytes(migration.BANKED_SMOKE_RESULT.read_bytes())

    with pytest.raises(ValueError, match="not an exact clean pass"):
        naive.verify_smoke_result(copied)


def test_aug11_block_preflight_accepts_only_replayed_banked_smoke(
    tmp_path: Path, monkeypatch
) -> None:
    live = {
        "total_credits_usd": 245.0,
        "total_usage_usd": 222.90,
        "balance_usd": 22.10,
    }
    main_ledger = tmp_path / "main-ledger.json"
    main_ledger.write_text(
        json.dumps(
            {
                "opening_total_usage_usd": 220.121013787,
                "recorded_actual_spend_usd": 2.78,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(daily.main_daily.LEDGERS, "a", main_ledger)
    monkeypatch.setattr(
        daily,
        "_main_predecessor",
        lambda block_id: {"verified": True, "block_id": block_id},
    )
    for block_id in daily.development.BLOCK_ORDER:
        monkeypatch.setitem(daily.BLOCK_DIRS, block_id, tmp_path / f"block-{block_id}")
        monkeypatch.setitem(
            daily.SUPPLEMENTAL_LEDGERS,
            block_id,
            tmp_path / f"ledger-{block_id}.json",
        )

    result = daily.preflight_block(
        block_id="a",
        now=datetime(2026, 8, 11, 9, tzinfo=ZoneInfo("Europe/London")),
        live_reader=lambda: live,
        catalog_reader=_catalog,
    )

    assert result["status"] == "ready_without_paid_calls"
    assert result["smoke_result_sha256"] == migration.BANKED_RESULT_SHA256
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0
