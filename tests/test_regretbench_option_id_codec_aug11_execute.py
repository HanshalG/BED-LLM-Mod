from __future__ import annotations

from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pytest

from scripts import regretbench_option_id_codec_aug11_execute as execute
from scripts import regretbench_option_id_codec_smoke as smoke
from tests.test_regretbench_option_id_codec_smoke import _Adapter


NOW = datetime(2026, 8, 11, 21, 30, tzinfo=ZoneInfo("Europe/London"))
PRIOR_SPEND = 220.132449014 - execute.OPENING_USAGE_USD
LIVE = {
    "total_credits_usd": 245.0,
    "total_usage_usd": execute.OPENING_USAGE_USD + PRIOR_SPEND,
    "balance_usd": 245.0 - execute.OPENING_USAGE_USD - PRIOR_SPEND,
}


def _catalog(prompt="0.00000008", completion="0.00000018"):
    return {
        "data": [
            {
                "id": smoke.MODEL_ID,
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["seed", "response_format", "structured_outputs"],
                "pricing": {"prompt": prompt, "completion": completion},
            }
        ]
    }


def _paths(monkeypatch, tmp_path):
    root = tmp_path / "option-id-codec"
    monkeypatch.setattr(execute, "ROOT", root)
    monkeypatch.setattr(execute, "RUN_DIR", root / "smoke-20260811")
    monkeypatch.setattr(execute, "STAGE_LEDGER", tmp_path / "stage-ledger.json")
    monkeypatch.setattr(execute, "DAILY_RESULT", root / "DAILY_RESULT_20260811.json")
    monkeypatch.setattr(execute, "DAILY_FAILURE", root / "DAILY_FAILURE_20260811.json")


def test_preflight_is_read_only_and_carries_prior_account_spend(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    ready = execute.preflight(
        now=NOW,
        live_reader=lambda: dict(LIVE),
        catalog_reader=_catalog,
    )
    assert ready["status"] == "ready_without_paid_calls"
    assert ready["budget"]["prior_account_spend_usd"] == pytest.approx(PRIOR_SPEND)
    assert ready["budget"]["run_cap_usd"] == 0.1
    assert ready["files_written"] == 0
    assert ready["model_calls_made"] == 0
    assert not execute.STAGE_LEDGER.exists()
    assert not execute.RUN_DIR.exists()


def test_exact_smoke_executes_verifies_and_replays_without_network(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    reads = 0

    def live():
        nonlocal reads
        reads += 1
        return dict(LIVE)

    adapter = _Adapter()
    result = execute.execute(
        now=NOW,
        live_reader=live,
        catalog_reader=_catalog,
        adapter_builder=lambda **kwargs: adapter,
    )
    assert result["status"] == "complete_reconciled"
    assert result["smoke_status"] == "passed"
    assert result["independent_replay_passed"] is True
    assert result["development_opened"] is False
    assert result["endpoint_outcomes_opened"] is False
    assert adapter.requests == 8
    verification = json.loads((execute.RUN_DIR / "VERIFICATION.json").read_text())
    assert verification["status"] == "verified"
    before = reads
    replay = execute.execute(
        now=NOW,
        live_reader=lambda: (_ for _ in ()).throw(AssertionError("live read")),
        catalog_reader=lambda: (_ for _ in ()).throw(AssertionError("catalog read")),
    )
    assert replay == result
    assert reads == before


def test_verified_semantic_null_completes_but_authorizes_nothing(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    result = execute.execute(
        now=NOW,
        live_reader=lambda: dict(LIVE),
        catalog_reader=_catalog,
        adapter_builder=lambda **kwargs: _Adapter(codec_mode="disagree"),
    )
    assert result["status"] == "complete_reconciled"
    assert result["smoke_status"] == "codec_failed"
    assert result["authorizes"] == "nothing"
    assert result["development_opened"] is False


def test_post_preflight_usage_race_opens_no_stage(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    raced_usage = execute.OPENING_USAGE_USD + 4.91
    values = [
        dict(LIVE),
        {
            "total_credits_usd": 245.0,
            "total_usage_usd": raced_usage,
            "balance_usd": 245.0 - raced_usage,
        },
    ]
    with pytest.raises(RuntimeError, match="usage changed"):
        execute.execute(
            now=NOW,
            live_reader=lambda: values.pop(0),
            catalog_reader=_catalog,
            adapter_builder=lambda **kwargs: _Adapter(),
        )
    assert not execute.STAGE_LEDGER.exists()
    assert not execute.RUN_DIR.exists()


class _SecondCodecPairFailure(_Adapter):
    def chat_complete_seeded_messages_batched_structured(self, *args, **kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        if "_environment_codec_" in name and self.requests == 6:
            raise RuntimeError("synthetic second codec pair transport failure")
        return super().chat_complete_seeded_messages_batched_structured(*args, **kwargs)


def test_partial_failure_banks_six_calls_and_cannot_retry(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    adapter = _SecondCodecPairFailure()
    with pytest.raises(RuntimeError, match="synthetic second codec pair"):
        execute.execute(
            now=NOW,
            live_reader=lambda: dict(LIVE),
            catalog_reader=_catalog,
            adapter_builder=lambda **kwargs: adapter,
        )
    assert adapter.requests == 6
    failure = json.loads(execute.DAILY_FAILURE.read_text())
    assert failure["status"] == "failed_closed"
    assert failure["authorizes"] == "nothing"
    raw = json.loads((execute.RUN_DIR / "private/RAW_RESPONSES.json").read_text())
    assert len(raw["proposals"]) == 2
    assert len(raw["evaluations"]) == 2
    assert len(raw["codec_responses"]) == 2
    replay = execute.execute(
        now=NOW,
        live_reader=lambda: (_ for _ in ()).throw(AssertionError("live read")),
        catalog_reader=lambda: (_ for _ in ()).throw(AssertionError("catalog read")),
    )
    assert replay == failure


def test_post_run_account_outage_banks_local_cost_and_fails(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    reads = 0

    def live():
        nonlocal reads
        reads += 1
        if reads >= 3:
            raise RuntimeError("synthetic account outage")
        return dict(LIVE)

    adapter = _Adapter()
    with pytest.raises(RuntimeError, match="synthetic account outage"):
        execute.execute(
            now=NOW,
            live_reader=live,
            catalog_reader=_catalog,
            adapter_builder=lambda **kwargs: adapter,
        )
    ledger = json.loads(execute.STAGE_LEDGER.read_text())
    assert adapter.requests == 8
    assert ledger["stage"]["status"] == "failed_closed"
    assert ledger["reconciliation"]["live_read_fallback"] is True
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(
        PRIOR_SPEND + adapter.usage_snapshot()["adapter_cost_usd"]
    )


def test_live_price_increase_fails_before_writes(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="reservation"):
        execute.preflight(
            now=NOW,
            live_reader=lambda: dict(LIVE),
            catalog_reader=lambda: _catalog(completion="0.000001"),
        )
    assert not execute.STAGE_LEDGER.exists()


def test_terminal_result_tamper_is_rejected(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    execute.execute(
        now=NOW,
        live_reader=lambda: dict(LIVE),
        catalog_reader=_catalog,
        adapter_builder=lambda **kwargs: _Adapter(),
    )
    daily = json.loads(execute.DAILY_RESULT.read_text())
    daily["development_opened"] = True
    execute.DAILY_RESULT.write_text(json.dumps(daily))
    with pytest.raises(RuntimeError, match="does not replay exactly"):
        execute.execute(
            now=NOW,
            live_reader=lambda: (_ for _ in ()).throw(AssertionError("live read")),
            catalog_reader=lambda: (_ for _ in ()).throw(AssertionError("catalog read")),
        )


@pytest.mark.parametrize(
    "changed",
    [
        {**LIVE, "total_usage_usd": float("nan")},
        {**LIVE, "balance_usd": -1.0},
        {**LIVE, "balance_usd": LIVE["balance_usd"] - 1.0},
    ],
)
def test_malformed_live_account_fails_before_writes(monkeypatch, tmp_path, changed):
    _paths(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="account values"):
        execute.preflight(
            now=NOW,
            live_reader=lambda: changed,
            catalog_reader=_catalog,
        )
    assert not execute.STAGE_LEDGER.exists()
    assert not execute.RUN_DIR.exists()


def test_nonpristine_run_path_fails_before_external_reads(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    execute.RUN_DIR.mkdir(parents=True)
    (execute.RUN_DIR / "unexpected.json").write_text("{}")
    with pytest.raises(RuntimeError, match="not pristine"):
        execute.preflight(
            now=NOW,
            live_reader=lambda: (_ for _ in ()).throw(AssertionError("live read")),
            catalog_reader=lambda: (_ for _ in ()).throw(AssertionError("catalog read")),
        )


def test_stale_execution_binding_fails_before_external_reads(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    binding = json.loads(execute.EXECUTION_BINDING.read_text())
    binding["producer"]["sha256"] = "0" * 64
    stale = tmp_path / "EXECUTION_BINDING.json"
    stale.write_text(json.dumps(binding))
    monkeypatch.setattr(execute, "EXECUTION_BINDING", stale)
    with pytest.raises(RuntimeError, match="producer binding changed"):
        execute.preflight(
            now=NOW,
            live_reader=lambda: (_ for _ in ()).throw(AssertionError("live read")),
            catalog_reader=lambda: (_ for _ in ()).throw(AssertionError("catalog read")),
        )
