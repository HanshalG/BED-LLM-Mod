from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import regretbench_deepseek_smc_support_recovery_daily as daily
from scripts import regretbench_deepseek_smc_support_recovery_verify as verify


def _root_raw(cig, truth, question: str) -> str:
    alias = str((truth.slots or {})["answer_aliases"]).split("|")[0].strip()
    hypotheses = [
        {
            "interpretation": f"interpretation {index} for {cig.cig_id}",
            "final_answer": alias if index == 0 else f"distractor answer {index}",
            "prior_weight": 1,
        }
        for index in range(8)
    ]
    return json.dumps(
        {
            "hypotheses": hypotheses,
            "questions": [
                question,
                "Which period is intended?",
                "Which category is intended?",
                "Which comparison is intended?",
            ],
        }
    )


def _child_from_messages(messages) -> str:
    payload = json.loads(messages[1]["content"])
    children = []
    for parent in payload["parent_particles"]:
        index = parent["parent_index"]
        retained = index < 2
        children.append(
            {
                "parent_index": index,
                "revision_type": "retained" if retained else "revised",
                "interpretation": parent["interpretation"]
                if retained
                else parent["interpretation"] + " revised",
                "final_answer": parent["final_answer"],
                "prior_weight": 1,
            }
        )
    return json.dumps(
        {
            "hypotheses": children,
            "questions": [
                "Which scope is intended?",
                "Which period is intended?",
                "Which category is intended?",
                "Which comparison is intended?",
            ],
        }
    )


class _Adapter:
    def __init__(self) -> None:
        self.messages = []
        self.seeds = []

    def chat_complete_seeded_messages_batched_structured(
        self,
        messages,
        seeds,
        *,
        temperature,
        response_format,
        max_new_tokens,
    ):
        self.messages = list(messages)
        self.seeds = list(seeds)
        assert temperature == 0.7
        assert max_new_tokens == 2_200
        assert response_format["json_schema"]["strict"] is True
        return [_child_from_messages(item) for item in messages]

    def usage_snapshot(self):
        return {
            "adapter_requests": 128,
            "http_attempts": 128,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
            "adapter_prompt_tokens": 10_000,
            "adapter_completion_tokens": 10_000,
        }


def _install_primary(tmp_path: Path, monkeypatch) -> Path:
    primary_dir = tmp_path / "primary"
    private = primary_dir / "private"
    private.mkdir(parents=True)
    roots = []
    controls = []
    for index, cig in enumerate(verify._cigs()):
        truth_index, truth = verify._truth(cig, verify.TRUTH_SEED_START + index)
        question = next(
            item.text.strip().rstrip("?") + "?"
            for item in cig.reference_questions
            if verify._map(
                cig, item.text.strip().rstrip("?") + "?", truth
            )["supported"]
        )
        roots.append(_root_raw(cig, truth, question))
        controls.append(
            {
                "task_id": cig.cig_id,
                "truth_index": truth_index,
                "question": question,
                "mapping": verify._map(cig, question, truth),
                "aliases": str((truth.slots or {})["answer_aliases"]),
            }
        )
    (private / "RAW_RESPONSES.json").write_text(
        json.dumps({"stage": "development", "root": roots, "branches": []})
    )
    (private / "CONTROLS.json").write_text(
        json.dumps({"stage": "development", "roots": controls, "privacy": []})
    )
    result = {
        "schema_version": 1,
        "interface_version": verify.PRIMARY_INTERFACE,
        "status": "gated_null",
        "authorizes": "nothing",
        "protocol": {
            "stage": "development",
            "model": verify.MODEL_ID,
            "support_recovery_endpoint_accessed": True,
            "policy_endpoint_opened": False,
        },
        "mechanics_gates": {"all_pass": True},
    }
    (primary_dir / "RESULT.json").write_text(json.dumps(result))
    verification = {
        "status": "verified",
        "result_status": "gated_null",
        "mismatches": [],
        "model_calls": 0,
        "cost_usd": 0.0,
        "artifact_sha256": {
            "RESULT.json": daily.primary.sha256_file(primary_dir / "RESULT.json"),
            "private/RAW_RESPONSES.json": daily.primary.sha256_file(
                private / "RAW_RESPONSES.json"
            ),
            "private/CONTROLS.json": daily.primary.sha256_file(
                private / "CONTROLS.json"
            ),
        },
    }
    (primary_dir / "VERIFICATION.json").write_text(json.dumps(verification))
    monkeypatch.setattr(daily, "PRIMARY_DIR", primary_dir)
    return primary_dir


def _catalog() -> dict:
    return {
        "data": [
            {
                "id": daily.core.MODEL_ID,
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["seed", "structured_outputs"],
                "top_provider": {
                    "context_length": 1_048_576,
                    "max_completion_tokens": 65_536,
                },
                "pricing": {"prompt": "0.00000009", "completion": "0.00000018"},
            }
        ]
    }


def _install_daily_predecessor(tmp_path: Path, monkeypatch) -> Path:
    primary_dir = _install_primary(tmp_path, monkeypatch)
    daily_root = tmp_path / "primary-daily"
    daily_root.mkdir()
    ledger = tmp_path / "primary-ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "date": "2026-08-08",
                "timezone": daily.TIMEZONE,
                "daily_cap_usd": 5.0,
                "opening_total_usage_usd": 100.0,
                "recorded_actual_spend_usd": 0.2,
                "account_wide_usage_counts_against_cap": True,
                "stages": {
                    "smoke": {"status": "passed"},
                    "development": {"status": "gated_null"},
                },
            }
        )
    )
    daily_result = {
        "status": "complete_reconciled",
        "development_status": "gated_null",
        "development_opened": True,
        "policy_endpoint_opened": False,
        "confirmation_opened": False,
        "independent_replay_passed": True,
        "development_result_sha256": daily.primary.sha256_file(
            primary_dir / "RESULT.json"
        ),
        "development_verification_sha256": daily.primary.sha256_file(
            primary_dir / "VERIFICATION.json"
        ),
        "ledger_sha256": daily.primary.sha256_file(ledger),
    }
    (daily_root / "DAILY_RESULT.json").write_text(json.dumps(daily_result))
    monkeypatch.setattr(daily.primary_daily, "ROOT", daily_root)
    monkeypatch.setattr(daily.primary_daily, "LEDGER", ledger)
    monkeypatch.setattr(daily, "_forbidden_primary_descendants", lambda: [])
    return primary_dir


def test_full_scale_producer_and_independent_verifier(tmp_path, monkeypatch) -> None:
    primary_dir = _install_primary(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    adapter = _Adapter()

    result = daily.run_experiment(
        output_dir=run_dir,
        adapter=adapter,
        daily_budget_status={"authorized": True},
        bootstrap_samples=1_000,
    )
    verification = verify.verify(run_dir, primary_dir=primary_dir)

    assert len(adapter.messages) == 128
    assert adapter.seeds[0:4] == [202608270000, 202608270000, 202608270001, 202608270001]
    assert result["mechanics_gates"]["all_pass"] is True
    assert result["status"] == "gated_null"
    assert result["authorizes"] == "nothing"
    assert result["protocol"]["initial_support_calls_repeated"] is False
    assert verification["status"] == "verified"
    assert verification["model_calls"] == 0
    assert verification["mismatches"] == []


def test_verifier_detects_public_result_tamper(tmp_path, monkeypatch) -> None:
    primary_dir = _install_primary(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    daily.run_experiment(
        output_dir=run_dir,
        adapter=_Adapter(),
        daily_budget_status={"authorized": True},
        bootstrap_samples=100,
    )
    result_path = run_dir / "RESULT.json"
    result = json.loads(result_path.read_text())
    result["tasks"][0]["conditioned_covered"] = False
    result_path.write_text(json.dumps(result))

    verification = verify.verify(run_dir, primary_dir=primary_dir)

    assert verification["status"] == "verification_failed"
    assert "$.tasks[0].conditioned_covered" in verification["mismatches"]


def test_preflight_rejects_wrong_date() -> None:
    with pytest.raises(RuntimeError, match="only on 2026-08-09"):
        daily.preflight(
            now=datetime(2026, 8, 8, 12, tzinfo=ZoneInfo("Europe/London"))
        )


def test_preflight_is_zero_call_and_inherits_aug8_boundary(
    tmp_path, monkeypatch
) -> None:
    _install_daily_predecessor(tmp_path, monkeypatch)
    monkeypatch.setattr(daily, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(daily, "DAILY_RESULT", tmp_path / "DAILY_RESULT.json")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "smc-ledger.json")
    live = {
        "total_credits_usd": 130.0,
        "total_usage_usd": 100.3,
        "balance_usd": 29.7,
    }

    result = daily.preflight(
        now=datetime(2026, 8, 9, 12, tzinfo=ZoneInfo("Europe/London")),
        live_reader=lambda: live,
        catalog_reader=_catalog,
    )

    assert result["status"] == "ready_without_paid_calls"
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0
    assert result["budget"]["opening_total_usage_boundary_usd"] == pytest.approx(
        100.2
    )
    assert result["budget"]["spent_before_smc_usd"] == pytest.approx(0.1)
    assert result["budget"]["remaining_after_full_cap_usd"] == pytest.approx(4.4)


def test_primary_pass_does_not_authorize_contingency(tmp_path) -> None:
    result = tmp_path / "RESULT.json"
    verification = tmp_path / "VERIFICATION.json"
    result.write_text(
        json.dumps(
            {
                "interface_version": verify.PRIMARY_INTERFACE,
                "status": "passed",
                "authorizes": "separately_preregistered_development_policy_only",
                "protocol": {
                    "stage": "development",
                    "model": verify.MODEL_ID,
                    "support_recovery_endpoint_accessed": True,
                    "policy_endpoint_opened": False,
                },
                "mechanics_gates": {"all_pass": True},
            }
        )
    )
    verification.write_text("{}")

    with pytest.raises(ValueError, match="does not authorize"):
        daily.core.validate_primary_null_predecessor(result, verification)


def test_budget_status_reserves_full_run_cap() -> None:
    ledger = {
        "date": daily.DATE,
        "timezone": daily.TIMEZONE,
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": 4.51,
    }
    live = {
        "total_credits_usd": 120.0,
        "total_usage_usd": 104.51,
        "balance_usd": 15.49,
    }

    with pytest.raises(RuntimeError, match="projected"):
        daily._budget_status(
            ledger,
            live,
            now=datetime(2026, 8, 9, 12, tzinfo=ZoneInfo("Europe/London")),
        )


def test_forbidden_descendant_detection_is_fail_closed(tmp_path, monkeypatch) -> None:
    opened = tmp_path / "policy" / "RESULT.json"
    opened.parent.mkdir()
    opened.write_text("{}")
    monkeypatch.setattr(daily, "_forbidden_primary_descendants", lambda: [opened])

    with pytest.raises(RuntimeError, match="already open"):
        daily._assert_no_primary_descendants_open()
