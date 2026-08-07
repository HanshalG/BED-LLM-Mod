from __future__ import annotations

import json

import pytest

from scripts import regretbench_aug8_execute as chain


def test_binding_tamper_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(chain, "POLICY_DAILY_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="policy daily binding changed"):
        chain.validate_bindings()


def test_execution_binding_matches_orchestrator_and_components() -> None:
    path = (
        chain.Path(__file__).resolve().parents[1]
        / "results/nonmyopic/regretbench_aug8_chain/EXECUTION_BINDING.json"
    )
    binding = json.loads(path.read_text())

    assert binding["status"] == "frozen_before_aug8_calls"
    assert chain._sha256(
        chain.Path(__file__).resolve().parents[1]
        / binding["orchestrator"]["path"]
    ) == binding["orchestrator"]["sha256"]
    assert all(
        chain._sha256(chain.Path(__file__).resolve().parents[1] / relative)
        == expected
        for relative, expected in binding["components"].items()
    )
    assert binding["requirements"]["maximum_combined_exposure_usd"] == 4.8


def test_preflight_routes_to_pristine_baseline_first(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(chain, "validate_bindings", lambda: None)
    monkeypatch.setattr(chain.baseline, "SMOKE_RESULT", tmp_path / "baseline.json")
    monkeypatch.setattr(chain, "_refuse_downstream_before_baseline", lambda: None)
    monkeypatch.setattr(
        chain.baseline,
        "preflight_smoke",
        lambda **kwargs: {"status": "ready_without_paid_calls"},
    )

    result = chain.preflight(
        now=chain.datetime(2026, 8, 8, 10, tzinfo=chain.ZoneInfo(chain.TIMEZONE))
    )

    assert result["next_stage"] == "baseline_smoke"
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0


def test_preflight_routes_to_support_after_banked_baseline(
    tmp_path, monkeypatch
) -> None:
    baseline_result = tmp_path / "baseline.json"
    baseline_result.write_text(json.dumps({"status": "passed"}))
    monkeypatch.setattr(chain, "validate_bindings", lambda: None)
    monkeypatch.setattr(chain.baseline, "SMOKE_RESULT", baseline_result)
    monkeypatch.setattr(chain.support, "ROOT", tmp_path / "support")
    monkeypatch.setattr(chain, "_refuse_policy_before_support", lambda: None)
    monkeypatch.setattr(
        chain.support,
        "preflight",
        lambda **kwargs: {"status": "ready_without_paid_calls"},
    )

    result = chain.preflight(
        now=chain.datetime(2026, 8, 8, 10, tzinfo=chain.ZoneInfo(chain.TIMEZONE))
    )

    assert result["next_stage"] == "support_recovery"


def test_banked_baseline_null_is_terminal_on_resume(tmp_path, monkeypatch) -> None:
    baseline_result = tmp_path / "baseline.json"
    baseline_result.write_text(json.dumps({"status": "mechanics_failed"}))
    monkeypatch.setattr(chain, "validate_bindings", lambda: None)
    monkeypatch.setattr(chain.baseline, "SMOKE_RESULT", baseline_result)

    result = chain.preflight(
        now=chain.datetime(2026, 8, 8, 10, tzinfo=chain.ZoneInfo(chain.TIMEZONE))
    )

    assert result["status"] == "terminal_baseline_result"
    assert result["baseline_status"] == "mechanics_failed"
    assert result["next_stage"] is None


def test_support_null_is_terminal_and_never_routes_policy(
    tmp_path, monkeypatch
) -> None:
    baseline_result = tmp_path / "baseline.json"
    baseline_result.write_text(json.dumps({"status": "passed"}))
    support_root = tmp_path / "support"
    support_root.mkdir()
    (support_root / "DAILY_RESULT.json").write_text(
        json.dumps(
            {
                "status": "complete_reconciled",
                "development_status": "gated_null",
            }
        )
    )
    monkeypatch.setattr(chain, "validate_bindings", lambda: None)
    monkeypatch.setattr(chain.baseline, "SMOKE_RESULT", baseline_result)
    monkeypatch.setattr(chain.support, "ROOT", support_root)

    result = chain.preflight(
        now=chain.datetime(2026, 8, 8, 10, tzinfo=chain.ZoneInfo(chain.TIMEZONE))
    )

    assert result["status"] == "terminal_support_result"
    assert result["next_stage"] is None


def test_execute_runs_exact_order_and_stops_at_policy(monkeypatch) -> None:
    states = iter(
        [
            {"next_stage": "baseline_smoke"},
            {"next_stage": "support_recovery"},
            {"next_stage": "dynamic_policy"},
        ]
    )
    monkeypatch.setattr(chain, "preflight", lambda **kwargs: next(states))
    monkeypatch.setattr(
        chain.baseline, "execute_smoke", lambda **kwargs: {"status": "passed"}
    )
    monkeypatch.setattr(
        chain.support,
        "execute",
        lambda **kwargs: {"development_status": "passed"},
    )
    monkeypatch.setattr(
        chain.policy,
        "execute",
        lambda **kwargs: {
            "development_status": "gated_null",
            "development_result_sha256": "result",
            "development_verification_sha256": "verification",
        },
    )

    result = chain.execute()

    assert result["status"] == "complete"
    assert result["development_status"] == "gated_null"
    assert result["stages_completed_by_invocation"] == [
        "baseline_smoke",
        "support_recovery",
        "dynamic_policy",
    ]


def test_execute_does_not_open_policy_after_support_null(monkeypatch) -> None:
    states = iter(
        [{"next_stage": "support_recovery"}]
    )
    monkeypatch.setattr(chain, "preflight", lambda **kwargs: next(states))
    monkeypatch.setattr(
        chain.support,
        "execute",
        lambda **kwargs: {"development_status": "gated_null"},
    )
    called = False

    def policy_execute(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(chain.policy, "execute", policy_execute)

    result = chain.execute()

    assert result["status"] == "terminal_support_result"
    assert called is False
