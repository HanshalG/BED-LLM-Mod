from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import hiddenbench_dynamic_belief_v3_aug13_execute as execute
from scripts import hiddenbench_dynamic_belief_v3_serving as serving
from tests.test_hiddenbench_dynamic_belief_v3_serving import FakeAdapter, synthetic_views


def live(usage=220.178352166): return {"total_credits_usd":245.0,"total_usage_usd":usage,"balance_usd":245.0-usage}
def catalog(prompt=.08/1_000_000,completion=.18/1_000_000): return {"data":[{"id":"deepseek/deepseek-v4-flash-0731","architecture":{"input_modalities":["text"],"output_modalities":["text"]},"supported_parameters":["seed","response_format"],"pricing":{"prompt":str(prompt),"completion":str(completion)}}]}


def test_catalog_and_account_budget_math() -> None:
    model=execute.validate_catalog(catalog()); assert model["covered_prompt_tokens_at_live_price"]>=25_000
    assert execute.prior_spend(live())==pytest.approx(220.178352166-execute.OPENING_USAGE_USD)
    with pytest.raises(RuntimeError,match="price increased"): execute.validate_catalog(catalog(prompt=.10/1_000_000))
    with pytest.raises(RuntimeError,match="invalid"): execute.validate_live(live(execute.OPENING_USAGE_USD-.01))


def test_exact_execution_binding_passes() -> None:
    checked = execute.validate_bindings()
    assert checked["producer_sha256"] == "d9638a5cd92d0023481d76108959b467d339748432cfba64f2105947dc038e80"
    assert checked["verifier_sha256"] == "d48d6a661394e6d4b063432c4c7182a5429777db2e1104902ef62d3e5769de7e"


def test_endpoint_command_is_exact_and_unique() -> None:
    command=execute.endpoint_command(); joined=" ".join(command)
    assert "hiddenbench_dynamic_belief_v3_endpoint_custodian.py" in joined
    assert "--pass-token" in command and "--output" in command
    assert command.count(str(execute.SOURCE))==1


def test_synthetic_wrapper_transaction_orders_endpoint_after_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    paths = {
        "RUN_DIR": run_dir,
        "LEDGER": tmp_path / "ledger.json",
        "EXECUTION_BINDING": tmp_path / "binding.json",
        "DAILY_RESULT": tmp_path / "daily-result.json",
        "DAILY_FAILURE": tmp_path / "daily-failure.json",
        "SOURCE_MANIFEST": tmp_path / "manifest.json",
        "SOURCE_AUDIT": tmp_path / "audit.json",
        "RAW": run_dir / "private/RAW_RESPONSES.json",
        "LABEL_FREE": run_dir / "LABEL_FREE_RESULT.json",
        "VERIFICATION": run_dir / "VERIFICATION.json",
        "PASS_TOKEN": run_dir / "LABEL_FREE_PASS_TOKEN.json",
        "ENDPOINTS": run_dir / "private/ENDPOINTS.json",
        "ENDPOINT_RESULT": run_dir / "ENDPOINT_RESULT.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(execute, name, path)
    for name in ("EXECUTION_BINDING", "SOURCE_MANIFEST", "SOURCE_AUDIT"):
        paths[name].write_text(json.dumps({name: 1}))
    monkeypatch.setattr(execute, "preflight", lambda **kwargs: {"bindings": {"execution_binding_sha256": execute.sha256_file(paths["EXECUTION_BINDING"])} , "budget": {}})
    monkeypatch.setattr(serving, "load_views", lambda source_path: synthetic_views())
    monkeypatch.setattr(serving, "build_adapter", lambda run_id, output_dir: FakeAdapter())

    def fake_endpoint(command, check):
        assert check is True
        assert paths["PASS_TOKEN"].exists()
        assert paths["VERIFICATION"].exists()
        paths["ENDPOINTS"].parent.mkdir(parents=True, exist_ok=True)
        paths["ENDPOINTS"].write_text(json.dumps({"endpoints": [{"slot": f"T{i + 1}", "correct_option_id": "O3"} for i in range(4)]}))

    monkeypatch.setattr(execute.subprocess, "run", fake_endpoint)
    terminal = execute.execute(live_reader=lambda: live(), catalog_reader=lambda: catalog())
    assert terminal["status"] in {"mechanics_pass", "mechanics_null"}
    assert paths["PASS_TOKEN"].exists()
    assert paths["ENDPOINTS"].exists()
    assert paths["ENDPOINT_RESULT"].exists()
    assert paths["DAILY_RESULT"].exists()
    assert not paths["DAILY_FAILURE"].exists()


def test_label_free_null_opens_no_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    paths = {
        "RUN_DIR": run_dir,
        "LEDGER": tmp_path / "ledger.json",
        "EXECUTION_BINDING": tmp_path / "binding.json",
        "DAILY_RESULT": tmp_path / "daily-result.json",
        "DAILY_FAILURE": tmp_path / "daily-failure.json",
        "RAW": run_dir / "private/RAW_RESPONSES.json",
        "LABEL_FREE": run_dir / "LABEL_FREE_RESULT.json",
        "VERIFICATION": run_dir / "VERIFICATION.json",
        "PASS_TOKEN": run_dir / "LABEL_FREE_PASS_TOKEN.json",
        "ENDPOINTS": run_dir / "private/ENDPOINTS.json",
        "ENDPOINT_RESULT": run_dir / "ENDPOINT_RESULT.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(execute, name, path)
    paths["EXECUTION_BINDING"].write_text("{}")
    monkeypatch.setattr(execute, "preflight", lambda **kwargs: {"bindings": {"execution_binding_sha256": execute.sha256_file(paths["EXECUTION_BINDING"])} , "budget": {}})
    fake = FakeAdapter()
    monkeypatch.setattr(serving, "build_adapter", lambda run_id, output_dir: fake)
    def null_run(**kwargs):
        paths["RAW"].parent.mkdir(parents=True, exist_ok=True)
        paths["RAW"].write_text("{}")
        value={"status":"serving_failed_closed","authorizes":"nothing"}
        paths["LABEL_FREE"].write_text(json.dumps(value))
        return value
    monkeypatch.setattr(serving, "run_serving", null_run)
    terminal=execute.execute(live_reader=lambda: live(),catalog_reader=lambda: catalog())
    assert terminal["status"]=="serving_failed_closed"
    assert terminal["registered_answers_opened"] is False
    assert not paths["VERIFICATION"].exists()
    assert not paths["PASS_TOKEN"].exists()
    assert not paths["ENDPOINTS"].exists()
