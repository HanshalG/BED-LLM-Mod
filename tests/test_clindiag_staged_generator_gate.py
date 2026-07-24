from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from helpers import load_config
from scripts.clindiag_staged_generator_gate import (
    DEVELOPMENT_IDS,
    HOLDOUT_IDS,
    SMOKE_IDS,
    ClinDiagCase,
    _assert_no_lexical_target_leak,
    full_differential_messages,
    initial_differential_messages,
    load_selected_cases,
    summarize,
)


def test_clindiag_gate_config_is_nonreasoning_and_fail_closed() -> None:
    config = load_config(
        "configs/config_clindiag_staged_generator_gate_openrouter.yaml"
    )
    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4"
    assert config.model_pairs[0].questioner.thinking is None
    assert config.openrouter_budget_usd == pytest.approx(70.38480269545715)
    assert config.openrouter_run_budget_usd == pytest.approx(1.0)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.5)
    assert config.openrouter_concurrency == 64


def _case(source_id: str = "case1", diagnosis: str = "Target syndrome") -> ClinDiagCase:
    return ClinDiagCase(
        source_id=source_id,
        subset="rare" if source_id.startswith("rare") else "challenging",
        initial_information="A patient has fever and fatigue.",
        medical_history={"history": "Symptoms progressed over two weeks."},
        physical_examination={"findings": "A non-specific rash is present."},
        diagnostic_test={"laboratory": "Marker X is elevated."},
        final_diagnosis=diagnosis,
    )


def _write_case_zip(path: Path, case: ClinDiagCase) -> None:
    payloads = {
        "initial_information.json": {
            "initial_information": case.initial_information
        },
        "medical_history.json": case.medical_history,
        "physical_examination.json": case.physical_examination,
        "diagnostic_test.json": case.diagnostic_test,
        "diagnosis.json": {
            "diagnosis": {"final_diagnosis": case.final_diagnosis}
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        for filename, payload in payloads.items():
            archive.writestr(
                f"{case.source_id}/{filename}",
                json.dumps(payload),
            )


def test_clindiag_splits_are_fixed_balanced_disjoint() -> None:
    assert len(SMOKE_IDS) == 4
    assert len(DEVELOPMENT_IDS) == 20
    assert len(HOLDOUT_IDS) == 60
    assert not set(SMOKE_IDS).intersection(DEVELOPMENT_IDS)
    assert not set(SMOKE_IDS).intersection(HOLDOUT_IDS)
    assert not set(DEVELOPMENT_IDS).intersection(HOLDOUT_IDS)
    assert sum(value.startswith("rare") for value in DEVELOPMENT_IDS) == 10
    assert sum(value.startswith("rare") for value in HOLDOUT_IDS) == 30


def test_clindiag_loader_and_hash_switch(tmp_path: Path) -> None:
    case = _case()
    path = tmp_path / "benchmark.zip"
    _write_case_zip(path, case)
    loaded = load_selected_cases(path, [case.source_id], verify_hash=False)
    assert loaded == [case]
    with pytest.raises(ValueError, match="hash mismatch"):
        load_selected_cases(path, [case.source_id], verify_hash=True)


def test_lexical_target_leaks_are_rejected() -> None:
    _assert_no_lexical_target_leak(_case())
    leaking = _case()
    leaking = ClinDiagCase(
        **{
            **leaking.__dict__,
            "diagnostic_test": {"result": "Target syndrome was confirmed."},
        }
    )
    with pytest.raises(ValueError, match="diagnosis leaks"):
        _assert_no_lexical_target_leak(leaking)


def test_generation_prompts_never_include_measurement_target() -> None:
    case = _case(diagnosis="Secret diagnosis")
    initial = initial_differential_messages(case)
    full = full_differential_messages(case, ["Alternative one"])
    prompt = json.dumps([initial, full])
    assert "Secret diagnosis" not in prompt
    assert "final diagnosis" in prompt
    assert "Alternative one" in prompt


def test_summary_applies_frozen_gate_and_subgroups() -> None:
    records = []
    for index in range(20):
        initial_score = 0.9 if index < 4 else 0.1
        full_score = 0.9 if index < 18 else 0.2
        records.append(
            {
                "subset": "challenging" if index < 10 else "rare",
                "initial_measurement": {
                    "covered": initial_score >= 0.8,
                    "best_match_score": initial_score,
                },
                "full_measurement": {
                    "covered": full_score >= 0.8,
                    "best_match_score": full_score,
                },
            }
        )
    summary = summarize(records)
    assert summary["initial_covered"] == 4
    assert summary["full_generated_covered"] == 18
    assert summary["initially_omitted_recovered_by_full_evidence"] == 14
    assert summary["full_coverage_by_subset"] == {"challenging": 10, "rare": 8}
    assert summary["gates"]["all_pass"] is True
