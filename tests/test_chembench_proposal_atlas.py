from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import threading
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from environments.chembench_mopen.factored import FactoredModelBank
from environments.chembench_mopen.proposal_atlas import (
    ACTION_GROUPS,
    compile_response,
    feature_vector,
    parse_response,
    proposal_induced_risk,
    random_typed_candidates,
    render_prompt,
    response_json_schema,
    retrieve_atlas_candidates,
    score_response,
    standardized_features,
)
from scripts import chembench_llm_proposal_atlas as producer
from scripts import chembench_llm_proposal_atlas_aug15_execute as executor


INITIAL_NAMES = (
    "c0_michaelis_menten",
    "c1_competitive_inhibition",
    "c2_product_inhibition",
    "c3_arrhenius_temperature",
    "c5_pingpong_bisubstrate",
    "c6_uncompetitive_inhibition",
    "c7_substrate_inhibition",
    "c8_hill_cooperativity",
    "c9_noncompetitive_inhibition",
)
OUTSIDE_NAMES = (
    "c10_mm_competitive_arrhenius",
    "c23_pingpong_arrhenius",
    "c33_hill_competitive",
    "c52_sinh_arrhenius",
    "c65_ordered_bi_bi",
    "c67_allosteric_act",
    "c68_anticoop_hill",
    "c69_fractal_kinetics",
    "c70_mixed_inhibition",
    "c71_coop_inhibition",
    "c72_monotonic_ph",
    "c73_metal_activation",
    "c74_product_activation",
    "c75_two_substrate_inhibition",
)


def _bank() -> FactoredModelBank:
    names = INITIAL_NAMES + OUTSIDE_NAMES
    actions = ("C_A=0.1", "C_I=50,C_A=1", "C_B=0.1", "C_P=10", "T=368", "pH=10")
    rng = np.random.default_rng(104)
    likelihoods = rng.dirichlet(np.ones(3), size=(len(names), len(actions)))
    features = rng.normal(size=(len(names), 11))
    return FactoredModelBank(
        likelihoods,
        features,
        names,
        actions,
        ACTION_GROUPS,
        tuple(range(len(INITIAL_NAMES))),
        evidence_slots=8,
        diversity_slots=4,
    )


def _valid_payload() -> dict:
    return {
        "proposals": [
            {
                "parent_model_id": "c1_competitive_inhibition",
                "operation": "add_factor",
                "core_family": "michaelis_menten",
                "modifiers": ["competitive_inhibition", "arrhenius"],
                "residual_motif": "temperature-dependent competitive inhibition",
                "exposing_assay_group": "T",
                "falsifying_assay_group": "C_I",
            },
            {
                "parent_model_id": "c5_pingpong_bisubstrate",
                "operation": "add_factor",
                "core_family": "pingpong",
                "modifiers": ["arrhenius"],
                "residual_motif": "two-substrate residual changes with temperature",
                "exposing_assay_group": "T",
                "falsifying_assay_group": "C_B",
            },
            {
                "parent_model_id": "c8_hill_cooperativity",
                "operation": "add_factor",
                "core_family": "hill",
                "modifiers": ["competitive_inhibition"],
                "residual_motif": "cooperative curve shifts with inhibitor",
                "exposing_assay_group": "C_I",
                "falsifying_assay_group": "C_A",
            },
            {
                "parent_model_id": "c7_substrate_inhibition",
                "operation": "add_factor",
                "core_family": "substrate_inhibition",
                "modifiers": ["arrhenius"],
                "residual_motif": "high-substrate suppression changes with temperature",
                "exposing_assay_group": "T",
                "falsifying_assay_group": "C_A",
            },
        ]
    }


def _task(bank: FactoredModelBank, task_id: str, history: tuple[tuple[int, int], ...]) -> dict:
    state = bank.state(history, bank.initial_support, tried=bank.initial_support)
    report = bank.residual_report(state, remaining_budget=2)
    return {
        "task_id": task_id,
        "task_position": 0,
        "split": "atlas",
        "difficulty": "easy",
        "history_length": len(history),
        "remaining_budget": 2,
        "phase": "explore",
        "history": [
            {
                "action_index": action,
                "action_name": bank.action_names[action],
                "action_group": bank.action_groups[action],
                "outcome": ("low", "mid", "high")[outcome],
            }
            for action, outcome in history
        ],
        "residual_report": report,
    }


def test_strict_parser_and_typed_compiler_accept_valid_distinct_edits() -> None:
    bank = _bank()
    state = bank.initial_state()
    parsed = parse_response(json.dumps(_valid_payload()))
    compiled = compile_response(bank, state, parsed)
    assert parsed.schema_valid
    assert [item.candidate for item in compiled if item is not None] == [9, 10, 11, 12]
    score = score_response(bank, state, _valid_payload(), truth=9)
    assert score["schema_valid"]
    assert score["item_compile_rate"] == 1.0
    assert score["response_executable"]
    assert score["truth_recall_at_4"] == 1


@pytest.mark.parametrize(
    "payload,error",
    [
        ("not-json", "invalid_json"),
        ({"proposals": []}, "invalid_proposal_count"),
        ({"proposals": [{"bad": "shape"}] * 4}, "invalid_proposal_shape"),
        ({"proposals": [*_valid_payload()["proposals"], {}]}, "invalid_proposal_count"),
    ],
)
def test_strict_parser_rejects_malformed_responses(payload: object, error: str) -> None:
    parsed = parse_response(payload)
    assert not parsed.schema_valid
    assert parsed.error is not None and parsed.error.startswith(error)


def test_semantically_invalid_item_scores_zero_for_that_item_without_repair() -> None:
    bank = _bank()
    payload = _valid_payload()
    payload["proposals"][0]["core_family"] = "invented_family"
    score = score_response(bank, bank.initial_state(), payload, truth=9)
    assert score["schema_valid"]
    assert score["item_compile_rate"] == 0.75
    assert not score["response_executable"]
    assert score["truth_recall_at_4"] == 0


def test_prompt_boundary_and_schema_expose_only_current_parents() -> None:
    bank = _bank()
    task = _task(bank, "atlas-0", ((0, 2),))
    aware = render_prompt(bank, task, "residual_aware")
    blind = render_prompt(bank, task, "history_blind")
    assert "signed_innovation_by_group" in aware["user"]
    assert "signed_innovation_by_group" not in blind["user"]
    assert "C_A/(Km+C_A+C_A^2/Ki_s)" in aware["user"]
    assert "exp(-Ea/R*(1/T-1/T_ref))" in aware["user"]
    assert "C_A*C_B/(KiA*KmB+KmB*C_A+KmA*C_B+C_A*C_B)" in aware["user"]
    assert "c10_mm_competitive_arrhenius" not in aware["user"]
    assert len(aware["system"]) + len(aware["user"]) <= 24_000
    schema = response_json_schema(bank)
    parent_enum = schema["properties"]["proposals"]["items"]["properties"][
        "parent_model_id"
    ]["enum"]
    assert tuple(parent_enum) == INITIAL_NAMES


def test_feature_standardization_and_atlas_retrieval_are_deterministic() -> None:
    bank = _bank()
    atlas = [
        _task(bank, "a0", ((0, 0),)),
        _task(bank, "a1", ((1, 1),)),
        _task(bank, "a2", ((2, 2),)),
    ]
    target = _task(bank, "h0", ((0, 0),))
    standardized, metadata = standardized_features(
        {"easy": bank}, atlas, [target]
    )
    result = retrieve_atlas_candidates(
        target,
        atlas,
        standardized,
        {"a0": [[9, 10]], "a1": [[11]], "a2": [[12]]},
    )
    assert result["neighbors"][0]["task_id"] == "a0"
    assert result["candidate_indices"][0:2] == [9, 10]
    assert len(metadata["feature_names"]) == len(feature_vector(bank, target)[1])


def test_proposal_induced_risk_and_random_control_are_finite_and_reproducible() -> None:
    bank = _bank()
    history = ((0, 1),)
    first = random_typed_candidates(bank, bank.state(history, bank.initial_support), task_position=3)
    second = random_typed_candidates(bank, bank.state(history, bank.initial_support), task_position=3)
    assert first == second and len(set(first)) == 4
    result = proposal_induced_risk(bank, history, truth=9, candidates=(9, 10, 11, 12))
    assert np.isfinite(result["risk"])
    assert result["action_index"] not in {0}


def _catalog(*, expensive: bool = False) -> dict:
    return {
        "data": {
            "id": executor.MODEL_ID,
            "endpoints": [
                {
                    "provider_name": "cheap",
                    "status": 0,
                    "supported_parameters": ["seed", "reasoning", "response_format"],
                    "pricing": {"prompt": "0.00000014", "completion": "0.00000028"},
                },
                {
                    "provider_name": "ceiling",
                    "status": 0,
                    "supported_parameters": ["seed", "reasoning", "structured_outputs"],
                    "pricing": {
                        "prompt": "0.00000020",
                        "completion": "0.00000051" if expensive else "0.00000050",
                    },
                },
            ],
        }
    }


def _live() -> dict:
    usage = 220.348811737
    return {
        "total_credits_usd": 245.0,
        "total_usage_usd": usage,
        "balance_usd": 245.0 - usage,
    }


def _relocate_executor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = tmp_path / "run"
    paths = {
        "ROOT": tmp_path,
        "RUN": run,
        "MANIFEST": run / "PUBLIC_MANIFEST.json",
        "LABELS": run / "SEALED_LABELS.json",
        "SOURCE_SUMMARY": run / "SOURCE_SUMMARY.json",
        "SOURCE_VERIFICATION": run / "SOURCE_VERIFICATION.json",
        "BINDING": run / "EXECUTION_BINDING.json",
        "RAW": run / "private/RAW.json",
        "EVALUATION": run / "EVALUATION.json",
        "SEMANTIC_VERIFICATION": run / "VERIFY.json",
        "RESULT": run / "RESULT.json",
        "FAILURE": run / "FAILURE.json",
        "LEDGER": run / "LEDGER.json",
    }
    for name, value in paths.items():
        monkeypatch.setattr(executor, name, value)
    run.mkdir(parents=True)
    prompt = {"system": "system", "user": "user"}
    manifest = {
        "tasks": [
            {
                "task_id": "t0",
                "prompts": {
                    "residual_aware": prompt,
                    "history_blind": prompt,
                },
            }
        ],
        "requests": [
            {
                "request_id": f"r{index}",
                "task_id": "t0",
                "arm": "residual_aware" if index % 2 == 0 else "history_blind",
                "seed": index,
                "prompt_sha256": producer.payload_hash(prompt),
            }
            for index in range(126)
        ],
        "response_schema": {"type": "object"},
    }
    executor.MANIFEST.write_text(json.dumps(manifest))
    executor.SOURCE_VERIFICATION.write_text(json.dumps({"status": "passed"}))
    monkeypatch.setattr(
        executor,
        "validate_bindings",
        lambda: {"execution_binding_sha256": "a" * 64, "pushed_head": "test"},
    )


def test_executor_catalog_and_payload_are_parameter_constrained_nonreasoning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prices = executor.validate_catalog(_catalog())
    assert prices["prompt_price_usd_per_token"] == pytest.approx(0.20e-6)
    assert prices["completion_price_usd_per_token"] == pytest.approx(0.50e-6)
    with pytest.raises(RuntimeError, match="price ceiling"):
        executor.validate_catalog(_catalog(expensive=True))
    _relocate_executor(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-only")
    adapter = executor.build_adapter(request_cap=0.01, authorize=None)
    adapter._seed = threading.local()
    adapter._seed.value = 202608370000
    payload = adapter._payload(
        [{"role": "user", "content": "test"}],
        0.3,
        1,
        20,
        disable_reasoning=True,
        response_format={"type": "json_object"},
    )
    assert payload["provider"] == {"require_parameters": True}
    assert payload["seed"] == 202608370000
    assert payload["reasoning"]["enabled"] is False
    assert payload["reasoning"]["exclude"] is True


def test_executor_preflight_reserves_all_126_requests_without_writing_paid_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _relocate_executor(monkeypatch, tmp_path)
    ready = executor.preflight(
        now=datetime(2026, 8, 15, 18, tzinfo=ZoneInfo("Europe/London")),
        live_reader=_live,
        catalog_reader=_catalog,
    )
    assert ready["status"] == "ready_without_paid_calls"
    assert ready["authorization"]["exposure"]["request_count"] == 126
    assert ready["authorization"]["exposure"]["exact_stage_exposure_usd"] < 0.75
    assert ready["model_calls_made"] == ready["files_written"] == 0
    assert not executor.LEDGER.exists()
