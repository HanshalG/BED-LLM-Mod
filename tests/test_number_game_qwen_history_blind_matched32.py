from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from scripts import number_game_qwen_history_blind_matched32 as matched
from scripts import number_game_qwen_history_blind_serving_smoke as smoke


class _FakeFormalAdapter:
    def __init__(self, first_response: str, second_response: str) -> None:
        self.first_response = first_response
        self.second_response = second_response
        self.request_count = 0

    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages,
        seeds,
        **kwargs,
    ):
        del kwargs
        assert len(batch_messages) == matched.EXPECTED_REQUESTS
        assert len(seeds) == matched.EXPECTED_REQUESTS
        self.request_count = len(seeds)
        return [
            self.first_response if index % 2 == 0 else self.second_response
            for index in range(len(seeds))
        ]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.request_count,
            "http_attempts": self.request_count,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
        }


def _diagnostic(valid: int = 20) -> dict:
    return {
        "codec_mode": "strict_json",
        "valid_unique_count": valid,
    }


def _pool_row() -> dict:
    return {
        "valid_unique_count": 30,
        "diagnostic": {
            "draw_novel_contributions": [20, 10],
            "draw_diagnostics": [_diagnostic(), _diagnostic()],
        },
    }


def _stage(mse_difference: float, coverage_difference: float) -> dict:
    return {
        "conditional": {
            "posterior_predictive_mse": 0.1,
            "truth_extension_coverage": 0.7,
            "support_size": 30.0,
        },
        "history_blind": {
            "posterior_predictive_mse": 0.1 - mse_difference,
            "truth_extension_coverage": 0.7 - coverage_difference,
            "support_size": 20.0,
        },
        "differences": {
            "conditional_minus_history_blind_predictive_mse": (
                mse_difference
            ),
            "conditional_minus_history_blind_truth_coverage": (
                coverage_difference
            ),
            "conditional_minus_history_blind_support_size": 10.0,
        },
    }


def _score_row(
    *,
    contrast: float = 0.02,
    realized: float = 0.01,
    roots_differ: bool = True,
) -> dict:
    return {
        "tree_mean": {
            "first": _stage(-0.01, 0.1),
            "second": _stage(-0.02, 0.1),
        },
        "selected_roots": {
            "roots_differ": roots_differ,
            "dynamic_root_prompt_conditioning_benefit": 0.03,
            "fixed_root_prompt_conditioning_benefit": 0.03 - contrast,
            "dynamic_minus_fixed_root_prompt_conditioning_benefit": (
                contrast
            ),
            "realized_advantage": realized,
        },
    }


def test_frozen_seed_schedule_is_unique_and_slot_local() -> None:
    assert matched.control_seed(0, 0, 0) == 9_000_000
    assert matched.control_seed(0, 0, 1) == 9_000_001
    assert matched.control_seed(1, 0, 0) == 9_001_000
    seeds = {
        matched.control_seed(tree, history, draw)
        for tree in range(matched.TREE_COUNT)
        for history in range(matched.SLOTS_PER_TREE)
        for draw in range(matched.DRAWS_PER_SLOT)
    }
    assert len(seeds) == matched.EXPECTED_REQUESTS


def test_source_tree_expands_to_exact_frozen_branch_slots() -> None:
    _, trees, _ = matched.quality.load_and_validate_sources()
    slots = matched.branch_slots(trees["trees"][0])

    assert len(slots) == 48
    assert sum(slot["stage"] == "first" for slot in slots) == 16
    assert sum(slot["stage"] == "second" for slot in slots) == 32
    assert slots[0]["stage"] == "first"
    assert slots[16]["stage"] == "second"


def test_mechanics_gates_accept_exact_strict_accounting() -> None:
    controls = {"trees": [{} for _ in range(matched.TREE_COUNT)]}
    pool_rows = [
        _pool_row()
        for _ in range(matched.TREE_COUNT * matched.SLOTS_PER_TREE)
    ]
    usage = {
        "adapter_requests": matched.EXPECTED_REQUESTS,
        "http_attempts": matched.EXPECTED_REQUESTS + 2,
        "retry_count": 2,
        "provider_error_retries": 2,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 3.5,
    }

    gates = matched.mechanics_gates(
        usage=usage,
        controls=controls,
        pool_rows=pool_rows,
    )

    assert all(gates.values())


def test_scientific_summary_uses_frozen_difference_signs() -> None:
    rows = [
        _score_row(
            contrast=0.01 + index / 100000,
            realized=0.02 + index / 100000,
        )
        for index in range(matched.TREE_COUNT)
    ]

    first = matched.summarize_scores(rows, bootstrap_samples=100)
    second = matched.summarize_scores(rows, bootstrap_samples=100)

    assert first == second
    assert first["directionally_coherent"] is True
    assert all(first["scientific_gates"].values())
    assert (
        first["bootstrap"][
            "second_conditional_minus_history_blind_predictive_mse_95pct"
        ][1]
        < 0.0
    )


def test_serving_gates_require_strict_exact_ten() -> None:
    diagnostics = [_diagnostic() for _ in range(10)]
    pools = [
        {
            "valid_unique_count": 30,
            "draw_novel_contributions": [20, 10],
        }
        for _ in range(5)
    ]
    usage = {
        "adapter_requests": 10,
        "http_attempts": 10,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.02,
    }

    gates = smoke.serving_gates(
        draw_diagnostics=diagnostics,
        pools=pools,
        usage=usage,
    )

    assert gates["all_pass"] is True


def test_smoke_validation_is_hash_bound(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "RESULT.json"
    payload = {
        "status": "passed",
        "protocol": {
            "interface_version": smoke.INTERFACE_VERSION,
            "model": smoke.MODEL_ID,
            "expected_requests": 10,
            "efficacy_used_for_authorization": False,
        },
        "gates": {"all_pass": True},
    }
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(matched, "SMOKE_RESULT_SHA256", digest)

    validated = matched.validate_smoke_result(path)

    assert validated == payload
    path.write_text(json.dumps({**payload, "status": "gated_null"}))
    with pytest.raises(ValueError, match="hash changed"):
        matched.validate_smoke_result(path)


def test_summary_requires_complete_cohort() -> None:
    with pytest.raises(ValueError, match="32 trees"):
        matched.summarize_scores(
            [deepcopy(_score_row())],
            bootstrap_samples=10,
        )


def test_zero_call_full_formal_path(
    tmp_path,
    monkeypatch,
) -> None:
    targets = json.loads(matched.SOURCE_TARGETS.read_text())["targets"]

    def response(items) -> str:
        return json.dumps(
            {
                "hypotheses": [
                    {
                        "name": item["name"],
                        "expression": item["expression"],
                    }
                    for item in items
                ]
            }
        )

    adapter = _FakeFormalAdapter(
        response(targets[:24]),
        response(targets[9:33]),
    )
    smoke_path = tmp_path / "SMOKE.json"
    smoke_payload = {
        "status": "passed",
        "protocol": {
            "interface_version": smoke.INTERFACE_VERSION,
            "model": smoke.MODEL_ID,
            "expected_requests": 10,
            "efficacy_used_for_authorization": False,
        },
        "usage": {"run_cost_usd": 0.0},
        "gates": {"all_pass": True},
    }
    smoke_path.write_text(json.dumps(smoke_payload))
    monkeypatch.setattr(
        matched,
        "SMOKE_RESULT_SHA256",
        hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
    )

    result = matched.run_formal(
        output_dir=tmp_path / "formal",
        run_id="zero-call-integration",
        smoke_result_path=smoke_path,
        adapter=adapter,
        remaining_credit=10.0,
        bootstrap_samples=20,
    )

    assert adapter.request_count == matched.EXPECTED_REQUESTS
    assert all(result["mechanics_gates"].values())
    assert len(result["trees"]) == matched.TREE_COUNT
    assert result["usage"]["run_cost_usd"] == 0.0
    assert (tmp_path / "formal" / "CONTROLS.json").exists()
    assert (tmp_path / "formal" / "RESULT.json").exists()
