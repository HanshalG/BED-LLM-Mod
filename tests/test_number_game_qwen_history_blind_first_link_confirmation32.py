from __future__ import annotations

import hashlib
import json

from scripts import number_game_qwen_history_blind_first_link_confirmation32 as run
from scripts import number_game_qwen_history_blind_matched32 as base
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
        assert len(batch_messages) == base.EXPECTED_REQUESTS
        assert len(seeds) == base.EXPECTED_REQUESTS
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


def _score_row(index: int) -> dict:
    centered = (index - 15.5) / 1000.0
    return {
        "tree_mean": {
            "first": _stage(-0.01, 0.1),
            "second": _stage(-0.02, 0.1),
        },
        "selected_roots": {
            "roots_differ": True,
            "dynamic_root_prompt_conditioning_benefit": centered,
            "fixed_root_prompt_conditioning_benefit": 0.0,
            "dynamic_minus_fixed_root_prompt_conditioning_benefit": (
                centered
            ),
            "realized_advantage": centered,
        },
    }


def _response(items) -> str:
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


def _smoke_path(tmp_path, monkeypatch):
    path = tmp_path / "SMOKE.json"
    payload = {
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
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        base,
        "SMOKE_RESULT_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return path


def test_development_result_and_controls_are_hash_bound() -> None:
    result = run.validate_development()

    assert result["status"] == "gated_null"
    assert result["protocol"]["tree_indices"] == list(range(32))
    assert (
        result["analysis"]["selected_root_prompt_conditioning"][
            "prompt_benefit_contrast_to_realized_advantage_spearman"
        ]
        == run.DEVELOPMENT_SPEARMAN
    )


def test_confirmation_manifest_uses_disjoint_source_block() -> None:
    result, trees, _ = base.quality.load_and_validate_sources()
    scored = result["trees"][32:64]

    with run.configured_base():
        manifest = base.request_manifest(trees["trees"][32:64])

    assert (
        sum(
            row["selection"]["crossfit_depth_three_root"]
            != row["selection"]["fixed_support_depth_three_root"]
            for row in scored
        )
        == 21
    )
    assert len(manifest) == base.EXPECTED_REQUESTS
    assert manifest[0]["tree_index"] == 32
    assert manifest[0]["local_tree_index"] == 0
    assert manifest[0]["tree_seed"] == 80_032
    assert manifest[0]["seed"] == 9_600_000
    assert manifest[-1]["tree_index"] == 63
    assert manifest[-1]["local_tree_index"] == 31
    assert manifest[-1]["tree_seed"] == 80_063
    assert len({item["seed"] for item in manifest}) == len(manifest)


def test_confirmation_context_restores_base_globals() -> None:
    names = (
        "INTERFACE_VERSION",
        "SOURCE_TREE_START",
        "SOURCE_TREE_SEEDS",
        "CONTROL_SEED_START",
        "BOOTSTRAP_SEED",
        "mechanics_gates",
        "summarize_scores",
    )
    originals = {name: getattr(base, name) for name in names}

    with run.configured_base():
        assert base.SOURCE_TREE_START == 32
        assert base.SOURCE_TREE_SEEDS == tuple(range(80_032, 80_064))
        assert base.control_seed(0, 0, 0) == 9_600_000
        assert base.summarize_scores is run.summarize_scores

    assert {name: getattr(base, name) for name in names} == originals


def test_confirmation_uses_correlation_not_prior_mean_gate() -> None:
    rows = [_score_row(index) for index in range(base.TREE_COUNT)]

    summary = run.summarize_scores(rows, bootstrap_samples=200)

    assert summary["directionally_coherent"] is True
    assert all(summary["scientific_gates"].values())
    assert (
        summary["descriptive_selected_root_mean_gate"][
            "changed_root_prompt_benefit_contrast_ci_above_zero"
        ]
        is False
    )
    assert (
        summary["descriptive_selected_root_mean_gate"][
            "used_for_confirmation_success"
        ]
        is False
    )


def test_zero_call_full_confirmation_and_replay(
    tmp_path,
    monkeypatch,
) -> None:
    targets = json.loads(base.SOURCE_TARGETS.read_text())["targets"]
    adapter = _FakeFormalAdapter(
        _response(targets[:24]),
        _response(targets[9:33]),
    )
    smoke_path = _smoke_path(tmp_path, monkeypatch)
    output_dir = tmp_path / "formal"

    result = run.run_formal(
        output_dir=output_dir,
        run_id="zero-call-first-link-confirmation",
        smoke_result_path=smoke_path,
        adapter=adapter,
        remaining_credit=10.0,
        bootstrap_samples=20,
    )
    replay = run.replay_saved_controls(
        controls_path=output_dir / "CONTROLS.json",
        result_path=output_dir / "RESULT.json",
        bootstrap_samples=20,
    )

    assert adapter.request_count == base.EXPECTED_REQUESTS
    assert all(result["mechanics_gates"].values())
    assert result["protocol"]["tree_indices"] == list(range(32, 64))
    assert result["protocol"]["tree_seeds"] == list(range(80_032, 80_064))
    assert result["protocol"]["control_seed_start"] == 9_600_000
    assert result["protocol"]["bootstrap_seed"] == 9_700_000
    assert result["protocol"]["development_responses_reused"] is False
    assert result["second_draw_novelty"]["used_as_gate"] is False
    assert [row["tree_index"] for row in result["trees"]] == list(
        range(32, 64)
    )
    assert replay["analysis"] == result["analysis"]
    assert replay["trees"] == result["trees"]
