import copy

import pytest

from scripts import (
    number_game_deepseek_v4_flash_paired_efficacy32 as efficacy,
)


def _scored_tree(seed: int, candidate: float, baseline: float) -> dict:
    difference = candidate - baseline
    return {
        "tree_seed": seed,
        "endpoint": {
            "crossfit_depth_three": {
                "mean_posterior_predictive_brier": candidate,
                "mean_best_hamming_error": 0.03,
                "truth_extension_coverage_rate": 0.8,
            },
            "myopic_eig": {
                "mean_posterior_predictive_brier": baseline,
                "mean_best_hamming_error": 0.04,
                "truth_extension_coverage_rate": 0.7,
            },
        },
        "comparisons": {
            "myopic_eig": {
                "candidate_minus_baseline_brier": difference,
                "candidate_minus_baseline_hamming": -0.01,
                "coverage_difference": 0.1,
            }
        },
    }


def _live_tree() -> dict:
    diagnostic = {"raw_count": 24}
    return {
        "initial_diagnostics": diagnostic,
        "first_branch_diagnostics": {
            str(index): diagnostic for index in range(16)
        },
        "second_branch_diagnostics": {
            str(index): diagnostic for index in range(32)
        },
        "mechanics": {
            "initial_valid": 20,
            "minimum_first_branch_valid": 10,
            "minimum_second_branch_valid": 6,
        },
        "usage": {
            "adapter_requests": 49,
            "http_attempts": 49,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "run_cost_usd": 0.005,
        },
    }


def test_frozen_bank_and_exact_ten_smoke_are_hash_bound() -> None:
    smoke = efficacy.validate_smoke_result()
    trees, targets, result = efficacy.load_frozen_bank()

    assert smoke["protocol"]["model"] == efficacy.MODEL_ID
    assert len(trees["trees"]) == 32
    assert len(targets["targets"]) == 33
    assert len(result["trees"]) == 32
    assert [tree["tree_seed"] for tree in trees["trees"]] == list(
        efficacy.TREE_SEEDS
    )


def test_local_target_adapter_is_parser_valid_and_has_zero_usage() -> None:
    _, targets, _ = efficacy.load_frozen_bank()
    adapter = efficacy.LocalTargetAdapter(targets["targets"])

    responses = adapter.chat_complete_messages_batched_structured([[]])
    parsed, diagnostics = efficacy.depth.parse_proposals(responses[0])

    assert len(parsed) == 24
    assert diagnostics["raw_count"] == 24
    assert adapter.calls == 1
    assert adapter.usage_snapshot()["adapter_requests"] == 0
    assert adapter.usage_snapshot()["adapter_cost_usd"] == 0.0


def test_mechanics_require_only_the_1568_planner_requests() -> None:
    live = [_live_tree() for _ in range(32)]
    usage = efficacy.aggregate_usage(live)

    gates = efficacy.mechanics_gates(
        live_trees=live,
        usage=usage,
        local_target_calls=32,
    )

    assert all(gates.values())
    assert usage["adapter_requests"] == 1568
    assert gates["zero_provider_target_or_validation_calls"]


def test_intelligence_and_noninferiority_use_frozen_thresholds(
    monkeypatch,
) -> None:
    monkeypatch.setattr(efficacy, "BOOTSTRAP_SAMPLES", 500)
    candidate = [
        _scored_tree(seed, candidate=0.09, baseline=0.11)
        for seed in efficacy.TREE_SEEDS
    ]
    myopic = efficacy.comparison_with_frozen_bootstrap(
        candidate,
        baseline="myopic_eig",
    )
    assert all(efficacy.intelligence_gates(myopic).values())

    qwen = [
        _scored_tree(seed, candidate=0.086, baseline=0.11)
        for seed in efficacy.TREE_SEEDS
    ]
    comparison = efficacy.qwen_noninferiority(candidate, qwen)
    assert comparison["mean_candidate_minus_qwen_brier"] == pytest.approx(
        0.004
    )
    assert comparison["upper_ci_below_margin"]

    worse = [
        _scored_tree(seed, candidate=0.092, baseline=0.11)
        for seed in efficacy.TREE_SEEDS
    ]
    assert not efficacy.qwen_noninferiority(worse, qwen)[
        "upper_ci_below_margin"
    ]


def test_run_reuses_source_validations_without_target_provider_calls(
    tmp_path,
    monkeypatch,
) -> None:
    source_trees, _, source_result = efficacy.load_frozen_bank()
    source_by_seed = {
        int(tree["tree_seed"]): tree for tree in source_trees["trees"]
    }
    qwen_by_seed = {
        int(tree["tree_seed"]): tree for tree in source_result["trees"]
    }
    target_adapters = []

    def fake_runner(**kwargs):
        target = kwargs["target_adapter"]
        target.chat_complete_messages_batched_structured([[]])
        target_adapters.append(target)
        source = copy.deepcopy(source_by_seed[int(kwargs["tree_seed"])])
        return _live_tree(), {"raw": {"fake": True}, "public": source}

    def fake_score_crossfit(tree):
        return {"tree_seed": tree["tree_seed"]}

    def fake_score_fixed(tree, metrics, endpoints):
        del metrics
        assert endpoints[0]
        qwen_brier = qwen_by_seed[int(tree["tree_seed"])]["endpoint"][
            "crossfit_depth_three"
        ]["mean_posterior_predictive_brier"]
        return _scored_tree(
            int(tree["tree_seed"]),
            candidate=float(qwen_brier),
            baseline=float(qwen_brier) + 0.02,
        )

    monkeypatch.setattr(
        efficacy,
        "score_crossfit_public_tree",
        fake_score_crossfit,
    )
    monkeypatch.setattr(efficacy, "score_fixed_tree", fake_score_fixed)
    monkeypatch.setattr(
        efficacy,
        "aggregate_scored_trees",
        lambda trees: {"comparisons": {}},
    )
    monkeypatch.setattr(efficacy, "BOOTSTRAP_SAMPLES", 500)

    result = efficacy.run_efficacy(
        output_dir=tmp_path,
        run_id="test-deepseek-v4-flash-efficacy",
        tree_runner=fake_runner,
    )

    written = efficacy.json.loads(
        (tmp_path / "TREES.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "passed"
    assert len(target_adapters) == 32
    assert all(
        adapter.usage_snapshot()["adapter_requests"] == 0
        for adapter in target_adapters
    )
    assert (
        written["trees"][7]["validation_supports"]
        == source_trees["trees"][7]["validation_supports"]
    )
    assert result["protocol"]["provider_target_generation_calls"] == 0
    assert result["protocol"]["provider_validation_generation_calls"] == 0
