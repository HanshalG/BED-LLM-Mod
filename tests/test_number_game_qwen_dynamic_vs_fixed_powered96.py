from __future__ import annotations

import json
import re

import pytest

from scripts import (
    number_game_crossfit_depth_three_confirmation as crossfit,
    number_game_qwen_dynamic_vs_fixed_powered96 as run,
)


def scored_tree() -> dict:
    return {
        "mechanics": {
            "initial_valid": 24,
            "minimum_first_branch_valid": 12,
            "minimum_retained_second_branch_valid": 8,
            "validation_support_count": 16,
            "minimum_validation_support_valid": 16,
        }
    }


def test_frozen_counts_and_disjoint_seeds() -> None:
    assert run.REQUESTS_PER_TREE == 115
    assert run.EXPECTED_REQUESTS == 11_040
    assert run.EXPECTED_POOLED_PARSE_EVENTS == 4_704
    assert run.EXPECTED_PARSE_EVENTS == 6_336
    validation = {
        seed
        for index in range(run.TREE_COUNT)
        for seed in range(
            run.VALIDATION_SEED_START
            + index * run.VALIDATION_DRAWS_PER_TREE,
            run.VALIDATION_SEED_START
            + (index + 1) * run.VALIDATION_DRAWS_PER_TREE,
        )
    }
    assert len(validation) == 1_536
    assert not (
        set(run.TREE_SEEDS)
        | set(run.TARGET_SEEDS)
    ) & validation


def test_committed_smoke_remains_bound() -> None:
    result = run.validate_smoke_result(run.SMOKE_RESULT)
    assert result["status"] == "passed"
    assert result["usage"]["adapter_requests"] == 10


def test_starting_balance_gate() -> None:
    run.require_starting_balance(run.MIN_STARTING_BALANCE_USD)
    with pytest.raises(RuntimeError):
        run.require_starting_balance(run.MIN_STARTING_BALANCE_USD - 0.01)


def test_mechanics_accept_exact_frozen_counts() -> None:
    usage = {
        "adapter_requests": run.EXPECTED_REQUESTS,
        "http_attempts": run.EXPECTED_REQUESTS,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 14.0,
    }
    targets = [
        type("Target", (), {"extension": (index,)})()
        for index in range(run.TARGET_COUNT)
    ]
    parse_summary = {
        "parse_events": run.EXPECTED_PARSE_EVENTS,
        "pooled_parse_events": run.EXPECTED_POOLED_PARSE_EVENTS,
        "provider_draws_parsed": run.EXPECTED_REQUESTS,
        "item_salvaged_draws": 0,
    }
    gates = run.mechanics_gates(
        scored_trees=[scored_tree() for _ in range(run.TREE_COUNT)],
        usage=usage,
        targets=targets,
        parse_summary=parse_summary,
    )
    assert all(gates.values())


def test_dynamic_gates_treat_ties_as_no_treatment() -> None:
    comparison = {
        "relative_brier_reduction": 0.03,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.01, -0.001],
        "brier_tree_wins": 38,
        "brier_tree_losses": 37,
        "brier_tree_ties": 21,
    }
    assert all(
        run.dynamic_support_gates(
            comparison=comparison,
            root_differences=48,
        ).values()
    )
    comparison["brier_tree_losses"] = 38
    assert not run.dynamic_support_gates(
        comparison=comparison,
        root_differences=48,
    )["dynamic_wins_exceed_losses"]


def test_frozen_comparison_records_losses_and_ties(monkeypatch) -> None:
    trees = [
        {
            "comparisons": {
                "fixed_support_depth_three": {
                    "candidate_minus_baseline_brier": value,
                    "candidate_minus_baseline_hamming": value,
                    "coverage_difference": -value,
                }
            }
        }
        for value in (-0.2, 0.0, 0.1)
    ]
    monkeypatch.setattr(
        run.engine,
        "aggregate_scored_trees",
        lambda _: {
            "comparisons": {
                "fixed_support_depth_three": {
                    "relative_brier_reduction": 0.1,
                    "brier_tree_wins": 1,
                }
            }
        },
    )
    comparison = run.comparison_with_frozen_bootstrap(
        trees,
        baseline="fixed_support_depth_three",
        seed=7,
        samples=100,
    )
    assert comparison["brier_tree_wins"] == 1
    assert comparison["brier_tree_losses"] == 1
    assert comparison["brier_tree_ties"] == 1


def test_configured_engine_uses_powered_seeds_and_counts() -> None:
    original_qwen_draws = run.qwen.VALIDATION_DRAWS_PER_TREE
    with run.configured_engine(run.SMOKE_RESULT, []):
        assert run.qwen.VALIDATION_DRAWS_PER_TREE == 16
        assert run.engine.TREE_SEEDS == run.TREE_SEEDS
        assert run.engine.TARGET_SEEDS == run.TARGET_SEEDS
        assert run.engine.EXPECTED_REQUESTS == 11_040
        assert run.engine.validation_seeds_for_tree(0) == tuple(
            range(69_000, 69_016)
        )
        assert run.engine.validation_seeds_for_tree(95) == tuple(
            range(70_520, 70_536)
        )
    assert run.qwen.VALIDATION_DRAWS_PER_TREE == original_qwen_draws


class _SyntheticAdapter:
    def __init__(self, *, request_seed: int) -> None:
        self.request_seed = request_seed
        self.requests = 0

    @staticmethod
    def _observations(messages) -> list[tuple[int, bool]]:
        text = messages[-1]["content"]
        return [
            (int(number), label == "True")
            for number, label in re.findall(
                r"n=(\d+) MUST return (True|False)",
                text,
            )
        ]

    def chat_complete_messages_batched_structured(
        self,
        batch_messages,
        **kwargs,
    ):
        del kwargs
        self.requests += len(batch_messages)
        responses = []
        for messages in batch_messages:
            observations = self._observations(messages)
            hypotheses = []
            seed_offset = self.request_seed % 23
            for index, modulus in enumerate(range(2, 26)):
                remainder = (index + seed_offset) % modulus
                expression = f"n % {modulus} == {remainder}"
                for number, label in observations:
                    if label:
                        expression = f"({expression}) or n == {number}"
                    else:
                        expression = f"({expression}) and n != {number}"
                hypotheses.append(
                    {
                        "name": f"synthetic_{self.request_seed}_{index}",
                        "expression": expression,
                    }
                )
            responses.append(json.dumps({"hypotheses": hypotheses}))
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
        }


def test_one_tree_synthetic_run_exercises_powered_wrapper(
    tmp_path,
    monkeypatch,
) -> None:
    adapters = []

    def fake_adapter(**kwargs):
        adapter = _SyntheticAdapter(
            request_seed=int(kwargs["request_seed"])
        )
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(run.pooled.depth, "_adapter", fake_adapter)
    monkeypatch.setattr(run, "TREE_SEEDS", (90_000,))
    monkeypatch.setattr(run, "TARGET_SEEDS", (90_100,))
    monkeypatch.setattr(run, "TREE_COUNT", 1)
    monkeypatch.setattr(run, "VALIDATION_SEED_START", 90_200)
    monkeypatch.setattr(run, "EXPECTED_REQUESTS", run.REQUESTS_PER_TREE)
    monkeypatch.setattr(
        run,
        "EXPECTED_POOLED_PARSE_EVENTS",
        run.PLANNING_HISTORIES_PER_TREE,
    )
    monkeypatch.setattr(
        run,
        "EXPECTED_PARSE_EVENTS",
        run.PARSE_EVENTS_PER_TREE,
    )
    monkeypatch.setattr(crossfit, "BOOTSTRAP_SAMPLES", 100)

    result = run.run_powered_replication(
        output_dir=tmp_path,
        run_id="synthetic-one-tree",
    )

    assert result["status"] == "gated_null"
    assert result["usage"]["adapter_requests"] == 115
    assert sum(adapter.requests for adapter in adapters) == 115
    accounting = result["protocol"]["parse_accounting"]
    assert accounting["parse_events"] == 66
    assert accounting["pooled_parse_events"] == 49
    assert accounting["provider_draws_parsed"] == 115
    assert (tmp_path / "RESULT.json").is_file()
