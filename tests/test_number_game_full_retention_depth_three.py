import json
import re

import pytest

from scripts import number_game_depth_three_development as depth
from scripts import number_game_full_retention_depth_three as full
from scripts.number_game_full_retention_depth_three import (
    BRIER_TOLERANCE,
    EXPECTED_REQUESTS,
    MIN_STARTING_BALANCE_USD,
    RUN_BUDGET_USD,
    TARGET_SEEDS,
    TREE_SEEDS,
    powered_gates,
)
from scripts.number_game_retained_depth_three import score_public_tree


def _comparison(
    *,
    gain: float = 0.06,
    wins: int = 8,
) -> dict:
    return {
        "relative_brier_reduction": gain,
        "brier_tree_wins": wins,
        "mean_candidate_minus_baseline_brier": -0.01,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.02, -0.001],
        "mean_candidate_minus_baseline_hamming": -0.01,
        "mean_coverage_difference": 0.01,
    }


def test_powered_protocol_constants_are_frozen():
    assert TREE_SEEDS == tuple(range(28000, 28020))
    assert TARGET_SEEDS == tuple(range(28100, 28120))
    assert EXPECTED_REQUESTS == 1000
    assert RUN_BUDGET_USD == pytest.approx(3.60)
    assert MIN_STARTING_BALANCE_USD == pytest.approx(3.25)
    assert BRIER_TOLERANCE == pytest.approx(0.005)


def test_powered_gate_is_conjunctive():
    scored = [
        {
            "mechanics": {
                "initial_valid": 20,
                "minimum_first_branch_valid": 10,
                "minimum_retained_second_branch_valid": 10,
                "mean_first_branch_valid": 22.0,
                "mean_generated_first_branch_valid": 15.0,
                "mean_retained_second_branch_valid": 20.0,
                "mean_generated_second_branch_valid": 14.0,
                "target_valid": 20,
                "novel_targets": 10,
            }
        }
    ]
    live = [
        {
            "mechanics": {
                "minimum_first_branch_valid": 10,
                "minimum_second_branch_valid": 10,
            }
        }
    ]
    aggregate = {
        "comparisons": {
            "predictive_bayes_risk_depth_two": _comparison(gain=0.02),
            "retained_parent_only_depth_three": _comparison(gain=0.02),
            "generated_only_depth_three": _comparison(),
            "myopic_eig": _comparison(gain=0.06),
            "fixed_support_depth_three": _comparison(gain=0.06),
            "uniform_random_candidate_root": _comparison(),
        },
        "root_differences": {
            "predictive_bayes_risk_depth_two_root": 7,
            "retained_parent_only_depth_three_root": 6,
            "generated_only_depth_three_root": 4,
        },
        "ranking": {
            "predictive_risk_spearman_brier": {"mean": 0.5},
            "predictive_pairwise_concordance": {"mean": 0.7},
        },
        "novel_target_mean_differences": {
            "candidate_minus_baseline_brier": -0.01,
            "candidate_minus_baseline_hamming": -0.01,
            "coverage_difference": 0.01,
        },
    }
    usage = {
        "adapter_requests": 1000,
        "http_attempts": 1000,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 3.2,
    }

    gates = powered_gates(
        scored_trees=scored,
        live_trees=live,
        usage=usage,
        aggregate=aggregate,
    )

    assert all(gates.values())
    aggregate["comparisons"]["predictive_bayes_risk_depth_two"][
        "tree_cluster_brier_difference_95pct_bootstrap"
    ][1] = 0.0
    assert not powered_gates(
        scored_trees=scored,
        live_trees=live,
        usage=usage,
        aggregate=aggregate,
    )["brier_cluster_ci_vs_depth_two_below_zero"]


class _DepthFakeAdapter:
    def __init__(self, stage_offsets: list[int]) -> None:
        self.stage_offsets = stage_offsets
        self.batch_index = 0
        self.requests = 0

    @staticmethod
    def _response(
        observations: list[tuple[int, bool]],
        *,
        offset: int,
    ) -> str:
        if not observations and offset == 0:
            expressions = []
            for modulus in range(2, 12):
                for remainder in range(modulus):
                    expressions.append(
                        f"n % {modulus} == {remainder}"
                    )
                    if len(expressions) == 24:
                        return json.dumps(
                            {
                                "hypotheses": [
                                    {
                                        "name": f"fake_prior_{index}",
                                        "expression": expression,
                                    }
                                    for index, expression in enumerate(
                                        expressions
                                    )
                                ]
                            }
                        )
            raise AssertionError("fake prior pool was too small")
        positives = [
            number for number, label in observations if label
        ]
        observed = {number for number, _ in observations}
        extras = []
        for step in range(101):
            number = (offset + step) % 101
            if number in observed:
                continue
            extras.append(number)
            if len(extras) == 24:
                break
        hypotheses = []
        for index, extra in enumerate(extras):
            members = [*positives, extra]
            expression = " or ".join(
                f"n == {number}" for number in members
            )
            hypotheses.append(
                {
                    "name": f"fake_{offset}_{index}",
                    "expression": expression,
                }
            )
        return json.dumps({"hypotheses": hypotheses})

    def chat_complete_messages_batched_structured(
        self,
        messages,
        **kwargs,
    ):
        del kwargs
        offset = self.stage_offsets[self.batch_index]
        self.batch_index += 1
        self.requests += len(messages)
        responses = []
        for request in messages:
            prompt = request[-1]["content"]
            observations = [
                (int(number), label == "YES")
                for number, label in re.findall(
                    r"Is (\d+) in the concept\? (YES|NO)\.",
                    prompt,
                )
            ]
            responses.append(
                self._response(observations, offset=offset)
            )
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
        }


class _CreditResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args

    def read(self):
        return json.dumps(self.payload).encode()


def test_openrouter_balance_preflight_uses_live_credit(
    monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_urlopen(request, *, timeout):
        assert request.get_header("Authorization") == "Bearer test-key"
        assert timeout == pytest.approx(30.0)
        return _CreditResponse(
            {
                "data": {
                    "total_credits": 10.0,
                    "total_usage": 6.75,
                }
            }
        )

    monkeypatch.setattr(
        full.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    remaining = full.openrouter_remaining_credit()

    assert remaining == pytest.approx(3.25)
    full.require_starting_balance(remaining)
    with pytest.raises(RuntimeError, match="below the frozen"):
        full.require_starting_balance(3.249)


def test_full_retention_rehearsal_exercises_both_refreshes(
    tmp_path,
    monkeypatch,
):
    planning = _DepthFakeAdapter([0, 30, 60])
    target = _DepthFakeAdapter([70])
    adapters = iter((planning, target))
    adapter_run_ids = []

    def fake_adapter(**kwargs):
        adapter_run_ids.append(kwargs["run_id"])
        return next(adapters)

    monkeypatch.setattr(depth, "_adapter", fake_adapter)

    tree, artifacts = depth.run_tree_depth_three(
        tree_index=0,
        tree_seed=28000,
        target_seed=28100,
        output_dir=tmp_path,
        run_id="fake-full-retention",
        shared_budget_run_id="fake-shared-budget",
        first_support_mode=depth.FIRST_SUPPORT_RETAINED_REJUVENATION,
        second_support_mode=depth.SECOND_SUPPORT_RETAINED_REJUVENATION,
        brier_tolerance=BRIER_TOLERANCE,
    )
    scored = score_public_tree(artifacts["public"])

    assert planning.requests == 49
    assert target.requests == 1
    assert adapter_run_ids == [
        "fake-shared-budget",
        "fake-shared-budget",
    ]
    assert tree["mechanics"]["exact_50_requests"]
    assert tree["first_support_mode"] == "retained_rejuvenation"
    assert tree["second_support_mode"] == "retained_rejuvenation"
    assert (
        scored["mechanics"]["mean_first_branch_valid"]
        > scored["mechanics"]["mean_generated_first_branch_valid"]
    )
    assert (
        scored["mechanics"]["mean_retained_second_branch_valid"]
        > scored["mechanics"]["mean_generated_second_branch_valid"]
    )
    assert scored["mechanics"]["minimum_first_branch_valid"] >= 24
    assert (
        scored["mechanics"]["minimum_retained_second_branch_valid"]
        >= 24
    )
