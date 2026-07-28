import copy
import json
from pathlib import Path

import pytest

from scripts import number_game_crossfit_depth_three_confirmation as crossfit
from scripts import number_game_depth_three_development as depth


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TREES = (
    ROOT
    / "results/nonmyopic/number_game_full_retention_depth_three_powered"
    / "number-game-full-retention-depth-three-powered-20260728/TREES.json"
)


def test_validation_seeds_are_unique_and_disjoint() -> None:
    all_seeds = [
        seed
        for tree_index in range(len(crossfit.TREE_SEEDS))
        for seed in crossfit.validation_seeds_for_tree(tree_index)
    ]
    assert len(all_seeds) == crossfit.VALIDATION_SUPPORT_COUNT * len(
        crossfit.TREE_SEEDS
    )
    assert len(set(all_seeds)) == len(all_seeds)
    assert not set(all_seeds) & set(crossfit.TREE_SEEDS)
    assert not set(all_seeds) & set(crossfit.TARGET_SEEDS)


def test_expected_request_count_includes_all_validation_draws() -> None:
    assert crossfit.EXPECTED_REQUESTS_PER_TREE == 58
    assert crossfit.EXPECTED_REQUESTS == 1856


def test_starting_balance_check_is_fail_closed() -> None:
    with pytest.raises(RuntimeError):
        crossfit.require_starting_balance(5.79)
    crossfit.require_starting_balance(5.80)


def test_public_crossfit_scorer_uses_separate_validation_supports() -> None:
    source = json.loads(SOURCE_TREES.read_text())["trees"]
    tree = copy.deepcopy(source[0])
    tree["validation_supports"] = [
        source[index]["targets"] for index in range(1, 9)
    ]

    scored = crossfit.score_public_tree(tree)

    assert scored["mechanics"]["validation_support_count"] == 8
    assert scored["selection"]["crossfit_depth_three_root"] in tree["roots"]
    assert scored["selection"]["crossfit_depth_two_root"] in tree["roots"]
    assert (
        scored["endpoint"]["crossfit_depth_three"][
            "mean_posterior_predictive_brier"
        ]
        >= 0.0
    )
    assert "crossfit_depth_two" in scored["comparisons"]


class _ValidationAdapter:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched_structured(self, messages, **kwargs):
        del kwargs
        self.requests += len(messages)
        hypotheses = [
            {
                "name": f"mod_{modulus}_{remainder}",
                "expression": f"n % {modulus} == {remainder}",
            }
            for modulus in range(2, 8)
            for remainder in range(modulus)
        ][:24]
        return [json.dumps({"hypotheses": hypotheses}) for _ in messages]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
        }


def test_validation_generation_uses_eight_unique_seeded_calls(
    tmp_path,
    monkeypatch,
) -> None:
    seeds = []
    adapters = []

    def fake_adapter(**kwargs):
        seeds.append(kwargs["request_seed"])
        adapter = _ValidationAdapter()
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(depth, "_adapter", fake_adapter)

    supports, diagnostics, responses, snapshots = (
        crossfit._generate_validation_supports(
            tree_index=2,
            output_dir=tmp_path,
            run_id="shared-budget",
        )
    )

    assert seeds == list(crossfit.validation_seeds_for_tree(2))
    assert len(supports) == len(diagnostics) == len(responses) == 8
    assert all(len(support) >= 16 for support in supports)
    assert sum(snapshot["adapter_requests"] for snapshot in snapshots) == 8
    assert all(adapter.requests == 1 for adapter in adapters)
