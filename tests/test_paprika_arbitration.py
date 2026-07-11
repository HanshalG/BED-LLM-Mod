from __future__ import annotations

import numpy as np

from core import BeliefState, list_methods
from core.defaults import register_defaults
from environments.paprika_customer_service.prompts import arbitration_candidate_messages
from helpers import load_config
from methods.paprika_arbitration import (
    PaprikaNaivePrimaryArbitration,
    PaprikaNaivePrimaryCandidate0,
    categorical_eig_standard_error,
    select_naive_primary_index,
)


def test_arbitration_prompt_orders_native_default_without_beliefs() -> None:
    messages = arbitration_candidate_messages("A device is broken.", ())
    text = "\n".join(message["content"] for message in messages)
    assert "exactly 3 candidates" in text
    assert "Candidate 0" in text
    assert "natural" in text
    assert "hypoth" not in text.casefold()


def test_arbitration_margin_keeps_native_unless_gap_clears_combined_se() -> None:
    selected, gap, threshold = select_naive_primary_index(
        [0.10, 0.20, 0.05], [0.08, 0.08, 0.01]
    )
    assert selected == 0
    assert gap < threshold
    selected, gap, threshold = select_naive_primary_index(
        [0.10, 0.40, 0.05], [0.08, 0.08, 0.01]
    )
    assert selected == 1
    assert gap > threshold


def test_arbitration_scores_only_three_native_proposals() -> None:
    class FakeEnvironment:
        def generate_arbitration_actions(self, belief_state, history, model, config):
            del belief_state, history, model, config
            return ["native", "informative", "other"]

        def outcome_likelihoods(self, hypotheses, action):
            del hypotheses
            if action == "informative":
                return np.asarray([[1.0, 0.0], [0.0, 1.0]])
            return np.asarray([[0.5, 0.5], [0.5, 0.5]])

    method = PaprikaNaivePrimaryArbitration()
    belief = BeliefState(hypotheses=("h0", "h1"), probabilities=(0.5, 0.5))
    chosen = method.select_action([], belief, FakeEnvironment(), None, [], object())
    assert chosen.action == "informative"
    assert chosen.extras is not None
    assert chosen.extras["candidate_queries"] == ["native", "informative", "other"]
    assert chosen.extras["native_overridden"] is True
    assert chosen.extras["selected_index"] == 1


def test_arbitration_standard_error_is_zero_for_equal_contributions() -> None:
    likelihoods = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert categorical_eig_standard_error((0.5, 0.5), likelihoods) == 0.0


def test_arbitration_is_registered_only_for_paprika() -> None:
    register_defaults(force=True)
    assert "NaivePrimaryArbitration" in list_methods("paprika_customer_service")
    assert "NaivePrimaryCandidate0" in list_methods("paprika_customer_service")
    assert "NaivePrimaryArbitration" not in list_methods("location_finding")
    assert "NaivePrimaryCandidate0" not in list_methods("location_finding")


def test_prompt_matched_candidate0_is_belief_free_and_always_uses_index_zero() -> None:
    class FakeEnvironment:
        def generate_arbitration_actions(self, belief_state, history, model, config):
            del belief_state, history, model, config
            return ["native", "alternative-1", "alternative-2"]

    method = PaprikaNaivePrimaryCandidate0()
    assert method.requires_belief_state(FakeEnvironment(), object()) is False
    chosen = method.select_action(
        [], BeliefState(), FakeEnvironment(), None, [], object()
    )
    assert chosen.action == "native"
    assert chosen.score == 0.0
    assert chosen.extras == {
        "metric_name": "selected_eig",
        "candidate_queries": ["native", "alternative-1", "alternative-2"],
        "native_default_index": 0,
        "selected_index": 0,
        "native_overridden": False,
        "selection_rule": "prompt_matched_candidate_0",
    }


def test_prompt_matched_candidate0_batch_uses_first_proposal_per_history() -> None:
    class FakeEnvironment:
        def generate_arbitration_actions_many(
            self, belief_states, histories, model, config
        ):
            del belief_states, model, config
            return [
                [f"native-{index}", "a", "b"]
                for index, _history in enumerate(histories)
            ]

    chosen = PaprikaNaivePrimaryCandidate0().select_actions(
        [[], []],
        [BeliefState(), BeliefState()],
        FakeEnvironment(),
        None,
        [[], []],
        object(),
    )
    assert [item.action for item in chosen] == ["native-0", "native-1"]
    assert all(item.extras["selected_index"] == 0 for item in chosen if item.extras)


def test_headline_configs_freeze_method_order_range_and_concurrency() -> None:
    triplet = load_config(
        "configs/config_paprika_arbitration_headline_triplet_openrouter.yaml"
    )
    assert triplet.method_names == [
        "NaivePrimaryArbitration",
        "NaivePrimaryCandidate0",
        "naive",
    ]
    assert triplet.paprika_task_offset == 10
    assert triplet.paprika_num_trials == 1
    assert triplet.paprika_num_rounds == 5
    assert triplet.openrouter_concurrency == 25
    assert triplet.model_pairs[0].questioner.thinking is True
    assert triplet.model_pairs[0].questioner.thinking_max_new_tokens == 8192

    nonthinking = load_config(
        "configs/config_paprika_arbitration_headline_nonthinking_openrouter.yaml"
    )
    assert nonthinking.method_names == ["naive"]
    assert nonthinking.paprika_task_offset == 10
    assert nonthinking.paprika_num_trials == 10
    assert nonthinking.paprika_trial_batch_size == 10
    assert nonthinking.openrouter_concurrency == 51
    assert nonthinking.model_pairs[0].questioner.thinking is False
