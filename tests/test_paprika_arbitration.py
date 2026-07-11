from __future__ import annotations

import numpy as np

from core import BeliefState, list_methods
from core.defaults import register_defaults
from environments.paprika_customer_service.prompts import arbitration_candidate_messages
from methods.paprika_arbitration import (
    PaprikaNaivePrimaryArbitration,
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
    assert "NaivePrimaryArbitration" not in list_methods("location_finding")
