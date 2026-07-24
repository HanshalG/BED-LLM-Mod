from __future__ import annotations

import math
from pathlib import Path

import pytest

from core import BeliefState
from helpers import load_config
from scripts.animals_cabed_shared_tree_v10 import (
    DeterministicMechanicsModel,
    bayes_update,
    immediate_eig,
    run_stage,
    select_valid_questions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config_animals_cabed_shared_tree_v10.yaml"


def test_immediate_eig_and_bayes_update_match_perfect_binary_split():
    belief = BeliefState.uniform(["cat", "shark"])
    yes_probabilities = [1.0, 0.0]

    score = immediate_eig(belief, yes_probabilities)
    posterior = bayes_update(belief, yes_probabilities, "Yes")

    assert score == pytest.approx(math.log(2.0))
    assert posterior.hypotheses[0] == "cat"
    assert posterior.probabilities[0] == pytest.approx(1.0)


def test_select_valid_questions_removes_history_duplicates_and_direct_guesses():
    selected = select_valid_questions(
        [
            "Is it warm-blooded?",
            "Is it a cat?",
            "Does it have fur?",
            "does   it have fur",
            "Does it lay eggs?",
            "Does it live underwater?",
        ],
        width=3,
        history=[("Is it warm-blooded", "Yes")],
        support=["cat", "shark"],
    )

    assert selected == (
        "Does it have fur?",
        "Does it lay eggs?",
        "Does it live underwater?",
    )


def test_v10_config_fixes_shared_tree_bayes_protocol():
    config = load_config(str(CONFIG_PATH))

    assert len(config.animals[0]) == 64
    assert config.animals_belief_update_mode == "bayes_fixed_support"
    assert config.animals_likelihood_confidence == pytest.approx(0.7)
    assert config.belief_generation_enabled is False
    assert config.belief_filtering_enabled is False
    assert config.belief_prior_mode == "uniform"


def test_zero_cost_smoke_builds_complete_shared_trees_with_cached_likelihoods():
    config = load_config(str(CONFIG_PATH))
    model = DeterministicMechanicsModel()

    payload = run_stage(
        config,
        stage="serving_smoke",
        questioner=model,
        answerer=model,
    )

    assert payload["status"] == "passed"
    assert payload["summary"]["num_states"] == 2
    assert payload["summary"]["gates"]["all_states_complete"] is True
    assert payload["likelihood_cache_entries"] > 0
    assert model.probability_calls == 3
    for record in payload["records"]:
        assert len(record["roots"]) == 4
        assert all(
            len(branch["candidates"]) == 3
            for root in record["roots"]
            for branch in root["branches"]
        )
        assert set(record["selections"]) == {
            "depth_one",
            "depth_two",
            "random",
        }
