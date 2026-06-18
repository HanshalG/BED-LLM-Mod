"""End-to-end integration tests that drive BEDRunner through ``core.experiment``.

These prove the new architecture works for the full Config → registry →
Environment + Method → BEDRunner → RunResult slice without going through
``main.py`` (which still has W&B and run-management side effects).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import core.defaults as core_defaults
from core import ActionScore, BEDRunner, BeliefState, Environment, Method, RunResult, build_environment, build_method
from core.registry import register_environment, register_method
from environments.location_finding import LocationBEDEnvironment
from helpers import Config, load_config
from core.experiment import run_from_config


# ---------------------------------------------------------------------------
# Stub Model that the adapters / methods can call.
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """Model stub that returns canned completions / probabilities in order."""

    def __init__(
        self,
        completions: list[str] | None = None,
        batched_completions: list[list[str]] | None = None,
        probabilities: list[list[dict[str, float]]] | None = None,
    ) -> None:
        self._completions = list(completions or [])
        self._batched = list(batched_completions or [])
        self._probabilities = list(probabilities or [])
        self.complete_calls: list[Any] = []
        self.batched_calls: list[Any] = []
        self.probability_calls: list[Any] = []

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float = 0.0, num_responses: int = 1
    ) -> list[str]:
        self.complete_calls.append(messages)
        if not self._completions:
            raise AssertionError("scripted model ran out of completions")
        return [self._completions.pop(0)]

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float = 0.0,
        block_size: int = 50,
        max_new_tokens: int = 8192,
    ) -> list[str]:
        self.batched_calls.append(batch_messages)
        if not self._batched:
            raise AssertionError("scripted model ran out of batched completions")
        canned = self._batched.pop(0)
        if len(canned) != len(batch_messages):
            raise AssertionError(
                f"batched canned length {len(canned)} != batch size {len(batch_messages)}"
            )
        return list(canned)

    def chat_probabilities_messages_batched(
        self,
        messages: list[list[dict[str, str]]],
        responses: list[str],
        temperature: float = 0.0,
        block_size: int = 50,
    ) -> list[dict[str, float]]:
        self.probability_calls.append(messages)
        if not self._probabilities:
            raise AssertionError("scripted model ran out of probability rows")
        canned = self._probabilities.pop(0)
        if len(canned) != len(messages):
            raise AssertionError(
                f"probability canned length {len(canned)} != batch size {len(messages)}"
            )
        return list(canned)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_defaults():
    core_defaults.register_defaults(force=True)
    yield
    core_defaults.register_defaults(force=True)


def test_location_finding_runs_one_full_trial_through_bed_runner():
    """Drive LocationBEDEnvironment + Naive method end-to-end with a stub model."""
    config = Config(
        task="location_finding",
        method_names=["Naive"],
        location_num_rounds=2,
        location_num_trials=1,
        location_num_sources=2,
        location_dim=2,
        location_noise_sd=0.5,
        location_max_total_beliefs=50,
        location_max_llm_prompt_beliefs=10,
        location_num_generated_hypotheses=4,
        location_target_num_candidates=3,
        location_search_depth=1,
        location_eig_quadrature_order=5,
        location_seed=42,
        generation_temperature_diverse=0.0,
        belief_distribution_num_calls=1,
    )

    # The location_finding Naive method uses the direct naive query prompt, not
    # EIG candidate generation:
    # - Initial belief generation
    # - Per-round naive location
    # - Per-round belief refresh
    hypothesis_json = (
        '{"hypotheses":[[[0.0,0.0],[1.0,1.0]],[[1.0,-1.0],[-1.0,1.0]]]}'
    )
    location_a_json = '{"location":[0.1,0.1]}'
    location_b_json = '{"location":[0.5,-0.5]}'
    estimate_json = '{"sources":[[0.0,0.0],[1.0,1.0]]}'
    questioner = _ScriptedModel(
        completions=[
            hypothesis_json,    # initial belief generation
            location_a_json,    # round 1 naive location
            hypothesis_json,    # round 1 belief refresh
            estimate_json,      # round 1 naive source estimate
            location_b_json,    # round 2 naive location
            hypothesis_json,    # round 2 belief refresh
            estimate_json,      # round 2 naive source estimate
        ],
    )

    run_result, summary = run_from_config(
        config,
        questioner=questioner,
        answerer=None,
        method_name="Naive",
    )

    assert isinstance(run_result, RunResult)
    assert len(run_result.trials) == 1
    trial = run_result.trials[0]
    assert len(trial.rounds) == 2
    assert "source_rmse" in summary.metrics
    assert "top_probability" in summary.metrics
    assert len(summary.metrics["source_rmse"]) == 2


def test_location_finding_observe_uses_seeded_rng_for_reproducibility():
    """Same seed → same hidden state → identical observations across runs."""
    config = Config(
        task="location_finding",
        location_num_sources=2,
        location_dim=2,
        location_noise_sd=0.5,
    )
    env = LocationBEDEnvironment(config=config)

    rng_a = np.random.default_rng(7)
    theta_a = env.sample_hidden_state(rng_a)

    rng_b = np.random.default_rng(7)
    theta_b = env.sample_hidden_state(rng_b)

    np.testing.assert_array_equal(theta_a, theta_b)


def test_registry_dispatch_does_not_require_main_py():
    """The new architecture replaces main.py's if/else dispatch with a registry."""
    config = Config(task="animals", method_names=["EIG"], animals=[["dog"]])
    questioner = _ScriptedModel()
    answerer = _ScriptedModel()

    env = build_environment("animals", config, questioner, answerer)
    method = build_method("animals", "EIG", config)

    assert env.name == "animals"
    assert method.name == "EIG"


def test_unknown_task_in_config_raises_keyerror_through_run_from_config():
    config = Config(task="space_invaders")
    with pytest.raises(KeyError, match="space_invaders"):
        run_from_config(config, questioner=None, answerer=None, method_name="EIG")


def test_unknown_method_in_config_raises_keyerror_through_run_from_config():
    config = Config(task="animals", method_names=["WhirligigEIG"], animals=[["dog"]])
    with pytest.raises(KeyError, match="WhirligigEIG"):
        run_from_config(config, questioner=None, answerer=None)


class _ThirdEnvironment(Environment[str, str, str, str]):
    @property
    def name(self) -> str:
        return "third_env"

    def sample_hidden_state(self, rng):
        return "h"

    def observe(self, action, hidden_state, rng):
        return "o"

    def log_prior(self, hypothesis):
        return 0.0

    def log_likelihood(self, hypothesis, action, observation):
        return 0.0

    def initial_belief_state(self, model, config):
        return BeliefState.uniform(("h",))

    def update_belief_state(self, belief_state, history, model, config):
        return belief_state

    def generate_candidate_actions(self, belief_state, history, model, config):
        return ["a"]

    def round_metrics(self, belief_state, history, hidden_state):
        return {"ran": 1.0}

    def trial_count(self, config):
        return 1

    def round_count(self, config):
        return 1


class _ThirdMethod(Method[str, str, str, str]):
    @property
    def name(self):
        return "third"

    def select_action(self, candidates, belief_state, environment, model, history, config):
        return ActionScore(action=candidates[0], score=0.0)


def test_run_from_config_accepts_registered_third_environment_without_core_edits():
    register_environment("third_env", lambda config, questioner, answerer: _ThirdEnvironment())
    register_method("third_env", "third", lambda config, environment=None: _ThirdMethod())
    config = Config(task="third_env", method_names=["third"])

    run_result, summary = run_from_config(config, questioner=None, answerer=None)

    assert len(run_result.trials) == 1
    assert summary.metrics["ran"] == [1.0]


def test_yaml_loaded_third_environment_runs_without_loader_edits(tmp_path):
    register_environment("third_yaml_env", lambda config, questioner, answerer: _ThirdEnvironment())
    register_method("third_yaml_env", "third", lambda config, environment=None: _ThirdMethod())
    config_path = tmp_path / "third.yaml"
    config_path.write_text(
        """
task: third_yaml_env
environment:
  num_rounds: 1
model_pairs:
  - questioner:
      model: "Qwen/Qwen3.5-4B"
      thinking: false
    answerer:
      model: "Qwen/Qwen3.5-4B"
      thinking: false
method_names:
  - third
""".strip(),
        encoding="utf-8",
    )
    config = load_config(str(config_path))

    run_result, summary = run_from_config(config, questioner=None, answerer=None)

    assert config.task == "third_yaml_env"
    assert config.environment == {"num_rounds": 1}
    assert len(run_result.trials) == 1
    assert summary.metrics["ran"] == [1.0]


def test_runner_accepts_generic_trial_batch_size():
    class CountingEnvironment(_ThirdEnvironment):
        def trial_count(self, config):
            return 3

    register_environment("batched_third_env", lambda config, questioner, answerer: CountingEnvironment())
    register_method("batched_third_env", "third", lambda config, environment=None: _ThirdMethod())
    config = Config(
        task="batched_third_env",
        method_names=["third"],
        environment={"trial_batch_size": 2},
    )

    run_result, summary = run_from_config(config, questioner=None, answerer=None)

    assert [trial.trial_index for trial in run_result.trials] == [0, 1, 2]
    assert summary.metrics["ran"] == [1.0]
