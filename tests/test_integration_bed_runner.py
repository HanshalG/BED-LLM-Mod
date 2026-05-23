"""End-to-end integration tests that drive BEDRunner through ``main_via_runner``.

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
from core import BEDRunner, RunResult, build_environment, build_method
from environments.location_finding import LocationBEDEnvironment
from helpers import Config
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
        location_query_bounds=[-2.0, 2.0],
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

    # Plain Naive: query from observation history + end-of-round source estimate only.
    location_a_json = '{"location":[0.1,0.1]}'
    location_b_json = '{"location":[0.5,-0.5]}'
    estimate_json = '{"sources":[[0.0,0.0],[1.0,1.0]]}'
    questioner = _ScriptedModel(
        completions=[
            location_a_json,    # round 1 naive location
            estimate_json,      # round 1 naive source estimate
            location_b_json,    # round 2 naive location
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


def test_location_naive_skips_belief_generation(monkeypatch):
    """Plain Naive must not call hypothesis generation or posterior scoring."""
    from environments.location_finding import runner as lf_runner

    hypothesis_json = (
        '{"hypotheses":[[[0.0,0.0],[1.0,1.0]],[[1.0,-1.0],[-1.0,1.0]]]}'
    )
    calls: list[str] = []

    original = lf_runner.generate_location_hypotheses

    def _track(*args, **kwargs):
        calls.append(str(kwargs.get("label", "")))
        return original(*args, **kwargs)

    monkeypatch.setattr(lf_runner, "generate_location_hypotheses", _track)

    config = Config(
        task="location_finding",
        method_names=["Naive"],
        location_num_rounds=1,
        location_num_trials=1,
        location_num_sources=2,
        location_dim=2,
        location_noise_sd=0.5,
        location_query_bounds=[-2.0, 2.0],
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
    questioner = _ScriptedModel(
        completions=[
            '{"location":[0.1,0.1]}',
            '{"sources":[[0.0,0.0],[1.0,1.0]]}',
        ],
    )
    run_from_config(config, questioner=questioner, answerer=None, method_name="Naive")
    assert calls == []

    calls.clear()
    questioner_belief = _ScriptedModel(
        completions=[
            hypothesis_json,
            '{"location":[0.1,0.1]}',
            hypothesis_json,
            '{"sources":[[0.0,0.0],[1.0,1.0]]}',
        ],
    )
    run_from_config(
        config,
        questioner=questioner_belief,
        answerer=None,
        method_name="naive+belief",
    )
    assert "initial belief generation" in calls
    assert "belief refresh" in calls


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


def test_hyperbolic_discounting_runs_one_trial_through_run_from_config():
    config = Config(
        task="hyperbolic_discounting",
        method_names=["EIG"],
        htd_num_rounds=1,
        htd_num_trials=1,
        htd_noise_sd=0.25,
        htd_target_num_candidates=2,
        htd_search_depth=1,
        htd_posterior_mode="analytical_likelihood",
        htd_max_total_beliefs=20,
        htd_max_llm_prompt_beliefs=6,
        htd_num_generated_hypotheses=4,
        generation_temperature_diverse=0.0,
    )
    hypothesis_json = (
        '{"hypotheses":[{"k":0.5,"alpha":1.0},{"k":1.5,"alpha":0.8}]}'
    )
    designs_json = (
        '{"designs":[{"iR":10,"dR":20,"days":7},{"iR":5,"dR":30,"days":14}]}'
    )
    questioner = _ScriptedModel(
        completions=[
            hypothesis_json,
            designs_json,
            hypothesis_json,
        ],
    )
    run_result, summary = run_from_config(
        config,
        questioner=questioner,
        answerer=None,
        method_name="EIG",
    )
    assert len(run_result.trials) == 1
    assert len(run_result.trials[0].rounds) == 1
    assert "parameter_rmse" in summary.metrics


def test_hyperbolic_discounting_batched_run_from_config():
    config = Config(
        task="hyperbolic_discounting",
        method_names=["EIG"],
        htd_num_rounds=1,
        htd_num_trials=2,
        htd_trial_batch_size=2,
        htd_target_num_candidates=2,
        htd_search_depth=1,
        htd_posterior_mode="analytical_likelihood",
        htd_max_total_beliefs=20,
        htd_max_llm_prompt_beliefs=6,
        htd_num_generated_hypotheses=4,
        generation_temperature_diverse=0.0,
    )
    hypothesis_json = '{"hypotheses":[{"k":0.5,"alpha":1.0},{"k":1.5,"alpha":0.8}]}'
    designs_json = '{"designs":[{"iR":10,"dR":20,"days":7},{"iR":5,"dR":30,"days":14}]}'
    questioner = _ScriptedModel(
        completions=[],
        batched_completions=[
            [hypothesis_json, hypothesis_json],
            [designs_json, designs_json],
            [hypothesis_json, hypothesis_json],
        ],
    )
    run_result, summary = run_from_config(
        config,
        questioner=questioner,
        answerer=None,
        method_name="EIG",
    )
    assert len(run_result.trials) == 0
    assert "parameter_rmse" in summary.metrics
    assert "implied_choice_accuracy" in summary.metrics


def test_unknown_task_in_config_raises_keyerror_through_run_from_config():
    config = Config(task="space_invaders")
    with pytest.raises(KeyError, match="space_invaders"):
        run_from_config(config, questioner=None, answerer=None, method_name="EIG")


def test_unknown_method_in_config_raises_keyerror_through_run_from_config():
    config = Config(task="animals", method_names=["WhirligigEIG"], animals=[["dog"]])
    with pytest.raises(KeyError, match="WhirligigEIG"):
        run_from_config(config, questioner=None, answerer=None)
