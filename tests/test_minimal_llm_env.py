"""Minimal binary environment exercised through all global BED methods."""

from __future__ import annotations

from typing import Any

import pytest

from core import BEDRunner
from environments.minimal_binary import MinimalBinaryEnvironment
from methods import Naive, StrategyEIG
from methods.eig import build_eig_method


class _LookupModel:
  def __init__(self, yes_probabilities: dict[tuple[str, str], float]):
      self._table = yes_probabilities

  def chat_probabilities_messages_batched(self, messages, responses, temperature=0.0, block_size=50):
      out = []
      for convo in messages:
          text = convo[0]["content"]
          hypothesis, action = text.split("|", 1)
          p_yes = self._table.get((hypothesis, action), 0.5)
          out.append({"Yes": p_yes, "No": 1.0 - p_yes})
      return out


@pytest.fixture
def env_and_model():
    model = _LookupModel(
        {
            ("h0", "q1"): 1.0,
            ("h1", "q1"): 0.0,
            ("h0", "q2"): 0.5,
            ("h1", "q2"): 0.5,
        }
    )
    env = MinimalBinaryEnvironment(
        yes_probabilities=model._table,
        config=type("C", (), {"answer_temperature": 0.0, "batched_block_size": 8})(),
    )
    return env, model


def _run(env, method, model, rounds=1):
    return BEDRunner(
        environment=env,
        method=method,
        model=model,
        config=env.config,
        num_trials=1,
        num_rounds=rounds,
    ).run_single_trial(0)


def test_minimal_env_eig(env_and_model):
    env, model = env_and_model
    method = build_eig_method(env.config, environment=env)
    trial = _run(env, method, model)
    assert trial.rounds[0].chosen.action == "q1"


def test_minimal_env_naive(env_and_model):
    env, model = env_and_model
    trial = _run(env, Naive(method_name="naive"), model)
    assert trial.rounds[0].chosen.action == "naive-q"


def test_minimal_env_strategy(env_and_model):
    env, model = env_and_model
    trial = _run(env, StrategyEIG(), model)
    assert trial.rounds[0].chosen.action == "strategy-q"
