"""Parity tests: legacy metric shapes vs BEDRunner + summarize_run."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import core.defaults as core_defaults
from core.experiment import run_from_config
from helpers import Config


class _ScriptedModel:
    def __init__(self, completions: list[str], probabilities: list[list[dict[str, float]]] | None = None):
        self._completions = list(completions)
        self._probabilities = list(probabilities or [])

    def chat_complete(self, messages, temperature=0.0, num_responses=1):
        return [self._completions.pop(0)]

    def chat_complete_messages_batched(self, batch_messages, temperature=0.0, block_size=50, max_new_tokens=8192):
        return [self._completions.pop(0) for _ in batch_messages]

    def chat_probabilities_messages_batched(self, messages, responses, temperature=0.0, block_size=50):
        row = self._probabilities.pop(0)
        return [row for _ in messages]


@pytest.fixture(autouse=True)
def _reset_registry():
    core_defaults.register_defaults(force=True)
    yield
    core_defaults.register_defaults(force=True)


def test_location_naive_bed_runner_matches_legacy_smoke(tmp_path):
    from environments.location_finding.runner import run_location_finding

    location_json = '{"location":[0.1,0.1]}'
    estimate_json = '{"sources":[[0.0,0.0],[1.0,1.0]]}'
    completions = [
        location_json,
        estimate_json,
    ]
    legacy_model = _ScriptedModel(list(completions))
    bed_model = _ScriptedModel(list(completions))
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

    legacy = run_location_finding(
        legacy_model, config, rng=np.random.default_rng(1), output_dir=tmp_path, method_name="Naive"
    )
    _run, summary = run_from_config(
        config, bed_model, method_name="Naive", output_dir=tmp_path
    )

    assert len(summary.metrics["source_rmse"]) == len(legacy.source_rmse)
    assert summary.metrics["source_rmse"][0] == pytest.approx(legacy.source_rmse[0], rel=0.5)


def test_animals_naive_metrics_include_accuracy_series():
    from core import BEDRunner
    from core.bed_runner import RunResult
    from environments.animals import AnimalsBEDEnvironment
    from methods import Naive

    config = Config(
        task="animals",
        animals=[["dog"]],
        animals_num_rounds=2,
        belief_generation_enabled=True,
        belief_filtering_enabled=False,
        belief_state_mode="uniform",
        generation_temperature_simple=0.0,
        answer_temperature=0.0,
        batched_block_size=8,
    )
    env = AnimalsBEDEnvironment(config=config, answerer=_ScriptedModel(["Yes"] * 25), target_animals=["dog"])
    model = _ScriptedModel(["dog", "cat", "Is it big?", "Is it small?"] * 25)
    trial = BEDRunner(
        environment=env,
        method=Naive(method_name="naive"),
        model=model,
        config=config,
        num_trials=1,
        num_rounds=2,
    ).run_single_trial(0)
    summary = env.summarize_run(RunResult(trials=(trial,)), config)
    assert "accuracy" in summary.metrics
    assert len(summary.metrics["accuracy"]) == 2
