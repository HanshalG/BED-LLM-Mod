"""Registry-based experiment driver for BED experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core import BEDRunner, RunResult, build_environment, build_method
from core.experiment_summary import ExperimentSummary
from core.defaults import register_defaults
from core.trial_batching import supports_trial_batching, trial_batch_size


def run_from_config(
    config: Any,
    questioner: Any,
    answerer: Any = None,
    *,
    method_name: str | None = None,
    output_dir: Path | None = None,
) -> tuple[RunResult, ExperimentSummary]:
    """Construct environment/method/runner from ``config`` and run."""
    register_defaults()

    env_name = getattr(config, "task")
    selected_method = method_name or _first_method_name(config)

    env = build_environment(env_name, config, questioner, answerer)
    method = build_method(env_name, selected_method, config)

    if env_name == "location_finding":
        if config.location_dim != 2:
            raise ValueError("Location Finding currently supports 2D source locations")
        if config.location_noise_sd != 0.5:
            raise ValueError(
                "The initial Location Finding implementation requires known noise_sd=0.5"
            )
    if env_name == "hyperbolic_discounting":
        if config.htd_noise_sd <= 0.0:
            raise ValueError("htd_noise_sd must be positive")

    batch_size = trial_batch_size(config, env_name)
    if batch_size > 1:
        if not supports_trial_batching(env, batch_size):
            raise ValueError(
                f"task {env_name!r} does not support trial_batch_size={batch_size}"
            )
        summary = env.run_batched_experiment(
            questioner,
            method,
            config,
            output_dir=output_dir,
            rng=np.random.default_rng(_seed(config, env_name)),
        )
        return RunResult(trials=()), summary

    num_trials, num_rounds = _trial_round_counts(config, env_name)
    seed = _seed(config, env_name)

    if env_name == "animals":
        env = _configure_animals_trials(env, config)

    runner = BEDRunner(
        environment=env,
        method=method,
        model=questioner,
        config=config,
        num_trials=num_trials,
        num_rounds=num_rounds,
        rng=np.random.default_rng(seed),
    )
    run_result = runner.run()
    summary = env.summarize_run(run_result, config)
    if output_dir is not None:
        env.save_artifacts(run_result, output_dir, config)
    return run_result, summary


def run_configured_experiments(
    config: Any,
    models: dict[Any, Any],
    *,
    output_dir: Path | None = None,
) -> dict[tuple[str, str, str], ExperimentSummary]:
    """Run every configured model-pair/method combination."""
    register_defaults()
    results: dict[tuple[str, str, str], ExperimentSummary] = {}

    for pair in config.model_pairs:
        questioner_model = models[pair.questioner]
        answerer_model = (
            None
            if config.task in {"location_finding", "hyperbolic_discounting"}
            else models[pair.answerer]
        )
        for method_name in config.method_names:
            build_method(config.task, method_name, config)
            key = (pair.questioner.model, pair.answerer.model, method_name)
            _run_result, summary = run_from_config(
                config,
                questioner_model,
                answerer_model,
                method_name=method_name,
                output_dir=output_dir,
            )
            results[key] = summary
    return results


def _configure_animals_trials(env: Any, config: Any) -> Any:
    """Set target animal pool on the environment (including prior-sampling mode)."""
    target_animals = list(config.animals[config.version])
    if config.answerer_sample_from_prior:
        from helpers import get_answerer_prior

        base_prior = get_answerer_prior(config)
        if base_prior is None or len(base_prior.beliefs) == 0:
            raise ValueError("answerer_sample_from_prior=true requires a non-empty configured prior")
        num_trials = config.answerer_num_prior_trials
        if num_trials is None:
            num_trials = len(base_prior.beliefs)
        rng = np.random.default_rng(config.answerer_prior_seed)
        target_animals = []
        for _trial_idx in range(num_trials):
            if config.answerer_randomize_prior_order_per_trial:
                prior_order = [
                    base_prior.beliefs[int(index)]
                    for index in rng.permutation(len(base_prior.beliefs))
                ]
            else:
                prior_order = list(base_prior.beliefs)
            config.active_answerer_prior_animals = prior_order
            try:
                trial_prior = get_answerer_prior(config)
            finally:
                config.active_answerer_prior_animals = None
            if trial_prior is None or len(trial_prior.beliefs) == 0:
                raise ValueError("answerer_sample_from_prior=true requires a non-empty configured prior")
            sampled_index = int(rng.choice(len(trial_prior.beliefs), p=trial_prior.probabilities))
            target_animals.append(trial_prior.beliefs[sampled_index])
    env.target_animals = target_animals
    object.__setattr__(env, "_animal_pool", list(target_animals))
    return env


def _first_method_name(config: Any) -> str:
    methods = getattr(config, "method_names", None) or []
    if not methods:
        raise ValueError(
            "config.method_names is empty; pass method_name= explicitly or set it on the config"
        )
    return methods[0]


def _trial_round_counts(config: Any, env_name: str) -> tuple[int, int]:
    if env_name == "location_finding":
        return int(config.location_num_trials), int(config.location_num_rounds)
    if env_name == "hyperbolic_discounting":
        return int(config.htd_num_trials), int(config.htd_num_rounds)
    if env_name == "animals":
        animals_table = list(getattr(config, "animals", []) or [])
        version = int(getattr(config, "version", 0))
        pool = animals_table[version] if 0 <= version < len(animals_table) else []
        trials = max(1, len(pool))
        num_rounds = int(getattr(config, "animals_num_rounds", 20))
        return trials, num_rounds
    raise ValueError(f"unknown task {env_name!r}")


def _seed(config: Any, env_name: str) -> int | None:
    if env_name == "location_finding":
        return getattr(config, "location_seed", None)
    if env_name == "hyperbolic_discounting":
        return getattr(config, "htd_seed", None)
    return getattr(config, "seed", None)
