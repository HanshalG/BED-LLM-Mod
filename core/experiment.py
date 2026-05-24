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
    env.validate_config(config)
    env = env.configure_for_run(config)
    method = build_method(env_name, selected_method, config, environment=env)

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
            rng=np.random.default_rng(env.run_seed(config)),
        )
        return RunResult(trials=()), summary

    runner = BEDRunner(
        environment=env,
        method=method,
        model=questioner,
        config=config,
        num_trials=env.trial_count(config),
        num_rounds=env.round_count(config),
        rng=np.random.default_rng(env.run_seed(config)),
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
        answerer_model = None if config.task == "location_finding" else models[pair.answerer]
        for method_name in config.method_names:
            env = build_environment(config.task, config, questioner_model, answerer_model)
            env.validate_config(config)
            env = env.configure_for_run(config)
            build_method(config.task, method_name, config, environment=env)
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


def _first_method_name(config: Any) -> str:
    methods = getattr(config, "method_names", None) or []
    if not methods:
        raise ValueError(
            "config.method_names is empty; pass method_name= explicitly or set it on the config"
        )
    return methods[0]
