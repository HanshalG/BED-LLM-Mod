"""Registry-based experiment driver for BED experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core import BEDRunner, RunResult, build_environment, build_method
from core.experiment_summary import ExperimentSummary
from core.defaults import register_defaults
from core.trial_batching import trial_batch_size


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

    if (
        env_name == "paprika_customer_service"
        and questioner is not None
        and answerer is not None
        and bool(getattr(questioner, "thinking", False))
        and not bool(getattr(answerer, "thinking", False))
    ):
        setattr(questioner, "_paprika_evaluation_model", answerer)

    env = build_environment(env_name, config, questioner, answerer)
    env.validate_config(config)
    env = env.configure_for_run(config)
    method = build_method(env_name, selected_method, config, environment=env)

    batch_size = trial_batch_size(config, env_name)
    runner = BEDRunner(
        environment=env,
        method=method,
        model=questioner,
        config=config,
        num_trials=env.trial_count(config),
        num_rounds=env.round_count(config),
        trial_batch_size=batch_size,
        rng=np.random.default_rng(env.run_seed(config)),
    )
    run_result = runner.run()
    summary = env.summarize_run(run_result, config)
    if env_name == "location_finding" and getattr(config, "location_eig_bounds_enabled", False):
        from environments.location_finding.eig_bounds import estimate_eig_bounds_from_run_result

        contrastive_seed = getattr(config, "location_eig_bounds_seed", None)
        if contrastive_seed is None:
            base_seed = getattr(config, "location_seed", None)
            contrastive_seed = None if base_seed is None else int(base_seed) + 1
        eig_bound_metrics = estimate_eig_bounds_from_run_result(
            run_result,
            config,
            rng=np.random.default_rng(contrastive_seed),
        ).as_metric_traces()
        summary = ExperimentSummary(
            metrics={**summary.metrics, **eig_bound_metrics},
            logs=summary.logs,
            artifacts=summary.artifacts,
        )
    if output_dir is not None:
        artifacts = env.save_artifacts(run_result, output_dir, config)
        summary = ExperimentSummary(
            metrics=summary.metrics,
            logs=summary.logs,
            artifacts=artifacts,
        )
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
    required_roles = required_model_roles_for_config(config)

    for pair in config.model_pairs:
        questioner_model = models[pair.questioner] if "questioner" in required_roles else None
        answerer_model = models[pair.answerer] if "answerer" in required_roles else None
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


def required_model_roles_for_config(config: Any) -> tuple[str, ...]:
    """Return model-pair roles required by the configured environment."""
    register_defaults()
    env = build_environment(getattr(config, "task"), config, None, None)
    roles = getattr(env, "required_model_roles", lambda _config: ("questioner", "answerer"))(config)
    return tuple(dict.fromkeys(roles))


def _first_method_name(config: Any) -> str:
    methods = getattr(config, "method_names", None) or []
    if not methods:
        raise ValueError(
            "config.method_names is empty; pass method_name= explicitly or set it on the config"
        )
    return methods[0]
