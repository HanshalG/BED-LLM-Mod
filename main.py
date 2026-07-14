def main():
    import argparse
    from pathlib import Path

    try:
        import wandb
    except ImportError:
        class _NoOpWandb:
            @staticmethod
            def init(*args, **kwargs):
                return None

            @staticmethod
            def log(*args, **kwargs):
                return None

            @staticmethod
            def login(*args, **kwargs):
                return None

        wandb = _NoOpWandb()

    import numpy as np

    from core.experiment import required_model_roles_for_config, run_from_config
    from helpers import build_models, format_config_for_log, load_config, resolve_run_id, write_to_log
    from model_factory import build_model_adapter

    import time

    def usage_snapshot(adapters):
        unique = {id(adapter): adapter for adapter in adapters if adapter is not None}
        snapshots = [
            adapter.usage_snapshot()
            for adapter in unique.values()
            if callable(getattr(adapter, "usage_snapshot", None))
        ]
        return {
            "cost_usd": sum(item.get("adapter_cost_usd", 0.0) for item in snapshots),
            "requests": sum(item.get("adapter_requests", 0) for item in snapshots),
            "prompt_tokens": sum(item.get("adapter_prompt_tokens", 0) for item in snapshots),
            "completion_tokens": sum(item.get("adapter_completion_tokens", 0) for item in snapshots),
            "reasoning_tokens": sum(item.get("adapter_reasoning_tokens", 0) for item in snapshots),
            "forced_exits": sum(item.get("forced_exits", 0) for item in snapshots),
        }

    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", help="Optional human-readable name for this run directory")
    parser.add_argument(
        "--paprika-task-offset",
        type=int,
        help="Override the first Paprika task index for an isolated paired shard",
    )
    parser.add_argument(
        "--mediq-task-offset",
        type=int,
        help="Override the first MediQ task index for an isolated paired shard",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Directory under which per-invocation run directories are created",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    print(f"[main] Loading config from {config_path}")
    config = load_config(str(config_path))
    if args.paprika_task_offset is not None:
        if config.task != "paprika_customer_service":
            parser.error("--paprika-task-offset is only valid for paprika_customer_service")
        if args.paprika_task_offset < 0:
            parser.error("--paprika-task-offset must be non-negative")
        config.paprika_task_offset = args.paprika_task_offset
        if isinstance(config.environment, dict):
            config.environment["task_offset"] = args.paprika_task_offset
    if args.mediq_task_offset is not None:
        if config.task != "mediq":
            parser.error("--mediq-task-offset is only valid for mediq")
        if args.mediq_task_offset < 0:
            parser.error("--mediq-task-offset must be non-negative")
        config.mediq_task_offset = args.mediq_task_offset
        if isinstance(config.environment, dict):
            config.environment["task_offset"] = args.mediq_task_offset
    config.run_id = resolve_run_id()
    print(
        f"[main] Loaded config with {len(config.model_pairs)} model pair(s), "
        f"{len(config.method_names)} method(s), task={config.task}"
    )
    print(f"[main] Using run ID {config.run_id}")

    from run_management import (
        add_item_artifact,
        create_run_context,
        item_base_metadata,
        set_item_metrics,
    )

    run_context = create_run_context(
        output_root=args.output_root,
        run_id=config.run_id,
        run_name=args.run_name,
        config_path=config_path,
        task=config.task,
        cwd=Path.cwd(),
    )
    config.log_path = run_context.log_path
    run_context.write_config_snapshot(config)
    print(f"[main] Run directory ready at {run_context.run_dir.resolve()}")

    try:
        print("[main] Initializing Weights & Biases run")
        wandb.init(
            project="BED-LLM-reproduction",
            name=f"{config.run_id}_{run_context.run_name}",
            config={
                "run_id": config.run_id,
                "run_name": run_context.run_name,
                "run_dir": str(run_context.run_dir.resolve()),
                "output_root": str(args.output_root.resolve()),
                "model_pairs": [
                    {
                        "questioner": pair.questioner,
                        "answerer": pair.answerer,
                    }
                    for pair in config.model_pairs
                ],
                "methods": config.method_names,
                "task": config.task,
                "environment": config.environment,
            }
        )

        required_roles = required_model_roles_for_config(config)
        models = build_models(
            config.model_pairs,
            lambda spec: build_model_adapter(spec, config=config),
            roles=required_roles,
        )
        print(f"[main] Preparing {len(models)} unique model adapter(s)")
        print("[main] Model adapters ready")

        item_index = 0
        for pair in config.model_pairs:
            questioner = pair.questioner.model
            answerer = pair.answerer.model
            questioner_model = models[pair.questioner] if "questioner" in required_roles else None
            answerer_model = models[pair.answerer] if "answerer" in required_roles else None

            for method_name in config.method_names:
                item = run_context.new_item(
                    item_index,
                    method_name,
                    item_base_metadata(config, method_name, pair),
                )
                item_index += 1
                write_to_log(f"Config file: {config_path}\n", config)
                write_to_log(f"Config parameters:\n{format_config_for_log(config)}\n\n", config)
                write_to_log(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n", config)
                print(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n")

                usage_before = usage_snapshot([questioner_model, answerer_model])

                _run_result, summary = run_from_config(
                    config,
                    questioner_model,
                    answerer_model,
                    method_name=method_name,
                    output_dir=item.item_dir,
                )
                metrics = summary.metrics

                usage_after = usage_snapshot([questioner_model, answerer_model])
                usage_delta = {
                    key: usage_after[key] - usage_before[key]
                    for key in usage_after
                }

                metric_payload = {}
                for metric_name, series in sorted(metrics.items()):
                    metric_path = item.item_dir / f"{metric_name}.npy"
                    write_to_log(f"{metric_name} trace: {series}\n", config)
                    print(f"[main] Saving {metric_name} trace for method {method_name} to {metric_path}")
                    np.save(metric_path, np.array(series))
                    add_item_artifact(item, metric_name, metric_path, run_context)
                    metric_payload[metric_name] = series
                if pair.questioner.backend == "openrouter" or pair.answerer.backend == "openrouter":
                    for key, value in usage_delta.items():
                        metric_payload[f"backend_{key}"] = [value]
                for artifact_name, artifact_path in sorted(summary.artifacts.items()):
                    add_item_artifact(item, artifact_name, artifact_path, run_context)
                set_item_metrics(item, metric_payload)
                run_context.write_metadata()
                run_context.write_metrics()
                print(f"Metrics: {metric_payload}\n\n")
                wandb.log(metric_payload)
        end_time = time.perf_counter()
        print(f"[main] Total time: {end_time - start_time:.2f} seconds")
        write_to_log(f"Total time: {end_time - start_time:.2f} seconds\n", config)
        run_context.finish(status="completed")
    except Exception as exc:
        run_context.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
