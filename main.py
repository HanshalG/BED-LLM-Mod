def main():
    import argparse
    from pathlib import Path

    import wandb

    import numpy as np

    from core.experiment import run_from_config
    from helpers import build_models, format_config_for_log, load_config, resolve_run_id, write_to_log
    from model import build_model_adapter

    import time

    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", help="Optional human-readable name for this run directory")
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
    config.run_id = resolve_run_id()
    if config.task == "location_finding":
        target_description = f"{config.location_num_trials} Location Finding trial(s)"
    else:
        target_description = f"{len(config.animals[config.version])} target animal(s)"
    print(
        f"[main] Loaded config with {len(config.model_pairs)} model pair(s), "
        f"{len(config.method_names)} method(s), and {target_description}"
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
                "guessing": config.animals[config.version] if config.task == "animals" else "location_finding",
                "task": config.task,
                "search_depth": config.search_depth,
                "location_search_depth": config.location_search_depth,
            }
        )

        if config.task == "location_finding":
            questioner_specs = []
            seen_questioner_specs = set()
            for pair in config.model_pairs:
                if pair.questioner not in seen_questioner_specs:
                    questioner_specs.append(pair.questioner)
                    seen_questioner_specs.add(pair.questioner)
            models = {
                spec: build_model_adapter(spec, config=config)
                for spec in questioner_specs
            }
        else:
            models = build_models(config.model_pairs, lambda spec: build_model_adapter(spec, config=config))
        print(f"[main] Preparing {len(models)} unique model adapter(s)")
        print("[main] Model adapters ready")

        item_index = 0
        for pair in config.model_pairs:
            questioner = pair.questioner.model
            answerer = pair.answerer.model
            questioner_model = models[pair.questioner]
            answerer_model = None if config.task == "location_finding" else models[pair.answerer]

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

                _run_result, summary = run_from_config(
                    config,
                    questioner_model,
                    answerer_model,
                    method_name=method_name,
                    output_dir=item.item_dir,
                )
                metrics = summary.metrics

                if config.task == "location_finding":
                    source_rmse = metrics.get("source_rmse", [])
                    top_probability = metrics.get("top_probability", [])
                    selected_eig = metrics.get("selected_eig", [])
                    source_rmse_path = item.item_dir / "source_rmse.npy"
                    top_probability_path = item.item_dir / "top_probability.npy"
                    selected_eig_path = item.item_dir / "selected_eig.npy"
                    write_to_log(f"Source RMSE trace: {source_rmse}\n", config)
                    write_to_log(f"Top probability trace: {top_probability}\n", config)
                    write_to_log(f"Selected EIG trace: {selected_eig}\n", config)
                    print(f"[main] Saving Location Finding source RMSE trace to {source_rmse_path}")
                    np.save(source_rmse_path, np.array(source_rmse))
                    add_item_artifact(item, "source_rmse", source_rmse_path, run_context)
                    print(f"[main] Saving Location Finding top probability trace to {top_probability_path}")
                    np.save(top_probability_path, np.array(top_probability))
                    add_item_artifact(item, "top_probability", top_probability_path, run_context)
                    print(f"[main] Saving Location Finding selected EIG trace to {selected_eig_path}")
                    np.save(selected_eig_path, np.array(selected_eig))
                    add_item_artifact(item, "selected_eig", selected_eig_path, run_context)
                    for plot_idx, plot_path in enumerate(sorted(item.item_dir.glob("location_trial_*.png")), start=1):
                        add_item_artifact(item, f"location_trial_plot_{plot_idx:03d}", plot_path, run_context)
                    set_item_metrics(
                        item,
                        {
                            "source_rmse": source_rmse,
                            "top_probability": top_probability,
                            "selected_eig": selected_eig,
                        },
                    )
                    run_context.write_metadata()
                    run_context.write_metrics()
                    print(f"Source RMSE: {source_rmse}\n\n")
                    wandb.log({
                        "source_rmse": source_rmse,
                        "top_probability": top_probability,
                        "selected_eig": selected_eig,
                    })
                    continue

                accuracy = metrics.get("accuracy", [])
                correct_belief_mass = metrics.get("correct_belief_mass", [0.0] * len(accuracy))
                accuracy_path = item.item_dir / "accuracy.npy"
                correct_belief_mass_path = item.item_dir / "correct_belief_mass.npy"
                write_to_log(f"Accuracy trace: {accuracy}\n", config)
                write_to_log(f"Correct belief mass trace: {correct_belief_mass}\n", config)
                print(f"[main] Saving accuracy trace for method {method_name} to {accuracy_path}")
                np.save(
                    accuracy_path,
                    np.array(accuracy),
                )
                add_item_artifact(item, "accuracy", accuracy_path, run_context)
                print(
                    f"[main] Saving correct belief mass trace for method {method_name} "
                    f"to {correct_belief_mass_path}"
                )
                np.save(
                    correct_belief_mass_path,
                    np.array(correct_belief_mass),
                )
                add_item_artifact(item, "correct_belief_mass", correct_belief_mass_path, run_context)
                set_item_metrics(
                    item,
                    {
                        "accuracy": accuracy,
                        "correct_belief_mass": correct_belief_mass,
                    },
                )
                run_context.write_metadata()
                run_context.write_metrics()
                print(f"Accuracy: {accuracy}\n\n")
                print(f"Correct belief mass: {correct_belief_mass}\n\n")
                wandb.log({
                    "accuracy": accuracy,
                    "correct_belief_mass_trace": correct_belief_mass,
                })
        end_time = time.perf_counter()
        print(f"[main] Total time: {end_time - start_time:.2f} seconds")
        write_to_log(f"Total time: {end_time - start_time:.2f} seconds\n", config)
        run_context.finish(status="completed")
    except Exception as exc:
        run_context.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
