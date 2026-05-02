from helpers import write_to_log
def main():
    import argparse
    from pathlib import Path

    import wandb

    import numpy as np

    from helpers import build_models, build_output_stem, format_config_for_log, load_config, resolve_run_id, write_to_log
    from model import build_model_adapter
    from questions_game import twenty_questions_animals

    import time

    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    print(f"[main] Loading config from {args.config}")
    config = load_config(args.config)
    config.run_id = resolve_run_id()
    print(f"[main] Loaded config with {len(config.model_pairs)} model pair(s), {len(config.method_names)} method(s), and {len(config.animals[config.version])} target animal(s)")
    print(f"[main] Using run ID {config.run_id}")

    print("[main] Initializing Weights & Biases run")
    wandb.init(
        project="BED-LLM-reproduction",
        config={
            "model_pairs": [
                {
                    "questioner": pair.questioner,
                    "answerer": pair.answerer,
                }
                for pair in config.model_pairs
            ],
            "methods": config.method_names,
            "guessing": config.animals[config.version],
            "search_depth": config.search_depth,
        }
    )

    models = build_models(config.model_pairs, lambda spec: build_model_adapter(spec, config=config))
    print(f"[main] Preparing {len(models)} unique model adapter(s)")
    print("[main] Model adapters ready")

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"[main] Logs directory ready at {logs_dir.resolve()}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print(f"[main] Results directory ready at {results_dir.resolve()}")

    for pair in config.model_pairs:
        questioner = pair.questioner.model
        answerer = pair.answerer.model
        questioner_model = models[pair.questioner]
        answerer_model = models[pair.answerer]

        for method_name in config.method_names:
            output_stem = build_output_stem(
                config.run_id,
                method_name,
                pair.questioner,
                pair.answerer,
                config.version,
                belief_state_mode=config.belief_state_mode,
                search_depth=config.search_depth,
            )
            config.log_path = logs_dir / f"{output_stem}.log"
            results_path = results_dir / f"{output_stem}.npy"
            correct_belief_mass_path = results_dir / f"{output_stem}_correct_belief_mass.npy"
            write_to_log(f"Config file: {Path(args.config).resolve()}\n", config)
            write_to_log(f"Config parameters:\n{format_config_for_log(config)}\n\n", config)
            write_to_log(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n", config)
            print(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n")
            game_metrics = twenty_questions_animals(
                questioner_model,
                answerer_model,
                config.animals[config.version],
                method_name,
                config,
            )
            if hasattr(game_metrics, "correct_guess") and hasattr(game_metrics, "correct_belief_mass"):
                accuracy = game_metrics.correct_guess
                correct_belief_mass = game_metrics.correct_belief_mass
            else:
                accuracy = list(game_metrics)
                correct_belief_mass = [0.0] * len(accuracy)
            write_to_log(f"Accuracy trace: {accuracy}\n", config)
            write_to_log(f"Correct belief mass trace: {correct_belief_mass}\n", config)
            print(f"[main] Saving accuracy trace for method {method_name} to {results_path}")
            np.save(
                results_path,
                np.array(accuracy),
            )
            print(
                f"[main] Saving correct belief mass trace for method {method_name} "
                f"to {correct_belief_mass_path}"
            )
            np.save(
                correct_belief_mass_path,
                np.array(correct_belief_mass),
            )
            print(f"Accuracy: {accuracy}\n\n")
            print(f"Correct belief mass: {correct_belief_mass}\n\n")
            wandb.log({
                "accuracy": accuracy,
                "correct_belief_mass_trace": correct_belief_mass,
            })
    end_time = time.perf_counter()
    print(f"[main] Total time: {end_time - start_time:.2f} seconds")
    write_to_log(f"Total time: {end_time - start_time:.2f} seconds\n", config)


if __name__ == "__main__":
    main()
