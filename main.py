from helpers import write_to_log
def main():
    import argparse
    from pathlib import Path

    try:
        import wandb
    except ModuleNotFoundError:
        class _NoOpWandb:
            @staticmethod
            def init(*args, **kwargs):
                return None

            @staticmethod
            def log(*args, **kwargs):
                return None

        wandb = _NoOpWandb()

    import numpy as np

    from helpers import ModelSpec, build_models, build_output_stem, format_config_for_log, load_config, resolve_run_id, write_to_log
    from wordle_game import load_wordle_words, run_wordle

    import time

    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    print(f"[main] Loading config from {args.config}")
    config = load_config(args.config)
    config.run_id = resolve_run_id()
    if config.game == "wordle":
        target_count = len(load_wordle_words(config.wordle_solution_words_path or ""))
        print(f"[main] Loaded Wordle config with {len(config.method_names)} method(s) and {target_count} target word(s)")
    else:
        target_count = len(config.animals[config.version])
        print(f"[main] Loaded config with {len(config.model_pairs)} model pair(s), {len(config.method_names)} method(s), and {target_count} target animal(s)")
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
            "game": config.game,
            "guessing": config.animals[config.version] if config.game == "animals" else config.wordle_solution_words_path,
            "search_depth": config.search_depth,
        }
    )

    models = {}
    if config.game == "animals":
        from model import build_model_adapter

        models = build_models(config.model_pairs, lambda spec: build_model_adapter(spec, config=config))
        print(f"[main] Preparing {len(models)} unique model adapter(s)")
        print("[main] Model adapters ready")

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"[main] Logs directory ready at {logs_dir.resolve()}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print(f"[main] Results directory ready at {results_dir.resolve()}")

    if config.game == "wordle":
        wordle_spec = ModelSpec(model="deterministic-wordle")
        for method_name in config.method_names:
            output_stem = build_output_stem(
                config.run_id,
                method_name,
                wordle_spec,
                wordle_spec,
                config.version,
                belief_state_mode=config.belief_state_mode,
                search_depth=config.search_depth,
                game=config.game,
            )
            config.log_path = logs_dir / f"{output_stem}.log"
            results_path = results_dir / f"{output_stem}.npy"
            write_to_log(f"Config file: {Path(args.config).resolve()}\n", config)
            write_to_log(f"Config parameters:\n{format_config_for_log(config)}\n\n", config)
            write_to_log(f"Starting Wordle method {method_name}\n\n", config)
            print(f"Starting Wordle method {method_name}\n\n")
            accuracy = run_wordle(method_name, config)
            write_to_log(f"Accuracy trace: {accuracy}\n", config)
            print(f"[main] Saving Wordle accuracy trace for method {method_name} to {results_path}")
            np.save(
                results_path,
                np.array(accuracy),
            )
            print(f"Accuracy: {accuracy}\n\n")
            wandb.log({
                "accuracy": accuracy,
            })
    for pair in (config.model_pairs if config.game == "animals" else []):
        from questions_game import twenty_questions_animals

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
                game=config.game,
            )
            config.log_path = logs_dir / f"{output_stem}.log"
            results_path = results_dir / f"{output_stem}.npy"
            write_to_log(f"Config file: {Path(args.config).resolve()}\n", config)
            write_to_log(f"Config parameters:\n{format_config_for_log(config)}\n\n", config)
            write_to_log(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n", config)
            print(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n")
            accuracy = twenty_questions_animals(questioner_model, answerer_model, config.animals[config.version], method_name, config)
            write_to_log(f"Accuracy trace: {accuracy}\n", config)
            print(f"[main] Saving accuracy trace for method {method_name} to {results_path}")
            np.save(
                results_path,
                np.array(accuracy),
            )
            print(f"Accuracy: {accuracy}\n\n")
            wandb.log({
                "accuracy": accuracy,
            })
    end_time = time.perf_counter()
    print(f"[main] Total time: {end_time - start_time:.2f} seconds")
    write_to_log(f"Total time: {end_time - start_time:.2f} seconds\n", config)


if __name__ == "__main__":
    main()
