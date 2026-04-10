def main():
    import argparse
    from pathlib import Path

    import wandb

    import numpy as np

    from helpers import build_models, build_output_stem, load_config, resolve_run_id, write_to_log
    from model import build_model_adapter
    from questions_game import twenty_questions_animals

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
        }
    )

    models = build_models(config.model_pairs, build_model_adapter)
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
            output_stem = build_output_stem(config.run_id, method_name, pair.questioner, pair.answerer, config.version)
            config.log_path = logs_dir / f"{output_stem}.log"
            results_path = results_dir / f"{output_stem}.npy"
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


if __name__ == "__main__":
    main()
