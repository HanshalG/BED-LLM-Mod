def main():
    import argparse
    from datetime import datetime
    from pathlib import Path

    import wandb

    import numpy as np

    from helpers import build_output_stem, load_config, write_to_log
    from model import VLLMAdapter
    from questions_game import twenty_questions_animals

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    print(f"[main] Loading config from {args.config}")
    config = load_config(args.config)
    config.run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"[main] Loaded config with {len(config.model_names)} model pair(s), {len(config.method_names)} method(s), and {len(config.animals[config.version])} target animal(s)")
    print(f"[main] Using run ID {config.run_id}")

    print("[main] Initializing Weights & Biases run")
    wandb.init(
        project="BED-LLM-reproduction",
        config={
            "models": config.model_names,
            "methods": config.method_names,
            "guessing": config.animals[config.version],
        }
    )

    model_names = sorted({model_name for pair in config.model_names for model_name in pair})
    print(f"[main] Preparing {len(model_names)} unique model adapter(s)")
    models = {
        model_name: VLLMAdapter(model_name=model_name)
        for model_name in model_names
    }
    print("[main] Model adapters ready")

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"[main] Logs directory ready at {logs_dir.resolve()}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print(f"[main] Results directory ready at {results_dir.resolve()}")

    for questioner, answerer in config.model_names:
        questioner_model = models[questioner]
        answerer_model = models[answerer]

        for method_name in config.method_names:
            output_stem = build_output_stem(config.run_id, method_name, questioner, answerer, config.version)
            config.log_path = logs_dir / f"{output_stem}.log"
            results_path = results_dir / f"{output_stem}.npy"
            write_to_log(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n", config)
            print(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n")
            accuracy = twenty_questions_animals(questioner_model, answerer_model, config.animals[config.version], method_name, config)
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
