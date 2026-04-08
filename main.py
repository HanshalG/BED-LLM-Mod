def main():
    import argparse
    from pathlib import Path

    import wandb

    import numpy as np

    from helpers import load_config, write_to_log
    from model import VLLMAdapter
    from questions_game import twenty_questions_animals

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    print(f"[main] Loading config from {args.config}")
    config = load_config(args.config)
    print(f"[main] Loaded config with {len(config.model_names)} model pair(s), {len(config.method_names)} method(s), and {len(config.animals[config.version])} target animal(s)")

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

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print(f"[main] Results directory ready at {results_dir.resolve()}")

    for questioner, answerer in config.model_names:
        questioner_model = models[questioner]
        answerer_model = models[answerer]

        for method_name in config.method_names:
            write_to_log(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n", config.version)
            print(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n")
            accuracy = twenty_questions_animals(questioner_model, answerer_model, config.animals[config.version], method_name, config)
            print(f"[main] Saving accuracy trace for method {method_name}")
            np.save(
                results_dir / f"{method_name}_Q:{questioner.replace('/', '_')},A:{answerer.replace('/', '_')}_{config.version}_animals.npy",
                np.array(accuracy),
            )
            print(f"Accuracy: {accuracy}\n\n")
            wandb.log({
                "accuracy": accuracy,
            })


if __name__ == "__main__":
    main()
