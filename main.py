def main():
    import argparse
    import wandb

    import numpy as np

    from helpers import load_config, write_to_log
    from model import HuggingFaceAdapter, VLLMAdapter
    from questions_game import twenty_questions_animals

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)

    wandb.init(
        project="BED-LLM-reproduction",
        config={
            "models": config.model_names,
            "methods": config.method_names,
            "guessing": config.animals[config.version],
        }
    )

    models = {}
    for questioner, answerer in config.model_names:
        if (questioner == "Llama" or answerer == "Llama") and "Llama" not in models.keys():
            models["Llama"] = VLLMAdapter(model_name="meta-llama/Llama-3.3-70B-Instruct")
        if (questioner == "Qwen" or answerer == "Qwen") and "Qwen" not in models.keys():
            models["Qwen"] = VLLMAdapter(model_name="Qwen/Qwen3.5-4B")

    for questioner, answerer in config.model_names:
        questioner_model = models[questioner]
        answerer_model = models[answerer]

        for method_name in config.method_names:
            write_to_log(f"Starting with models Q: {questioner}, A: {answerer}, method {method_name}\n\n", config.version)
            accuracy = twenty_questions_animals(questioner_model, answerer_model, config.animals[config.version], method_name, config)
            np.save(f"results/{method_name}_Q:{questioner},A:{answerer}_{str(config.version)}_animals.npy", np.array(accuracy))
            wandb.log({
                "accuracy": accuracy,
            })


if __name__ == "__main__":
    main()
