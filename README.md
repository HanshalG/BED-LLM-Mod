# BED-LLM Sequential Experimental Design Framework

This repository contains a framework for non-myopic sequential Bayesian
experimental design (BED) with LLM-driven proposal and posterior components.
It currently includes two environments:

- `animals`: 20 Questions over animal identities with LLM answer likelihoods.
- `location_finding`: continuous source-localization with analytical or
  LLM-scored posteriors.

The shared runner, belief container, method registry, and common action
selection modes live under `core/` and `methods/`. Environment-specific
simulation, prompts, likelihoods, posterior refresh, metrics, and artifacts
live under `environments/`.

## Common Modes

The built-in environments expose the same primary method names:

- `naive`: direct LLM action proposal from task history.
- `naive+belief`: direct LLM action proposal conditioned on the current belief.
- `EIG`: expected information gain over candidate actions.
- `StrategyEIG`: generate and evaluate high-level LLM strategies before acting.
- `StrategyEIG+root`: generate strategies with fixed root actions and ask the
  selected root action directly.

Animals also keeps compatibility modes `Entropy` and `split` for reproduction
comparisons.

## Running Experiments

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate 20_questions_env
```

Optional model dependencies:

```bash
pip install accelerate
pip install flash-attn --no-build-isolation
```

Run a configured experiment:

```bash
python main.py -c config.yaml
```

Each invocation writes a self-contained run directory under `runs/`. Use
`--run-name` to choose the suffix and `--output-root` to place run directories
elsewhere.

## Adding An Environment

1. Implement `core.Environment[S, H, A, O]`.
2. Provide hidden-state sampling, observation simulation, likelihoods, belief
   initialization/update, candidate generation, and metrics.
   Environment belief hooks should consume and return `core.BeliefState[H]`
   directly; use `.hypotheses` and `.probabilities` for support access.
3. Put task-specific options under the top-level `environment:` config mapping
   and validate them in the environment adapter.
4. Override optional hooks when needed: `validate_config`, `trial_count`,
   `round_count`, `run_seed`, `configure_for_run`, `save_artifacts`, or
   strategy/naive hooks.
5. Register the environment and supported methods in `core.defaults`.

For binary LLM likelihoods, use `core.llm_likelihood` and expose
`observation_labels`, `build_likelihood_messages`, and `get_questioner`. For
continuous Gaussian observations, expose `predictive_means` and, optionally,
depth-2 scoring hooks.

## Public Surfaces

Use `run_from_config(...)` or `python main.py -c ...` for configured runs. New
environment code should import from `core/`, `methods/`, and
`environments/<task>/`; the root-level location-finding entry point has been
removed.

Task-specific YAML should use this shape:

```yaml
task: location_finding
method_names: ["EIG"]
model_pairs:
  - questioner: {model: "Qwen/Qwen3.5-4B", thinking: false}
    answerer: {model: "Qwen/Qwen3.5-4B", thinking: false}
environment:
  num_rounds: 20
  num_trials: 1
  trial_batch_size: 1
```

The runner owns trial batching for all environments via `trial_batch_size`;
environment-specific batched experiment loops should not bypass `BEDRunner`.

## Hardware Notes

The model adapter supports transformers and vLLM backends. Per-model vLLM
settings can be supplied under each `questioner` or `answerer` entry, including
`cuda_visible_devices`, `tensor_parallel_size`, `gpu_memory_utilization`, and
`max_model_len`.

## Paper Reproducibility

The workshop paper's headline Number Game and Rock Diagnosis values are bound
to committed public result artifacts in `paper/claim_manifest.json`. Validate
the artifact hashes and exact reported fields with:

```bash
python scripts/validate_paper_claim_manifest.py
python scripts/validate_paper_draft.py
```
