# RockSample[15,15] Direct-vLLM Replication

Registered 2026-07-22 after the OpenRouter confirmation was blocked without an
endpoint and before any direct-vLLM response or policy endpoint on this geometry.

## Purpose

Test whether the 15-rock root-slot result transfers across serving backends using
the same Gemma 4 26B A4B checkpoint. This is a fresh replication, not a replacement
or continuation of the pending OpenRouter seed-24100 confirmation.

## Frozen Design

- Frozen POBAX `15-15` geometry, start `(0,7)`, exact 32,768-state belief,
  diagnosis-only actions, and half-efficiency distance `log(2)`.
- Hugging Face checkpoint `google/gemma-4-26B-A4B-it`, direct vLLM on one A100,
  bfloat16, non-thinking, temperature zero, maximum context 32,768, and maximum
  completion 2,048 tokens.
- Fresh seed `24101`; 30 paired trials; 15 rounds; K4; h2;
  `branch_policy_v2`; 10,000 paired bootstrap replicates.
- Trial concurrency one because the in-process vLLM engine owns one GPU. This is
  an execution setting; every statistical arm remains paired within trial.
- Arms and endpoints are identical to the OpenRouter preregistration: StrategyEIG,
  shared-roots d1, exhaustive d1 width with matched exact-scorer units, matched
  random strategies, and exhaustive d2; entropy AUC is primary.
- StrategyEIG must have a strictly positive 95% paired bootstrap lower bound
  against shared d1, width, and random. All three truth-log-AUC lower bounds must
  also be positive for corroboration. All legality, pairing, compute-match,
  random-K, terminal, and zero-rollout-LLM mechanics must pass.

A fresh ten-cell actual-prompt cluster smoke must pass every cell within one repair
before the formal job is submitted. Model download, load, and smoke are interface
checks and cannot alter the checkpoint, prompt, seed, K, horizon, rounds, controls,
or endpoints. The jobs use `msc,llm`, exclude `oat12`, and request one A100.

## Frozen Commands

```bash
sbatch --job-name=r15-vllm-smoke scripts/run_nonmyopic_rock_strategy_a100.sh \
  scripts/nonmyopic_rock_branch_strategy_smoke.py \
  --config configs/config_nonmyopic_rocksample_15_15_vllm.yaml \
  --maps 15-15 --probe-states-per-map 10 --num-strategies 4 --concurrency 1 \
  --run-id nonmyopic-rocksample-15-15-vllm-smoke-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_vllm_smoke_20260722

sbatch --job-name=r15-vllm-formal scripts/run_nonmyopic_rock_strategy_a100.sh \
  scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_15_15_vllm.yaml --maps 15-15 \
  --run-id nonmyopic-rocksample-15-15-vllm-replication-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_vllm_replication_20260722 \
  --num-trials-per-map 30 --num-rounds 15 --num-strategies 4 \
  --seed 24101 --bootstrap-replicates 10000 --trial-concurrency 1 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```

## Execution Status

The first smoke allocation, job `106098`, failed before model load or any LLM
response because the fresh checkout's `.env` symlink pointed to a retired workspace.
The symlink was corrected to the existing cluster secret file without reading or
changing its contents. The identical job `106101` then reached A100 node `oat10`
but failed before model initialization because the existing user conda environment
lacked the repo-pinned `pomdp-py==1.3.5.1`. That exact package and its SciPy
dependency were installed into the user environment, and an import preflight
constructed the 32,768-state model successfully under vLLM `0.19.1rc1.dev367`.
Job `106106` revealed that `/scratch-ssd` is node-local: the login-node install did
not modify `oat10`'s environment, so it failed at the same pre-import stage. The
launcher now checks `pomdp_py` after activating the allocation-local environment and,
only when absent, installs the pinned wheel under the cluster's existing package
lock before asserting the frozen map import. No model or policy endpoint has yet
been observed.
