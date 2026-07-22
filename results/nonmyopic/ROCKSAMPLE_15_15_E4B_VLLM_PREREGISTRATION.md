# RockSample[15,15] Gemma 4 E4B Transfer Test

Registered 2026-07-22 after the mechanics-only E4B serving gate passed and before
any E4B policy trajectory or formal endpoint on this geometry.

## Purpose

Test whether the confirmed non-myopic 15-rock result transfers from Gemma 4 26B A4B
to the substantially smaller Gemma 4 E4B proposer while the exact strategy verifier,
environment, controls, and compute accounting remain fixed. The smoke's weak
descriptive exact-score fraction is recorded in its own preregistration and does not
change this protocol.

## Frozen Design

- Frozen POBAX `15-15` geometry, start `(0,7)`, exact 32,768-state belief,
  diagnosis-only actions, and half-efficiency distance `log(2)`.
- Hugging Face checkpoint `google/gemma-4-E4B-it`, direct vLLM on one A100,
  bfloat16, non-thinking, temperature zero, maximum context 32,768, and maximum
  completion 2,048 tokens.
- Fresh seed `24105`; 30 paired trials; 15 rounds; K4; h2;
  `branch_policy_v2`; 10,000 paired bootstrap replicates; trial concurrency one.
- Arms: StrategyEIG, shared-roots d1, exhaustive d1 width with matched exact-scorer
  units, matched random strategies, and exhaustive d2. Paired hidden states and
  observation outcomes use the runner's common-random-number protocol.
- Primary endpoint: entropy AUC. Truth-log-posterior AUC is the preregistered
  truth-anchored corroborating endpoint. Final entropy and exact-d2 comparisons are
  secondary diagnostics.

The smaller-model transfer claim passes only if StrategyEIG has a strictly positive
95% paired bootstrap lower bound against shared d1, matched width, and matched random
on entropy AUC, and all three corresponding truth-log-AUC lower bounds are also
strictly positive. All legality, pairing, compute-match, random-K, terminal, and
zero-rollout-LLM mechanics must pass. A failed seed is reported as a failed transfer;
it will not be replaced or pooled with the 26B trials to rescue the decision.

The job uses `msc,llm`, excludes `oat12`, and requests one A100. It will be launched
from the committed preregistration revision.

## Frozen Command

```bash
sbatch --job-name=r15-e4b-formal scripts/run_nonmyopic_rock_strategy_a100.sh \
  scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_15_15_e4b_vllm.yaml --maps 15-15 \
  --run-id nonmyopic-rocksample-15-15-e4b-vllm-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_e4b_vllm_20260722 \
  --num-trials-per-map 30 --num-rounds 15 --num-strategies 4 \
  --seed 24105 --bootstrap-replicates 10000 --trial-concurrency 1 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```

## Execution Status

Pending.
