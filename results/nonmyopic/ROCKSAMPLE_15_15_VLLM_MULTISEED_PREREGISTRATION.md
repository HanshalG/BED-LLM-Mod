# RockSample[15,15] Direct-vLLM Multi-Seed Robustness

Registered 2026-07-22 after the seed-`24101` direct-vLLM endpoint and before any
policy response or endpoint under seeds `24102` or `24103`.

## Question

Does the positive 32,768-state StrategyEIG result persist under fresh policy,
truth, observation, and bootstrap seeds with the identical direct-vLLM model and
root-slot interface?

## Frozen Design

- Frozen POBAX RockSample[15,15] diagnosis geometry, exact joint belief, prior,
  sensor, action set, and start position from the completed seed-`24101` run.
- Direct vLLM `google/gemma-4-26B-A4B-it` on one A100 per run, bfloat16,
  non-thinking, temperature zero, 32,768-token context, and 2,048-token completion
  cap.
- Fresh seeds `24102` and `24103`. Each seed has 30 paired trials, 15 rounds, K4,
  horizon two, `branch_policy_v2`, trial concurrency one, and 10,000 paired
  bootstrap replicates.
- Arms are StrategyEIG, shared-roots d1, exhaustive d1 width with matched exact
  scorer units, matched random strategies, and exhaustive d2.
- Exact posterior updates, simulated outcomes, rollout scoring, and deployed
  observations make no LLM calls. The LLM proposes policy cells only.
- Jobs use `msc,llm`, request one A100, and exclude `oat12`. They may run in
  parallel because their seed streams and result directories are disjoint.

The completed seed-`24101` actual-prompt smoke covers this unchanged serving and
grammar interface; no new smoke or prompt adjustment is permitted.

## Frozen Gates

Each fresh seed is evaluated independently. For a seed to pass:

1. the 95% paired bootstrap lower bound for StrategyEIG entropy-AUC gain must be
   strictly positive against shared-roots d1, exhaustive d1 width, and matched
   random strategies;
2. the corresponding three truth-log-posterior-AUC lower bounds must be strictly
   positive; and
3. all legality, pairing, matched-compute, random-K, terminal-cell, and zero
   rollout-LLM mechanics checks must pass.

No seed may be replaced. The two-seed robustness claim passes only if both seeds
pass all six statistical gates. After those decisions are fixed, the three direct-
vLLM seeds `24101`--`24103` may be pooled descriptively using a paired bootstrap
over all 90 trials. Cross-seed pooling cannot rescue a failed per-seed gate.

## Frozen Commands

```bash
sbatch --job-name=r15-vllm-s24102 scripts/run_nonmyopic_rock_strategy_a100.sh \
  scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_15_15_vllm.yaml --maps 15-15 \
  --run-id nonmyopic-rocksample-15-15-vllm-seed-24102-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_vllm_seed_24102_20260722 \
  --num-trials-per-map 30 --num-rounds 15 --num-strategies 4 \
  --seed 24102 --bootstrap-replicates 10000 --trial-concurrency 1 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc

sbatch --job-name=r15-vllm-s24103 scripts/run_nonmyopic_rock_strategy_a100.sh \
  scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_15_15_vllm.yaml --maps 15-15 \
  --run-id nonmyopic-rocksample-15-15-vllm-seed-24103-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_vllm_seed_24103_20260722 \
  --num-trials-per-map 30 --num-rounds 15 --num-strategies 4 \
  --seed 24103 --bootstrap-replicates 10000 --trial-concurrency 1 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```
