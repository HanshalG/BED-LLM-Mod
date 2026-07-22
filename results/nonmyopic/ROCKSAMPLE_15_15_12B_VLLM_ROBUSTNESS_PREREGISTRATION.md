# RockSample[15,15] Gemma 4 12B Multi-Seed Robustness

Registered 2026-07-22 after seed `24106` passed its frozen transfer test and before
any response for seeds `24107` or `24108`.

## Purpose

Test whether the non-thinking Gemma 4 12B result is robust across fresh hidden-state
and observation seeds, matching the existing three-seed standard for the 26B model.
This is a replication of the fixed 15-rock protocol, not a tuning sweep.

## Frozen Design

- Frozen POBAX `15-15` geometry, exact 32,768-state belief, diagnosis-only actions,
  and half-efficiency distance `log(2)`.
- `google/gemma-4-12B-it` under direct vLLM 0.23.0 in the official
  `vllm/vllm-openai:v0.23.0-cu129` container, one A100 per run, bfloat16,
  non-thinking, temperature zero, 32,768-token context, and 2,048-token completion.
- Fresh seeds `24107` and `24108`; each run has 30 paired trials, 15 rounds, K4,
  h2, `branch_policy_v2`, 10,000 paired bootstrap replicates, and trial concurrency
  one.
- Arms and common-random-number pairing are unchanged: StrategyEIG, shared-roots d1,
  matched exhaustive d1 width, matched random strategies, and exhaustive d2.
- Primary endpoint is entropy AUC. Truth-log-posterior AUC is the independently
  required corroborating endpoint.

Each fresh seed must independently have a strictly positive 95% paired-bootstrap
lower bound against shared d1, width, and random for entropy AUC and for truth-log
AUC. All legality, pairing, compute-match, random-K, terminal, and zero-rollout-LLM
mechanics must pass. The robustness claim fails if either seed fails any interval;
seeds are not replaced. An equal-seed-weight stratified 90-pair bootstrap with base
seed `24109` is secondary and cannot rescue an individual failure.

Jobs use `msc,llm`, exclude `oat12`, and request one A100 each. No OpenRouter calls
or reasoning-mode policy calls are used. A software failure may be repaired and
resumed under the same seed only if reported before further responses, as in the
seed-24106 verifier incident.

## Frozen Commands

```bash
sbatch --nodelist=oat14 --job-name=r15-12b-s24107 \
  scripts/run_nonmyopic_rock_strategy_a100_singularity.sh \
  scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_15_15_12b_vllm.yaml --maps 15-15 \
  --run-id nonmyopic-rocksample-15-15-12b-vllm-seed-24107-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_12b_vllm_seed_24107_20260722 \
  --num-trials-per-map 30 --num-rounds 15 --num-strategies 4 --seed 24107 \
  --bootstrap-replicates 10000 --trial-concurrency 1 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc

sbatch --nodelist=oat14 --job-name=r15-12b-s24108 \
  scripts/run_nonmyopic_rock_strategy_a100_singularity.sh \
  scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_15_15_12b_vllm.yaml --maps 15-15 \
  --run-id nonmyopic-rocksample-15-15-12b-vllm-seed-24108-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_12b_vllm_seed_24108_20260722 \
  --num-trials-per-map 30 --num-rounds 15 --num-strategies 4 --seed 24108 \
  --bootstrap-replicates 10000 --trial-concurrency 1 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```

## Execution Status

Completed from preregistration commit `b55c4e6` on 2026-07-22. Jobs `106288`
(seed `24107`) and `106289` (seed `24108`) ran concurrently on `oat14` in the
`msc` partition with `oat12` excluded. Both completed without a resume, rejected
response, terminal failure, reasoning token, forced exit, or rollout-scoring LLM
call.

The independent per-seed audits passed all six required intervals for each fresh
seed. The locked equal-seed-weight aggregate also passed: across 90 paired trials,
entropy-AUC gains were `+0.6417 [0.6171, 0.6672]` against shared d1, `+0.6075
[0.5813, 0.6340]` against width, and `+0.6194 [0.5930, 0.6451]` against random,
with `90/0/0` wins/ties/losses each. The corresponding truth-log-AUC intervals
were all strictly positive. See
`ROCKSAMPLE_15_15_12B_VLLM_ROBUSTNESS_RESULT.md` and the stored independent
audits for the complete decision record.
