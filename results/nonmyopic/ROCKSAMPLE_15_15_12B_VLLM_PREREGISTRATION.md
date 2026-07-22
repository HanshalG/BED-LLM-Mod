# RockSample[15,15] Gemma 4 12B Transfer Test

Registered 2026-07-22 after the frozen 12B proposal-quality smoke passed and before
any 12B policy trajectory or formal endpoint on this geometry.

## Purpose

Test whether the non-myopic 15-rock gain transfers to a dense middle-capacity,
non-thinking Gemma 4 12B proposer. The exact strategy verifier, environment,
controls, and compute accounting remain fixed. The prerequisite smoke achieved
mean best-proposed/exhaustive-d2 value `0.4217` and exceeded `0.20` in seven of eight
h2 probes, between the failed E4B and successful 26B proposer screens.

## Frozen Design

- Frozen POBAX `15-15` geometry, start `(0,7)`, exact 32,768-state belief,
  diagnosis-only actions, and half-efficiency distance `log(2)`.
- Hugging Face checkpoint `google/gemma-4-12B-it`, direct vLLM 0.23.0 in the official
  `vllm/vllm-openai:v0.23.0-cu129` container on one A100, bfloat16, non-thinking,
  temperature zero, maximum context 32,768, and maximum completion 2,048 tokens.
- Fresh seed `24106`; 30 paired trials; 15 rounds; K4; h2;
  `branch_policy_v2`; 10,000 paired bootstrap replicates; trial concurrency one.
- Arms: StrategyEIG, shared-roots d1, exhaustive d1 width with matched exact-scorer
  units, matched random strategies, and exhaustive d2. Paired hidden states and
  observation outcomes use the runner's common-random-number protocol.
- Primary endpoint: entropy AUC. Truth-log-posterior AUC is the preregistered
  truth-anchored corroborating endpoint. Final entropy and exact-d2 comparisons are
  secondary diagnostics.

The 12B transfer claim passes only if StrategyEIG has a strictly positive 95% paired
bootstrap lower bound against shared d1, matched width, and matched random on entropy
AUC, and all three corresponding truth-log-AUC lower bounds are also strictly
positive. All legality, pairing, compute-match, random-K, terminal, and
zero-rollout-LLM mechanics must pass. A failed seed is reported as a failed transfer;
it will not be replaced or pooled with 26B trials to rescue the decision.

The job uses `msc,llm`, excludes `oat12`, and requests one A100. It will be launched
from the committed preregistration revision. No OpenRouter calls are used.

## Frozen Command

```bash
sbatch --job-name=r15-12b-formal scripts/run_nonmyopic_rock_strategy_a100_singularity.sh \
  scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_15_15_12b_vllm.yaml --maps 15-15 \
  --run-id nonmyopic-rocksample-15-15-12b-vllm-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_12b_vllm_20260722 \
  --num-trials-per-map 30 --num-rounds 15 --num-strategies 4 \
  --seed 24106 --bootstrap-replicates 10000 --trial-concurrency 1 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```

## Execution Status

Job `106227` launched from commit `d954aa0` on `msc` node `oat14`. It stopped before
an endpoint after 373 model requests because the exact verifier enumerated an
observation with a tiny positive complementary scalar probability but exactly zero
joint likelihood, then correctly rejected the impossible posterior update. The
failed-closed artifact contains 372 accepted cells, zero invalid responses, and no
policy metrics.

Before any additional 12B response, the numerical repair is frozen: predictive
branch probability will use the same joint likelihood normalizer as the posterior,
so exactly impossible branches have zero expectation weight. A regression test
covers the cancellation case. The run will resume with `--resume-failure` from the
failed-closed artifact, revalidating and reusing all 372 accepted cells. Model,
seed `24106`, hidden trials, prompts, K, horizon, controls, endpoints, and gates are
unchanged. This is a reported same-seed software-failure resume, not a replacement
seed or a fresh endpoint attempt.

The crashing 373rd response consumed tokens but escaped before it could be recorded
as an accepted or rejected cell. Cumulative final serving usage will therefore be
the replayable physical-cell count plus one. The resume accounting merges the prior
and new vLLM process snapshots and preserves that discrepancy explicitly.

Resume job `106265` completed on `msc` node `oat14` from repair/accounting commit
`c6133c7`. The independent trace/bootstrap audit passed the primary entropy-AUC gate
and the separate truth-log corroboration gate. No seed, endpoint, candidate budget,
or control was replaced.
