# RockSample[7,8] GPT-5.4 Mini Replication Preregistration

Registered 2026-07-21 before any GPT-5.4 Mini strategy cell or policy endpoint was
generated on Rock Diagnosis.

## Claim

Test whether the positive branch-policy StrategyEIG result on the canonical
eight-rock geometry transfers from Gemma 4 26B A4B to a second model family. This is
a model-robustness replication, not a model leaderboard.

## Frozen Setting

- Diagnosis-only RockSample[7,8] geometry and exact 256-state model from the completed
  scale extension: seven-by-seven grid, eight static rock types, uniform prior,
  distance-dependent binary sensor, and exact posterior/EIG.
- `openai/gpt-5.4-mini` through OpenRouter, reasoning disabled, temperature zero,
  2,048-token output cap, concurrency 64.
- Fresh seed `24073`; 30 paired trajectories, 10 rounds, K=6 explicit
  `branch_policy_v2` policies, receding horizon two, and 10,000 paired bootstrap
  resamples.
- Arms and accounting are unchanged: StrategyEIG, the identical LLM roots scored at
  d1, exhaustive d1 width with matched logical calls and exact scorer units, K=6
  random legal branch policies, and exhaustive d2 as proposal-coverage oracle.

## Gates

Before the endpoint, run the same ten-cell initial/posterior serving screen used for
the 26B scale extension. All ten cells must validate within one semantic retry, with
no terminal failure, legal branch actions, the required distinct movement/check root
mix, and move-then-check continuation. A failed serving gate stops the replication.

The primary endpoint is paired mean posterior-entropy AUC. The replication passes
only if StrategyEIG's 95% paired bootstrap CI lower endpoint is positive against all
three frozen controls: shared-roots d1, exhaustive d1 width, and matched random
strategies. Truth-log-posterior AUC corroborates when all three point estimates are
positive and no interval is entirely negative. Final entropy, movement rate,
exhaustive-value fraction, parse failures, repairs, and cost are secondary. All
mechanics checks must pass and rollout scoring must make zero LLM calls.

The prior Gemma result is used only to motivate this replication. It is not pooled
into the primary intervals. Cross-model effect sizes are descriptive.

## Serving Recovery And Cost

The fail-closed accepted-cell resume mechanism frozen for the scale extension applies
unchanged: accepted cells may be revalidated and reused after a process-level terminal
cell, while unresolved cells are generated under the same bounded validator and all
recovery provenance remains in the final artifact.

The ten-cell smoke is expected below `$0.10`. The full run has a `$3.00` projection
and a hard `$4.00` run cap inside the existing `$40.00` project ledger.

## Frozen Commands

```bash
set -a; source .env; set +a
python scripts/nonmyopic_rock_branch_strategy_smoke.py \
  --config configs/config_nonmyopic_rocksample_7_8_gpt54mini_openrouter.yaml \
  --maps 7-8 --probe-states-per-map 10 \
  --output-dir results/nonmyopic/rocksample_7_8_gpt54mini_smoke_20260721 \
  --run-id nonmyopic-rocksample-7-8-gpt54mini-smoke-20260721 \
  --num-strategies 6 --concurrency 10

python scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_7_8_gpt54mini_openrouter.yaml \
  --maps 7-8 --run-id nonmyopic-rocksample-7-8-gpt54mini-replication-20260721 \
  --output-dir results/nonmyopic/rocksample_7_8_gpt54mini_replication_20260721 \
  --num-trials-per-map 30 --num-rounds 10 --num-strategies 6 \
  --seed 24073 --bootstrap-replicates 10000 --trial-concurrency 32 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```
