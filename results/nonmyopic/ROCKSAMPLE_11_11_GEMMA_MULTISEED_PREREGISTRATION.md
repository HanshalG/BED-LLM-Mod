# RockSample[11,11] Gemma Multi-Seed Robustness Replication

Registered 2026-07-21 after the seed-24079 Gemma and seed-24080 GPT endpoints were
known, but before any response or endpoint from seeds 24081--24083.

## Claim

Test whether the strongest RockSample[11,11] StrategyEIG result is robust to fresh
policy/history seeds under the identical Gemma root-slot interface. This is a
replication across independently generated trajectories and proposal histories, not
a hyperparameter search. The previously observed seed 24079 is not pooled into the
primary gate.

## Frozen Design

- Geometry and sensor: registered `11-11` map, start `(0,5)`, diagnosis-only action
  set, uniform prior over 11 binary rocks, and half-efficiency distance `log(2)`.
- Model: `google/gemma-4-26b-a4b-it` through OpenRouter, reasoning disabled,
  temperature zero, 2,048-token output cap, and serving concurrency 64.
- Three fresh policy seeds: `24081`, `24082`, and `24083`. Each seed has 30 paired
  trials, 12 rounds, K6, h2, `branch_policy_v2`, 10,000 paired bootstrap replicates,
  and trial concurrency 32.
- Ordered root slots: up to three currently legal movement roots are machine assigned
  in canonical order. Gemma chooses their follow-up checks, all direct-check roots,
  observation-contingent followups, names, and rationales.
- Arms: StrategyEIG, shared-roots d1, exhaustive d1 width with matched exact-scorer
  units and one LLM ordering call, matched random strategies, and exhaustive d2.
- All exact posterior updates, observation branches, rollout scores, and policy
  selection use the frozen finite simulator. Rollout scoring makes zero LLM calls.

## Gates

A fresh ten-cell actual-prompt serving smoke must pass all cells within one bounded
repair, with legal roots/followups and no terminal failure.

For each fresh seed separately, StrategyEIG minus shared d1, exhaustive d1 width, and
matched random strategies must have a strictly positive 95% paired bootstrap lower
bound for posterior entropy AUC. The same six per-seed comparisons must each have a
strictly positive truth-log-posterior-AUC lower bound. Thus all 18 entropy/truth
intervals must exclude zero in the favorable direction. Failure of any one interval
fails the multi-seed primary claim; no selective seed removal or replacement is
allowed.

All legality, pairing, shared-root, width-compute, random-K, and terminal mechanics
must pass for every seed. The pooled 90-pair estimates, movement rates, exhaustive
fractions, exact-d2 gaps, rejects, repairs, and costs are secondary. The prior
seed-24079 Gemma run may be shown only as an external replication point after the
fresh three-seed gate is evaluated.

## Serving And Cost

The smoke ceiling is `$0.05`. Recent identical-interface Gemma runs cost about `$0.30`
per seed; the three-run projection is `$0.95`. Each process retains the existing
`$2.00` hard run cap and the project retains its `$40.00` cap. Seeds run sequentially
so concurrent processes cannot race on the shared spend ledger. The accepted-cell
fail-closed resume protocol is allowed only for an identical frozen seed/config/run.

## Frozen Commands

```bash
set -a; source .env; set +a
python scripts/nonmyopic_rock_branch_strategy_smoke.py \
  --config configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml \
  --maps 11-11 --probe-states-per-map 10 \
  --output-dir results/nonmyopic/rocksample_11_11_gemma_multiseed_smoke_20260721 \
  --run-id nonmyopic-rocksample-11-11-gemma-multiseed-smoke-20260721 \
  --num-strategies 6 --concurrency 10

for seed in 24081 24082 24083; do
  python scripts/nonmyopic_rock_strategy_prior.py \
    --config configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml \
    --maps 11-11 \
    --run-id "nonmyopic-rocksample-11-11-gemma-seed-${seed}-20260721" \
    --output-dir "results/nonmyopic/rocksample_11_11_gemma_seed_${seed}_20260721" \
    --num-trials-per-map 30 --num-rounds 12 --num-strategies 6 \
    --seed "$seed" --bootstrap-replicates 10000 --trial-concurrency 32 \
    --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
done
```
