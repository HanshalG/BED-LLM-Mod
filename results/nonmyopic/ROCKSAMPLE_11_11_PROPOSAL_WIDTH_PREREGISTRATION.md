# RockSample[11,11] StrategyEIG Proposal-Width Robustness

Registered 2026-07-21 before any response or endpoint for seed 24085 or the K2/K4/K6
comparison. Prior K6 endpoints and the exact oracle's preference for the first
machine-assigned movement root are known.

## Claim

Test whether the non-myopic StrategyEIG gain on RockSample[11,11] is robust across a
threefold proposal-width range, from only two candidate branch policies to the
established K6 setting. This is a robustness and quality-versus-verifier-cost test,
not a preregistered monotonic-improvement claim: the exact oracle's enabling root is
already present at K2, so more candidates may legitimately plateau.

## Frozen Design

- Geometry, sensor, prior, exact simulator, and root-slot interface are identical to
  the confirmed RockSample[11,11] experiments.
- Model: `google/gemma-4-26b-a4b-it` through OpenRouter, reasoning disabled,
  temperature zero, 2,048-token output cap, and serving concurrency 64.
- One fresh paired seed, `24085`, used independently at K2, K4, and K6. Each width has
  the same 30 trial indices/truth seeds, 12 rounds, h2, `branch_policy_v2`, 10,000
  bootstrap replicates, and trial concurrency 32. Observation uniforms are keyed by
  seed, trial, state, action, and repeat count, giving common random numbers whenever
  policies reach the same experiment.
- K2 has one machine-assigned movement root and one direct-check root; K4 has two of
  each; K6 has three of each whenever that many moves are legal. The LLM chooses all
  followups and direct-check identities.
- Every width retains StrategyEIG, shared-roots d1, matched exact-scorer-width d1,
  grammar-matched random strategies of the same K, and exhaustive d2.
- Rollout scoring, Bayesian updates, observations, and endpoint computation use the
  exact finite simulator and make zero LLM calls.

## Gates

Fresh actual-prompt serving smokes at K2, K4, and K6 must each pass 10/10 cells within
one bounded repair, with all root-slot, followup, and terminal mechanics valid.

Within each K separately, StrategyEIG minus shared d1, exhaustive d1 width, and
matched random strategies must each have a strictly positive 95% paired bootstrap
lower bound for entropy AUC. The corresponding three truth-log-posterior-AUC lower
bounds must also be positive. All 18 width-by-control-by-endpoint intervals must pass;
no K may be removed, replaced, or rerun after endpoints are opened.

Cross-K paired entropy/truth differences, exact-d2 gaps, movement rates, exhaustive
fractions, request/token cost, and exact scorer nodes per decision are secondary.
The quality curve may increase, plateau, or decrease; only the all-width positive
gain is primary.

## Cost

Based on completed K6 Gemma runs, the three formal widths are projected at `$0.75`
total and the 30-call smoke wave below `$0.02`. Each process retains the existing
`$2.00` hard cap and the project retains its `$40.00` cap. Runs are sequential to
avoid races in the shared spend ledger. Fail-closed resume is permitted only for the
same frozen K, seed, run ID, and config.

## Frozen Commands

```bash
set -a; source .env; set +a
for k in 2 4 6; do
  python scripts/nonmyopic_rock_branch_strategy_smoke.py \
    --config configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml \
    --maps 11-11 --probe-states-per-map 10 \
    --output-dir "results/nonmyopic/rocksample_11_11_width_k${k}_smoke_20260721" \
    --run-id "nonmyopic-rocksample-11-11-width-k${k}-smoke-20260721" \
    --num-strategies "$k" --concurrency 10
done

for k in 2 4 6; do
  python scripts/nonmyopic_rock_strategy_prior.py \
    --config configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml \
    --maps 11-11 \
    --run-id "nonmyopic-rocksample-11-11-width-k${k}-seed-24085-20260721" \
    --output-dir "results/nonmyopic/rocksample_11_11_width_k${k}_seed_24085_20260721" \
    --num-trials-per-map 30 --num-rounds 12 --num-strategies "$k" \
    --seed 24085 --bootstrap-replicates 10000 --trial-concurrency 32 \
    --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
done
```
