# RockSample 7-8 StrategyEIG Scale Extension Preregistration

Registered: 2026-07-21, before any exact-policy or LLM-policy outcome on this
adaptation was computed.

## Claim

Test whether the positive branch-policy StrategyEIG result extends from three- and
five-rock maps to the canonical eight-rock `RockSample[7,8]` benchmark geometry. The
adaptation remains diagnosis-only: the latent target is the static eight-bit rock-type
vector, and sampling, reward, and exit actions are excluded.

## Frozen Map And Model

- Source: Smith and Simmons (2004), *Heuristic Search Value Iteration for POMDPs*,
  Figure 4, page 5.
- Seven-by-seven grid, zero-based rock coordinates `(1,0)`, `(5,1)`, `(2,2)`,
  `(3,2)`, `(6,3)`, `(0,5)`, `(3,5)`, `(2,6)`, and displayed start `(0,3)`.
- Uniform prior over all 256 good/bad vectors. Exact `pomdp_py==1.3.5.1` transition
  and distance-dependent observation likelihoods with half-efficiency distance
  `log(2)`. Exact posterior updates and MAP decode.
- The posterior remains factored because the prior is independent and each check
  likelihood involves one rock. LLM prompts therefore receive all eight exact
  marginals, a sufficient representation of the full 256-state posterior.

## Stage A: Zero-LLM Qualification

- Seed `24071`, 1,000 paired trajectories, 10 rounds, 10,000 paired bootstrap
  resamples.
- Candidate width `12`, which contains every legal action at every state. Thus the
  depth-two arm is exhaustive d2 and both one-step arms are exhaustive d1; the
  call-matched width arm remains as an allocation/mechanics check.
- Primary endpoint: final posterior-entropy reduction for d2 relative to exhaustive
  d1. Qualification requires a strictly positive lower endpoint of the paired 95%
  bootstrap interval, with all pairing, width, and legality checks true.
- If qualification fails, no paid StrategyEIG measurement is run on this map.

## Stage B: Paid Paired Confirmation

Run only if Stage A passes:

- `google/gemma-4-26b-a4b-it`, OpenRouter, non-thinking, temperature zero.
- Fresh seed `24072`, 30 paired trajectories, 10 rounds, K=6 explicit
  `branch_policy_v2` policies, receding horizon two, 10,000 paired bootstrap
  resamples, trial concurrency 32.
- Arms: StrategyEIG; the identical LLM roots scored at d1; exhaustive d1 width with
  matched logical LLM calls and exact scorer units; K=6 random legal branch policies;
  and exhaustive d2 as a proposal-coverage oracle.
- Primary endpoint: paired entropy-AUC gain for StrategyEIG against shared d1,
  exhaustive d1 width, and matched random strategies. The scale-extension gate passes
  only if all three paired 95% CI lower endpoints are positive.
- Truth-log-posterior AUC corroborates when all three point estimates are positive and
  no interval is entirely negative. Final entropy, MAP accuracy, movement rate,
  exhaustive-value fraction, parse failures, repairs, and token cost are secondary.
- Mechanics must match the earlier confirmation: all actions legal, shared initial
  strategy cells, width compute matched, K random policies present, no failed terminal
  cell, and zero rollout-scoring LLM calls.

## Serving And Cost Gate

Before Stage B, run ten actual prompt cells spanning initial and posterior states. All
ten must return valid complete cells within one semantic retry, with no terminal
failure. Project cost from that smoke. Stage B has a declared `$1.25` projection and a
hard `$2.00` run cap inside the existing `$40` ledger.

## Frozen Commands

```bash
python scripts/nonmyopic_rock_diagnosis_oracle.py \
  --map 7-8 --num-trials 1000 --num-rounds 10 --candidate-widths 12 \
  --seed 24071 --bootstrap-replicates 10000 \
  --output-dir results/nonmyopic/rocksample_7_8_exact_qualification_20260721

set -a; source .env; set +a
python scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml \
  --maps 7-8 --run-id nonmyopic-rocksample-7-8-scale-20260721 \
  --output-dir results/nonmyopic/rocksample_7_8_scale_20260721 \
  --num-trials-per-map 30 --num-rounds 10 --num-strategies 6 \
  --seed 24072 --bootstrap-replicates 10000 --trial-concurrency 32 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```

## Serving-Recovery Amendment

Added 2026-07-21 after the first Stage B process failed closed and before any Stage B
endpoint was produced or inspected. The process accepted 803 cells, then one later
horizon-two cell repeated a movement root on both its initial response and its single
semantic retry. It wrote only `L1_FAILURE.json`; no trial traces, paired estimates, or
gate result existed.

The continuation may revalidate and cache every accepted cell from that failure
artifact, replay the deterministic exact trajectories, and request only cells that
were not accepted. The failed cell is generated afresh under the same one-feedback-
retry validator. Accepted cells are never regenerated, the map, trials, seed, arms,
scoring, endpoints, and bootstrap remain frozen, and the completed artifact must
record the failure path, prior error, reused-cell count, preserved invalid-response
count, and cumulative run usage. A resumed run is valid when it has no unresolved
terminal cell and all original mechanics checks pass.

```bash
set -a; source .env; set +a
python scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml \
  --maps 7-8 --run-id nonmyopic-rocksample-7-8-scale-20260721 \
  --output-dir results/nonmyopic/rocksample_7_8_scale_20260721 \
  --resume-failure results/nonmyopic/rocksample_7_8_scale_20260721/L1_FAILURE.json \
  --num-trials-per-map 30 --num-rounds 10 --num-strategies 6 \
  --seed 24072 --bootstrap-replicates 10000 --trial-concurrency 32 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```
