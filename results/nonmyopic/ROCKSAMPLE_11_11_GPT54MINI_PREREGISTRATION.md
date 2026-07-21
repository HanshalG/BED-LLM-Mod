# RockSample[11,11] GPT-5.4 Mini Scale Replication

Registered 2026-07-21 before any GPT-5.4 Mini response or policy endpoint on the
eleven-rock geometry.

## Claim

Test whether the preregistered non-myopic gain on the standard 2,048-state
RockSample[11,11] diagnosis instance transfers from Gemma 4 26B A4B to a second model
family under the identical ordered root-slot interface. This is a robustness
replication, not a model leaderboard.

## Frozen Design

- Geometry and sensor: registered `11-11` map, start `(0,5)`, diagnosis-only action
  set, uniform prior over 11 binary rocks, and half-efficiency distance `log(2)`.
- Model: `openai/gpt-5.4-mini` through OpenRouter, reasoning disabled, temperature
  zero, 2,048-token output cap, and serving concurrency 64.
- Fresh policy seed `24080`; 30 paired trials; 12 rounds; K6; h2;
  `branch_policy_v2`; 10,000 paired bootstrap replicates; trial concurrency 32.
- Ordered movement-root slots: up to three currently legal movement roots are machine
  assigned in canonical action order. GPT chooses their follow-up checks, all direct
  check roots, observation-contingent followups, names, and rationales.
- Arms: StrategyEIG, shared-roots d1, exhaustive d1 width with matched exact-scorer
  units and one LLM ordering call, matched random strategies, and exhaustive d2.
- All exact posterior updates, observations, rollout branches, and policy scores use
  the frozen finite simulator. Rollout scoring makes zero LLM calls.

## Gates

Before the endpoint, a fresh ten-cell actual-prompt smoke must pass all cells within
one bounded repair. Every selected and branch action must be legal, all h2 cells must
have the frozen three-move/three-check mix and move-then-check continuations, and there
may be no terminal cell failure.

The primary endpoint is posterior entropy AUC. StrategyEIG minus shared d1, exhaustive
d1 width, and matched random strategies must each have a strictly positive 95% paired
bootstrap lower bound. Truth-log-posterior AUC corroborates only if all three lower
bounds are also positive. Final entropy, MAP accuracy, movement rate, exhaustive
fraction, exact-d2 gap, rejects, repairs, and serving cost are secondary.

All legality, pairing, shared-root, width-compute, random-K, and terminal mechanics
must pass. The prior Gemma and eight-rock GPT results are not pooled into any interval;
cross-model effect-size differences are descriptive only.

## Serving And Cost

The smoke ceiling is `$0.10`. Based on the completed eight-rock GPT run and 21% more
logical cells, the formal projection is `$2.60` with the unchanged `$4.00` hard run
cap inside the `$40.00` project ledger. The accepted-cell fail-closed resume protocol
is allowed only for this identical frozen prompt, config, seed, and interface, with
all rejects and recovery provenance retained.

## Frozen Commands

```bash
set -a; source .env; set +a
python scripts/nonmyopic_rock_branch_strategy_smoke.py \
  --config configs/config_nonmyopic_rocksample_7_8_gpt54mini_openrouter.yaml \
  --maps 11-11 --probe-states-per-map 10 \
  --output-dir results/nonmyopic/rocksample_11_11_gpt54mini_slot_smoke_20260721 \
  --run-id nonmyopic-rocksample-11-11-gpt54mini-slot-smoke-20260721 \
  --num-strategies 6 --concurrency 10

python scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_7_8_gpt54mini_openrouter.yaml \
  --maps 11-11 \
  --run-id nonmyopic-rocksample-11-11-gpt54mini-slot-replication-20260721 \
  --output-dir results/nonmyopic/rocksample_11_11_gpt54mini_slot_replication_20260721 \
  --num-trials-per-map 30 --num-rounds 12 --num-strategies 6 \
  --seed 24080 --bootstrap-replicates 10000 --trial-concurrency 32 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```

## Pre-Endpoint Resume Accounting Amendment

Added after the first formal process failed closed and before any policy endpoint was
produced. One direct-check policy twice selected a branch move illegal at `(0,5)`.
The failure artifact contains 1,038 accepted cells and 20 rejected responses from
1,058 physical requests; it cost `$2.51745885`. No trace or metric exists.

The accepted-cell resume is authorized unchanged, but the original reusable config's
`$2.00` startup reservation plus the banked spend exceeds the frozen `$4.00` run cap.
A startup attempt therefore stopped locally before constructing an adapter or making
a request. For the 12 remaining logical cells only, the resume command substitutes
`configs/config_nonmyopic_rocksample_11_11_gpt54mini_resume_openrouter.yaml`, whose
sole experimental difference is `openrouter_projected_cost_usd: 0.10`. The project
budget, cumulative `$4.00` hard run cap, model, prompts, concurrency, retries, token
limit, seed, L1 config, endpoints, and gates are unchanged. The loader revalidates all
banked accepted cells against that frozen L1 config.

```bash
set -a; source .env; set +a
python scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_11_11_gpt54mini_resume_openrouter.yaml \
  --maps 11-11 \
  --run-id nonmyopic-rocksample-11-11-gpt54mini-slot-replication-20260721 \
  --output-dir results/nonmyopic/rocksample_11_11_gpt54mini_slot_replication_20260721 \
  --num-trials-per-map 30 --num-rounds 12 --num-strategies 6 \
  --seed 24080 --bootstrap-replicates 10000 --trial-concurrency 32 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc \
  --resume-failure \
  results/nonmyopic/rocksample_11_11_gpt54mini_slot_replication_20260721/L1_FAILURE.json
```

## Outcome

The fresh serving smoke passed 10/10 cells on the first response with zero rejects,
reasoning tokens, forced exits, or terminal failures, costing `$0.03416175`.

The resumed formal run completed all 1,050 logical policy cells. It reused 1,038
accepted cells from the fail-closed artifact, made 11 additional requests, and
retained all 20 rejected responses. Cumulatively it made 1,069 physical requests,
used 2,100,085 prompt and 267,488 completion tokens, had zero reasoning tokens,
forced exits, terminal failures, or rollout-scoring LLM calls, and cost
`$2.54133255` within the frozen `$4.00` cap.

All preregistered gates passed. StrategyEIG's entropy-AUC gains were +0.5811 (95%
CI [0.4672, 0.6909]) against shared-roots d1, +0.5528 [0.4374, 0.6613] against
exhaustive d1 width, and +0.4688 [0.3364, 0.5919] against matched random
strategies, with wins/ties/losses of 29/0/1, 27/0/3, and 27/0/3. The corresponding
truth-log-AUC gains were +0.6053 [0.4244, 0.7750], +0.6066 [0.4033, 0.7988], and
+0.5183 [0.3255, 0.7011]. StrategyEIG moved on 231/360 decisions, captured 62.6%
of exhaustive d2 value over eligible h2 rounds, and remained 0.5459 entropy-AUC
nats below exhaustive d2. The positive sign therefore replicates across model
families on the hardest map, while proposal quality remains model-dependent.
