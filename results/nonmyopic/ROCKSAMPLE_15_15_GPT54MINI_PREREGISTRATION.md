# RockSample[15,15] GPT-5.4 Mini Cross-Family Replication

Registered 2026-07-22 before any GPT-5.4 Mini response or policy endpoint on the
fifteen-rock geometry.

## Claim

Test whether the preregistered non-myopic gain at 32,768 latent states transfers
from the independently replicated Gemma 4 12B and 26B results to GPT-5.4 Mini under
the identical K4 ordered root-slot interface. This is a sign and robustness
replication, not a model leaderboard. Prior model-family results are not pooled
into any interval, and cross-model effect-size differences are descriptive.

## Frozen Design

- Geometry and sensor: registered `15-15` map, start `(0,7)`, diagnosis-only
  actions, uniform prior over 15 binary rocks, and half-efficiency distance
  `log(2)`.
- Model: non-thinking `openai/gpt-5.4-mini` through OpenRouter, temperature zero,
  2,048-token output cap, and serving concurrency 64.
- Fresh policy seed `24114`; 30 paired trials; 15 rounds; K4; h2;
  `branch_policy_v2`; 10,000 paired bootstrap replicates; trial concurrency four.
- Ordered movement-root slots are machine assigned. GPT selects their follow-up
  checks, all direct-check roots, observation-contingent followups, names, and
  rationales.
- Arms: StrategyEIG, shared-roots d1, exhaustive d1 width with matched exact-scorer
  units and one LLM ordering call, matched random strategies, and exhaustive d2.
- Exact finite-state code controls every observation, posterior update, branch,
  and strategy score. Rollout scoring makes zero LLM calls.

## Gates

A fresh ten-cell actual-prompt smoke must first complete with every cell accepted
within one bounded repair, zero terminal failures, zero reasoning tokens and forced
exits, legal roots and followups, both movement and check roots in every h2 cell,
and a check after every proposed movement. Smoke proposal scores are descriptive
and cannot qualify the policy endpoint.

The primary endpoint is posterior entropy AUC. StrategyEIG minus shared d1,
exhaustive d1 width, and matched random strategies must each have a strictly
positive 95% paired-bootstrap lower bound. Truth-log-posterior AUC corroborates
only if all three corresponding lower bounds are positive. Final entropy, MAP
accuracy, movement rate, exhaustive fraction, exact-d2 gap, rejects, repairs,
latency, and serving cost are secondary.

All legality, pairing, shared-root, width-compute, random-K, and terminal mechanics
must pass. No threshold, seed, model, prompt, width, or endpoint may change after a
response. The accepted-cell failed-closed resume protocol is allowed only under the
identical frozen config and interface, retaining all rejection and cost provenance.

## Serving And Cost

The smoke ceiling is `$0.10`. The completed 11-rock GPT run cost `$2.5413` for
1,069 physical requests; the 15-rock formal is projected at `$4.00` with a `$6.00`
hard run cap. The user added `$40` of OpenRouter credit after the Gated Sensor S1.
The project ledger ceiling is conservatively raised from `$40` to `$70`, leaving
`$39.2677` of ledger headroom at registration, slightly below the authenticated
account balance.

## Frozen Commands

```bash
set -a; source .env; set +a
python scripts/nonmyopic_rock_branch_strategy_smoke.py \
  --config configs/config_nonmyopic_rocksample_15_15_gpt54mini_openrouter.yaml \
  --maps 15-15 --probe-states-per-map 10 \
  --output-dir results/nonmyopic/rocksample_15_15_gpt54mini_smoke_20260722 \
  --run-id nonmyopic-rocksample-15-15-gpt54mini-smoke-20260722 \
  --num-strategies 4 --concurrency 10

python scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_15_15_gpt54mini_openrouter.yaml \
  --maps 15-15 \
  --run-id nonmyopic-rocksample-15-15-gpt54mini-replication-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_gpt54mini_replication_20260722 \
  --num-trials-per-map 30 --num-rounds 15 --num-strategies 4 \
  --seed 24114 --bootstrap-replicates 10000 --trial-concurrency 4 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```
