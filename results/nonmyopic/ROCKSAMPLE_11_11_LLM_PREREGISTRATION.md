# RockSample[11,11] Root-Slot StrategyEIG Confirmation

Registered 2026-07-21 after the exact structural gate and before any LLM response or
policy endpoint on this geometry.

## Motivation

The frozen 500-pair exact qualification passed: exhaustive d2 beat exhaustive d1 by
`+1.1042` entropy-AUC nats (95% CI `[+1.1015,+1.1068]`, 500/0/0) on the standard
2,048-state SARSOP RockSample[11,11] instance. This confirmation asks whether the
same root-slot LLM-Modulo policy that succeeded on RockSample[7,8] retains useful
search bias at eleven rocks.

## Frozen Design

- Geometry and sensor: exact registered `11-11` map, start `(0,5)`, diagnosis-only
  action set, and half-efficiency distance `log(2)`.
- Model: `google/gemma-4-26b-a4b-it`, OpenRouter, non-thinking, temperature zero.
- Fresh policy seed `24079`; 30 paired trials; 12 rounds; K6; h2;
  `branch_policy_v2`; 10,000 paired bootstrap replicates; trial concurrency 32.
- Ordered movement-root slots from commit `0aff861`: up to three currently legal
  movement roots are machine assigned. The LLM selects their follow-up checks, all
  direct-check roots, observation-contingent followups, names, and rationales.
- Arms: StrategyEIG, shared-roots d1, exhaustive d1 width with matched exact-scorer
  units and one LLM ordering call, matched random strategies, and exhaustive d2.
- Primary endpoint: posterior entropy AUC. StrategyEIG minus shared d1, width, and
  random must each have a strictly positive 95% paired bootstrap lower bound.
- Truth-log-posterior AUC corroborates only if all three lower bounds are positive.
  Final entropy, MAP, movement rate, exhaustive fraction, exact-d2 gap, rejects, and
  serving cost are secondary.
- All legality, pairing, shared-root, width-compute, random-K, and terminal mechanics
  must pass. Exact rollout scoring must make zero LLM calls.

The prior eight-rock results and the 11-rock exact qualification are not pooled into
the primary intervals. A fresh ten-cell actual-prompt smoke must pass every cell within
one bounded repair before the formal launch. The smoke ceiling is `$0.05`; the formal
run uses the existing `$1.25` projection and `$2.00` hard cap within the `$40` project
ledger. The accepted-cell failed-closed resume protocol is allowed only under the
identical frozen interface and config.

## Frozen Commands

```bash
set -a; source .env; set +a
python scripts/nonmyopic_rock_branch_strategy_smoke.py \
  --config configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml \
  --maps 11-11 --probe-states-per-map 10 \
  --output-dir results/nonmyopic/rocksample_11_11_gemma_slot_smoke_20260721 \
  --run-id nonmyopic-rocksample-11-11-gemma-slot-smoke-20260721 \
  --num-strategies 6 --concurrency 10

python scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml \
  --maps 11-11 \
  --run-id nonmyopic-rocksample-11-11-gemma-slot-confirmation-20260721 \
  --output-dir results/nonmyopic/rocksample_11_11_gemma_slot_confirmation_20260721 \
  --num-trials-per-map 30 --num-rounds 12 --num-strategies 6 \
  --seed 24079 --bootstrap-replicates 10000 --trial-concurrency 32 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```
