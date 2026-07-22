# RockSample[15,15] Root-Slot StrategyEIG Confirmation

Registered 2026-07-22 after the exact structural gate and before any LLM response
or StrategyEIG policy endpoint on this geometry.

## Motivation

The frozen 100-pair exact qualification passed: exhaustive d2 beat exhaustive d1
by `+0.58884` entropy-AUC nats (95% CI `[+0.58006,+0.59706]`, 100/0/0) on the
32,768-state RockSample[15,15] diagnosis instance. This confirmation tests whether
the same root-slot LLM-Modulo policy that succeeded at eight and eleven rocks
retains useful search bias at fifteen rocks.

K4 is frozen because the registered eleven-rock width ablation found a positive
K4-minus-K2 gain (`+0.5655`, CI `[+0.4974,+0.6418]`) but no resolved K6-minus-K4
gain (`+0.0102`, CI `[-0.0156,+0.0337]`).

## Frozen Design

- Geometry and sensor: the exact registered `15-15` map, start `(0,7)`,
  diagnosis-only action set, 32,768-state full joint belief, and half-efficiency
  distance `log(2)`.
- Model: `google/gemma-4-26b-a4b-it`, OpenRouter, non-thinking, temperature zero.
- Fresh policy seed `24100`; 30 paired trials; 15 rounds; K4; h2;
  `branch_policy_v2`; 10,000 paired bootstrap replicates; trial concurrency 4.
- Ordered movement-root slots are machine assigned. The LLM selects their
  follow-up checks, all direct-check roots, observation-contingent followups,
  names, and rationales.
- Arms: StrategyEIG, shared-roots d1, exhaustive d1 width with matched exact-scorer
  units and one LLM ordering call, matched random strategies, and exhaustive d2.
- Primary endpoint: posterior entropy AUC. StrategyEIG minus shared d1, width, and
  random must each have a strictly positive 95% paired bootstrap lower bound.
- Truth-log-posterior AUC corroborates only if all three lower bounds are positive.
  Final entropy, MAP, movement rate, exhaustive fraction, exact-d2 gap, rejects,
  latency, and serving cost are secondary.
- All legality, pairing, shared-root, width-compute, random-K, and terminal
  mechanics must pass. Exact rollout scoring must make zero LLM calls.

No endpoint from a paid policy run on this geometry may alter the registered
design. A deterministic dry run may test runtime and mechanics only. A fresh
ten-cell actual-prompt smoke must pass every cell within one bounded repair before
the formal launch. The smoke ceiling is `$0.05`; the formal projection is `$0.75`
with a `$1.50` hard run cap inside the `$40` project ledger. The accepted-cell
failed-closed resume protocol is allowed only under the identical config and
interface.

## Frozen Commands

```bash
set -a; source .env; set +a
python scripts/nonmyopic_rock_branch_strategy_smoke.py \
  --config configs/config_nonmyopic_rocksample_15_15_openrouter.yaml \
  --maps 15-15 --probe-states-per-map 10 \
  --output-dir results/nonmyopic/rocksample_15_15_gemma_slot_smoke_20260722 \
  --run-id nonmyopic-rocksample-15-15-gemma-slot-smoke-20260722 \
  --num-strategies 4 --concurrency 10

python scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_15_15_openrouter.yaml \
  --maps 15-15 \
  --run-id nonmyopic-rocksample-15-15-gemma-slot-confirmation-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_gemma_slot_confirmation_20260722 \
  --num-trials-per-map 30 --num-rounds 15 --num-strategies 4 \
  --seed 24100 --bootstrap-replicates 10000 --trial-concurrency 4 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```

## Serving Gate Outcome

The fresh actual-prompt smoke passed all 10/10 cells on the first attempt. It had
zero rejected attempts, terminal failures, reasoning tokens, or forced exits; all
cells contained the required movement and check roots, and every movement branch
continued to a check. The ten requests cost `$0.00438036`, below the registered
`$0.05` ceiling. This authorized the unchanged formal command above.
