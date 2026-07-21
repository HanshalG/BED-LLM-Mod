# RockSample[7,8] Gemma Root-Slot Replication

Registered 2026-07-21 before any new serving response or policy endpoint.

## Purpose

The positive canonical result used Gemma 4 26B A4B with a semantic distinct-root
instruction. The cross-family GPT-5.4 Mini replication used the later ordered physical
movement-root slots. This fresh run removes that interface difference by evaluating
Gemma under exactly the GPT slot interface.

## Frozen Design

- Map: canonical diagnosis-only RockSample[7,8].
- Model: `google/gemma-4-26b-a4b-it`, OpenRouter, non-thinking, temperature zero.
- Fresh seed `24077`; 30 paired trials; 10 rounds; K6; h2; `branch_policy_v2`.
- Ordered movement-root slots from commit `0aff861`; the LLM still chooses movement
  follow-up checks, direct-check roots, observation branches, names, and rationales.
- Arms: StrategyEIG, shared-roots d1, exhaustive d1 width with matched exact-scorer
  units and one ordering call, matched random strategies, and exhaustive d2 reference.
- Primary endpoint: posterior entropy AUC. Each of StrategyEIG minus shared d1, width,
  and random must have a strictly positive 95% paired bootstrap lower bound over
  10,000 replicates.
- Truth-log-posterior AUC corroborates only when all three lower bounds are positive.
  Final entropy, MAP, movement, exhaustive fraction, rejections, and cost are secondary.
- All mechanics must pass; exact rollout scoring makes zero LLM calls.

The prior Gemma and GPT runs are not pooled into primary intervals. Cross-run effect
sizes are descriptive because seeds differ.

## Serving and Recovery

A fresh ten-cell slot-interface smoke must pass all cells within one bounded repair
before formal launch. The accepted-cell failed-closed resume protocol remains frozen:
only responses that revalidate under this exact interface and config may be reused.
The smoke should cost below `$0.05`; the formal run retains the config's `$1.25`
projection and `$2.00` hard run cap within the `$40` project ledger.

## Frozen Commands

```bash
set -a; source .env; set +a
python scripts/nonmyopic_rock_branch_strategy_smoke.py \
  --config configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml \
  --maps 7-8 --probe-states-per-map 10 \
  --output-dir results/nonmyopic/rocksample_7_8_gemma_slot_smoke_20260721 \
  --run-id nonmyopic-rocksample-7-8-gemma-slot-smoke-20260721 \
  --num-strategies 6 --concurrency 10

python scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml \
  --maps 7-8 --run-id nonmyopic-rocksample-7-8-gemma-slot-replication-20260721 \
  --output-dir results/nonmyopic/rocksample_7_8_gemma_slot_replication_20260721 \
  --num-trials-per-map 30 --num-rounds 10 --num-strategies 6 \
  --seed 24077 --bootstrap-replicates 10000 --trial-concurrency 32 \
  --strategy-schema branch_policy_v2 --primary-endpoint entropy_auc
```

## Pre-Endpoint Terminal-Horizon Amendment

The first slot-interface smoke failed closed at 9/10 cells before any policy endpoint.
Both rejected responses came from one h1 cell. Although movement slots apply only at
h2, its prompt still contained `MOVEMENT_ROOT_SLOTS=[]` and described "the first 0"
machine-assigned slots. Gemma emitted six empty slot placeholders plus six real root
strategies on both bounded attempts, violating the exact-six schema. The other nine
cells passed; the attempt used 11 requests and `$0.00501294`.

The repair removes every movement-slot and h2-followup instruction from h1 prompts.
H1 continues to require exactly six distinct legal roots with empty followups. H2
prompts, parser constraints, model, budgets, controls, seed, endpoints, and gates are
unchanged. The failed smoke remains diagnostic only. A fresh ten-cell smoke in
`rocksample_7_8_gemma_slot_smoke_v2_20260721` must pass before the formal command above
is authorized.
