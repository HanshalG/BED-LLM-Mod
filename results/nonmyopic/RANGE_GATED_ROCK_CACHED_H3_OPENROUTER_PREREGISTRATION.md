# Range-Gated Rock Cached-h3 OpenRouter Preregistration

Status: frozen before any live Rock response from the repaired OpenRouter adapter.

## Motivation and Multiplicity

The direct-vLLM cached-h3 S0 is pending for `msc,llm` capacity and remains the
primary cluster line. Separately, the OpenRouter adapter now preserves reasoning-only
length-stopped completions and requests one bounded non-reasoning final. A synthetic
Gemma 4 26B calibration exercised this path successfully for `$0.00005187`.

This is a distinct provider replication, not a replacement run. Both provider
outcomes will be reported. No outcome from this line can alter, cancel, or relabel
the direct-vLLM protocol.

## S0

- Model: OpenRouter `google/gemma-4-26b-a4b-it`, thinking enabled.
- Reasoning/final budgets: 4,096 and 256 tokens.
- Fresh model seed: `24196`.
- Same 12 fixed trajectory-prefix cells, successor-grounded K4 interface,
  JSON-prefix parser, exact stable verifier, and one correction attempt as the
  direct-vLLM S0.
- Run cap: `$0.25`.

The unchanged S0 gates require 12 complete legal cells, exact
`move-SOUTH, move-SOUTH, check-5` selected roots on the first three prefixes, at
least 75% exhaustive-d3 root agreement over all cells, complete request/reasoning/
forced-final accounting, and no scoring-time model calls. Failure stops this
OpenRouter line without repair.

## Conditional S1

S1 runs only after OpenRouter S0 passes.

- Fresh truth/model seed: `24197`, unused by deterministic or live trajectory runs.
- 50 paired truths, eight rounds, and exactly the four registered arms:
  cached Gemma h3, identical-root random h3, exhaustive d2, exhaustive d3.
- H3 proposals in rounds 1--6 and common exact-d2 tails in rounds 7--8.
- Exact full-message SHA-256 cache, 300 logical cells, hard cap of 64 physical
  proposal cells.
- Common deterministic observation uniforms.
- Run cap: `$1.00`.

The producer and independent-audit gates are unchanged from
`RANGE_GATED_ROCK_CACHED_H3_TRAJECTORY_PREREGISTRATION.md`: all four entropy/truth
95% lower bounds versus exhaustive d2 and random h3 must be positive, exact-d3
recovery must be at least 60%, both two-south and on-site-by-round-three rates must
be at least 75%, all cache/legality/pairing mechanics must pass, and the fresh local
audit must reproduce every decision and pass its own four intervals.
