# Range-Gated Rock JSON-Prefix Cluster 26B Preregistration

Status: frozen after parser implementation and deterministic tests, before any fresh
model response under this parser mode.

## Motivation and bounded repair

The direct-vLLM successor-grounded seed `24183` line failed its strict full-string
JSON gate after six accepted cells. Every accepted policy and both rejected response
prefixes contained the exact delayed route `move-SOUTH, move-SOUTH, check-5`. The two
rejected responses began with a complete valid four-key object, then appended
self-correction prose. That seed remains failed and is never reused.

This fresh line changes serving only:

1. After existing code-fence normalization, `json.JSONDecoder.raw_decode` reads one
   JSON value starting at the first character.
2. The decoded value must still have exactly keys `r0` through `r3`, exactly two
   string actions per key, the machine-fixed roots, and complete dynamic legality.
3. Any prefix text, malformed first object, missing branch, wrong type, wrong length,
   or illegal action fails exactly as before.
4. Trailing text is ignored for compilation but retained verbatim with the compiled
   plan for audit.
5. The standalone run now writes `run.log`, retaining full first-stage reasoning,
   forced-finalization traces, token events, raw finals, and correction events.

The model, prompt, successor table, exact verifier, 4,096-token reasoning pass,
512-token bounded final pass, and all scientific thresholds are unchanged.

## S0 serving and route smoke

- Fresh seed `24185`; ten distinct start-position posterior cells.
- Gemma 4 26B A4B direct vLLM on `msc,llm`, excluding `oat12`.

All requirements must pass:

1. Ten cells complete with four dynamically legal fixed-root plans.
2. Exactly ten accepted logical cells; all physical generations, corrections,
   forced-finalization events, raw text, and reasoning traces are accounted for.
3. `move-SOUTH, move-SOUTH, check-5` appears in at least `8/10` cells.
4. No empty final, traceback, OOM, malformed first object, or scoring-time model call.

Any failure stops this parser mode without a seed, prompt, budget, or threshold change.

## Conditional S1 proposal-quality gate

S1 runs only after a complete S0 pass.

- Fresh seed `24186`; sixteen distinct strict h3-over-h2 opportunities.
- Identical-root random tails, shared-plan d2, strong exhaustive d2 root, exhaustive
  open-loop h3, and 5,000 paired bootstraps.

All requirements must pass:

1. Positive 95% lower bounds versus matched random, shared-plan d2, and strong d2.
2. At least 75% exhaustive h3 route-root selection.
3. At least 60% mean recovery of the h3-over-strong-d2 opportunity.
4. Complete serving/mechanical accounting and an independent exact replay of roots,
   plans, values, controls, aggregates, and gates.

A pass authorizes only a separately frozen paired trajectory confirmation. Local
cluster inference adds no OpenRouter spend.
