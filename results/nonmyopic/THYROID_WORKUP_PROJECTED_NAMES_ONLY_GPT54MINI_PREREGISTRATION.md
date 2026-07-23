# UCI Thyroid Projected Names-Only GPT-5.4 Mini Preregistration

Status: frozen after the projected-utility confirmation and before any names-only
projected-interface GPT-5.4 Mini response or endpoint is observed.

## Causal ablation

Projected utility combined two changes relative to the failed names-only transfer:
calibrated branch-local utility cards and bounded exact legal projection. This fresh
factorial cell removes the utility cards while retaining the projection compiler.

The model receives the original names-only prompt: fixed roots, branch probabilities,
class probabilities, action names/descriptions, and legal menus. It sees no expected
entropy or information-gain value. After one invalid-response correction, the compiler
preserves all valid branches and projects only invalid or missing branches to a legal
minimum-entropy continuation. Projection remains capped so it cannot silently supply
the policy.

## S0 smoke

- Fresh seed `24163`; 12 late-state cells at history lengths
  `0,1,2,3,4,5,6,6,5,4,3,2`.
- `openai/gpt-5.4-mini` via OpenRouter, non-thinking, temperature zero, 1,024-token
  output cap, one registered correction.
- Required: all cells complete/legal; no root repeat; all lengths covered; zero
  projected cells; zero utility cards in requests; zero reasoning, forced exits, or
  scoring-time LLM calls. Any failure stops the ablation.

## Conditional S1 confirmation

- Fresh seed `24164`; 50 patient rows without replacement; eight paired actions.
- Arms: projected names-only GPT depth two, matched-random policies on identical
  roots, exact depth one, and exhaustive depth two; common exact final action.
- Primary target-entropy AUC and corroborating truth-log-posterior AUC; 10,000 paired
  bootstraps; exactly 350 logical GPT cells.

All gates are unchanged and required:

1. Positive paired 95% lower bounds for both endpoints versus exact depth one.
2. Positive paired 95% lower bounds for both endpoints versus matched random.
3. At least 60% recovery of exhaustive depth two's entropy gain over depth one.
4. At least 75% first-round blood collection.
5. At most 5% projected cells and at most 1% projected branches.
6. Complete legal pairing, no utility cards in model requests, zero reasoning/forced/
   scoring calls, and independent replay of decisions, controls, projections, and
   fresh-bootstrap gates.

No alternate seed, prompt repair, threshold change, or replacement run follows a
failure. Expected OpenRouter cost is below `$0.60`, with the existing `$6` run cap.
Project spend is `$36.18786751 / $110`, leaving `$73.81213249` before S0.
