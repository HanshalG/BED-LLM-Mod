# UCI Thyroid Workup 26B Named-Policy Preregistration

Status: frozen before any live response from this interface and before any
proposal-quality endpoint is computed.

## Claim boundary

The exact UCI ann-thyroid qualification at seed 24148 passed independently. This
registration asks a narrower transfer question: can non-thinking Gemma 4 26B supply
useful observation-contingent second actions when the candidate roots are fixed by
the machine? A pass would support an LLM continuation-proposal claim, not an
end-to-end language-only Bayesian-planning claim.

The model never selects or formats the root set. Every cell has four machine-fixed
roots: blood collection plus the three strongest immediate query roots. The model
returns explicit legal action names for every positive-probability outcome branch.
There are no menu indices, no model-enforced root-diversity requirement, and no LLM
calls during exact policy scoring.

## S0 serving gate

- Seed: `24151`.
- Ten deterministic uncollected posterior cells with zero, one, or two history queries.
- Model: `google/gemma-4-26B-A4B-it`, direct vLLM on one A100, non-thinking.
- Four fixed roots, one named branch-policy response per cell, temperature zero,
  1,024-token cap, and one bounded validation-feedback retry.
- Required: ten accepted cells, exact branch coverage, all roots/follow-ups legal,
  machine-fixed collection root first, zero reasoning tokens, zero forced exits, and
  zero rollout/scoring calls.
- S0 choices are mechanics-only and are never proposal-quality evidence.
- Any S0 failure stops this interface line. There is no format-repair rerun.

## Conditional S1 proposal gate

S1 is launched only if S0 passes every serving gate.

- Fresh seed: `24152`.
- 32 distinct uncollected posterior histories from the full 7,200-row empirical prior.
- Every cell is selected mechanically before responses: exhaustive depth 2 must choose
  blood collection and have strictly lower exact two-step cost than the depth-1 root
  with its exact best continuation.
- The LLM supplies one named follow-up for every branch of every fixed root. The exact
  verifier scores all four complete branch policies and chooses the minimum-cost root.
- Controls use the identical fixed roots: matched-random named continuations, the exact
  best continuation under the depth-1 root, and exhaustive depth 2.
- Five thousand paired bootstrap replicates; all scoring and bootstraps make zero LLM calls.

Frozen S1 gates, all required:

1. The 95% paired lower bound of matched-random-minus-LLM cost is above zero.
2. The 95% paired lower bound of exact-depth-1-root-minus-LLM cost is above zero.
3. The exact verifier selects the collection-root LLM policy in at least 75% of cells.
4. Mean recovery of the exact depth-2 opportunity is at least 0.60.
5. All 32 histories are distinct exact depth-2 collection opportunities; roots are
   paired across LLM/random; every policy is exactly scored; all cells are accepted;
   reasoning, forced exits, and scoring-time LLM calls are zero.

A failed S1 stops the thyroid LLM-policy line. A pass authorizes only a separately
preregistered paired trajectory confirmation on fresh patient rows.

No OpenRouter spend is authorized by this registration; the direct-vLLM path is used.
